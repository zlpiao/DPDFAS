# =========================================================
# infer.py
# Inference-only code for multimodal FAS (review version)
# =========================================================

import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import clip
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from utils.statistic import get_EER_states, get_HTER_at_thr


# -----------------------
# Args
# -----------------------
def get_args():
    parser = argparse.ArgumentParser("Multimodal FAS Inference")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--list_dir', type=str, required=True)
    parser.add_argument('--test_domain', type=str, required=True,
                        choices=['c', 'p', 's', 'w'])
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained .pth model')
    parser.add_argument('--model_name', type=str, default='ViT-B/16')
    parser.add_argument('--test_modality', type=str, default='rgb_ir_depth',
                        choices=['rgb_ir_depth', 'rgb_ir', 'rgb_depth', 'rgb'])
    parser.add_argument('--temp', type=float, default=0.07)
    return parser.parse_args()


args = get_args()
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Transforms
# -----------------------
transform_rgb_ir = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )
])
transform_depth = transform_rgb_ir


def load_rgb(path):
    return Image.open(path).convert('RGB')


def load_ir(path):
    return Image.open(path).convert('RGB')


def load_depth(path):
    return Image.open(path).convert('RGB')


# -----------------------
# Model (STRUCTURE-ALIGNED)
# -----------------------
class MultimodalCLIP(nn.Module):
    """
    NOTE:
    This model definition is STRUCTURE-ALIGNED with the training code.
    Training-related logic is NOT included.
    """
    def __init__(self, model_name='ViT-B/16', device='cuda', temp=0.07):
        super().__init__()
        self.device = device

        # CLIP
        self.clip_model, _ = clip.load(model_name, device=device, jit=False)
        for name, p in self.clip_model.named_parameters():
            if not name.startswith('visual'):
                p.requires_grad = False

        self.clip_dim = getattr(self.clip_model.visual, 'output_dim', 512)

        # Depth branch
        resnet = models.resnet18(pretrained=False)
        self.depth_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.depth_proj = nn.Linear(512, self.clip_dim)

        # Fusion modules (same as training)
        self.fuse_rgb_ir = nn.Sequential(
            nn.Linear(self.clip_dim * 2, self.clip_dim),
            nn.Sigmoid()
        )
        self.fuse_depth = nn.Sequential(
            nn.Linear(self.clip_dim * 2, self.clip_dim),
            nn.Sigmoid()
        )

        # Auxiliary classifier (required for state_dict compatibility)
        self.classifier = nn.Linear(self.clip_dim, 2)

        # Temperature (buffer)
        self.temp = nn.Parameter(torch.tensor(temp), requires_grad=False)

    @torch.no_grad()
    def encode_rgb_ir(self, rgb, ir):
        b = rgb.size(0)
        x = torch.cat([rgb, ir], dim=0)
        f = self.clip_model.encode_image(x)
        rgb_f, ir_f = f[:b], f[b:]
        g = self.fuse_rgb_ir(torch.cat([rgb_f, ir_f], dim=1))
        return rgb_f * g + ir_f * (1 - g)

    @torch.no_grad()
    def encode_depth(self, depth):
        x = self.depth_backbone(depth).flatten(1)
        return self.depth_proj(x)

    @torch.no_grad()
    def forward(self, rgb=None, depth=None, ir=None, test_modality='rgb_ir_depth'):
        if test_modality == 'rgb_ir_depth':
            fused = self.encode_rgb_ir(rgb, ir)
            d = self.encode_depth(depth)
            g = self.fuse_depth(torch.cat([fused, d], dim=1))
            return fused * g + d * (1 - g)
        elif test_modality == 'rgb_ir':
            return self.encode_rgb_ir(rgb, ir)
        elif test_modality == 'rgb_depth':
            rgb_f = self.clip_model.encode_image(rgb)
            d = self.encode_depth(depth)
            g = self.fuse_depth(torch.cat([rgb_f, d], dim=1))
            return rgb_f * g + d * (1 - g)
        elif test_modality == 'rgb':
            return self.clip_model.encode_image(rgb)
        else:
            raise ValueError(f"Invalid modality: {test_modality}")


# -----------------------
# Text templates (paper)
# -----------------------
spoof_templates = [
    'This is an example of a spoof face',
    'This is an example of an attack face',
    'This is not a real face',
    'This is how a spoof face looks like',
    'a photo of a spoof face',
]
real_templates = [
    'This is an example of a real face',
    'This is a bonafide face',
    'This is a real face',
    'This is how a real face looks like',
    'a photo of a real face',
]


# -----------------------
# Dataset (test-only)
# -----------------------
class TestDataset:
    def __init__(self, list_file, data_root):
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                rgb, depth, ir, label = parts[:4]
                self.samples.append((
                    os.path.join(data_root, rgb),
                    os.path.join(data_root, depth),
                    os.path.join(data_root, ir),
                    int(label)
                ))


# -----------------------
# Evaluation
# -----------------------
@torch.no_grad()
def evaluate(model, dataset):
    model.eval()

    # text features
    spoof_tok = clip.tokenize(spoof_templates).to(device)
    real_tok = clip.tokenize(real_templates).to(device)

    spoof_feat = model.clip_model.encode_text(spoof_tok).mean(0, keepdim=True)
    real_feat = model.clip_model.encode_text(real_tok).mean(0, keepdim=True)

    txt_feat = torch.cat([spoof_feat, real_feat], dim=0)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    scores, labels = [], []

    for rgb_p, depth_p, ir_p, y in dataset.samples:
        rgb = transform_rgb_ir(load_rgb(rgb_p)).unsqueeze(0).to(device)
        ir  = transform_rgb_ir(load_ir(ir_p)).unsqueeze(0).to(device)
        dp  = transform_depth(load_depth(depth_p)).unsqueeze(0).to(device)

        feat = model(rgb=rgb, depth=dp, ir=ir, test_modality=args.test_modality)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        logits = (feat @ txt_feat.T) / model.temp
        prob_real = torch.softmax(logits, dim=1)[0, 1].item()

        scores.append(prob_real)
        labels.append(y)

    scores = np.array(scores)
    labels = np.array(labels)

    eer, thr, _, _ = get_EER_states(scores, labels)
    hter = get_HTER_at_thr(scores, labels, thr)
    auc = roc_auc_score(labels, scores)

    print(f"HTER = {hter:.4f}, AUC = {auc:.4f}")


# -----------------------
# Main
# -----------------------
def main():
    domain_map = {
        'c': 'cefa_all.txt',
        'p': 'padisi_all.txt',
        's': 'surf_all.txt',
        'w': 'wmca_all.txt'
    }

    list_file = os.path.join(args.list_dir, domain_map[args.test_domain])
    dataset = TestDataset(list_file, args.data_root)

    model = MultimodalCLIP(
        model_name=args.model_name,
        device=device,
        temp=args.temp
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if unexpected:
        print("⚠️ Unexpected keys:", unexpected)
    if missing:
        print("⚠️ Missing keys:", missing)

    evaluate(model, dataset)


if __name__ == "__main__":
    main()
