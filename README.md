<p align="center">

  <h2 align="center"><strong>DPD: Expert-Driven Visual Encoding and Spoof-Aware Text Prompting for Multimodal Face Anti-Spoofing</strong></h2>

</p>


<p align="center">
</p>



##  Updates :
- 30-01-2026: Inference code has been released, on Baidu Netdisk, share the optimal models based .
- The training code will be made publicly available on GitHub.




## Instruction for code usage :

### Setup
- Get Code
```shell
 git clone https://github.com/zlpiao/DPDFAS.git
```



## File Shared via Cloud Storage: 

best_model_run_w.pth
Link: https://pan.baidu.com/s/1Sc-7M0hhyKtqKnUcJuKJaw?pwd=k7at 提取码: k7at 

best_model_run_c.pth
Link: https://pan.baidu.com/s/1jlwKOxsJI5GJJDgfjuCU1Q?pwd=gp6m 提取码: gp6m 

best_model_run_p.pth
Link: https://pan.baidu.com/s/1sR3KHjbc6L5dVrH8Yj_lsA?pwd=jtf4 提取码: jtf4 

best_model_run_s.pth
Link: https://pan.baidu.com/s/17E6wurmLFV75fUVYNeFk7Q?pwd=wjkw 提取码: wjkw 


## Inference
```shell
python infer.py \
  --data_root /data \
  --list_dir ./lists \
  --test_domain w \
  --model_path best_model_run_w.pth
```

```shell
  python infer.py \
  --data_root /data \
  --list_dir ./lists \
  --test_domain p \
  --model_path best_model_run_p.pth
```

```shell
    python infer.py \
  --data_root /data \
  --list_dir ./lists \
  --test_domain c \
  --model_path best_model_run_c.pth
```

```shell
    python infer.py \
  --data_root /data \
  --list_dir ./lists \
  --test_domain s \
  --model_path best_model_run_s.pth
```
