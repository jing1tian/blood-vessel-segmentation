# blood-vessel-segmentation
Training code for [SenNet + HOA - Hacking the Human Vasculature in 3D](https://www.kaggle.com/competitions/blood-vessel-segmentation/) (1st place solution).

It's recommended to save memmap files to shared memory(/dev/shm) to accelerate 3d rotation. 

```
# generate memmap files
python prepare_data.py \
    -s [dataset path] \ 
    -o [save dir]

# train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --memmap_dir [memmap path]
```