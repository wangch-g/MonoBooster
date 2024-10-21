## Introduction
This is the implementation of our IEEE Signal Processing Letters paper "MonoBooster: Semi-Dense Skip Connection with Cross-Level Attention for Boosting Self-Supervised Monocular Depth Estimation".

## Dependency
We use python 3.8.13/cuda 11.4/torch 1.10.0/torchvision 0.11.0/opencv 3.4.8 for training and evaluation.

## Data Preparation
### KITTI depth
For KITTI depth, download KITTI raw dataset from the <a href="http://www.cvlibs.net/download.php?file=raw_data_downloader.zip">script</a> provided on the official website. The data structure should be:
```
raw_data
  | 2011_09_26
  | 2011_09_28
  | 2011_09_29
  | 2011_09_30
  | 2011_10_03
```

## Training
In the main directory, run:
```
 python main.py --gpu [gpu id] --dataset kitti_raw --kitti_raw_root [/path/to/your/kitti/raw_data/root] --kitti_raw_txt ./splits/eigen_zhou/train_files.txt
```

## Evaluation
We provide the pre-trained models <a href="https://drive.google.com/drive/folders/1sxt7FkYpAZQP7ZaXYgonnK0G89w1DxXx?usp=drive_link">here</a> for evaluating.

### KITTI depth
Run the following commands to generate the ground truth files for testing in eigen split.
```
cd ./splits/eigen
python export_gt_depth.py --data_path /path/to/your/kitti/raw_data/root 
```
In the main directory, run:
```
 python eval_kitti.py --gpu [gpu id] --pretrained_model [/path/to/saved/checkpoints] --raw_base_dir [/path/to/your/kitti/raw_data/root]
```


## License
The code is released under the [MIT license](LICENSE).



## Related Projects
https://github.com/nianticlabs/monodepth2
