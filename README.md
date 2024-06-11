# AICUP 參數組 Team 5009 Source Code

> [**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**](https://arxiv.org/abs/2206.14651)
> 
> Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky

> [!IMPORTANT]  
> **This code is based on the code released by the original author of [BoT-SORT](https://github.com/NirAharon/BoT-SORT) and modified version from Ministry of Education(Taiwan) AI competition and labeled data acquisition project[AICUP Baseline: BoT-SORT](https://github.com/ricky-696/AICUP_Baseline_BoT-SORT.git).**


## Installation

**The code was tested on Ubuntu 20.04**
 
### Setup with Conda
**Step 1.** Create Conda environment and install pytorch.
```shell
conda create -n botsort python=3.7
conda activate botsort
```
**Step 2.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).<br>
The code was tested using torch 1.11.0+cu113 and torchvision==0.12.0 

**Step 3.** Fork this Repository and clone your Repository to your device

**Step 4.** **Install numpy first!!**
```shell
pip install numpy
```

**Step 5.** Install `requirements.txt`
```shell
pip install -r requirements.txt
```

**Step 6.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

**Step 7.** Others
```shell
# Cython-bbox
pip install cython_bbox

# faiss cpu / gpu
pip install faiss-cpu
pip install faiss-gpu

# for .ipynb on local(optional)
pip install ipykernel
```

## Data Preparation

Download the [AI_CUP dataset](https://tbrain.trendmicro.com.tw/Competitions/Details/32)

### Prepare ReID Dataset

For training the ReID, detection patches must be generated as follows:   

```shell
cd <BoT-SORT_dir>

# For AICUP 
python fast_reid/datasets/generate_AICUP_patches.py --data_path train
```

### Prepare YOLOv7 Dataset

run the `yolov7/tools/AICUP_to_YOLOv7.py` by the following command:
```
cd <BoT-SORT_dir>
python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir train --YOLOv7_dir yolov7Data
```

## Training (Fine-tuning)

### Train the ReID Module for AICUP

After generating the AICUP ReID dataset as described in the 'Data Preparation' section.

```shell
cd <BoT-SORT_dir>

# For training AICUP 
python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"
```
The training results are stored by default in ```logs/AICUP/bagtricks_R50-ibn```.

> [!IMPORTANT] 
> - Here we have already trained the ReID Module and save the result on [AICUP Team 5009 Weights](https://drive.google.com/drive/folders/1RhgtsKqOjap2nYfvxrmmxtAfFsnqcz-5?usp=sharing).
> - Please download and place it in ```results/ReID/model_0058.pth```.

### Fine-tune YOLOv7 for AICUP

We use [`yolov7-d6_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt) as our pretrained weight.

``` shell
cd <BoT-SORT_dir>
python yolov7/train.py --device 0 --batch-size 6 --epochs 50 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-d6.pt' --name yolov7-AICUP --hyp data/hyp.scratch.custom.yaml
```

The training results will be saved by default at `runs/train`.

> [!IMPORTANT] 
> - Here we have already trained the YOLOv7 and save the result on [AICUP Team 5009 Weights](https://drive.google.com/drive/folders/1RhgtsKqOjap2nYfvxrmmxtAfFsnqcz-5?usp=sharing).
> - Please download and place it in ```results/Yolov7/best.pt```.

## Tracking(Demo)

> [!IMPORTANT]
> - We write the `mc_demo_yolov7.ipynb` for inference all `<timestamps>` for AICUP.
> - You should change the path in block `Inference parameters settings` if needed.

Track all `<timestamps>` by run all `mc_demo_yolov7.ipynb`.

The submission file and visualized images will be saved by default at `results/detect/<timestamp>`.

## Citation

```
@article{aharon2022bot,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```

## Acknowledgement

A large part of the codes, ideas and results are borrowed from
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT)
- [FastReID](https://github.com/JDAI-CV/fast-reid)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOv7](https://github.com/wongkinyiu/yolov7)

Thanks for their excellent work!