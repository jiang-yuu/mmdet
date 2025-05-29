# mmdet
# 在VOC数据集上训练并测试模型 Mask R-CNN 和 Sparse R-CNN 
### 1.配置环境
训练环境：Python 3.9.7\pytorch=2.0.0\cuda=118 \torchvision=0.15.1\mmcv==2.0.0\MMDetection v3.3.0
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
#安装mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" 指详细说明，或更多的输出
# "-e" 表示在可编辑模式下安装项目，因此对代码所做的任何本地修改都会生效，从而无需重新安装。
```
### 2.数据集VOC
数据集格式如下
```
mmdetection/data/VOCdevkit/
└── VOC2007/
    ├── Annotations/
    ├── JPEGImages/
    └── ImageSets/
        └── Main/
            ├── train.txt
            ├── val.txt
```
### 3.修改configs利用VOC数据集训练和测试目标检测模型Mask R-CNN 和Sparse R-CNN
#### 训练Mask R-CNN
```
python tools/train.py configs/my_configs/mask_rcnn_r50_fpn_1x_voc.py 
```
#### 测试Mask R-CNN
```
python tools/test.py "configs/my_configs/mask_rcnn_r50_fpn_1x_voc.py" work_dirs/voc_mask_model/epoch_12.pth 
```
#### 训练Sparse R-CNN
```
python tools/train.py configs/my_configs/sparse-rcnn_r50_fpn_1x_voc.py
```
#### 测试Sparse R-CNN
```
python tools/test.py "configs/my_configs/sparse-rcnn_r50_fpn_1x_voc.py" "D:/mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc/epoch_12.pth"
```
### 4.tensorboard可视化
```
tensorboard --logdir=work_dirs
```
