# 路径：configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/voc2007_cocoformat_instance.py',  # 修改这里
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
evaluation = dict(interval=1, metric=['bbox', 'segm'])
