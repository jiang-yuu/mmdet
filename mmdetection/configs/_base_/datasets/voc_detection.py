import os

# dataset settings
dataset_type = 'CocoDataset'  # 使用 COCO 风格的数据集加载器，假设你的 JSON 文件符合 COCO 格式
data_root = 'data/VOCdevkit/'  # VOC 数据集路径

# VOC2007类别定义（对于 COCO 风格，确保类别名称正确）
class_names = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'potted plant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

backend_args = None

# 数据预处理
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # 加载边界框和掩码
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # 加载边界框和掩码
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 训练数据加载器
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=os.path.join(data_root, 'annotations', 'instances_train2007.json'),  # 使用 COCO 格式的训练标注文件
        data_prefix=dict(img=os.path.join(data_root, 'VOC2007', 'JPEGImages')),  # VOC 图像所在路径
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
        # class_names 不需要传递给 dataset，应该在模型里配置
    ))

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=os.path.join(data_root, 'annotations', 'instances_val2007.json'),  # 使用 COCO 格式的验证标注文件
        data_prefix=dict(img=os.path.join(data_root, 'VOC2007', 'JPEGImages')),  # VOC 图像所在路径
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        # class_names 不需要传递给 dataset，应该在模型里配置
    ))

test_dataloader = val_dataloader  # 测试数据集与验证数据集相同

# 评估器
val_evaluator = dict(
    type='CocoMetric',  # 使用 COCO 风格的评估方法
    ann_file=os.path.join(data_root, 'annotations', 'instances_val2007.json'),  # 使用 COCO 格式的验证集标注文件
    metric=['bbox','segm'],  # 评估目标检测任务的bbox和实例分割任务的segm（掩码）
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator  # 测试评估器与验证评估器相同
