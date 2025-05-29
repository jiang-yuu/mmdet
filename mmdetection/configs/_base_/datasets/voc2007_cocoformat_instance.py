import os

# dataset settings
dataset_type = 'CocoDataset'  # 使用 COCO 格式的 dataset 类型

# 动态获取 VOC 数据集路径
if '__file__' in globals():
    data_root = os.path.join(os.path.dirname(__file__), 'data', 'VOCdevkit')  # 动态获取VOC数据集路径
else:
    data_root = os.path.join(os.getcwd(), 'data', 'VOCdevkit')  # 当前工作目录

# VOC2007类别定义（如果需要，可以自定义类名）
class_names = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
    'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# 训练 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # 加载边界框和掩码
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# 测试 pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # 加载边界框和掩码
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# COCO 数据集标注文件路径（已转换为 JSON 格式）
annotations_file = os.path.join(data_root, 'annotations', 'instances_train2007.json')  # 训练集 JSON 文件路径
val_annotations_file = os.path.join(data_root, 'annotations', 'instances_val2007.json')  # 验证集 JSON 文件路径

# 训练 dataloader 配置
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=annotations_file,  # COCO 格式的标注文件路径
        data_prefix=dict(img=os.path.join(data_root, 'JPEGImages')),  # 图像文件夹路径
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ))

# 验证 dataloader 配置
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_annotations_file,  # COCO 格式的验证集标注文件路径
        data_prefix=dict(img=os.path.join(data_root, 'JPEGImages')),  # 图像文件夹路径
        test_mode=True,
        pipeline=test_pipeline,
    ))

# 测试 dataloader 配置
test_dataloader = val_dataloader

# 评估器配置（如果你使用的是 COCO 格式的数据，通常使用 COCO 评估器）
val_evaluator = dict(
    type='CocoMetric',  # 使用 COCO 评估器
    ann_file=val_annotations_file,  # 使用 COCO 格式的验证集标注
    metric=['bbox', 'segm'],  # 评估目标检测任务的bbox和实例分割任务的segm（掩码）
    format_only=False,
)

test_evaluator = val_evaluator  # 测试评估器与验证评估器相同