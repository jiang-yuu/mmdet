import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from matplotlib.patches import Rectangle
from mmdet.utils import register_all_modules

# 可视化最大得分框
def visualize_best(model, image_path, save_path, score_thr=0.5, show_proposals=False):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"无法加载图像：{image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 获取预测结果
    result = inference_detector(model, image_path)
    pred_instances = result.pred_instances
    if hasattr(result, 'proposals') and show_proposals:
        proposals = result.proposals.cpu().numpy()
    else:
        proposals = None

    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy() if hasattr(pred_instances, 'masks') else None

    class_names = model.dataset_meta['classes']

    # 找到得分最大的框
    max_score_index = np.argmax(scores)
    max_bbox = bboxes[max_score_index]
    max_score = scores[max_score_index]
    max_label = labels[max_score_index]
    caption = f'{class_names[max_label]}: {max_score:.2f}'

    # 创建图像并绘制
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.axis('off')

    # 可视化 proposal（第一阶段）
    if proposals is not None:
        for p in proposals[:100]:  # 只画前100个 proposal
            x1, y1, x2, y2 = p
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor='cyan', facecolor='none', linewidth=1, linestyle='--'))

    # 可视化得分最大的框
    x1, y1, x2, y2 = max_bbox
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                           edgecolor='lime', facecolor='none', linewidth=2))
    ax.text(x1, y1 - 2, caption, color='white', fontsize=9,
            bbox=dict(facecolor='green', alpha=0.5, pad=0))

    # 掩码可视化（如果有）
    if masks is not None:
        mask = masks[max_score_index].astype(bool)
        color = np.random.rand(3)
        image_rgb[mask] = image_rgb[mask] * 0.5 + np.array(color) * 255 * 0.5

    ax.imshow(image_rgb)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"保存至：{save_path}")

# 加载模型
def load_model(config_path, checkpoint_path, device='cuda:0'):
    register_all_modules()
    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, checkpoint_path, device=device)
    return model

# 主函数
if __name__ == "__main__":
    # 图像路径
    test_images = [
        'data/VOCdevkit/VOC2007/JPEGImages/000012.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000142.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000036.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000039.jpg'
    ]

    # 模型路径（替换为你训练好的路径）
    mask_config = 'configs/my_configs/mask_rcnn_r50_fpn_1x_voc.py'
    mask_ckpt = 'work_dirs/voc_mask_model/epoch_12.pth'

    sparse_config = 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'
    sparse_ckpt = 'work_dirs/sparse-rcnn_r50_fpn_1x_voc/epoch_12.pth'

    # 加载模型
    mask_model = load_model(mask_config, mask_ckpt)
    sparse_model = load_model(sparse_config, sparse_ckpt)

    for img_path in test_images:
        base_name = os.path.basename(img_path)

        # 可视化 Mask R-CNN（仅绘制得分最大的框）
        visualize_best(mask_model, img_path, f'outputs/mask_rcnn_best/{base_name}', score_thr=0.5, show_proposals=True)

        # 可视化 Sparse R-CNN（仅绘制得分最大的框）
        visualize_best(sparse_model, img_path, f'outputs/sparse_rcnn_best/{base_name}', score_thr=0.5, show_proposals=False)
