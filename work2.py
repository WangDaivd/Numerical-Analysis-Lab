import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def image_super_resolution(input_lr_path, input_hr_truth_path=None, scale_factor=2):
    # 读取低分辨率图像并转换为灰度图
    img_lr = cv2.imread(input_lr_path, cv2.IMREAD_GRAYSCALE)
    h, w = img_lr.shape

    # 使用最近邻插值放大
    img_nn = cv2.resize(
        img_lr,
        (w * scale_factor, h * scale_factor),
        interpolation=cv2.INTER_NEAREST
    )

    # 使用双线性插值放大
    img_bilinear = cv2.resize(
        img_lr,
        (w * scale_factor, h * scale_factor),
        interpolation=cv2.INTER_LINEAR
    )

    # 如果有真实高分辨率图像，计算质量指标
    if input_hr_truth_path:
        img_hr_truth = cv2.imread(input_hr_truth_path, cv2.IMREAD_GRAYSCALE)

        # 裁剪到相同尺寸（防止尺寸不匹配）
        img_hr_truth = cv2.resize(img_hr_truth, (img_nn.shape[1], img_nn.shape[0]))

        # 计算PSNR和SSIM
        psnr_nn = psnr(img_hr_truth, img_nn, data_range=255)
        ssim_nn = ssim(img_hr_truth, img_nn, data_range=255)

        psnr_bilinear = psnr(img_hr_truth, img_bilinear, data_range=255)
        ssim_bilinear = ssim(img_hr_truth, img_bilinear, data_range=255)
    else:
        psnr_nn, ssim_nn = None, None
        psnr_bilinear, ssim_bilinear = None, None

    # 可视化结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_nn, cmap='gray')
    plt.title(f'Nearest Neighbor\nPSNR: {psnr_nn:.2f} dB (if available)\nSSIM: {ssim_nn:.4f}')

    plt.subplot(1, 3, 2)
    plt.imshow(img_bilinear, cmap='gray')
    plt.title(f'Bilinear\nPSNR: {psnr_bilinear:.2f} dB (if available)\nSSIM: {ssim_bilinear:.4f}')

    if input_hr_truth_path:
        plt.subplot(1, 3, 3)
        plt.imshow(img_hr_truth, cmap='gray')
        plt.title('Ground Truth HR Image')

    plt.tight_layout()
    plt.show()

# 示例调用（替换为实际路径）
image_super_resolution('flower_256x256.png', 'flower.jpg', scale_factor=2)