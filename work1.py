import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ==================== 一维插值方法实现 ====================

def lagrange_interp(x_nodes, y_nodes, x):
    """拉格朗日插值"""
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if j != i:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result


def newton_divided_differences(x_nodes, y_nodes):
    """构建牛顿差商表"""
    n = len(x_nodes)
    table = np.zeros((n, n))
    table[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_nodes[i + j] - x_nodes[i])
    return table


def newton_interp(x_nodes, y_nodes, x):
    """牛顿插值"""
    table = newton_divided_differences(x_nodes, y_nodes)
    n = len(x_nodes)
    result = table[0][0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (x - x_nodes[i - 1])
        result += table[0][i] * product_term
    return result


def piecewise_hermite(x_nodes, y_nodes, derivatives, x):
    """分段三次埃尔米特插值"""
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            h = x_nodes[i + 1] - x_nodes[i]
            t = (x - x_nodes[i]) / h
            return ((1 - 3 * t ** 2 + 2 * t ** 3) * y_nodes[i] +
                    (3 * t ** 2 - 2 * t ** 3) * y_nodes[i + 1] +
                    h * t * (1 - t) ** 2 * derivatives[i] +
                    h * t ** 2 * (t - 1) * derivatives[i + 1])
    return 0


# ==================== 数据生成与插值测试 ====================

# 生成原始数据
x_nodes = np.linspace(0, 2 * np.pi, 10)
y_nodes = np.sin(x_nodes)
derivatives = np.cos(x_nodes)  # 用于埃尔米特插值的导数值

# 生成密集采样点用于绘图
x_fine = np.linspace(0, 2 * np.pi, 200)
y_true = np.sin(x_fine)

# 计算各插值方法结果
y_lagrange = [lagrange_interp(x_nodes, y_nodes, x) for x in x_fine]
y_newton = [newton_interp(x_nodes, y_nodes, x) for x in x_fine]
y_linear = np.interp(x_fine, x_nodes, y_nodes)
cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
y_spline = cs(x_fine)
pchip = PchipInterpolator(x_nodes, y_nodes)
y_hermite = pchip(x_fine)


# ==================== 误差计算 ====================

def calculate_errors(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    max_error = np.max(np.abs(y_true - y_pred))
    return mse, max_error


errors = {
    'Lagrange': calculate_errors(y_true, y_lagrange),
    'Newton': calculate_errors(y_true, y_newton),
    'Linear': calculate_errors(y_true, y_linear),
    'Spline': calculate_errors(y_true, y_spline),
    'Hermite': calculate_errors(y_true, y_hermite)
}

# ==================== 可视化结果 ====================

plt.figure(figsize=(12, 8))

# 绘制插值曲线对比
plt.subplot(2, 1, 1)
plt.plot(x_fine, y_true, 'k-', label='True Function')
plt.plot(x_fine, y_lagrange, '--', label='Lagrange')
plt.plot(x_fine, y_newton, '-.', label='Newton')
plt.plot(x_fine, y_linear, ':', label='Linear')
plt.plot(x_fine, y_spline, '-', label='Cubic Spline')
plt.plot(x_fine, y_hermite, '-', label='Hermite')
plt.scatter(x_nodes, y_nodes, c='red', label='Nodes')
plt.title('Interpolation Comparison')
plt.legend()
plt.grid(True)

# 绘制误差曲线
plt.subplot(2, 1, 2)
for method in errors:
    plt.plot(x_fine, np.abs(y_true - eval(f'y_{method.lower()}')), label=method)
plt.title('Absolute Error Comparison')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==================== 图像超分辨率插值 ====================

# 读取并处理图像
def image_interpolation_demo(input_path):
    img_low = Image.open(input_path).convert('L')
    img_low = np.array(img_low)

    # 插值放大
    scale_factor = 2
    h, w = img_low.shape
    img_nn = cv2.resize(img_low, (w * scale_factor, h * scale_factor),
                        interpolation=cv2.INTER_NEAREST)
    img_bilinear = cv2.resize(img_low, (w * scale_factor, h * scale_factor),
                              interpolation=cv2.INTER_LINEAR)

    # 计算质量指标（假设有真实高分辨率图像）
    img_high = cv2.resize(img_low, (w * scale_factor, h * scale_factor),
                          interpolation=cv2.INTER_CUBIC)  # 模拟真实HR

    psnr_nn = peak_signal_noise_ratio(img_high, img_nn)
    ssim_nn = structural_similarity(img_high, img_nn)

    psnr_bilinear = peak_signal_noise_ratio(img_high, img_bilinear)
    ssim_bilinear = structural_similarity(img_high, img_bilinear)

    # 显示结果
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_nn, cmap='gray')
    plt.title(f'Nearest Neighbor\nPSNR: {psnr_nn:.2f} dB\nSSIM: {ssim_nn:.4f}')

    plt.subplot(1, 3, 2)
    plt.imshow(img_bilinear, cmap='gray')
    plt.title(f'Bilinear\nPSNR: {psnr_bilinear:.2f} dB\nSSIM: {ssim_bilinear:.4f}')

    plt.subplot(1, 3, 3)
    plt.imshow(img_high, cmap='gray')
    plt.title('Original High Resolution')

    plt.tight_layout()
    plt.show()


# 使用示例（需替换为实际图像路径）
# image_interpolation_demo('input_image.jpg')

# ==================== 误差分析表格 ====================
print("\n=== 一维插值误差分析 ===")
print("{:<10} | {:<15} | {:<15}".format('Method', 'MSE', 'Max Error'))
print("-" * 40)
for method, (mse, max_err) in errors.items():
    print("{:<10} | {:<15.4e} | {:<15.4e}".format(method, mse, max_err))