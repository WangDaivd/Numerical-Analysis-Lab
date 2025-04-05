# 依赖安装：pip install numpy matplotlib scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 定义原函数和生成数据
def original_curve(x):
    return 1 / (1 + x**2)

x_fine = np.linspace(-5, 5, 1000)
y_true = original_curve(x_fine)
x_nodes = np.linspace(-5, 5, 11)
y_nodes = original_curve(x_nodes)

# 拉格朗日插值实现
def lagrange_interp(x_nodes, y_nodes, x):
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if j != i:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result

y_lagrange = [lagrange_interp(x_nodes, y_nodes, x) for x in x_fine]

# 三次样条插值
cs = CubicSpline(x_nodes, y_nodes, bc_type='natural')
y_spline = cs(x_fine)

# 分段线性插值
y_linear = np.interp(x_fine, x_nodes, y_nodes)

# 误差计算
def calculate_errors(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    max_error = np.max(np.abs(y_true - y_pred))
    return mse, max_error

errors = {
    'Lagrange': calculate_errors(y_true, y_lagrange),
    'Cubic Spline': calculate_errors(y_true, y_spline),
    'Linear': calculate_errors(y_true, y_linear)
}

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(x_fine, y_true, 'k-', linewidth=2, label='y=1/(1+x*x)')
plt.plot(x_fine, y_lagrange, '--', label='lagrange')
plt.plot(x_fine, y_spline, '-.', label='spline')
plt.plot(x_fine, y_linear, ':', label='linear')
plt.scatter(x_nodes, y_nodes, c='red', s=50, label='interpolation knot', zorder=5)
plt.xlabel('x'), plt.ylabel('y')
plt.title('Comparision of Interpolation Curves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 误差曲线
plt.figure(figsize=(12, 6))
results = {
    'lagrange': y_lagrange,
    'cubic_spline': y_spline,
    'linear': y_linear
}

for method in ['Lagrange', 'Cubic Spline', 'Linear']:
    method_key = method.lower().replace(" ", "_")
    y_pred = results[method_key]
    plt.plot(x_fine, np.abs(y_true - y_pred), label=method)

plt.yscale('log')
plt.xlabel('x'), plt.ylabel('Absolute error (logarithmic scale)')
plt.title('Comparison of Interpolation Method Errors')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 打印误差表格
print("\n=== 误差分析 ===")
print("{:<12} | {:<15} | {:<15}".format('方法', 'MSE', '最大误差'))
print("-" * 45)
for method, (mse, max_err) in errors.items():
    print("{:<12} | {:<15.2e} | {:<15.2e}".format(method, mse, max_err))
