import numpy as np
import matplotlib.pyplot as plt

# 打开 testfile.txt 以读取模式
with open('testfile.txt', 'r') as file:
    # 逐行读取文件内容
    data = [line.strip().split(',') for line in file]

# 将数据分成 x 和 y 值
x_values = [float(item[0]) for item in data]
y_values = [float(item[1]) for item in data]

# 创建一个范围用于绘图的 x 值
x = np.linspace(min(x_values), max(x_values), 100)

# 创建一个子图，分为三行一列
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

# 读取并绘制三个多项式
polynomial_files = ['LSE.txt', 'steepest.txt', 'newtons.txt']
titles = ['LSE', 'Steepest', 'Newtons']  # 添加标题列表

for i, file_name in enumerate(polynomial_files):
    coefficients = np.loadtxt(file_name)
    y = np.polyval(coefficients, x)
    
    # 绘制散点图
    axes[i].scatter(x_values, y_values, label='Scatter Plot')
    
    # 绘制多项式曲线
    axes[i].plot(x, y, label=f'{coefficients.tolist()}')
    
    # 添加标签和标题
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    axes[i].set_title(titles[i])  # 设置不同的标题
    axes[i].legend()

# 调整子图的间距
plt.tight_layout()

# 显示图形
plt.show()
