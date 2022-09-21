import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")

plt.rcParams['font.sans-serif'] = ['Simsun']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 原函数
def func(x):
    return x ** 3 + 0.2 * x ** 2 + 2 * x


# 预测形式
def predict(a0, a1, a2, x):
    return a0 + a1 * x + a2 * pow(x, 2)


# 梯度下降法确定拟合系数
def gradient_d(end, learn, data_x, data_y, a0, a1, a2):
    """
    计算总数据量，算出各个系数

    :param end: 迭代次数
    :param learn: 学习率
    :param data_x: x轴数据集
    :param data_y: y轴数据集
    :param a0: 常数项
    :param a1: 一次
    :param a2: 二次
    :param a3: 三次
    :return:
    """
    for item in range(end):
        a0_t = 0
        a1_t = 0
        a2_t = 0
        # a3_t = 0
        m = float(len(data_x))
        # 损失函数对各个系数的求导，累加和的求导转换为求导的累加和
        # 损失函数方便求导时多定义了个1/2，此处刚好抵消
        for i in range(len(data_x)):
            a0_t += (a0 + a1 * data_x[i] + a2 * pow(data_x[i], 2)) - data_y[i]
            a1_t += ((a0 + a1 * data_x[i] + a2 * pow(data_x[i], 2)) - data_y[i]) * data_x[i]
            a2_t += ((a0 + a1 * data_x[i] + a2 * pow(data_x[i], 2)) - data_y[i]) * pow(data_x[i], 2)

        # 导数的最终结果
        a0_t = a0_t / m
        a1_t = a1_t / m
        a2_t = a2_t / m
        # 向损失函数减小的方向更新
        a0 = a0 - (learn * a0_t)
        a1 = a1 - (learn * a1_t)
        a2 = a2 - (learn * a2_t)

    return a0, a1, a2


# 最小二乘法确定拟合参数


def ls(x_data, y_data):
    # 二阶拟合，1000个数据，规定A为[1000,3]的矩阵
    A = np.zeros((3, 1000))
    for i in range(3):
        A[i] = pow(x_data, i)
    A = A.T
    A_T = A.T
    b = np.zeros(1000)
    b = y_data
    b = b.T

    x = np.zeros((3, 1))
    x = np.dot(np.linalg.inv(np.dot(A_T, A)), (np.dot(A_T, b)))

    return x


# 规定参数
learn = 0.001
a0 = 0
a1 = 0
a2 = 0

# 数据初始化
x = np.zeros(1000)
x = np.linspace(-1, 3, num=1000)

y = np.zeros(1000)
y = func(x)

noise = np.random.randn(1000)
after_noise = y + noise * 5

# 写入
origin = np.array([x, after_noise])
dataNew = 'D:\\learn_software\\program_file\\PycharmProject\\class-03\\original_data.mat'
scio.savemat(dataNew, {'original_data': origin})

# 数据导入
dataFile = 'D:\\learn_software\\program_file\\PycharmProject\\class-03\\original_data.mat'
data = scio.loadmat(dataFile)
original_data = data['original_data']
x_data = original_data[0]
y_data = original_data[1]
print(original_data)

# 最小二乘法计算参数
theta = []
theta = ls(x_data, y_data)
predict_ls = np.zeros(1000)
predict_ls = predict(theta[0], theta[1], theta[2], x_data)

# 梯度下降法多项式拟合
print("running...")
a0, a1, a2 = gradient_d(1000, learn, x, after_noise, a0, a1, a2)
after = predict(a0, a1, a2, x)

# 结果的处理与比较
sum_gradient = 0
average_gradient = 0
sum_ls = 0
average_ls = 0
for i in range(1000):
    sum_gradient += (predict(a0, a1, a2, x[i]) - func(x[i])) ** 2 / 1000
    average_gradient += abs(predict(a0, a1, a2, x[i]) - func(x[i])) / 1000
    sum_ls += (predict_ls[i] - func(x[i])) ** 2 / 1000
    average_ls += abs(predict_ls[i] - func(x[i])) / 1000
print(f"梯度下降法方差为:{sum_gradient}")
print(f"梯度下降法误差均值为:{average_gradient}")
print(f"最小二乘法方差为:{sum_ls}")
print(f"最小二乘法误差均值为:{average_ls}")
gd = [a0, a1, a2]
print(f"梯度下降法拟合系数{gd}")
print(f"最小二乘法拟合系数{theta}")

# 可视化
fig = plt.figure(num='fig1', figsize=(9, 5), dpi=80, edgecolor='y', linewidth=10, frameon=True)

plt.subplot(311)
plt.plot(x, y_data, color='g', linestyle=':', label='原始数据')
plt.plot(x, after, color='r', linestyle='-.', label='最梯度下降法拟合')
plt.plot(x, predict_ls, color='b', linestyle='dashed', label='最小二乘法拟合')
plt.legend(loc='lower right', fontsize=8)
plt.subplot(312)
plt.plot(x, y_data, color="b", linestyle=':', label='原始数据')
plt.plot(x, after, color='r', linestyle='-.', label='梯度下降法拟合')
plt.legend(loc='lower right', fontsize=8)
plt.subplot(313)
plt.plot(x, predict_ls, color="r", linestyle='-.', label='原始数据')
plt.plot(x, y_data, color='b', linestyle=':', label='最小二乘法拟合')
plt.legend(loc='lower right', fontsize=8)
plt.show()
