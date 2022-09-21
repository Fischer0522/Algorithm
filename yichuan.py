import numpy as np
import  matplotlib.pyplot as plt

def fitness_func(X):
    a = 10
    pi =np.pi
    x = X[:,0]
    y = X[:,1]
    return 2 * a + x ** 2 - a * np.cos(2 * pi * x) + y ** 2 - a * np.cos(2 * pi * y)

def decode(x,a,b):
    xt = 0
    for i in range (len(x)):
        xt = xt + x[i] * np.power(2,i)
    # 归一化解码
    return a + xt * (b - a) / np.power(2,len(x) - 1)

def decode_X(X:np.array):
    # shape获取维数，0 为行数 1为列数
    X2 = np.zeros((X.shape[0],2))
    for i in range(X.shape[0]):
        # 前20位位变量x 后20位为变量y
        xi = decode(X[i,:20],-5,5)
        yi = decode(X[i,20:],-5,5)
        X2[i,:] = np.array([xi,yi])

    return X2

def select(X,fitness):
    fitness = 1/fitness
    fitness = fitness/fitness.sum()
    idx = np.array(list(range(X.shape[0])))

    # X2_idx为选出的个体的索引
    X2_idx = np.random.choice(idx,size=X.shape[0],p=fitness)
    # 通过索引获取到个体
    X2 = X[X2_idx,:]
    return X2

def crossover(X,c):
    # 步长为2，则取出来的为一对，令这一对进行交叉
    for i in range(0,X.shape[0],2):
        xa = X[i,:]
        xb = X[i + 1,:]
        for j in range(X.shape[1]):
            if np.random.rand() <= c:
                xa[j],xb[j] = xb[j],xa[j]

        X[i,:] = xa
        X[i+1,:] = xb
    return X

def mutation(X,m):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.random.rand() <= m:
                # 0 1 转换
                X[i,j] = (X[i,j] + 1) % 2
    return X


def ga():
    c = 0.3
    m = 0.05
    best_fitness = []
    best_xy = []
    iter_num = 100
    # 生成随机的染色体，基因为要么为0要么为1
    X0 = np.random.randint(0,2,(50,40))
    # 迭代100次
    for i in range(iter_num):
        X1 = decode_X(X0)
        fitness = fitness_func(X1)
        X2 = select(X0,fitness)
        X3 = crossover(X2,c)
        X4 = mutation(X3,m)
        X5 = decode_X(X4)
        fitness = fitness_func(X5)
        best_fitness.append(fitness.min())
        x, y = X5[fitness.argmin()]
        best_xy.append((x,y))
        # 继续下一轮
        X0 = X4
    print("最优值：%.5f" % best_fitness[-1])

    print("最优解x=%.5f,y=%.5f" % best_xy[-1])

    plt.plot(best_fitness,color = "r")
    plt.show()

ga()
