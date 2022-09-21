import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def fitness_func(X):

    A = 10
    pi = np.pi
    x = X[:,0]
    y = X[:,1]
    return 2 * A + x ** 2 - A * np.cos(2 * pi * x) + y ** 2 - A * np.cos(2 * pi * y)

def velocity_update(V,X,pbest,gbest,c1,c2,w,max_val):
    size = X.shape[0]
    r1 = np.random.random((size,1))
    r2 = np.random.random((size,1))
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)

    V[V < -max_val] = -max_val
    V[V > max_val] = max_val

    return V

def position_update(X,V):

    return X + V


def pso():
    w = 1
    c1 = 2
    c2 = 2
    r1 = None
    r2 = None
    dim = 2
    size = 20
    iter_num = 1000
    max_val = 0.5
    best_fitness = float(9e10)
    fitness_value_list = []

    X = np.random.uniform(-5,5,size=(size,dim))
    V = np.random.uniform(-0.5,0.5,size=(size,dim))
    p_fitness = fitness_func(X)
    g_fitness = p_fitness.min()
    fitness_value_list.append(g_fitness)
    pbest = X
    gbest = X[p_fitness.argmin()]

    for i in range(1,iter_num):
        V = velocity_update(V,X,pbest,gbest,c1,c2,w,max_val)
        X = position_update(X,V)
        p_fitness2 = fitness_func(X)
        g_fitness2 = p_fitness2.min()

        for j in range(size):
            if p_fitness[j] > p_fitness2[j]:
                pbest[j] = X[j]
                p_fitness[j] = p_fitness2[j]
        if g_fitness > g_fitness2:
            gbest = X[p_fitness2.argmin()]
            g_fitness = g_fitness2

        fitness_value_list.append(g_fitness)

        i += 1


    print("最优值：%.5f" % fitness_value_list[-1])
    print("最优解是x=%.5f,y=%.5f"  % gbest)

    plt.plot(fitness_value_list,color='r')

    plt.title("迭代过程")


pso()
