import numpy as np
import matplotlib.pyplot as plt

#函数及导数
def origin(x):

    return x**4+2*x**3-3*x**2-2*x
def dao(x):
    return 4*x**3+6*x**2-6*x-2





# 梯度下降法模拟
result_x=[]
result_y=[]
history_theta=[]
start:float=-4

learn=0.001
step=0.01
def gradient_d(start,end,learn,derta):
    theta=start

    flag = 0
    # print(theta)
    while flag < end:
        flag += 1
        history_theta.append(theta)
        #print(flag)
        last = theta
        gradient = dao(theta)
        theta = theta - (learn * gradient)
        if abs(dao(theta)) < derta:
            if len(result_x)>0 and abs(theta - result_x[-1])>=10e-3:
                result_x.append((theta))
                result_y.append(origin(theta))
            elif len(result_x)==0:
                result_x.append((theta))
                result_y.append(origin(theta))

            break


x=np.zeros(5000)
x=np.linspace(-4,3,5000)
y=np.zeros(5000)
y=origin(x)
while(start<3):
    gradient_d(start,1e4,learn,1e-8)
    start+=step
history_theta=np.array(history_theta)
history=np.zeros(len(history_theta))
history=origin(history_theta)
print(f"result x: {result_x}")
print(f"result y: {result_y}")
print(history_theta)
plt.style.use("seaborn-darkgrid")
plt.plot(result_x,result_y,'*',color='b')
plt.plot(x,y,color='r',linestyle=':')
plt.plot(history_theta,history,'+',color="g")
flag=0
while(flag<len(result_x)):

    plt.text(result_x[flag],result_y[flag],s=f"{round(result_x[flag],5)},{round(result_y[flag],5)}",fontsize=10)
    flag+=1
plt.show()




