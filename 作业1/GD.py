#输入数据和标签
x_data = [338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data = [640.,633.,619.,393.,428.,27.,193.,66.,226.,1591]

#初始值和超参数设置
b = -150
w = 0
# lr = 0.0000001
lr = 1e-6
iteration = 500000

#w_list和b_list用于记录每一轮迭代的w和b值，用于绘图
w_list = [float]*iteration
b_list = [float]*iteration
start = 0
end = len(x_data)
length = len(x_data)
losses = []
for i in range(iteration):
    loss = 0
    b_grad = 0
    w_grad = 0
    sum = 0
    #填空部分，实现梯度下降算法
    for index in range(len(x_data)):
        sum = ((x_data[index]*w + b) - y_data[index])
        loss += ((x_data[index]*w + b) - y_data[index])**2
        w_grad += 2*x_data[index]*sum
        b_grad += 2*sum
    loss /= 10.
    print("loss is :" ,loss)
    losses.append(loss)
    w = w - w_grad*lr
    b = b - b_grad*lr
    w_list[i] = w
    b_list[i] = b
#绘图部分
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm, plot
fig = plt.figure()
plt.xlim(-200,-80)
plt.ylim(-4,4)

#设置背景
xmin, xmax = xlim = -200,-80
ymin, ymax = ylim = -4,4
ax = fig.add_subplot(111, xlim=xlim, ylim=ylim,
                     autoscale_on=False)
X = [[4, 4],[4, 4],[4, 4],[1, 1]]
ax.imshow(X, interpolation='bicubic', cmap=cm.Spectral,
          extent=(xmin, xmax, ymin, ymax), alpha=1)
ax.set_aspect('auto')

#绘制每一个数据点
plt.scatter(b_list,w_list,s=2,c='black',label=(lr,iteration))
plt.legend()
plt.show()