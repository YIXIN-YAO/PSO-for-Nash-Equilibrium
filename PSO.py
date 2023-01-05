import numpy as np
import random
import matplotlib.pyplot as plt

# 【余谦，王先甲． 基于粒子群优化算法求解纳什均衡的演化算法[J]． 武汉大学学报，2006，52(1):25- 29】 第三节例1
# Payoff_Matrix=np.array([[3,1,6],[0,0,4],[1,2,5]])
# Payoff_Matrix2=np.array([[3,0,1],[1,0,2],[6,4,5]])

# 【余谦，王先甲． 基于粒子群优化算法求解纳什均衡的演化算法[J]． 武汉大学学报，2006，52(1):25- 29】 第三节例2
# Payoff_Matrix=np.array([[1,235,0,0.1],[0,1,235,0.1],[235,0,1,0.1],[1.1,1.1,1.1,0]])
# Payoff_Matrix2=np.array([[1,0,235,1.1],[235,1,0,1.1],[0,235,1,1.1],[0.1,0.1,0.1,0]])

# 【余谦，王先甲． 基于粒子群优化算法求解纳什均衡的演化算法[J]． 武汉大学学报，2006，52(1):25- 29】 第三节例3
# Payoff_Matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
# Payoff_Matrix2 = np.array([[0,1,0],[0,0,1],[1,0,0]])

# 我的论文3.3.2节中求的普通支付矩阵。我论文里当时写的结果不一定是全局最优，以算法多次运行得到的稳定结果为准。
# from PayoffMatrix import Payoff_Matrix, Payoff_Matrix2

# 我的论文4.3.3节中求的相对贴进度矩阵。我论文里当时写的结果不一定是全局最优，以算法多次运行得到的稳定结果为准。
from Relative_Entropy import TJD1, TJD2
Payoff_Matrix, Payoff_Matrix2 = TJD1, TJD2

m = n = Payoff_Matrix.shape[0]
# 粒子群算法超参数。注意粒子群算法并不能保证求得全局最优解，可以通过调节超参数，增大粒子群规模和最大迭代次数提高求解稳定性，代价是耗时更长
size = 20  # 粒子群规模
iter_num = 1000  # 最大迭代次数
x_max = 2  # 粒子位置范围
max_vel = 0.3  # 粒子最大速度
wmax = 0.9  # 惯性权重最大值
wmin = 0.4  # 惯性权重最小值


def fit_fun(X, m, n):  # 适应函数
    x = np.zeros(shape=(1, m))
    y = np.zeros(shape=(1, n))
    for i in range(m):
        x[0][i] = X[i]
    for i in range(m, m + n):
        y[0][i - m] = X[i]

    return max(max(np.dot(Payoff_Matrix[i], y.T) - np.dot(np.dot(x, Payoff_Matrix), y.T) for i in range(m)), 0) \
           + max(max(np.dot(x, Payoff_Matrix2[:, i]) - np.dot(np.dot(x, Payoff_Matrix2), y.T) for i in range(n)), 0)


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim, m, n):
        self.__pos = [random.uniform(1, 100) for i in range(dim)]
        # 粒子的位置初始化在0-矩阵最大行数。这只适用于敌我策略数量一致的情况，否则要单独给第二维赋值
        he1 = 0
        he2 = 0
        for i in range(m):
            he1 = self.__pos[i] + he1
        for i in range(m, dim):
            he2 = self.__pos[i] + he2

        for i in range(m):
            self.__pos[i] = self.__pos[i] / he1
        for i in range(m, dim):
            self.__pos[i] = self.__pos[i] / he2
        # print(self.__pos)

        self.__vel = [random.uniform(-max_vel, max_vel) for i in range(dim)]  # 粒子的速度
        he11 = 0
        he22 = 0
        for i in range(m - 1):
            he11 = self.__vel[i] + he11
        self.__vel[m - 1] = -he11
        for i in range(m, dim - 1):
            he22 = self.__vel[i] + he22
        self.__vel[dim - 1] = -he22  # 让初始化的速度和为0

        self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos, m, n)  # 适应度函数值

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, i, value):
        self.__bestPos[i] = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, i, value):
        self.__vel[i] = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, dim, size, iter_num, x_max, max_vel, m, n, wmax, wmin, best_fitness_value=float('Inf'), C1=2,
                 C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max  # 粒子位置范围
        self.max_vel = max_vel  # 粒子最大速度
        self.m = m
        self.n = n
        self.best_fitness_value = best_fitness_value  # 最优适应度，初始值为无限大
        self.best_position = [0.0 for i in range(dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim, self.m, self.n) for i in range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part, iter_num_now):

        for i in range(self.dim):
            vel_value = (wmax - iter_num_now * (wmax - wmin) / self.iter_num) * part.get_vel()[
                i] + self.C1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])

            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        aa = 99999999.0
        pos_value_sum1 = 0
        pos_value_sum2 = 0
        for i in range(self.dim):

            pos_value = part.get_pos()[i] + part.get_vel()[i]
            if pos_value < 0:

                x = -part.get_pos()[i] / part.get_vel()[i]

                if x >= 0 and x < aa:
                    aa = x

        if aa == 99999999:
            for i in range(self.m):
                pos_value = part.get_pos()[i] + part.get_vel()[i]
                pos_value_sum1 = pos_value_sum1 + pos_value
            for i in range(self.m, self.dim):
                pos_value = part.get_pos()[i] + part.get_vel()[i]
                pos_value_sum2 = pos_value_sum2 + pos_value

            for i in range(self.m):
                value = (part.get_pos()[i] + part.get_vel()[i]) / pos_value_sum1
                if value < 0:
                    value = 0

                part.set_pos(i, value)

            for i in range(self.m, self.dim):
                value = (part.get_pos()[i] + part.get_vel()[i]) / pos_value_sum2
                if value < 0:
                    value = 0
                # 由于计算精度的问题，这个值可能小于0，小于0的数存进去会对算法产生很大影响，负数会被不断放大
                # 改这个bug花了我3个小时，现在是5.26凌晨两点，5.27就交论文了。
                part.set_pos(i, value)

        else:
            for i in range(self.m):
                pos_value = part.get_pos()[i] + aa * part.get_vel()[i]
                pos_value_sum1 = pos_value_sum1 + pos_value
            for i in range(self.m, self.dim):
                pos_value = part.get_pos()[i] + aa * part.get_vel()[i]
                pos_value_sum2 = pos_value_sum2 + pos_value
            # 为归一化处理计算和
            for i in range(self.m):
                value = (part.get_pos()[i] + aa * part.get_vel()[i]) / pos_value_sum1
                if value < 0:
                    value = 0
                part.set_pos(i, value)
                vel_value = aa * part.get_vel()[i]
                part.set_vel(i, vel_value)  # 把更改后的速度存进去
            for i in range(self.m, self.dim):
                value = (part.get_pos()[i] + aa * part.get_vel()[i]) / pos_value_sum2
                if value < 0:
                    value = 0
                part.set_pos(i, value)  # 这么长一段其实就是个归一化处理
                vel_value = aa * part.get_vel()[i]
                part.set_vel(i, vel_value)  # 把更改后的速度存进去

        value = fit_fun(part.get_pos(), self.m, self.n)
        if value < part.get_fitness_value():  # 如果适应度小于当前粒子最优适应度，则更新该粒子的最优值和最优位置
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])

        if value < self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            # print(value,end='    ')
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])
            #     print(part.get_pos()[i],end=' ')
            # print()

    def update(self):
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part, i)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
        return self.fitness_val_list, self.get_bestPosition()


dim = m + n

pso = PSO(dim, size, iter_num, x_max, max_vel, m, n, wmax, wmin)
fit_var_list, best_pos = pso.update()
best_pos = np.array(best_pos)
# best_pos中小于1e-2的值赋0
best_pos[best_pos < 1e-2] = 0.0
print('我方最优策略：', end=' ')
for i in range(m):
    print(best_pos[i], end=' ')
print()
print('敌方最优策略：', end=' ')
for i in range(m, dim):
    print(best_pos[i], end=' ')
print()
print("最优解:" + str(fit_var_list[-1]))
ax = np.zeros(shape=(iter_num))
for i in range(iter_num):
    ax[i] = float(fit_var_list[i])
plt.plot(np.linspace(0, iter_num, iter_num), ax, c="r", alpha=0.5)
plt.xlabel("iterations")
plt.ylabel("fitness")
plt.show()
