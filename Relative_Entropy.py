# 这份代码对应论文的第四章，引入不确定信息，最后得到相对贴近度矩阵
import numpy as np
import math

np.set_printoptions(suppress=True)  # 不用科学计数法输出
np.set_printoptions(linewidth=400)  # 输出不自动换行
Vi = [0, 80, 71]# 我方价值
Vj = [
    [0, 0],
    [125, 127],
    [126, 128]
]  # 敌方价值
Vaa = 25.2  # 25.2  # 我方导弹价值
Vbb = [42.5, 44.5]  # 42.5  # 敌方导弹价值

num_w = 2
num_d = 2
# Payoff_Matrix_min = np.zeros(shape=(9, 9, 2))
Payoff_Matrix = [[[0 for c in range(2)] for x in range(9)] for y in range(9)]  # 初始化三维数组
Payoff_Matrix2 = [[[0 for c in range(2)] for x in range(9)] for y in range(9)]

# Payoff_Matrix_max = np.zeros(shape=(9, 9))
Pij = [
    [[0, 0], [0, 0], [0, 0]],
    [[0, 0], [0.86, 0.88], [0.65, 0.80]],  # 我方1号无人机对敌杀伤概率.三维数组
    [[0, 0], [0.88, 0.90], [0.86, 0.88]],  # 我方2号无人机对敌杀伤概率
    # 敌方在攻击状态
    [[0, 0], [0.85, 0.87], [0.50, 0.67]],
    [[0, 0], [0.80, 0.85], [0.85, 0.87]]

    # 敌方在防御状态

]
Pij1 = [
    [[0, 0], [0, 0], [0, 0]],
    [[0, 0], [0.60, 0.65], [0.69, 0.72]],  # 敌1对我方杀伤概率
    [[0, 0], [0.54, 0.56], [0.60, 0.63]],  # 敌2对我方杀伤概率
    # 我方在攻击状态
    [[0, 0], [0.57, 0.61], [0.69, 0.71]],
    [[0, 0], [0.50, 0.54], [0.55, 0.58]]

    # 我方在防御状态
]
X = np.array([
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])  # 我方的九种策略，前两个数代表我方一号无人机攻击敌方哪个无人机；后两个数代表我方二号无人机攻击敌方哪个无人机，都为0则表示防御
Y = np.array([
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])


def Rmin(i, j, k):
    return (Vj[j][0] * Pij[i][j][0] * (Y[k][j * 2 - 1] or Y[k][j * 2 - 2]) + Vj[j][0] * Pij[i + num_w][j][0] * (
        not (Y[k][j * 2 - 1] or Y[k][j * 2 - 2])) - Vaa)  # 最小打击概率x最小可能价值-最大我方导弹价值


# 这个逻辑运算相当于实现了一个if语句的功能，敌方攻击或防守时使用不同的毁伤概率。


def Cmax(i, j, m):  # 这里的j是指我方无人机，i是敌方无人机

    return (Vi[j] * Pij1[i][j][1] * (X[m][j * 2 - 1] or X[m][j * 2 - 2]) + Vi[j] * Pij1[i + num_d][j][1] * (
        not (X[m][j * 2 - 1] or X[m][j * 2 - 2])) - Vbb[0])  # 敌方对我方最大杀伤概率x最大我方价值-最小敌方导弹价值


def Rmax(i, j, k):
    return (Vj[j][1] * Pij[i][j][1] * (Y[k][j * 2 - 1] or Y[k][j * 2 - 2]) + Vj[j][1] * Pij[i + num_w][j][1] * (
        not (Y[k][j * 2 - 1] or Y[k][j * 2 - 2])) - Vaa)  # 最大打击概率x最大可能价值-最小我方导弹价值


def Cmin(i, j, m):  # 这里的j是指我方无人机，i是敌方无人机

    return (Vi[j] * Pij1[i][j][0] * (X[m][j * 2 - 1] or X[m][j * 2 - 2]) + Vi[j] * Pij1[i + num_d][j][0] * (
        not (X[m][j * 2 - 1] or X[m][j * 2 - 2])) - Vbb[1])  # 敌方对我方最小杀伤概率x最小我方价值-最大敌方导弹价值


def Fmin(m, k):
    RRmin = np.array([
        [Rmin(1, 1, k)],
        [Rmin(1, 2, k)],
        [Rmin(2, 1, k)],
        [Rmin(2, 2, k)],
    ])
    # 根据敌方采取策略k，我方几种攻击情况的收益都算出来，然后与我方实际策略X[m]点乘
    CCmax = np.array([
        [Cmax(1, 1, m)],
        [Cmax(1, 2, m)],
        [Cmax(2, 1, m)],
        [Cmax(2, 2, m)],
    ])
    x = (np.dot(X[m], RRmin) - np.dot(Y[k], CCmax))

    return x[0]  # 点乘后得到的是一个矩阵，我想返回一个值


def Fmax(m, k):
    RRmax = np.array([
        [Rmax(1, 1, k)],
        [Rmax(1, 2, k)],
        [Rmax(2, 1, k)],
        [Rmax(2, 2, k)],
    ])
    # 根据敌方采取策略k，我方几种攻击情况的收益都算出来，然后与我方实际策略X[m]点乘
    CCmin = np.array([
        [Cmin(1, 1, m)],
        [Cmin(1, 2, m)],
        [Cmin(2, 1, m)],
        [Cmin(2, 2, m)],
    ])
    x = (np.dot(X[m], RRmax) - np.dot(Y[k], CCmin))

    return x[0]  # 点乘后得到的是一个矩阵，我想返回一个值


def Relative_Entropy(a, b1, b2):

    I1 = 0.5 * (a * math.log2(a / (0.5 * (a + b1))) + (1 - a) * math.log2((1 - a) / (1 - (0.5 * (a + b1))))) + \
         0.5 * (a * math.log2(a / (0.5 * (a + b2))) + (1 - a) * math.log2((1 - a) / (1 - (0.5 * (a + b2)))))
    I2 = 0.5 * (b1 * math.log2(b1 / (0.5 * (b1 + a))) + (1 - b1) * math.log2((1 - b1) / (1 - (0.5 * (b1 + a))))) + \
         0.5 * (b2 * math.log2(b2 / (0.5 * (b2 + a))) + (1 - b2) * math.log2((1 - b2) / (1 - (0.5 * (b2 + a)))))
    D = I1 + I2
    return D

for m in range(0, 9):  # m是我方策略
    for k in range(0, 9):  # k是敌方策略
        Payoff_Matrix[m][k][0] = round(Fmin(m, k), 1)  # 保留四位小数
        Payoff_Matrix2[m][k][0] = -Payoff_Matrix[m][k][0]

for m in range(0, 9):  # m是我方策略
    for k in range(0, 9):  # k是敌方策略
        Payoff_Matrix[m][k][1] = round(Fmax(m, k), 1)  # 保留四位小数
        Payoff_Matrix2[m][k][1] = -Payoff_Matrix[m][k][1]

# for i in range(9):
#     print(Payoff_Matrix[i])


TJD1 = np.zeros(shape=(8, 8))
TJD2 = np.zeros(shape=(8, 8))
dmax1 = np.zeros(shape=(8, 8))
dmin1 = np.zeros(shape=(8, 8))
dmax2 = np.zeros(shape=(8, 8))
dmin2 = np.zeros(shape=(8, 8))


for i in range(8):  # 最后一行数据收益有负数，是明显无意义可以舍去的策略
    for j in range(8):
        dmax1[i][j] = Relative_Entropy(180, Payoff_Matrix[i][j][0], Payoff_Matrix[i][j][1])  # 最高收益与最低收益是人为设置的
        dmin1[i][j] = Relative_Entropy(20, Payoff_Matrix[i][j][0], Payoff_Matrix[i][j][1])
        TJD1[i][j] = round(dmin1[i][j]/(dmax1[i][j] + dmin1[i][j]),4)

        dmax2[i][j] = Relative_Entropy(-20, Payoff_Matrix2[i][j][0], Payoff_Matrix2[i][j][1])  # 最高收益与最低收益是人为设置的
        dmin2[i][j] = Relative_Entropy(-180, Payoff_Matrix2[i][j][0], Payoff_Matrix2[i][j][1])
        TJD2[i][j] = round(dmin2[i][j] / (dmax2[i][j] + dmin2[i][j]),4)


print('我方相对贴近度矩阵：')
print(TJD1)
print('敌方相对贴近度矩阵：')
print(TJD2)