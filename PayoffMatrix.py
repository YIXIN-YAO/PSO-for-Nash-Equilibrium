import numpy as np

np.set_printoptions(suppress=True)  # 不用科学计数法输出
np.set_printoptions(linewidth=400)  # 输出不自动换行
Vi = [0, 73, 71]  # 我方价值
Vj = [0, 125, 126]  # 敌方价值
# Vaa = 1000  # 我方导弹价值
# Vbb = 1000  # 敌方导弹价值
Vaa = 25.2#25.2  # 我方导弹价值
Vbb = 42.5#42.5  # 敌方导弹价值
Vmaxd = max(Vj)
Vmaxw = max(Vi)
num_w = 2
num_d = 2
Payoff_Matrix = np.zeros(shape=(9, 9))
Pij = [
    [0, 0, 0],
    [0, 0.86, 0.87],  # 我方1号无人机对敌杀伤概率
    [0, 0.88, 0.86],  # 我方2号无人机对敌杀伤概率
    # 敌方在攻击状态

    [0, 0.80, 0.8],
    [0, 0.85, 0.81]
    # 敌方在防御状态

]
Pij1 = [
    [0, 0, 0],
    [0, 0.60, 0.69],  # 敌1对我方杀伤概率
    [0, 0.54, 0.60],  # 敌2对我方杀伤概率

    # 我方在攻击状态

    [0, 0.55, 0.60],
    [0, 0.52, 0.59]
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


def R(i, j, k):
    return (Vj[j] * Pij[i][j] * (Y[k][j * 2 - 1] or Y[k][j * 2 - 2]) + Vj[j] * Pij[i + num_w][j] * (
        not (Y[k][j * 2 - 1] or Y[k][j * 2 - 2])) - Vaa) / Vmaxd


# 这个逻辑运算相当于实现了一个if语句的功能，敌方攻击或防守时使用不同的毁伤概率。


def C(i, j, m):  # 这里的j是指我方无人机，i是敌方无人机

    return (Vi[j] * Pij1[i][j] * (X[m][j * 2 - 1] or X[m][j * 2 - 2]) + Vi[j] * Pij1[i + num_d][j] * (
        not (X[m][j * 2 - 1] or X[m][j * 2 - 2])) - Vbb) / Vmaxw


def F(m, k):
    RR = np.array([
        [R(1, 1, k)],
        [R(1, 2, k)],
        [R(2, 1, k)],
        [R(2, 2, k)],
    ])
    # 根据敌方采取策略k，我方几种攻击情况的收益都算出来，然后与我方实际策略X[m]点乘
    CC = np.array([
        [C(1, 1, m)],
        [C(1, 2, m)],
        [C(2, 1, m)],
        [C(2, 2, m)],
    ])
    x = (np.dot(X[m], RR) - np.dot(Y[k], CC))

    return x[0]  # 点乘后得到的是一个矩阵，我想返回一个值


for m in range(0, 9):  # m是我方策略
    for k in range(0, 9):  # k是敌方策略
        Payoff_Matrix[m][k] = ('%.4f' % F(m, k))  # 保留四位小数



def fit_fun(X):  # 适应函数
    return max(max(Payoff_Matrix[:,X[1]]) - Payoff_Matrix[X[0]][X[1]], 0) + max(max(Payoff_Matrix2[X[0]]) -
                                                                                Payoff_Matrix2[X[0]][X[1]], 0)

x = 1000
y = -1
di = np.zeros(shape=(9, 9))
wo = np.zeros(shape=(9, 9))
for row in range(0, 9):
    x = min(Payoff_Matrix[row])
    for column in range(0, 9):
        if Payoff_Matrix[row][column] == x:
            y = column
    di[row][y] = 1

# print(di)

for column in range(0, 9):
    x = max(Payoff_Matrix[:, column])
    for row in range(0, 9):
        if Payoff_Matrix[row][column] == x:
            y = row
    wo[y][column] = 1
# print(wo)
# for row in range(0, 9):
#     for column in range(0, 9):
#         if di[row][column] == 1 and wo[row][column] == 1:
#             print()
#             print('我方采取策略：', row)
#             print('敌方采取策略：', column)
Payoff_Matrix2 = -Payoff_Matrix
print('我方支付矩阵:')
print(Payoff_Matrix)