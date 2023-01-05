# PSO-for-Nash-Equilibrium

2020年本科毕业设计写的代码。
当时想用粒子群算法求解纳什均衡，网上没有找到可用的代码，于是参照论文写了这份代码，花了挺多时间。算法成功达到了预期目标，效果不错，可以求解维度比较高的**矩阵博弈的混合策略**纳什均衡解，放在这里供大家参考。

PSO.py: 粒子群算法求解部分。如果只是想用粒子群算法求矩阵博弈的纳什均衡，直接运行这一个文件就可以了，代码开头注释部分有几个例子，你也可以带入自己的支付矩阵。注意粒子群算法并不能保证求得全局最优解，可以通过调节超参数，增大粒子群规模和最大迭代次数提高求解稳定性，代价是耗时更长。    

另外两个文件是根据我毕设提供的场景求支付矩阵用的，场景是无人机空战博弈。PayoffMatrix.py是我论文3.3.2节中求的普通支付矩阵；Relative_Entropy.py是4.3.3节中不确定信息情况下求的相对贴进度矩阵。对大家来说应该没什么用。  

如果对代码有疑问，可以联系我1325293476@qq.com

I wrote this code for my undergraduate graduation project in 2020.   
At the time, I wanted to use the particle swarm algorithm to solve the Nash equilibrium, but I couldn't find any usable code online, so I wrote this code based on the paper, which took a lot of time. The algorithm successfully achieved the expected goal and had good performance in solving the **mixed strategy Nash equilibrium of matrix games** with relatively high dimensions. I am sharing it here for reference.  

PSO.py: Part of the particle swarm algorithm solution. If you just want to use the particle swarm algorithm to solve the Nash equilibrium of matrix games, you can run this file. The beginning comments in the code have several examples, and you can also input your own payoff matrix. Note that the particle swarm algorithm cannot guarantee the global optimal solution, but you can improve the stability of the solution by adjusting the hyperparameters, increasing the size of the particle swarm and the maximum number of iterations, at the cost of longer processing time.  

The other two files are used to calculate the payoff matrix based on the scenario provided by my graduation project, which is a UAV air combat game. PayoffMatrix.py is the ordinary payoff matrix I calculated in section 3.3.2 of my thesis; Relative_Entropy.py is the relative entropy progress matrix calculated in section 4.3.3 under uncertain information. These two files are probably not useful for everyone.  

If you have any questions about the code, you can contact me at 1325293476@qq.com"  

算法主要参考：
- 余谦，王先甲． 基于粒子群优化算法求解纳什均衡的演化算法[J]． 武汉大学学报，2006，52(1):25- 29
- 贾文生, 向淑文, 杨剑锋,等. 基于免疫粒子群算法的非合作博弈 Nash 均衡问题求解[J]. 计算机应 用研究, 2012(01):34-37
- 算法流程：

![求解流程](https://user-images.githubusercontent.com/47975865/168188905-bde46cee-0ac2-4ffb-b08b-84019147c8c6.png#pic_center)
- 适应度曲线

![适应度曲线](https://user-images.githubusercontent.com/47975865/168188912-5b2bf4a6-fa21-41ea-8a83-ade71820b70f.png#pic_center)
