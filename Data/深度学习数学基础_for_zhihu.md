# 深度学习数学基础(一)

机器学习以及深度学习的学习和应用过程需要一定的数学知识，这就要求学习者掌握一定的数学知识，虽然说大多数知识已经在大学的前置数学课程中学习过了，但由于鄙人不够用心，学得差不多都还给老师了，就在这里做一个梳理和总结，以方便后续知识的学习。

**所需的数学知识：**

- 高等数学/微积分
- 线性代数与矩阵论
- 概率论与信息论
- 最优化方法
- 图论/离散数学

## 一、微积分

### 1.1 函数的部分性质

#### 1.1.1 上确界sup和下确界inf

​		一个实数集合*A*，若有一个实数*M*，使得*A*中任何数都不超过*M*，那么就称*M*是*A*的一个上界。即设有一实数集A⊂R，实数集A的上确界supA被定义为如下的数：

#### 1.1.2 函数的单调性

​		深度学习中常要考虑到函数的单调性，如在神经网络的激活函数，AdaBoost算法中都需要研究函数的单调性。

#### 1.1.3 函数的极值

​		函数的极值在机器学习中有着及其重要的意义，这是因为大部分优化问题都是连续优化问题，因此可以通过求导数为0的点求得函数的极值，以实现最小化损失函数，最大化似然函数的目标。

#### 1.1.4 函数的凹凸性

​		函数的凹凸性在凸优化和Jensen不等式中都有所应用。

**函数凹凸性的定义：**

 <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 定义在 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 上，若对 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 中的任意两点 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> ~1~和 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> ~2~和任意 <img src="https://www.zhihu.com/equation?tex=\lambda\in (0,1)" alt="\lambda\in (0,1)" class="ee_img tr_noresize" eeimg="1"> :

-  <img src="https://www.zhihu.com/equation?tex=f(\lambda x_1+(1-\lambda)x_2)\le \lambda f(x_1)+(1-\lambda )f(x_2)" alt="f(\lambda x_1+(1-\lambda)x_2)\le \lambda f(x_1)+(1-\lambda )f(x_2)" class="ee_img tr_noresize" eeimg="1"> ，则称 <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 为 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 上的凹函数(去掉等号是严格凹函数)

-  <img src="https://www.zhihu.com/equation?tex=f(\lambda x_1+(1-\lambda)x_2)\ge \lambda f(x_1)+(1-\lambda )f(x_2)" alt="f(\lambda x_1+(1-\lambda)x_2)\ge \lambda f(x_1)+(1-\lambda )f(x_2)" class="ee_img tr_noresize" eeimg="1"> ，则称 <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 为 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 上的凸函数(去掉等号是严格凸函数)

  ![凹函数](https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/深度学习数学基础/20190807114918637.png)      

**相关定理：**

1.  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 为 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 上的凸函数，对任意三点 <img src="https://www.zhihu.com/equation?tex=x_1<x_2<x_3" alt="x_1<x_2<x_3" class="ee_img tr_noresize" eeimg="1"> ，总有 <img src="https://www.zhihu.com/equation?tex=\frac {f(x_2)-f(x_1)}{x_2-x_1}\le\frac {f(x_3)-f(x_1)}{x_3-x_1}\le\frac {f(x_3)-f(x_2)}{x_3-x_2}" alt="\frac {f(x_2)-f(x_1)}{x_2-x_1}\le\frac {f(x_3)-f(x_1)}{x_3-x_1}\le\frac {f(x_3)-f(x_2)}{x_3-x_2}" class="ee_img tr_noresize" eeimg="1"> 凹函数可以得到类似的结论。
2.  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 上可导，则以下论断等价：
   -  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 上为凸函数
   -  <img src="https://www.zhihu.com/equation?tex=f^，" alt="f^，" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 上单调递增
   -  <img src="https://www.zhihu.com/equation?tex=\forall x1,x2\in I,f(x2)\ge f(x1)+f^,(x_1)(x_2-x_1)" alt="\forall x1,x2\in I,f(x2)\ge f(x1)+f^,(x_1)(x_2-x_1)" class="ee_img tr_noresize" eeimg="1"> 
3. 若 <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 在 <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 上二阶可导，则 <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> 为凸函数 <img src="https://www.zhihu.com/equation?tex=\iff " alt="\iff " class="ee_img tr_noresize" eeimg="1"> 

