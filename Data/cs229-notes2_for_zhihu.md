# 机器学习入门


# 斯坦福CS229课程笔记第二章（一）

### 第四部分 生成学习算法(Generative Learning algorithms)

目前为止，我们讲过的学习算法的模型都是 <img src="https://www.zhihu.com/equation?tex=p (y|x;\theta)" alt="p (y|x;\theta)" class="ee_img tr_noresize" eeimg="1"> ，也就是给定  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  下  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  的条件分布，以   <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">   为参数，这类学习算法构建得模型被称为判别模型。例如，逻辑回归中就是以  <img src="https://www.zhihu.com/equation?tex=h_\theta(x) = g(\theta^T x)" alt="h_\theta(x) = g(\theta^T x)" class="ee_img tr_noresize" eeimg="1">  作为  <img src="https://www.zhihu.com/equation?tex=p (y|x;\theta)" alt="p (y|x;\theta)" class="ee_img tr_noresize" eeimg="1">  的模型，这里的  <img src="https://www.zhihu.com/equation?tex=g" alt="g" class="ee_img tr_noresize" eeimg="1">  是一个 <img src="https://www.zhihu.com/equation?tex=sigmoid" alt="sigmoid" class="ee_img tr_noresize" eeimg="1"> 函数。接下来，我们要学习生成学习算法。

设想有这样一种分类问题，我们要学习基于一个动物的某个特征来辨别它是大象 <img src="https://www.zhihu.com/equation?tex=(y=1)" alt="(y=1)" class="ee_img tr_noresize" eeimg="1"> 还是小狗 <img src="https://www.zhihu.com/equation?tex=(y=0)" alt="(y=0)" class="ee_img tr_noresize" eeimg="1"> 。给定一个训练集，使用逻辑回归或者基础的感知机算法，这样的一法能找到一条直线，作为区分开大象和小狗的边界，这就是模型的判断边界。根据某动物的特征值的具体数值，将其在决策区域内定位，然后根据所落到的区域来给出预测。也就是说，我们学习了 <img src="https://www.zhihu.com/equation?tex=p (y|x;\theta)" alt="p (y|x;\theta)" class="ee_img tr_noresize" eeimg="1"> 后，给定一个输入 <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> ，我们就能根据模型作出预测，得到预测的 <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1"> 。

生成学习采用另一种方法。首先，观察大象，然后我们针对大象的特征来进行建模。然后，再观察小狗，针对小狗的特征建立另一个模型。最后把新动物分别用大象和小狗的模型来拟合，看看新动物更接近哪个训练集中已有的模型，判断一种新动物归属大象还是小狗，本质上是看求得的概率哪个更大。

**判别式学习算法**直接计算 <img src="https://www.zhihu.com/equation?tex=p(y|x)" alt="p(y|x)" class="ee_img tr_noresize" eeimg="1"> ，通过建立 <img src="https://www.zhihu.com/equation?tex=X\rightarrow Y" alt="X\rightarrow Y" class="ee_img tr_noresize" eeimg="1"> 的映射，判断对应  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  的值落到了空间中的哪个区域。和之前的这些判别式算法不同，下面我们要讲的新算法是对  <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=p(y)" alt="p(y)" class="ee_img tr_noresize" eeimg="1"> 来进行建模。这类算法叫**做生成学习算法（generative learning algorithms）**。例如如果  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  用来表示一个样例是  小狗  <img src="https://www.zhihu.com/equation?tex=(0)" alt="(0)" class="ee_img tr_noresize" eeimg="1">  或者  大象  <img src="https://www.zhihu.com/equation?tex=(1)" alt="(1)" class="ee_img tr_noresize" eeimg="1"> ，那么 <img src="https://www.zhihu.com/equation?tex=p(x|y = 0)" alt="p(x|y = 0)" class="ee_img tr_noresize" eeimg="1"> 就是对小狗特征分布的建模，而 <img src="https://www.zhihu.com/equation?tex=p(x|y = 1)" alt="p(x|y = 1)" class="ee_img tr_noresize" eeimg="1"> 就是对大象特征分布的建模。

对  <img src="https://www.zhihu.com/equation?tex=p(y)" alt="p(y)" class="ee_img tr_noresize" eeimg="1">  (**class priors**每个类别的概率) 和 <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1">  进行建模之后，我们的算法就是用**贝叶斯规则（Bayes rule）** 来推导对应给定  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  下  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  的**后验分布（posterior distribution）**：


<img src="https://www.zhihu.com/equation?tex=p(y|x)=\frac{p(x|y)p(y)}{p(x)}
" alt="p(y|x)=\frac{p(x|y)p(y)}{p(x)}
" class="ee_img tr_noresize" eeimg="1">

这里的**分母（denominator）** 为： <img src="https://www.zhihu.com/equation?tex=p(x) = p(x|y = 1)p(y = 1) + p(x|y = 0)p(y = 0)" alt="p(x) = p(x|y = 1)p(y = 1) + p(x|y = 0)p(y = 0)" class="ee_img tr_noresize" eeimg="1"> （这个等式关系可以根据概率的标准性质来推导验证`译者注：其实就是条件概率`），这样接下来就可以把它表示成我们熟悉的  <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1"> 和  <img src="https://www.zhihu.com/equation?tex=p(y)" alt="p(y)" class="ee_img tr_noresize" eeimg="1">  的形式了。实际上如果我们计算 <img src="https://www.zhihu.com/equation?tex=p(y|x)" alt="p(y|x)" class="ee_img tr_noresize" eeimg="1">  来进行预测，并不需要去计算这个分母，因为有下面的等式关系：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\arg \max_y p(y|x) & =\arg \max_y \frac{p(x|y)p(y)}{p(x)}\\
&= \arg \max_y p(x|y)p(y)
\end{aligned}" alt="\begin{aligned}
\arg \max_y p(y|x) & =\arg \max_y \frac{p(x|y)p(y)}{p(x)}\\
&= \arg \max_y p(x|y)p(y)
\end{aligned}" class="ee_img tr_noresize" eeimg="1">

#### 1 高斯判别分析（Gaussian discriminant analysis）

我们要学的第一个生成学习算法就是**高斯判别分析（Gaussian discriminant analysis** ，缩写为GDA。）在这个模型里面，我们**假设  <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1"> 是一个多元正态分布。** 所以首先简单讲一下多元正态分布的一些特点，然后再继续讲 GDA 高斯判别分析模型。

##### 1.1 多元正态分布（multivariate normal distribution）

 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 维多元正态分布，也叫做多变量高斯分布，参数为一个 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 维 **均值向量**  <img src="https://www.zhihu.com/equation?tex=\mu \in  R^n " alt="\mu \in  R^n " class="ee_img tr_noresize" eeimg="1"> ，以及一个 **协方差矩阵**  <img src="https://www.zhihu.com/equation?tex=\Sigma \in  R^{n\times n}" alt="\Sigma \in  R^{n\times n}" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=\Sigma \geq 0" alt="\Sigma \geq 0" class="ee_img tr_noresize" eeimg="1">  是一个对称（symmetric）的半正定（positive semi-definite）矩阵。当然也可以写成" <img src="https://www.zhihu.com/equation?tex=N (\mu, \Sigma)" alt="N (\mu, \Sigma)" class="ee_img tr_noresize" eeimg="1"> " 的分布形式，密度（density）函数为：


<img src="https://www.zhihu.com/equation?tex=p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
" alt="p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
" class="ee_img tr_noresize" eeimg="1">

在上面的等式中，" <img src="https://www.zhihu.com/equation?tex=|\Sigma|" alt="|\Sigma|" class="ee_img tr_noresize" eeimg="1"> "的意思是矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> 的行列式（determinant）。对于一个在  <img src="https://www.zhihu.com/equation?tex=N(\mu,\Sigma)" alt="N(\mu,\Sigma)" class="ee_img tr_noresize" eeimg="1"> 分布中的随机变量  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  ，注意这里的 <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1"> 也是n维的一个向量，而不是n+1维，没有 <img src="https://www.zhihu.com/equation?tex=x_0=1" alt="x_0=1" class="ee_img tr_noresize" eeimg="1"> 这一定义，其平均值是  <img src="https://www.zhihu.com/equation?tex=\mu" alt="\mu" class="ee_img tr_noresize" eeimg="1"> ：


<img src="https://www.zhihu.com/equation?tex=E[X]=\int_x xp(x;\mu,\Sigma)dx=\mu
" alt="E[X]=\int_x xp(x;\mu,\Sigma)dx=\mu
" class="ee_img tr_noresize" eeimg="1">

随机变量 <img src="https://www.zhihu.com/equation?tex=Z" alt="Z" class="ee_img tr_noresize" eeimg="1"> 是一个有值的向量， <img src="https://www.zhihu.com/equation?tex=Z" alt="Z" class="ee_img tr_noresize" eeimg="1">  的 **协方差（covariance）** 的定义是： <img src="https://www.zhihu.com/equation?tex=Cov(Z) = E[(Z-E[Z])(Z-E[Z])^T ]" alt="Cov(Z) = E[(Z-E[Z])(Z-E[Z])^T ]" class="ee_img tr_noresize" eeimg="1"> 。这是对实数随机变量的方差（variance）这一概念的泛化扩展。这个协方差还可以定义成 <img src="https://www.zhihu.com/equation?tex=Cov(Z) = E[ZZ^T]-(E[Z])(E[Z])^T" alt="Cov(Z) = E[ZZ^T]-(E[Z])(E[Z])^T" class="ee_img tr_noresize" eeimg="1"> （你可以自己证明一下这两个定义实际上是等价的。）如果  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  是一个多变量正态分布，即  <img src="https://www.zhihu.com/equation?tex=X \sim N (\mu, \Sigma)" alt="X \sim N (\mu, \Sigma)" class="ee_img tr_noresize" eeimg="1"> ，则有：


<img src="https://www.zhihu.com/equation?tex=Cov(X)=\Sigma
" alt="Cov(X)=\Sigma
" class="ee_img tr_noresize" eeimg="1">

下面这些样例是一些高斯分布的密度图，如下图所示：

![cs229note2f1](https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes2/cs229note2f1.png)

最左边的图，展示的是一个均值为 <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> （实际上是一个 <img src="https://www.zhihu.com/equation?tex=2\times 1" alt="2\times 1" class="ee_img tr_noresize" eeimg="1">  的零向量）的高斯分布，协方差矩阵就是 <img src="https://www.zhihu.com/equation?tex=\Sigma = I" alt="\Sigma = I" class="ee_img tr_noresize" eeimg="1">  （一个  <img src="https://www.zhihu.com/equation?tex=2\times 2" alt="2\times 2" class="ee_img tr_noresize" eeimg="1"> 的单位矩阵，identity matrix）。这种均值为 <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1">  并且协方差矩阵为单位矩阵的高斯分布也叫做**标准正态分布。** 中间的图中展示的是均值为 <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> 而协方差矩阵是 <img src="https://www.zhihu.com/equation?tex=0.6I" alt="0.6I" class="ee_img tr_noresize" eeimg="1">  的高斯分布的概率密度函数；最右边的展示的是协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma = 2I" alt="\Sigma = 2I" class="ee_img tr_noresize" eeimg="1"> 的高斯分布的概率密度函数。从这几个图可以看出，随着协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> 变大，高斯分布的形态就变得更宽平（spread-out），而如果协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> 变小，分布就会更加集中（compressed）。

看一下更多的样例：

![cs229note2f2](https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes2/cs229note2f2.png)

上面这几个图展示的是均值为 <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> ，但协方差矩阵各不相同的高斯分布，其中的协方差矩阵依次如下所示：


<img src="https://www.zhihu.com/equation?tex=\Sigma =\begin{bmatrix} 
1 & 0 \\ 0 & 1 \\ \end{bmatrix};
\Sigma =\begin{bmatrix} 
1 & 0.5 \\ 0.5 & 1 \\ 
\end{bmatrix};
\Sigma =\begin{bmatrix} 
1 & 0.8 \\ 0.8 & 1 \\ 
\end{bmatrix}
" alt="\Sigma =\begin{bmatrix} 
1 & 0 \\ 0 & 1 \\ \end{bmatrix};
\Sigma =\begin{bmatrix} 
1 & 0.5 \\ 0.5 & 1 \\ 
\end{bmatrix};
\Sigma =\begin{bmatrix} 
1 & 0.8 \\ 0.8 & 1 \\ 
\end{bmatrix}
" class="ee_img tr_noresize" eeimg="1">

第一幅图还跟之前的标准正态分布的样子很相似，然后我们发现随着增大协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1">  的反对角线（off-diagonal）的值，密度图像开始朝着  45° 方向 (也就是  <img src="https://www.zhihu.com/equation?tex=x_1 = x_2" alt="x_1 = x_2" class="ee_img tr_noresize" eeimg="1">  所在的方向)逐渐压缩。观察下面三个同样分布密度图的轮廓图（contours）能看得更明显，随着反对角线上的值的增加， <img src="https://www.zhihu.com/equation?tex=x_1" alt="x_1" class="ee_img tr_noresize" eeimg="1"> 与 <img src="https://www.zhihu.com/equation?tex=x_2" alt="x_2" class="ee_img tr_noresize" eeimg="1"> 之间的相关性逐渐增加：

![cs229note2f3](https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes2/cs229note2f3.png)

下面的是另外一组样例，调整了协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> :


<img src="https://www.zhihu.com/equation?tex=\Sigma =\begin{bmatrix} 
1 & 0.5 \\ 0.5 & 1 \\ 
\end{bmatrix};
\Sigma =\begin{bmatrix} 
1 & 0.8 \\ 0.8 & 1 \\ 
\end{bmatrix}
\Sigma =\begin{bmatrix} 
3 & 0.8 \\ 0.8 & 1 \\ \end{bmatrix};
" alt="\Sigma =\begin{bmatrix} 
1 & 0.5 \\ 0.5 & 1 \\ 
\end{bmatrix};
\Sigma =\begin{bmatrix} 
1 & 0.8 \\ 0.8 & 1 \\ 
\end{bmatrix}
\Sigma =\begin{bmatrix} 
3 & 0.8 \\ 0.8 & 1 \\ \end{bmatrix};
" class="ee_img tr_noresize" eeimg="1">

上面这三个图像对应的协方差矩阵分别如下所示：

![cs229note2f4](https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes2/cs229note2f4.png)

从最左边的到中间`注：注意，左边和中间的这两个协方差矩阵中，右上和左下的元素都是负值！`很明显随着协方差矩阵中右上左下这个对角线方向元素的值的降低，图像还是又被压扁了（compressed），只是方向是反方向的。最后，随着我们修改参数，通常生成的轮廓图（contours）都是椭圆（最右边的图就是一个例子）。

再举一些例子，固定协方差矩阵为单位矩阵，即 <img src="https://www.zhihu.com/equation?tex=\Sigma = I" alt="\Sigma = I" class="ee_img tr_noresize" eeimg="1"> ，然后调整均值 <img src="https://www.zhihu.com/equation?tex=\mu" alt="\mu" class="ee_img tr_noresize" eeimg="1"> ，我们就可以让密度图像随着均值而移动：

![cs229note2f5](https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes2/cs229note2f5.png)

上面的图像中协方差矩阵都是单位矩阵，即  <img src="https://www.zhihu.com/equation?tex=\Sigma = I" alt="\Sigma = I" class="ee_img tr_noresize" eeimg="1"> ，对应的均值 <img src="https://www.zhihu.com/equation?tex=\mu" alt="\mu" class="ee_img tr_noresize" eeimg="1"> 如下所示，图像的分布并无明显的变化：


<img src="https://www.zhihu.com/equation?tex=\mu =\begin{bmatrix} 
1 \\ 0 \\ 
\end{bmatrix};
\mu =\begin{bmatrix} 
-0.5 \\ 0 \\ 
\end{bmatrix};
\mu =\begin{bmatrix} 
-1 \\ -1.5 \\ 
\end{bmatrix};
" alt="\mu =\begin{bmatrix} 
1 \\ 0 \\ 
\end{bmatrix};
\mu =\begin{bmatrix} 
-0.5 \\ 0 \\ 
\end{bmatrix};
\mu =\begin{bmatrix} 
-1 \\ -1.5 \\ 
\end{bmatrix};
" class="ee_img tr_noresize" eeimg="1">

##### 1.2 高斯判别分析模型（Gaussian Discriminant Analysis model）

假如我们有一个分类问题，其中输入特征  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  是一系列的连续随机变量，那就可以使用高斯判别分析（Gaussian Discriminant Analysis ，缩写为 GDA）模型，其中对  <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1"> 用多元正态分布来进行建模。这个模型为：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
y & \sim Bernoulli(\phi)\\
x|y = 0 & \sim N(\mu_o,\Sigma)\\
x|y = 1 & \sim N(\mu_1,\Sigma)\\
\end{aligned}
" alt="\begin{aligned}
y & \sim Bernoulli(\phi)\\
x|y = 0 & \sim N(\mu_o,\Sigma)\\
x|y = 1 & \sim N(\mu_1,\Sigma)\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

这里的参数有 <img src="https://www.zhihu.com/equation?tex=\mu_0,\mu_1,\Sigma" alt="\mu_0,\mu_1,\Sigma" class="ee_img tr_noresize" eeimg="1"> ，也就是说正负样本均值不同，协方差相同。

分布写出来的具体形式如下：

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
p(y) & =\phi^y (1-\phi)^{1-y}\\
p(x|y=0) & = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp ( - \frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)  )\\
p(x|y=1) & = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp ( - \frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)  )\\
\end{aligned}
" alt="\begin{aligned}
p(y) & =\phi^y (1-\phi)^{1-y}\\
p(x|y=0) & = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp ( - \frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)  )\\
p(x|y=1) & = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} exp ( - \frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)  )\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

在上面的等式中，模型的参数包括 <img src="https://www.zhihu.com/equation?tex=\phi, \Sigma, \mu_0 和 \mu_1" alt="\phi, \Sigma, \mu_0 和 \mu_1" class="ee_img tr_noresize" eeimg="1"> 。（要注意，虽然这里有两个不同方向的均值向量 <img src="https://www.zhihu.com/equation?tex=\mu_0" alt="\mu_0" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\mu_1" alt="\mu_1" class="ee_img tr_noresize" eeimg="1"> ，针对这个模型还是一般只是用一个协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> 。）对数似然函数（log-likelihood）如下所示，这里默认各输入特征是相互独立的：

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
l(\phi,\mu_0,\mu_1,\Sigma) &= \log \prod^m_{i=1}p(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)\\
&= \log \prod^m_{i=1}p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma)p(y^{(i)};\phi)\\
\end{aligned}
" alt="\begin{aligned}
l(\phi,\mu_0,\mu_1,\Sigma) &= \log \prod^m_{i=1}p(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)\\
&= \log \prod^m_{i=1}p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma)p(y^{(i)};\phi)\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

通过使  <img src="https://www.zhihu.com/equation?tex=l" alt="l" class="ee_img tr_noresize" eeimg="1">  取得最大值，找到对应的参数组合，然后就能找到该参数组合对应的最大似然估计，结果如下所示（参考习题集1）：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\phi & = \frac {1}{m} \sum^m_{i=1}1\{y^{(i)}=1\}\\
\mu_0 & = \frac{\sum^m_{i=1}1\{y^{(i)}=0\}x^{(i)}}{\sum^m_{i=1}1\{y^{(i)}=0\}}\\
\mu_1 & = \frac{\sum^m_{i=1}1\{y^{(i)}=1\}x^{(i)}}{\sum^m_{i=1}1\{y^{(i)}=1\}}\\
\Sigma & = \frac{1}{m}\sum^m_{i=1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\\
\end{aligned}
" alt="\begin{aligned}
\phi & = \frac {1}{m} \sum^m_{i=1}1\{y^{(i)}=1\}\\
\mu_0 & = \frac{\sum^m_{i=1}1\{y^{(i)}=0\}x^{(i)}}{\sum^m_{i=1}1\{y^{(i)}=0\}}\\
\mu_1 & = \frac{\sum^m_{i=1}1\{y^{(i)}=1\}x^{(i)}}{\sum^m_{i=1}1\{y^{(i)}=1\}}\\
\Sigma & = \frac{1}{m}\sum^m_{i=1}(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

用图形化的方式来表述，这个算法可以按照下面的图示所表示：

![cs229note2f6](https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes2/cs229note2f6.png)

图中展示的点就是训练数据集，图中的两个高斯分布就是针对两类数据各自进行的拟合。要注意这两个高斯分布的轮廓图有同样的形状和拉伸方向，这是因为他们都有同样的协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> ，但他们有不同的均值 <img src="https://www.zhihu.com/equation?tex=\mu_0" alt="\mu_0" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\mu_1" alt="\mu_1" class="ee_img tr_noresize" eeimg="1">  。此外，图中的直线给出了 <img src="https://www.zhihu.com/equation?tex=p (y = 1|x) = 0.5" alt="p (y = 1|x) = 0.5" class="ee_img tr_noresize" eeimg="1">  这条边界线。在这条边界的一侧，我们预测  <img src="https://www.zhihu.com/equation?tex=y = 1" alt="y = 1" class="ee_img tr_noresize" eeimg="1"> 是最可能的结果，而另一侧，就预测  <img src="https://www.zhihu.com/equation?tex=y = 0" alt="y = 0" class="ee_img tr_noresize" eeimg="1"> 是最可能的结果。

##### 1.3 讨论：高斯判别分析（GDA）与逻辑回归（logistic regression）

高斯判别分析模型与逻辑回归有很有趣的相关性，一维的高斯判别分析模型在某种程度上可以用sigmoid函数进行刻画，如下图所示。

<img src="https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes2/image-20200802161848792.png" alt="image-20200802161848792" style="zoom:67%;" />

如果我们把变量 <img src="https://www.zhihu.com/equation?tex=p (y = 1|x; \phi, \mu_0, \mu_1, \Sigma)" alt="p (y = 1|x; \phi, \mu_0, \mu_1, \Sigma)" class="ee_img tr_noresize" eeimg="1">  作为一个  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  的函数，就会发现可以用如下的形式来表达：

<img src="https://www.zhihu.com/equation?tex=p(y=1|x;\phi,\Sigma,\mu_0,\mu_1)=\frac 1 {1+exp(-\theta^Tx)}
" alt="p(y=1|x;\phi,\Sigma,\mu_0,\mu_1)=\frac 1 {1+exp(-\theta^Tx)}
" class="ee_img tr_noresize" eeimg="1">

其中的   <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">   是对 <img src="https://www.zhihu.com/equation?tex=\phi" alt="\phi" class="ee_img tr_noresize" eeimg="1"> ,  <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> ,  <img src="https://www.zhihu.com/equation?tex=\mu_0" alt="\mu_0" class="ee_img tr_noresize" eeimg="1"> ,  <img src="https://www.zhihu.com/equation?tex=\mu_1" alt="\mu_1" class="ee_img tr_noresize" eeimg="1"> 的某种函数。这就是逻辑回归（也是一种判别分析算法）用来对 <img src="https://www.zhihu.com/equation?tex=p (y = 1|x)" alt="p (y = 1|x)" class="ee_img tr_noresize" eeimg="1">  建模的形式。

> 注：上面这里用到了一种转换，就是重新对 <img src="https://www.zhihu.com/equation?tex=x^{(i)}" alt="x^{(i)}" class="ee_img tr_noresize" eeimg="1"> 向量进行了定义，在右手侧（right-hand-side）增加了一个额外的坐标 <img src="https://www.zhihu.com/equation?tex=x_0^{(i)} = 1" alt="x_0^{(i)} = 1" class="ee_img tr_noresize" eeimg="1"> ，然后使之成为了一个  <img src="https://www.zhihu.com/equation?tex=n+1" alt="n+1" class="ee_img tr_noresize" eeimg="1"> 维的向量；

这两个模型中什么时候该选哪一个呢？一般来说，高斯判别分析（GDA）和逻辑回归，对同一个训练集，可能给出的判别曲线是不一样的。哪一个更好些呢？

我们刚刚已经表明，如果 <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1"> 是一个多变量的高斯分布（且具有一个共享的协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> ），那么 <img src="https://www.zhihu.com/equation?tex=p(y|x)" alt="p(y|x)" class="ee_img tr_noresize" eeimg="1"> 则必然符合logistic function。但如果反过来，这个命题是不成立的。例如假如 <img src="https://www.zhihu.com/equation?tex=p(y|x)" alt="p(y|x)" class="ee_img tr_noresize" eeimg="1"> 是一个逻辑函数，这并不能保证 <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1"> 一定是一个多变量的高斯分布。这就表明**高斯判别模型能比逻辑回归对数据进行更强的建模和假设（stronger modeling assumptions）。** 这也就意味着，**在这两种模型假设都可用的时候，高斯判别分析法去拟合数据是更好的，是一个更好的模型。** 尤其当 <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1"> 已经确定是一个高斯分布（有共享的协方差矩阵 <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> ），那么高斯判别分析是**渐进有效的（asymptotically efficient）。** 实际上，这也意味着，在面对非常大的训练集（训练样本规模  <img src="https://www.zhihu.com/equation?tex=m " alt="m " class="ee_img tr_noresize" eeimg="1"> 特别大）的时候，严格来说，可能就没有什么别的算法能比高斯判别分析更好（比如考虑到对  <img src="https://www.zhihu.com/equation?tex=p(y|x)" alt="p(y|x)" class="ee_img tr_noresize" eeimg="1"> 估计的准确度等等）。所以在这种情况下就表明，高斯判别分析（GDA）是一个比逻辑回归更好的算法；再扩展一下，即便对于小规模的训练集，我们最终也会发现高斯判别分析（GDA）是更好的。

但从另一个角度说，由于逻辑回归做出的假设要明显更弱一些（significantly weaker），所以因此逻辑回归给出的判断鲁棒性（robust）也更强，同时也对错误的建模假设不那么敏感。有很多不同的假设集合都能够将 <img src="https://www.zhihu.com/equation?tex=p(y|x)" alt="p(y|x)" class="ee_img tr_noresize" eeimg="1"> 引向逻辑回归函数。例如，如果 <img src="https://www.zhihu.com/equation?tex=x|y = 0\sim Poisson(\lambda_0)" alt="x|y = 0\sim Poisson(\lambda_0)" class="ee_img tr_noresize" eeimg="1">  是一个泊松分布，而 <img src="https://www.zhihu.com/equation?tex=x|y = 1\sim Poisson(\lambda_1)" alt="x|y = 1\sim Poisson(\lambda_1)" class="ee_img tr_noresize" eeimg="1"> 也是一个泊松分布，那么 <img src="https://www.zhihu.com/equation?tex=p(y|x)" alt="p(y|x)" class="ee_img tr_noresize" eeimg="1"> 也将是适合逻辑回归的（logistic）。逻辑回归也适用于这类的泊松分布的数据。但对这样的数据，如果我们强行使用高斯判别分析（GDA），然后用高斯分布来拟合这些非高斯数据，那么结果的可预测性就会降低，而且GDA这种方法也许可行，也有可能是不能用。

总结一下也就是：高斯判别分析方法（GDA）能够建立更强的模型假设，并且在数据利用上更加有效（比如说，需要更少的训练集就能有"还不错的"效果），当然前提是模型假设接近正确。逻辑回归建立的假设更弱，因此对于偏离的模型假设来说更加鲁棒（robust）。然而，如果训练集数据的确是非高斯分布的（non-Gaussian），而且是有限的大规模数据（in the limit of large datasets），那么逻辑回归几乎总是比GDA要更好的。因此，在实际中，逻辑回归的使用频率要比GDA高得多。（关于判别和生成模型的对比的相关讨论也适用于我们下面要讲的朴素贝叶斯算法（Naive Bayes），朴素贝叶斯算法也被认为是一个非常优秀也非常流行的分类算法。）



#### 2 朴素贝叶斯法（Naive Bayes）

在高斯判别分析（GDA）方法中，特征向量  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  是连续的值为实数的向量。下面要讲的是当  <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1">  是离散值时，我们使用的另外一种学习算法——朴素贝叶斯。

选用之前出现过的一个样例，使用机器学习的方法来尝试建立一个邮件筛选器。我们要对邮件信息进行分类，来判断一个邮件是商业广告邮件（就是垃圾邮件），还是非垃圾邮件。在学会之后，我们就可以让邮件阅读器能够自动对垃圾信息进行过滤，或者单独把这些垃圾邮件放进一个单独的文件夹中。对邮件进行分类是一个案例，属于文本分类这一更广泛问题集合。

假设我们有了一个训练集（也就是一堆已经标好了是否为垃圾邮件的邮件）。要构建垃圾邮件分选器，咱们先要开始确定用来描述一封邮件的特征 <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1"> 有哪些。

我们将用一个特征向量来表示一封邮件，这个向量的长度等于字典中单词的个数。如果邮件中包含了字典中的第  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  个单词，那么就令  <img src="https://www.zhihu.com/equation?tex=x_i = 1" alt="x_i = 1" class="ee_img tr_noresize" eeimg="1"> ；反之则 <img src="https://www.zhihu.com/equation?tex=x_i = 0" alt="x_i = 0" class="ee_img tr_noresize" eeimg="1"> 。例如下面这个向量：


<img src="https://www.zhihu.com/equation?tex=x=\begin{bmatrix}1\\0\\0\\\vdots \\1\\ \vdots \\0\end{bmatrix} \begin{matrix}\text{a}\\ \text{aardvark}\\ \text{aardwolf}\\ \vdots\\ \text{buy}\\ \vdots\\ \text{zygmurgy}\\ \end{matrix}
" alt="x=\begin{bmatrix}1\\0\\0\\\vdots \\1\\ \vdots \\0\end{bmatrix} \begin{matrix}\text{a}\\ \text{aardvark}\\ \text{aardwolf}\\ \vdots\\ \text{buy}\\ \vdots\\ \text{zygmurgy}\\ \end{matrix}
" class="ee_img tr_noresize" eeimg="1">

就用来表示一个邮件，其中包含了两个单词 "a" 和 "buy"，但没有单词 "aardvark"， "aardwolf" 或者 "zymurgy"  。这个单词集合编码整理成的特征向量也成为**词汇表（vocabulary,），** 所以特征向量  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  的维度就等于词汇表的长度。

> 注：实际应用中并不需要遍历整个英语词典来组成所有英语单词的列表，实践中更常用的方法是遍历一下训练集，然后把出现过一次以上的单词才编码成特征向量。这样做除了能够降低模型中单词表的长度之外，还能够降低运算量和空间占用，此外还有一个好处就是能够包含一些你的邮件中出现了而词典中没有的单词，比如本课程的缩写CS229。有时候（比如在作业里面），还要排除一些特别高频率的词汇，比如像冠词the，介词of 和and 等等；这些高频率但是没有具体意义的虚词也叫做stop words，因为很多文档中都要有这些词，用它们也基本不能用来判定一个邮件是否为垃圾邮件。对于中文邮件分类，常需要利用jieba库进行分词，生成我们需要的词向量。

选好了特征向量后，接下来就是建立一个生成模型。所以我们必须对 <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1"> 进行建模。但是，假如我们的单词有五万个词，则特征向量 <img src="https://www.zhihu.com/equation?tex=x \in  \{0, 1\}^{50000}" alt="x \in  \{0, 1\}^{50000}" class="ee_img tr_noresize" eeimg="1">  （即  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 是一个  <img src="https://www.zhihu.com/equation?tex=50000" alt="50000" class="ee_img tr_noresize" eeimg="1">  维的向量，其值是 <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> 或者 <img src="https://www.zhihu.com/equation?tex=1" alt="1" class="ee_img tr_noresize" eeimg="1"> ），如果我们要对这样的  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 进行多项式分布的建模，那么就可能有 <img src="https://www.zhihu.com/equation?tex=2^{50000}" alt="2^{50000}" class="ee_img tr_noresize" eeimg="1">  种可能的输出，然后就要用一个  <img src="https://www.zhihu.com/equation?tex=(2^{50000}-1)" alt="(2^{50000}-1)" class="ee_img tr_noresize" eeimg="1"> 维的参数向量。这样参数明显太多了。

要给 <img src="https://www.zhihu.com/equation?tex=p(x|y)" alt="p(x|y)" class="ee_img tr_noresize" eeimg="1"> 建模，先来做一个非常强的假设。我们**假设特征向量 <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1">  对于给定的  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  是独立的。** 这个假设也叫做**朴素贝叶斯假设（Naive Bayes ，NB assumption），** 基于此假设衍生的算法也就叫做**朴素贝叶斯分类器（Naive Bayes classifier）。** 例如，如果  <img src="https://www.zhihu.com/equation?tex=y = 1" alt="y = 1" class="ee_img tr_noresize" eeimg="1">  意味着一个邮件是垃圾邮件；然后其中"buy" 是第 <img src="https://www.zhihu.com/equation?tex=2087" alt="2087" class="ee_img tr_noresize" eeimg="1"> 个单词，而 "price"是第 <img src="https://www.zhihu.com/equation?tex=39831" alt="39831" class="ee_img tr_noresize" eeimg="1"> 个单词；那么接下来我们就假设，如果我告诉你  <img src="https://www.zhihu.com/equation?tex=y = 1" alt="y = 1" class="ee_img tr_noresize" eeimg="1"> ，也就是说某一个特定的邮件是垃圾邮件，那么对于 <img src="https://www.zhihu.com/equation?tex=x_{2087}" alt="x_{2087}" class="ee_img tr_noresize" eeimg="1">  （也就是单词 buy 是否出现在邮件里）的了解并不会影响你对 <img src="https://www.zhihu.com/equation?tex=x_{39831}" alt="x_{39831}" class="ee_img tr_noresize" eeimg="1">  （单词price出现的位置）的采信值。更正规一点，可以写成  <img src="https://www.zhihu.com/equation?tex=p(x_{2087}|y) = p(x_{2087}|y, x_{39831})" alt="p(x_{2087}|y) = p(x_{2087}|y, x_{39831})" class="ee_img tr_noresize" eeimg="1"> 。（要注意这个并不是说 <img src="https://www.zhihu.com/equation?tex=x_{2087}" alt="x_{2087}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=x_{39831}" alt="x_{39831}" class="ee_img tr_noresize" eeimg="1"> 这两个特征是独立的，那样就变成了 <img src="https://www.zhihu.com/equation?tex=p(x_{2087}) = p(x_{2087}|x_{39831})" alt="p(x_{2087}) = p(x_{2087}|x_{39831})" class="ee_img tr_noresize" eeimg="1"> ，我们这里是说在给定了  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  的这样一个条件下，二者才是有条件的独立。）

然后我们就得到了等式：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
p(x_1, ..., x_{50000}|y) & = p(x_1|y)p(x_2|y,x_1)p(x_3|y,x_1,x_2) ... p(x_{50000}|y,x_1,x_2,...,x_{49999})\\
& = p(x_1|y)p(x_2|y)p(x_3|y) ... p(x_{50000}|y)\\
& = \prod^n_{i=1}p(x_i|y)\\
\end{aligned}
" alt="\begin{aligned}
p(x_1, ..., x_{50000}|y) & = p(x_1|y)p(x_2|y,x_1)p(x_3|y,x_1,x_2) ... p(x_{50000}|y,x_1,x_2,...,x_{49999})\\
& = p(x_1|y)p(x_2|y)p(x_3|y) ... p(x_{50000}|y)\\
& = \prod^n_{i=1}p(x_i|y)\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

第一行的等式就是简单地来自概率的基本性质，第二个等式则使用了朴素贝叶斯假设。这里要注意，朴素贝叶斯假设也是一个很强的假设，产生的这个算法可以适用于很多种问题。

我们这个模型的参数为  <img src="https://www.zhihu.com/equation?tex=\phi_{i|y=1} = p (x_i = 1|y = 1), \phi_{i|y=0} = p (x_i = 1|y = 0)" alt="\phi_{i|y=1} = p (x_i = 1|y = 1), \phi_{i|y=0} = p (x_i = 1|y = 0)" class="ee_img tr_noresize" eeimg="1"> , 而  <img src="https://www.zhihu.com/equation?tex=\phi_y = p (y = 1)" alt="\phi_y = p (y = 1)" class="ee_img tr_noresize" eeimg="1"> 。和以往一样，给定一个训练集 <img src="https://www.zhihu.com/equation?tex=\{(x^{(i)},y^{(i)}); i = 1, ..., m\}" alt="\{(x^{(i)},y^{(i)}); i = 1, ..., m\}" class="ee_img tr_noresize" eeimg="1"> ，就可以写出下面的联合似然函数：


<img src="https://www.zhihu.com/equation?tex=\mathcal{L}(\phi_y,\phi_{j|y=0},\phi_{j|y=1})=\prod^m_{i=1}p(x^{(i)},y^{(i)})
" alt="\mathcal{L}(\phi_y,\phi_{j|y=0},\phi_{j|y=1})=\prod^m_{i=1}p(x^{(i)},y^{(i)})
" class="ee_img tr_noresize" eeimg="1">

找到使联合似然函数取得最大值的对应参数组合  <img src="https://www.zhihu.com/equation?tex=\phi_y , \phi_{i|y=0} 和 \phi_{i|y=1}" alt="\phi_y , \phi_{i|y=0} 和 \phi_{i|y=1}" class="ee_img tr_noresize" eeimg="1">  就给出了最大似然估计：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\phi_{j|y=1} &=\frac{\sum^m_{i=1}1\{x_j^{(i)} =1 \wedge y^{(i)} =1\} }{\sum^m_{i=1}1\{y^{(i)} =1\}} \\
\phi_{j|y=0} &= \frac{\sum^m_{i=1}1\{x_j^{(i)} =1 \wedge y^{(i)} =0\} }{\sum^m_{i=1}1\{y^{(i)} =0\}} \\
\phi_{y} &= \frac{\sum^m_{i=1}1\{y^{(i)} =1\}}{m}\\
\end{aligned}
" alt="\begin{aligned}
\phi_{j|y=1} &=\frac{\sum^m_{i=1}1\{x_j^{(i)} =1 \wedge y^{(i)} =1\} }{\sum^m_{i=1}1\{y^{(i)} =1\}} \\
\phi_{j|y=0} &= \frac{\sum^m_{i=1}1\{x_j^{(i)} =1 \wedge y^{(i)} =0\} }{\sum^m_{i=1}1\{y^{(i)} =0\}} \\
\phi_{y} &= \frac{\sum^m_{i=1}1\{y^{(i)} =1\}}{m}\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

在上面的等式中，" <img src="https://www.zhihu.com/equation?tex=\wedge" alt="\wedge" class="ee_img tr_noresize" eeimg="1"> (and)"这个符号的意思是逻辑"与"。这些参数有一个非常自然的解释。例如  <img src="https://www.zhihu.com/equation?tex=\phi_{j|y=1}" alt="\phi_{j|y=1}" class="ee_img tr_noresize" eeimg="1">  正是垃圾  <img src="https://www.zhihu.com/equation?tex=(y = 1)" alt="(y = 1)" class="ee_img tr_noresize" eeimg="1">  邮件中出现单词  <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1">  的邮件所占的比例。

拟合好了全部这些参数之后，要对一个新样本的特征向量  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  进行预测，只要进行如下的简单地计算：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
p(y=1|x)&=  \frac{p(x|y=1)p(y=1)}{p(x)}\\
&= \frac{(\prod^n_{i=1}p(x_i|y=1))p(y=1)}{(\prod^n_{i=1}p(x_i|y=1))p(y=1)+  (\prod^n_{i=1}p(x_i|y=0))p(y=0)}  \\
\end{aligned}
" alt="\begin{aligned}
p(y=1|x)&=  \frac{p(x|y=1)p(y=1)}{p(x)}\\
&= \frac{(\prod^n_{i=1}p(x_i|y=1))p(y=1)}{(\prod^n_{i=1}p(x_i|y=1))p(y=1)+  (\prod^n_{i=1}p(x_i|y=0))p(y=0)}  \\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

然后选择有最高后验概率的概率。

最后我们要注意，刚刚我们对朴素贝叶斯算法的使用中，特征向量  <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1">  都是二值化的，其实特征向量也可以是多个离散值，比如 <img src="https://www.zhihu.com/equation?tex=\{1, 2, ..., k_i\}" alt="\{1, 2, ..., k_i\}" class="ee_img tr_noresize" eeimg="1"> 这样也都是可以的。这时候只需要把对 <img src="https://www.zhihu.com/equation?tex=p(x_i|y)" alt="p(x_i|y)" class="ee_img tr_noresize" eeimg="1">  的建模从伯努利分布改成多项式分布。实际上，即便一些原始的输入值是连续值（比如我们第一个案例中的房屋面积），也可以转换成一个小规模的离散值的集合，然后再使用朴素贝叶斯方法。例如，如果我们用特征向量  <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1">  来表示住房面积，那么就可以按照下面所示的方法来对这一变量进行离散化：

|   居住面积   |  <img src="https://www.zhihu.com/equation?tex=<400" alt="<400" class="ee_img tr_noresize" eeimg="1">  |  <img src="https://www.zhihu.com/equation?tex=400-800" alt="400-800" class="ee_img tr_noresize" eeimg="1">  |  <img src="https://www.zhihu.com/equation?tex=800-1200" alt="800-1200" class="ee_img tr_noresize" eeimg="1">  |  <img src="https://www.zhihu.com/equation?tex=1200-1600" alt="1200-1600" class="ee_img tr_noresize" eeimg="1">  |  <img src="https://www.zhihu.com/equation?tex=>1600" alt=">1600" class="ee_img tr_noresize" eeimg="1">  |

| :----------: | :----: | :-------: | :--------: | :---------: | :-----: |

| 离散值  <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1">  |   <img src="https://www.zhihu.com/equation?tex=1" alt="1" class="ee_img tr_noresize" eeimg="1">    |     <img src="https://www.zhihu.com/equation?tex=2" alt="2" class="ee_img tr_noresize" eeimg="1">     |     <img src="https://www.zhihu.com/equation?tex=3" alt="3" class="ee_img tr_noresize" eeimg="1">      |      <img src="https://www.zhihu.com/equation?tex=4" alt="4" class="ee_img tr_noresize" eeimg="1">      |    <img src="https://www.zhihu.com/equation?tex=5" alt="5" class="ee_img tr_noresize" eeimg="1">    |


这样，对于一个面积为  <img src="https://www.zhihu.com/equation?tex=890" alt="890" class="ee_img tr_noresize" eeimg="1">  平方英尺的房屋，就可以根据上面这个集合中对应的值来把特征向量的这一项的 <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1"> 值设置为 <img src="https://www.zhihu.com/equation?tex=3" alt="3" class="ee_img tr_noresize" eeimg="1"> 。然后就可以用朴素贝叶斯算法，并且将 <img src="https://www.zhihu.com/equation?tex=p(x_i|y)" alt="p(x_i|y)" class="ee_img tr_noresize" eeimg="1"> 作为多项式分布来进行建模，就都跟前面讲过的内容一样了。当原生的连续值的属性不太容易用一个多元正态分布来进行建模的时候，将其**特征向量离散化**然后使用朴素贝叶斯法（NB）来替代高斯判别分析法（GDA），通常能形成一个更好的分类器。

##### 2.1 拉普拉斯平滑（Laplace smoothing）

刚刚讲过的朴素贝叶斯算法能够解决很多问题了，但还能对这种方法进行一点小调整来进一步提高效果，尤其是应对文本分类的情况。我们来简要讨论一下一个算法当前状态的一个问题，然后在讲一下如何解决这个问题。

还是考虑垃圾邮件分类的过程，设想你学完了CS229的课程，然后做了很棒的研究项目，之后你决定把自己的作品投稿到NIPS会议，这个NIPS是机器学习领域的一个顶级会议，递交论文的截止日期一般是六月末到七月初。你通过邮件来对这个会议进行了讨论，然后你也开始收到带有 nips 四个字母的信息。但这个是你第一个NIPS论文，而在此之前，你从来没有接到过任何带有 nips 这个单词的邮件；尤其重要的是，nips 这个单词就从来都没有出现在你的垃圾/正常邮件训练集里面。假如这个 nips 是你字典中的第 <img src="https://www.zhihu.com/equation?tex=35000" alt="35000" class="ee_img tr_noresize" eeimg="1"> 个单词，那么你的朴素贝叶斯垃圾邮件筛选器就要对参数 <img src="https://www.zhihu.com/equation?tex=\phi_{35000|y}" alt="\phi_{35000|y}" class="ee_img tr_noresize" eeimg="1">  进行最大似然估计，如下所示：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\phi_{35000|y=1} &=  \frac{\sum^m_{i=1}1\{x^{(i)}_{35000}=1 \wedge y^{(i)}=1  \}}{\sum^m_{i=1}1\{y^{(i)}=0\}}  &=0 \\
\phi_{35000|y=0} &=  \frac{\sum^m_{i=1}1\{x^{(i)}_{35000}=1 \wedge y^{(i)}=0  \}}{\sum^m_{i=1}1\{y^{(i)}=0\}}  &=0 \\
\end{aligned}
" alt="\begin{aligned}
\phi_{35000|y=1} &=  \frac{\sum^m_{i=1}1\{x^{(i)}_{35000}=1 \wedge y^{(i)}=1  \}}{\sum^m_{i=1}1\{y^{(i)}=0\}}  &=0 \\
\phi_{35000|y=0} &=  \frac{\sum^m_{i=1}1\{x^{(i)}_{35000}=1 \wedge y^{(i)}=0  \}}{\sum^m_{i=1}1\{y^{(i)}=0\}}  &=0 \\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

也就是说，因为之前程序从来没有在别的垃圾邮件或者正常邮件的训练样本中看到过 nips 这个词，所以它就认为看到这个词出现在这两种邮件中的概率都是 <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> 。因此当要决定一个包含 nips 这个单词的邮件是否为垃圾邮件的时候，他就检验这个类的后验概率，然后得到了：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
p(y=1|x) &= \frac{ \prod^n_{i=1} p(x_i|y=1)p(y=1) }   {\prod^n_{i=1} p(x_i|y=1)p(y=1) +\prod^n_{i=1} p(x_i|y=1)p(y=0)    }\\
&= \frac00\\
\end{aligned}
" alt="\begin{aligned}
p(y=1|x) &= \frac{ \prod^n_{i=1} p(x_i|y=1)p(y=1) }   {\prod^n_{i=1} p(x_i|y=1)p(y=1) +\prod^n_{i=1} p(x_i|y=1)p(y=0)    }\\
&= \frac00\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

这是因为对于"   <img src="https://www.zhihu.com/equation?tex=\prod^n_{i=1} p(x_i|y)" alt="\prod^n_{i=1} p(x_i|y)" class="ee_img tr_noresize" eeimg="1"> "中包含了 <img src="https://www.zhihu.com/equation?tex=p(x_{35000}|y) = 0" alt="p(x_{35000}|y) = 0" class="ee_img tr_noresize" eeimg="1"> 的都加了起来，也就还是 <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> 。所以我们的算法得到的就是  <img src="https://www.zhihu.com/equation?tex=\frac00" alt="\frac00" class="ee_img tr_noresize" eeimg="1"> ，也就是不知道该做出怎么样的预测了。

然后进一步拓展一下这个问题，统计学上来说，只因为你在自己以前的有限的训练数据集中没见到过一件事，就估计这个事件的概率为零，这明显不是个好主意。假设问题是估计一个多项式随机变量  <img src="https://www.zhihu.com/equation?tex=z" alt="z" class="ee_img tr_noresize" eeimg="1">  ，其取值范围在 <img src="https://www.zhihu.com/equation?tex=\{1,..., k\}" alt="\{1,..., k\}" class="ee_img tr_noresize" eeimg="1"> 之内。接下来就可以用 <img src="https://www.zhihu.com/equation?tex=\phi_i = p (z = i)" alt="\phi_i = p (z = i)" class="ee_img tr_noresize" eeimg="1">  来作为多项式参数。给定一个  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  个独立观测 <img src="https://www.zhihu.com/equation?tex=\{z^{(1)}, ..., z^{(m)}\}" alt="\{z^{(1)}, ..., z^{(m)}\}" class="ee_img tr_noresize" eeimg="1">  组成的集合，然后最大似然估计的形式如下：


<img src="https://www.zhihu.com/equation?tex=\phi_j=\frac{\sum^m_{i=1}1\{z^{(i)}=j\}}m
" alt="\phi_j=\frac{\sum^m_{i=1}1\{z^{(i)}=j\}}m
" class="ee_img tr_noresize" eeimg="1">

正如咱们之前见到的，如果我们用这些最大似然估计，那么一些 <img src="https://www.zhihu.com/equation?tex=\phi_j" alt="\phi_j" class="ee_img tr_noresize" eeimg="1"> 可能最终就是零了，这就是个问题了。要避免这个情况，我们可以引入**拉普拉斯平滑（Laplace smoothing），** 这种方法把上面的估计替换成：


<img src="https://www.zhihu.com/equation?tex=\phi_j=\frac{\sum^m_{i=1}1\{z^{(i)}=j\}+1}{m+k}
" alt="\phi_j=\frac{\sum^m_{i=1}1\{z^{(i)}=j\}+1}{m+k}
" class="ee_img tr_noresize" eeimg="1">

这里首先是对分子加 <img src="https://www.zhihu.com/equation?tex=1" alt="1" class="ee_img tr_noresize" eeimg="1"> ，然后对分母加 <img src="https://www.zhihu.com/equation?tex=k" alt="k" class="ee_img tr_noresize" eeimg="1"> ，要注意 <img src="https://www.zhihu.com/equation?tex=\sum^k_{j=1} \phi_j = 1" alt="\sum^k_{j=1} \phi_j = 1" class="ee_img tr_noresize" eeimg="1"> 依然成立（自己检验一下），这是一个必须有的性质，因为 <img src="https://www.zhihu.com/equation?tex=\phi_j" alt="\phi_j" class="ee_img tr_noresize" eeimg="1">  是对概率的估计，然后所有的概率加到一起必然等于 <img src="https://www.zhihu.com/equation?tex=1" alt="1" class="ee_img tr_noresize" eeimg="1"> 。另外对于所有的  <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1">  值，都有 <img src="https://www.zhihu.com/equation?tex=\phi_j \neq 0" alt="\phi_j \neq 0" class="ee_img tr_noresize" eeimg="1"> ，这就解决了刚刚的概率估计为零的问题了。在某些特定的条件下（相当强的假设条件下，arguably quite strong），可以发现拉普拉斯平滑还真能给出对参数 <img src="https://www.zhihu.com/equation?tex=\phi_j" alt="\phi_j" class="ee_img tr_noresize" eeimg="1">  的最佳估计（optimal estimator）。

回到我们的朴素贝叶斯分选器问题上，使用了拉普拉斯平滑之后，对参数的估计就写成了下面的形式：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\phi_{j|y=1} & =\frac{\sum^m_{i=1}1\{x_j^{(i)}=1\wedge y ^{(i)}=1\}+1}{\sum^m_{i=1}1{\{y^{(i)}=1\}}+2}\\
\phi_{j|y=0} & =\frac{\sum^m_{i=1}1\{x_j^{(i)}=1\wedge y ^{(i)}=10\}+1}{\sum^m_{i=1}1{\{y^{(i)}=0\}}+2}\\
\end{aligned}
" alt="\begin{aligned}
\phi_{j|y=1} & =\frac{\sum^m_{i=1}1\{x_j^{(i)}=1\wedge y ^{(i)}=1\}+1}{\sum^m_{i=1}1{\{y^{(i)}=1\}}+2}\\
\phi_{j|y=0} & =\frac{\sum^m_{i=1}1\{x_j^{(i)}=1\wedge y ^{(i)}=10\}+1}{\sum^m_{i=1}1{\{y^{(i)}=0\}}+2}\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

（在实际应用中，通常是否对 <img src="https://www.zhihu.com/equation?tex=\phi_y" alt="\phi_y" class="ee_img tr_noresize" eeimg="1">  使用拉普拉斯并没有太大影响，因为通常我们会对每个垃圾邮件和非垃圾邮件都有一个合适的划分比例，所以 <img src="https://www.zhihu.com/equation?tex=\phi_y" alt="\phi_y" class="ee_img tr_noresize" eeimg="1">  会是对 <img src="https://www.zhihu.com/equation?tex=p(y = 1)" alt="p(y = 1)" class="ee_img tr_noresize" eeimg="1">  的一个合理估计，无论如何都会与零点有一定距离。）

##### 2.2 针对文本分类的事件模型（Event models for text classification）

到这里就要给我们关于生成学习算法的讨论进行收尾了，所以就接着讲一点关于文本分类方面的另一个模型。上面的朴素贝叶斯方法能够解决很多分类问题，不过还有另一个相关的算法，在针对文本的分类效果还要更好。

在针对文本进行分类的特定背景下，上面讲的朴素贝叶斯方法使用的是一种叫做**多元伯努利事件模型（Multi-Variate Bernoulli event model）。** 在这个模型里面，我们假设邮件发送的方式，是随机确定的（根据先验类*class priors*，  <img src="https://www.zhihu.com/equation?tex=p(y)" alt="p(y)" class="ee_img tr_noresize" eeimg="1"> ），无论是不是垃圾邮件发送者，他是否给你发下一封邮件都是随机决定的。那么发件人就会遍历词典，决定在邮件中是否包含某个单词  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> ，各个单词之间互相独立，并且服从概率分布 <img src="https://www.zhihu.com/equation?tex=p(x_i=1|y)=\phi_{i|y}" alt="p(x_i=1|y)=\phi_{i|y}" class="ee_img tr_noresize" eeimg="1"> 。因此，一条消息的概率为： <img src="https://www.zhihu.com/equation?tex=p(y)\prod^n_{i=1}p(x_i|y)" alt="p(y)\prod^n_{i=1}p(x_i|y)" class="ee_img tr_noresize" eeimg="1"> 

 然后还有另外一个模型，叫做**多项式事件模型（Multinomial event model）。** 要描述这个模型，我们需要使用一个不同的记号和特征集来表征各种邮件。设  <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1">  表示单词中的第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 个单词。因此， <img src="https://www.zhihu.com/equation?tex=x_i" alt="x_i" class="ee_img tr_noresize" eeimg="1"> 现在就是一个整数，取值范围为 <img src="https://www.zhihu.com/equation?tex=\{1,...,|V|\}" alt="\{1,...,|V|\}" class="ee_img tr_noresize" eeimg="1"> ，这里的 <img src="https://www.zhihu.com/equation?tex=|V|" alt="|V|" class="ee_img tr_noresize" eeimg="1"> 是词汇列表（即字典）的长度。这样一个有  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  个单词的邮件就可以表征为一个长度为  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  的向量 <img src="https://www.zhihu.com/equation?tex=(x_1,x_2,...,x_n)" alt="(x_1,x_2,...,x_n)" class="ee_img tr_noresize" eeimg="1"> ；这里要注意，不同的邮件内容， <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  的取值可以是不同的。例如，如果一个邮件的开头是"A NIPS . . ." ，那么 <img src="https://www.zhihu.com/equation?tex=x_1 = 1" alt="x_1 = 1" class="ee_img tr_noresize" eeimg="1">  ("a" 是词典中的第一个)，而 <img src="https://www.zhihu.com/equation?tex=x_2 = 35000" alt="x_2 = 35000" class="ee_img tr_noresize" eeimg="1">  (这是假设 "nips"是词典中的第35000个)。

在多项式事件模型中，我们假设邮件的生成是通过一个随机过程的，而是否为垃圾邮件是首先决定的（根据 <img src="https://www.zhihu.com/equation?tex=p(y)" alt="p(y)" class="ee_img tr_noresize" eeimg="1"> ），这个和之前的模型假设一样。然后邮件的发送者写邮件首先是要生成  从对单词 <img src="https://www.zhihu.com/equation?tex=(p(x_1|y))" alt="(p(x_1|y))" class="ee_img tr_noresize" eeimg="1">  的某种多项式分布中生成  <img src="https://www.zhihu.com/equation?tex=x_1" alt="x_1" class="ee_img tr_noresize" eeimg="1"> 。然后第二步是独立于  <img src="https://www.zhihu.com/equation?tex=x_1" alt="x_1" class="ee_img tr_noresize" eeimg="1">  来生成  <img src="https://www.zhihu.com/equation?tex=x_2" alt="x_2" class="ee_img tr_noresize" eeimg="1"> ，但也是从相同的多项式分布中来选取，然后是  <img src="https://www.zhihu.com/equation?tex=x_3" alt="x_3" class="ee_img tr_noresize" eeimg="1"> , <img src="https://www.zhihu.com/equation?tex=x_4" alt="x_4" class="ee_img tr_noresize" eeimg="1">   等等，以此类推，直到生成了整个邮件中的所有的词。因此，一个邮件的总体概率就是 <img src="https://www.zhihu.com/equation?tex=p(y)\prod^n_{i=1}p(x_i|y)" alt="p(y)\prod^n_{i=1}p(x_i|y)" class="ee_img tr_noresize" eeimg="1"> 。要注意这个方程看着和我们之前那个多元伯努利事件模型里面的邮件概率很相似，但实际上这里面的意义完全不同了。尤其是这里的 <img src="https://www.zhihu.com/equation?tex=x_i|y" alt="x_i|y" class="ee_img tr_noresize" eeimg="1"> 现在是一个多项式分布了，而不是伯努利分布了。

我们新模型的参数还是 <img src="https://www.zhihu.com/equation?tex=\phi_y = p(y)" alt="\phi_y = p(y)" class="ee_img tr_noresize" eeimg="1"> ，这个跟以前一样，然后还有 <img src="https://www.zhihu.com/equation?tex=\phi_{k|y=1} = p(x_j =k|y=1)" alt="\phi_{k|y=1} = p(x_j =k|y=1)" class="ee_img tr_noresize" eeimg="1">  (对任何  <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1"> )以及  <img src="https://www.zhihu.com/equation?tex=\phi_{i|y=0} =p(x_j =k|y=0)" alt="\phi_{i|y=0} =p(x_j =k|y=0)" class="ee_img tr_noresize" eeimg="1"> 。要注意这里我们已经假设了对于任何 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1">  的值， <img src="https://www.zhihu.com/equation?tex=p(x_j|y)" alt="p(x_j|y)" class="ee_img tr_noresize" eeimg="1"> 这个概率都是相等的，也就是意味着在这个词汇生成的分布不依赖这个词在邮件中的位置 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1"> 。

如果给定一个训练集 <img src="https://www.zhihu.com/equation?tex=\{(x^{(i)},y^{(i)}); i = 1, ..., m\}" alt="\{(x^{(i)},y^{(i)}); i = 1, ..., m\}" class="ee_img tr_noresize" eeimg="1"> ，其中  <img src="https://www.zhihu.com/equation?tex=x^{(i)}  = ( x^{(i)}_{1} , x^{(i)}_{2} ,..., x^{(i)}_{n_i})" alt="x^{(i)}  = ( x^{(i)}_{1} , x^{(i)}_{2} ,..., x^{(i)}_{n_i})" class="ee_img tr_noresize" eeimg="1"> （这里的 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 是在第 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1"> 个训练样本中的单词数目），那么这个数据的似然函数如下所示：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\mathcal{L}(\phi,\phi_{k|y=0},\phi_{k|y=1})& = \prod^m_{i=1}p( x^{(i)},y^{(i)})\\
& = \prod^m_{i=1}(\prod^{n_i}_{j=1}p(x_j^{(i)}|y;\phi_{k|y=0},\phi_{k|y=1}))p( y^{(i)};\phi_y)\\
\end{aligned}
" alt="\begin{aligned}
\mathcal{L}(\phi,\phi_{k|y=0},\phi_{k|y=1})& = \prod^m_{i=1}p( x^{(i)},y^{(i)})\\
& = \prod^m_{i=1}(\prod^{n_i}_{j=1}p(x_j^{(i)}|y;\phi_{k|y=0},\phi_{k|y=1}))p( y^{(i)};\phi_y)\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

让上面的这个函数最大化就可以产生对参数的最大似然估计：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\phi_{k|y=1}&=  \frac{\sum^m_{i=1}\sum^{n_i}_{j=1}1\{x_j^{(i)}=k\wedge y^{(i)}=1\}}{\sum^m_{i=1}1\{y^{(i)}=1\}n_i} \\
\phi_{k|y=0}&=  \frac{\sum^m_{i=1}\sum^{n_i}_{j=1}1\{x_j^{(i)}=k\wedge y^{(i)}=0\}}{\sum^m_{i=1}1\{y^{(i)}=0\}n_i} \\
\phi_y&=   \frac{\sum^m_{i=1}1\{y^{(i)}=1\}}{m}\\
\end{aligned}
" alt="\begin{aligned}
\phi_{k|y=1}&=  \frac{\sum^m_{i=1}\sum^{n_i}_{j=1}1\{x_j^{(i)}=k\wedge y^{(i)}=1\}}{\sum^m_{i=1}1\{y^{(i)}=1\}n_i} \\
\phi_{k|y=0}&=  \frac{\sum^m_{i=1}\sum^{n_i}_{j=1}1\{x_j^{(i)}=k\wedge y^{(i)}=0\}}{\sum^m_{i=1}1\{y^{(i)}=0\}n_i} \\
\phi_y&=   \frac{\sum^m_{i=1}1\{y^{(i)}=1\}}{m}\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

如果使用拉普拉斯平滑（实践中会用这个方法来提高性能）来估计 <img src="https://www.zhihu.com/equation?tex=\phi_{k|y=0}" alt="\phi_{k|y=0}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=\phi_{k|y=1}" alt="\phi_{k|y=1}" class="ee_img tr_noresize" eeimg="1"> ，就在分子上加1，然后分母上加 <img src="https://www.zhihu.com/equation?tex=|V|" alt="|V|" class="ee_img tr_noresize" eeimg="1"> ，就得到了下面的等式：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\phi_{k|y=1}&=  \frac{\sum^m_{i=1}\sum^{n_i}_{j=1}1\{x_j^{(i)}=k\wedge y^{(i)}=1\}+1}{\sum^m_{i=1}1\{y^{(i)}=1\}n_i+|V|} \\
\phi_{k|y=0}&=  \frac{\sum^m_{i=1}\sum^{n_i}_{j=1}1\{x_j^{(i)}=k\wedge y^{(i)}=0\}+1}{\sum^m_{i=1}1\{y^{(i)}=0\}n_i+|V|} \\
\end{aligned}
" alt="\begin{aligned}
\phi_{k|y=1}&=  \frac{\sum^m_{i=1}\sum^{n_i}_{j=1}1\{x_j^{(i)}=k\wedge y^{(i)}=1\}+1}{\sum^m_{i=1}1\{y^{(i)}=1\}n_i+|V|} \\
\phi_{k|y=0}&=  \frac{\sum^m_{i=1}\sum^{n_i}_{j=1}1\{x_j^{(i)}=k\wedge y^{(i)}=0\}+1}{\sum^m_{i=1}1\{y^{(i)}=0\}n_i+|V|} \\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

当然了，朴素贝叶斯并不见得就是最好的分类算法，但由于大多数情况下贝叶斯的效果确实不错，所以这个方法就是一个很好的"首发选择"，因为它很简单又很好实现。后续我会陆续上传自己对于垃圾邮件的分类代码。

### 

## 参考链接

本笔记大量参考大佬在github上已经做好的学习笔记，感谢大佬的分享，特此声明。

**原作者**：[Andrew Ng  吴恩达](http://www.andrewng.org/)

**讲义翻译者**：[CycleUser](https://www.zhihu.com/people/cycleuser/columns)

**Github参考链接：**[Github 地址](https://github.com/Kivy-CN/Stanford-CS-229-CN)

[斯坦福大学 CS229 课程网站](http://cs229.stanford.edu/)

[知乎专栏](https://zhuanlan.zhihu.com/MachineLearn)

