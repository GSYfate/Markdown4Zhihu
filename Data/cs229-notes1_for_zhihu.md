# 斯坦福CS229


# 课程笔记第一章（一）

## 监督学习（Supervised learning）

**定义：**根据已有的数据集，知道输入和输出结果之间的关系。根据这种已知的关系，训练得到一个最优的模型。在监督学习中训练数据既有特征(feature)又有标签(label)，通过训练，让机器可以自己找到特征和标签之间的联系，在面对只有特征没有标签的数据时，可以判断出标签，也就是完成测试。

房价预测是使用监督学习来解决问题的一个实例。假如有这样一个数据集，里面的数据是俄勒冈州波特兰市的  <img src="https://www.zhihu.com/equation?tex=47" alt="47" class="ee_img tr_noresize" eeimg="1">  套房屋的面积和价格，如下表所示：

| 居住面积（平方英尺） | 价格（千美元） |

| :------------------: | :------------: |

|         <img src="https://www.zhihu.com/equation?tex=2104" alt="2104" class="ee_img tr_noresize" eeimg="1">         |      <img src="https://www.zhihu.com/equation?tex=400" alt="400" class="ee_img tr_noresize" eeimg="1">       |

|         <img src="https://www.zhihu.com/equation?tex=1600" alt="1600" class="ee_img tr_noresize" eeimg="1">         |      <img src="https://www.zhihu.com/equation?tex=330" alt="330" class="ee_img tr_noresize" eeimg="1">       |

|         <img src="https://www.zhihu.com/equation?tex=2400" alt="2400" class="ee_img tr_noresize" eeimg="1">         |      <img src="https://www.zhihu.com/equation?tex=369" alt="369" class="ee_img tr_noresize" eeimg="1">       |

|         <img src="https://www.zhihu.com/equation?tex=1416" alt="1416" class="ee_img tr_noresize" eeimg="1">         |      <img src="https://www.zhihu.com/equation?tex=232" alt="232" class="ee_img tr_noresize" eeimg="1">       |

|         <img src="https://www.zhihu.com/equation?tex=3000" alt="3000" class="ee_img tr_noresize" eeimg="1">         |      <img src="https://www.zhihu.com/equation?tex=540" alt="540" class="ee_img tr_noresize" eeimg="1">       |


将这些数据投影成图，方便进一步的分析：

<img src="https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes1/cs229note1f1.png" alt="cs229note1f1" style="zoom:67%;" />

取得这样的数据，我们要学会根据波特兰其他房屋的居住面积，预测这些房屋的价格。

首先先规范一下符号和含义，这些符号以后还会用到，假设  <img src="https://www.zhihu.com/equation?tex=x^{(i)}" alt="x^{(i)}" class="ee_img tr_noresize" eeimg="1">  表示 “输入的” 变量值（在这个例子中就是房屋面积），也可以叫做**输入特征**；用  <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">  来表示“输出值”，称之为**目标变量**，在这个例子里面就是房屋价格。这样的一对  <img src="https://www.zhihu.com/equation?tex=(x^{(i)},y^{(i)})" alt="(x^{(i)},y^{(i)})" class="ee_img tr_noresize" eeimg="1"> 就称为一组训练样本，作为机器进行学习的数据集，也就是一个长度为  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  的训练样本的列表 <img src="https://www.zhihu.com/equation?tex=\{(x^{(i)},y^{(i)}); i = 1,\dots ,m\}" alt="\{(x^{(i)},y^{(i)}); i = 1,\dots ,m\}" class="ee_img tr_noresize" eeimg="1"> 。这里的上标 <img src="https://www.zhihu.com/equation?tex=(i)" alt="(i)" class="ee_img tr_noresize" eeimg="1"> 只是作为训练集的索引记号，大写的 <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1"> 来表示 输入值的空间，大写的 <img src="https://www.zhihu.com/equation?tex=Y" alt="Y" class="ee_img tr_noresize" eeimg="1"> 表示输出值的空间。在本节的这个例子中，输入输出的空间都是实数域，所以  <img src="https://www.zhihu.com/equation?tex=X = Y = R" alt="X = Y = R" class="ee_img tr_noresize" eeimg="1"> 。

然后再用更加规范的方式来描述一下监督学习问题，我们的目标是，给定一个训练集，来让机器学习一个函数  <img src="https://www.zhihu.com/equation?tex=h: X → Y" alt="h: X → Y" class="ee_img tr_noresize" eeimg="1"> ，理想的结果是 <img src="https://www.zhihu.com/equation?tex=h(x)" alt="h(x)" class="ee_img tr_noresize" eeimg="1">  是一个与真实  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  值比较接近的评估值（预测值）。这个函数  <img src="https://www.zhihu.com/equation?tex=h" alt="h" class="ee_img tr_noresize" eeimg="1">  又被称为**假设（hypothesis）**。用一个图来表示的话，这个过程大概就是下面这样：

![](https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes1/cs229note1f2-1595488003652.png)

如果我们要预测的目标变量是连续的，这种学习问题就被称为**回归问题**，比如当前这个房屋价格-面积的案例，如果 <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1"> 只能取一小部分的离散的值（比如给定房屋面积，咱们要来确定这个房子是一个住宅还是公寓），这样的问题就叫做**分类问题。**分类问题的输出是有限的K个标签，而回归问题的输出是连续的值。


### 第一部分 线性回归

为了让我们的房屋案例更有意思，可以稍微对数据集进行一下补充，增加每一个房屋的卧室数目，进行更复杂的回归分析，具体数据如下表所示：

|居住面积（平方英尺）|卧室数目|价格（千美元）|

|:-:|:-:|:-:|

| <img src="https://www.zhihu.com/equation?tex=2104" alt="2104" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=3" alt="3" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=400" alt="400" class="ee_img tr_noresize" eeimg="1"> |

| <img src="https://www.zhihu.com/equation?tex=1600" alt="1600" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=3" alt="3" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=330" alt="330" class="ee_img tr_noresize" eeimg="1"> |

| <img src="https://www.zhihu.com/equation?tex=2400" alt="2400" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=3" alt="3" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=369" alt="369" class="ee_img tr_noresize" eeimg="1"> |

| <img src="https://www.zhihu.com/equation?tex=1416" alt="1416" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=2" alt="2" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=232" alt="232" class="ee_img tr_noresize" eeimg="1"> |

| <img src="https://www.zhihu.com/equation?tex=3000" alt="3000" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=4" alt="4" class="ee_img tr_noresize" eeimg="1"> | <img src="https://www.zhihu.com/equation?tex=540" alt="540" class="ee_img tr_noresize" eeimg="1"> |

| <img src="https://www.zhihu.com/equation?tex=\vdots" alt="\vdots" class="ee_img tr_noresize" eeimg="1">  | <img src="https://www.zhihu.com/equation?tex=\vdots" alt="\vdots" class="ee_img tr_noresize" eeimg="1">  | <img src="https://www.zhihu.com/equation?tex=\vdots" alt="\vdots" class="ee_img tr_noresize" eeimg="1">  |


现在，输入特征  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  就是在  <img src="https://www.zhihu.com/equation?tex=R^2" alt="R^2" class="ee_img tr_noresize" eeimg="1">  范围取值的一个二维向量了。  <img src="https://www.zhihu.com/equation?tex=x_1^{(i)}" alt="x_1^{(i)}" class="ee_img tr_noresize" eeimg="1">  就是训练集中第  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  个房屋的面积，而  <img src="https://www.zhihu.com/equation?tex=x_2^{(i)}" alt="x_2^{(i)}" class="ee_img tr_noresize" eeimg="1">   就是训练集中第  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  个房屋的卧室数目。（通常来说，设计一个学习算法的时候，选择哪些输入特征都取决于你，所以如果你不在波特兰收集房屋信息数据，你也完全可以选择包含其他的特征，例如房屋是否有壁炉，卫生间的数量等等。机器学习的一个重要的内容就是数据的预处理和特征的提取，如何从所给数据集中选出合适的特征参与训练，是我们在学习任务中首要思考的东西。关于特征筛选的内容会在后面给出，此处不再赘述。）

要进行监督学习，首先要确定好如何在计算机里面对**函数/假设**  <img src="https://www.zhihu.com/equation?tex=h" alt="h" class="ee_img tr_noresize" eeimg="1">  进行表示。首先可以以线性函数为例进行学习，把  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  假设为一个以  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  为变量的线性函数：


<img src="https://www.zhihu.com/equation?tex=h_\theta  (x) = \theta_0 + \theta_1 \times x_1 + \theta_2 \times x_2
" alt="h_\theta  (x) = \theta_0 + \theta_1 \times x_1 + \theta_2 \times x_2
" class="ee_img tr_noresize" eeimg="1">

这里的 <img src="https://www.zhihu.com/equation?tex=\theta_i" alt="\theta_i" class="ee_img tr_noresize" eeimg="1"> 是**参数**（也可以叫做**权重**），是从  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  到  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  的线性函数映射的空间参数。在不至于引起混淆的情况下，可以把 <img src="https://www.zhihu.com/equation?tex=h_\theta(x)" alt="h_\theta(x)" class="ee_img tr_noresize" eeimg="1">  里面的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">   省略掉，就简写成  <img src="https://www.zhihu.com/equation?tex=h(x)" alt="h(x)" class="ee_img tr_noresize" eeimg="1"> 。另外为了简化公式，可以设  <img src="https://www.zhihu.com/equation?tex=x_0 = 1" alt="x_0 = 1" class="ee_img tr_noresize" eeimg="1"> （这个为 **截距项 intercept term**）。这样简化之后就有了：


<img src="https://www.zhihu.com/equation?tex=h(x) = \sum^n_{i=0}  \theta_i x_i = \theta^T x
" alt="h(x) = \sum^n_{i=0}  \theta_i x_i = \theta^T x
" class="ee_img tr_noresize" eeimg="1">

等式最右边的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  都是向量，等式中的  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  是输入变量的个数（不包括 <img src="https://www.zhihu.com/equation?tex=x_0" alt="x_0" class="ee_img tr_noresize" eeimg="1"> ）。

现在，给定了一个训练集，咱们怎么来挑选/学习参数  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  呢？一个看上去比较合理的方法就是让  <img src="https://www.zhihu.com/equation?tex=h(x)" alt="h(x)" class="ee_img tr_noresize" eeimg="1">  尽量逼近  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1"> ，至少对咱已有的训练样本能适用。用公式的方式来表示的话，就要定义一个函数，来衡量对于每个不同的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  值， <img src="https://www.zhihu.com/equation?tex=h(x^{(i)})" alt="h(x^{(i)})" class="ee_img tr_noresize" eeimg="1">  与对应的  <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">  的距离。这样用如下的方式定义了一个 **成本函数 （cost function**）:


<img src="https://www.zhihu.com/equation?tex=J(\theta) = \frac 12 \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2
" alt="J(\theta) = \frac 12 \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2
" class="ee_img tr_noresize" eeimg="1">

如果之前你接触过线性回归，你会发现这个函数和**最小二乘法** 拟合模型中的最小二乘法成本函数非常相似。随后我们要引入一个特殊的算法——最小均方算法。

#### 1 最小均方算法（LMS algorithm）

训练过程中，我们希望选择一个能让  <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1"> 达到 最小的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  值。如何让  <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1"> 达到最小呢？可以先选用一个搜索的算法，从某一个对  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的“初始猜测值”（预设的初值），然后对  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  值不断进行调整，来让  <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1">  逐渐变小，最好是直到我们能够达到一个使  <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1">  最小的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 。具体可以考虑使用梯度下降法（gradient descent algorithm）。

梯度下降法就是 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  从一个预设的的初始值开始，按照下面的公式逐渐重复更新： <img src="https://www.zhihu.com/equation?tex=^1" alt="^1" class="ee_img tr_noresize" eeimg="1"> 

<img src="https://www.zhihu.com/equation?tex=\theta_j := \theta_j - \alpha \frac \partial {\partial\theta_j}J(\theta)
" alt="\theta_j := \theta_j - \alpha \frac \partial {\partial\theta_j}J(\theta)
" class="ee_img tr_noresize" eeimg="1">

（上面的这个更新要同时对应从  <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1">  到  <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1">  的所有 <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1">  值进行。）这里的  <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1">  也称为学习速率。这个算法是很自然的，逐步重复朝向  <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  降低最快的方向移动。这里 <img src="https://www.zhihu.com/equation?tex=\alpha " alt="\alpha " class="ee_img tr_noresize" eeimg="1"> 的大小选取一定要合适，如果过大会导致错过最低点，如果过小会导致降低速度过于缓慢。

>1 本文中  <img src="https://www.zhihu.com/equation?tex=:= " alt=":= " class="ee_img tr_noresize" eeimg="1">  表示的是计算机程序中的一种赋值操作，是把等号右边的计算结果赋值给左边的变量， <img src="https://www.zhihu.com/equation?tex=a := b" alt="a := b" class="ee_img tr_noresize" eeimg="1">  就表示用  <img src="https://www.zhihu.com/equation?tex=b" alt="b" class="ee_img tr_noresize" eeimg="1">  的值覆盖原有的 <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1"> 值。要注意区分，如果写的是  <img src="https://www.zhihu.com/equation?tex=a == b" alt="a == b" class="ee_img tr_noresize" eeimg="1">  则表示的是判断二者相等的关系。（译者注：在 Python 中，单个等号  <img src="https://www.zhihu.com/equation?tex==" alt="=" class="ee_img tr_noresize" eeimg="1">  就是赋值，两个等号  <img src="https://www.zhihu.com/equation?tex===" alt="==" class="ee_img tr_noresize" eeimg="1">   表示相等关系的判断。）

要实现这个算法，需要解决等号右边的导数项。首先来解决只有一组训练样本  <img src="https://www.zhihu.com/equation?tex=(x, y)" alt="(x, y)" class="ee_img tr_noresize" eeimg="1">  的情况，忽略掉等号右边对  <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  的求和。公式如下所示，求导后二阶函数变为一阶函数：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\frac \partial {\partial\theta_j}J(\theta) & = \frac \partial {\partial\theta_j} \frac  12(h_\theta(x)-y)^2\\
& = 2 \cdot\frac 12(h_\theta(x)-y)\cdot \frac \partial {\partial\theta_j}  (h_\theta(x)-y) \\
& = (h_\theta(x)-y)\cdot \frac \partial {\partial\theta_j}(\sum^n_{i=0} \theta_ix_i-y) \\
& = (h_\theta(x)-y) x_j
\end{aligned}
" alt="\begin{aligned}
\frac \partial {\partial\theta_j}J(\theta) & = \frac \partial {\partial\theta_j} \frac  12(h_\theta(x)-y)^2\\
& = 2 \cdot\frac 12(h_\theta(x)-y)\cdot \frac \partial {\partial\theta_j}  (h_\theta(x)-y) \\
& = (h_\theta(x)-y)\cdot \frac \partial {\partial\theta_j}(\sum^n_{i=0} \theta_ix_i-y) \\
& = (h_\theta(x)-y) x_j
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

对单个训练样本，更新规则如下所示：


<img src="https://www.zhihu.com/equation?tex=\theta_j := \theta_j + \alpha (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}
" alt="\theta_j := \theta_j + \alpha (y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}
" class="ee_img tr_noresize" eeimg="1">

这个规则也叫 **LMS** 更新规则 （LMS 是 “least mean squares” 的缩写，意思是最小均方），也被称为 **Widrow-Hoff** 学习规则。这个规则有几个自然直观的特性。例如，更新的大小与 <img src="https://www.zhihu.com/equation?tex=(y^{(i)} − h_\theta(x^{(i)}))" alt="(y^{(i)} − h_\theta(x^{(i)}))" class="ee_img tr_noresize" eeimg="1"> 成正比；另外，当我们遇到训练样本的预测值与  <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">  的真实值非常接近的情况下，就会发现基本没必要再对参数进行修改了；与此相反的情况是，如果我们的预测值  <img src="https://www.zhihu.com/equation?tex=h_\theta(x^{(i)})" alt="h_\theta(x^{(i)})" class="ee_img tr_noresize" eeimg="1">  与  <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">  的真实值有很大的误差（比如距离特别远），那就需要对参数进行更大地调整。

当只有一个训练样本的时候，我们推导出了 LMS 规则。当一个训练集有超过一个训练样本的时候，有两种对这个规则的修改方法。第一种就是下面这个算法：

$
\begin{aligned}
&\qquad 重复直到收敛 \{ \\
&\qquad\qquad\theta_j := \theta_j + \alpha \sum^m_{i=1}(y^{(i)}-h_\theta (x^{(i)}))x_j^{(i)}\quad(对每个j) \\
&\qquad\}
\end{aligned}
$

读者很容易能证明，在上面这个更新规则中求和项的值就是 <img src="https://www.zhihu.com/equation?tex=\frac {\partial J(\theta)}{\partial \theta_j}" alt="\frac {\partial J(\theta)}{\partial \theta_j}" class="ee_img tr_noresize" eeimg="1">  。所以这个更新规则实际上就是对原始的成本函数  <img src="https://www.zhihu.com/equation?tex=J " alt="J " class="ee_img tr_noresize" eeimg="1"> 进行简单的梯度下降。此时移动的距离就叫做步长，这一方法会在每一个步长内检查所有整个训练集中的所有样本，也叫做**批量梯度下降法（batch gradient descent**）。这里要注意，因为梯度下降法容易被局部最小值影响，而我们要解决的这个线性回归的优化问题需要的是一个全局的而不是局部的最优解；因此，梯度下降法应该总是收敛到全局最小值（假设学习速率  <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1">  不设置的过大）。 <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  很明确是一个凸二次函数。下面是一个样例，其中对一个二次函数使用了梯度下降法来找到最小值。

<img src="https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes1/cs229note1f3.png" alt="cs229note1f3" style="zoom:50%;" />

上图的椭圆就是一个二次函数的轮廓图。图中还有梯度下降法生成的规矩，初始点位置在 <img src="https://www.zhihu.com/equation?tex=(48,30)" alt="(48,30)" class="ee_img tr_noresize" eeimg="1"> 。图中的画的  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  标记了梯度下降法所经过的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的可用值。对之前的房屋数据集进行批量梯度下降来拟合  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  ，把房屋价格当作房屋面积的函数来进行预测，我们得到的结果是  <img src="https://www.zhihu.com/equation?tex=\theta_0 = 71.27, \theta_1 = 0.1345" alt="\theta_0 = 71.27, \theta_1 = 0.1345" class="ee_img tr_noresize" eeimg="1"> 。如果把  <img src="https://www.zhihu.com/equation?tex=h_{\theta}(x)" alt="h_{\theta}(x)" class="ee_img tr_noresize" eeimg="1">  作为一个定义域在  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  上的函数来投影，同时也投上训练集中的已有数据点，会得到下面这幅图：

<img src="https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes1/cs229note1f4.png" alt="cs229note1f4" style="zoom:67%;" />

如果在数据集中添加上卧室数目作为输入特征，那么得到的结果就是  <img src="https://www.zhihu.com/equation?tex=\theta_0 = 89.60, \theta_1 = 0.1392, \theta_2 = −8.738" alt="\theta_0 = 89.60, \theta_1 = 0.1392, \theta_2 = −8.738" class="ee_img tr_noresize" eeimg="1"> 

这个结果就是用批量梯度下降法来获得的。此外还有另外一种方法能够替代批量梯度下降法，这种方法效果也不错。如下所示：

$
\begin{aligned}
&\qquad循环：\{ \\
&\qquad\qquad i从1到m,\{   \\
&\qquad\qquad\qquad\theta_j := \theta_j  +\alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)} \qquad(对每个 j) \\
&\qquad\qquad\}  \\
&\qquad\}
\end{aligned}
$

在这个算法里，我们对整个训练集进行了循环遍历，每次遇到一个训练样本，根据每个单一训练样本的误差梯度来对参数进行更新。这个算法叫做**随机梯度下降法（stochastic gradient descent）**，或者叫**增量梯度下降法（incremental gradient descent）**。批量梯度下降法要在运行第一步之前先对整个训练集进行扫描遍历，当训练集的规模  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  变得很大的时候，引起的性能开销就很不划算了；随机梯度下降法就没有这个问题，而是可以立即开始，对查询到的每个样本都进行运算。通常情况下，随机梯度下降法查找到足够接近最低值的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的速度要比批量梯度下降法更快一些。（也要注意，也有可能会一直无法收敛（converge）到最小值，这时候  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  会一直在  <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1">  最小值附近震荡；不过通常情况下在最小值附近的这些值大多数其实也足够逼近了，足以满足咱们的精度要求，所以也可以用。 <img src="https://www.zhihu.com/equation?tex=^2" alt="^2" class="ee_img tr_noresize" eeimg="1"> ）由于这些原因，特别是在训练集很大的情况下，随机梯度下降往往比批量梯度下降更受青睐。

>2 当然更常见的情况通常是我们事先对数据集已经有了描述，并且有了一个确定的学习速率 <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1"> ，然后来运行随机梯度下降，同时逐渐让学习速率  <img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1">  随着算法的运行而逐渐趋于  <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> ，这样也能保证我们最后得到的参数会收敛到最小值，而不是在最小值范围进行震荡。

综上，更推荐使用随机梯度下降法而不是批量梯度下降


#### 2 正则方程（The normal equations）

上文中的梯度下降法是一种找出  <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  最小值的办法。事实上还有另一种实现方法——正则方程，这种方法寻找过程简单明了，而且不需要使用迭代算法。其基本思路是，通过特定方法直接找到导数为0的位置对应的的  <img src="https://www.zhihu.com/equation?tex=\theta_j" alt="\theta_j" class="ee_img tr_noresize" eeimg="1"> ，这样就能找到  <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  的最小值了。


##### 2.1 矩阵导数（Matrix derivatives）

假如有一个函数  <img src="https://www.zhihu.com/equation?tex=f: R^{m\times n} → R" alt="f: R^{m\times n} → R" class="ee_img tr_noresize" eeimg="1">  从  <img src="https://www.zhihu.com/equation?tex=m\times n" alt="m\times n" class="ee_img tr_noresize" eeimg="1">  大小的矩阵映射到实数域，那么就可以定义当矩阵为  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  的时候有导函数  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1">  如下所示：


<img src="https://www.zhihu.com/equation?tex=\nabla_A f(A)=\begin{bmatrix} \frac {\partial f}{\partial A_{11}} & \dots  & \frac {\partial f}{\partial A_{1n}} \\ \vdots  & \ddots & \vdots  \\ \frac {\partial f}{\partial A_{m1}} & \dots  & \frac {\partial f}{\partial A_{mn}} \\ \end{bmatrix}
" alt="\nabla_A f(A)=\begin{bmatrix} \frac {\partial f}{\partial A_{11}} & \dots  & \frac {\partial f}{\partial A_{1n}} \\ \vdots  & \ddots & \vdots  \\ \frac {\partial f}{\partial A_{m1}} & \dots  & \frac {\partial f}{\partial A_{mn}} \\ \end{bmatrix}
" class="ee_img tr_noresize" eeimg="1">

因此，这个梯度  <img src="https://www.zhihu.com/equation?tex=\nabla_A f(A)" alt="\nabla_A f(A)" class="ee_img tr_noresize" eeimg="1"> 本身也是一个  <img src="https://www.zhihu.com/equation?tex=m\times n" alt="m\times n" class="ee_img tr_noresize" eeimg="1">  的矩阵，其中的第  <img src="https://www.zhihu.com/equation?tex=(i,j)" alt="(i,j)" class="ee_img tr_noresize" eeimg="1">  个元素是  <img src="https://www.zhihu.com/equation?tex=\frac {\partial f}{\partial A_{ij}} " alt="\frac {\partial f}{\partial A_{ij}} " class="ee_img tr_noresize" eeimg="1">  。
假设  <img src="https://www.zhihu.com/equation?tex= A =\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \\ \end{bmatrix} " alt=" A =\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \\ \end{bmatrix} " class="ee_img tr_noresize" eeimg="1">  是一个  <img src="https://www.zhihu.com/equation?tex=2\times 2" alt="2\times 2" class="ee_img tr_noresize" eeimg="1">   矩阵，然后给定的函数  <img src="https://www.zhihu.com/equation?tex=f:R^{2\times 2} → R" alt="f:R^{2\times 2} → R" class="ee_img tr_noresize" eeimg="1">  为:

<img src="https://www.zhihu.com/equation?tex=f(A) = \frac 32A_{11}+5A^2_{12}+A_{21}A_{22}
" alt="f(A) = \frac 32A_{11}+5A^2_{12}+A_{21}A_{22}
" class="ee_img tr_noresize" eeimg="1">

这里面的  <img src="https://www.zhihu.com/equation?tex=A_{ij}" alt="A_{ij}" class="ee_img tr_noresize" eeimg="1">  表示的意思是矩阵  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  的第  <img src="https://www.zhihu.com/equation?tex=(i,j)" alt="(i,j)" class="ee_img tr_noresize" eeimg="1">  个元素。于是就可以计算梯度：


<img src="https://www.zhihu.com/equation?tex=\nabla _A f(A) =\begin{bmatrix} \frac  32 & 10A_{12} \\ A_{22} & A_{21} \\ \end{bmatrix}
" alt="\nabla _A f(A) =\begin{bmatrix} \frac  32 & 10A_{12} \\ A_{22} & A_{21} \\ \end{bmatrix}
" class="ee_img tr_noresize" eeimg="1">

还要引入 ** <img src="https://www.zhihu.com/equation?tex=trace" alt="trace" class="ee_img tr_noresize" eeimg="1"> ** 求迹运算，简写为  <img src="https://www.zhihu.com/equation?tex=“tr”" alt="“tr”" class="ee_img tr_noresize" eeimg="1"> 。对于一个给定的  <img src="https://www.zhihu.com/equation?tex=n\times n" alt="n\times n" class="ee_img tr_noresize" eeimg="1">  方形矩阵  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1"> ，它的迹定义为对角项和，即主对角线上元素之和：


<img src="https://www.zhihu.com/equation?tex=trA = \sum^n_{i=1} A_{ii}
" alt="trA = \sum^n_{i=1} A_{ii}
" class="ee_img tr_noresize" eeimg="1">

假如  <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1">  是一个实数，实际上  <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1">  就可以看做是一个  <img src="https://www.zhihu.com/equation?tex=1\times 1" alt="1\times 1" class="ee_img tr_noresize" eeimg="1">  的矩阵，那么就有  <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1">  的迹  <img src="https://www.zhihu.com/equation?tex=tr a = a" alt="tr a = a" class="ee_img tr_noresize" eeimg="1"> 。(如果你之前没有见到过这个“运算记号”，就可以把  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  的迹看成是  <img src="https://www.zhihu.com/equation?tex=tr(A)" alt="tr(A)" class="ee_img tr_noresize" eeimg="1"> ，或者理解成为一个对矩阵  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  进行操作的  <img src="https://www.zhihu.com/equation?tex=trace" alt="trace" class="ee_img tr_noresize" eeimg="1">  函数。不过通常情况都是写成不带括号的形式更多一些。) 

如果有两个矩阵  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  和 <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1"> ，能够满足  <img src="https://www.zhihu.com/equation?tex=AB" alt="AB" class="ee_img tr_noresize" eeimg="1">  为方阵， <img src="https://www.zhihu.com/equation?tex=trace" alt="trace" class="ee_img tr_noresize" eeimg="1">  求迹运算就有一个特殊的性质：  <img src="https://www.zhihu.com/equation?tex=trAB = trBA" alt="trAB = trBA" class="ee_img tr_noresize" eeimg="1">  (主对角线的元素和时不变的，可较简单的证明。

在此基础上进行推论，就能得到类似下面这样的等式关系：

<img src="https://www.zhihu.com/equation?tex=trABC=trCAB=trBCA \\
trABCD=trDABC=trCDAB=trBCDA
" alt="trABC=trCAB=trBCA \\
trABCD=trDABC=trCDAB=trBCDA
" class="ee_img tr_noresize" eeimg="1">

注意此处相对顺序不能改变，记忆是可以看做是一个循环滚动的形式。

下面这些和求迹运算相关的等量关系也很容易证明。其中  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1">  都是方形矩阵， <img src="https://www.zhihu.com/equation?tex=a" alt="a" class="ee_img tr_noresize" eeimg="1">  是一个实数：

<img src="https://www.zhihu.com/equation?tex=trA=trA^T \\
tr(A+B)=trA+trB \\
tr (a A)=a tr(A)
" alt="trA=trA^T \\
tr(A+B)=trA+trB \\
tr (a A)=a tr(A)
" class="ee_img tr_noresize" eeimg="1">

接下来提出一些矩阵导数（其中的一些直到本节末尾才用得上）。要注意等式 <img src="https://www.zhihu.com/equation?tex=(4)" alt="(4)" class="ee_img tr_noresize" eeimg="1"> 中的 <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  必须是**非奇异方阵（non-singular square matrices**），而  <img src="https://www.zhihu.com/equation?tex=|A|" alt="|A|" class="ee_img tr_noresize" eeimg="1">  表示的是矩阵  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  的行列式。那么我们就有下面这些等量关系：



<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
   \nabla_A tr AB & = B^T & \text{(1)}\\
   \nabla_{A^T} f(A) & = (\nabla_{A} f(A))^T &\text{(2)}\\
   \nabla_A tr ABA^TC& = CAB+C^TAB^T &\text{(3)}\\
   \nabla_A|A| & = |A|(A^{-1})^T &\text{(4)}\\
\end{aligned}
" alt="\begin{aligned}
   \nabla_A tr AB & = B^T & \text{(1)}\\
   \nabla_{A^T} f(A) & = (\nabla_{A} f(A))^T &\text{(2)}\\
   \nabla_A tr ABA^TC& = CAB+C^TAB^T &\text{(3)}\\
   \nabla_A|A| & = |A|(A^{-1})^T &\text{(4)}\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

为了使矩阵运算的记号更加具体，咱们就详细解释一下这些等式中的第一个。假如我们有一个确定的矩阵  <img src="https://www.zhihu.com/equation?tex=B \in R^{n\times m}" alt="B \in R^{n\times m}" class="ee_img tr_noresize" eeimg="1"> （注意顺序，是 <img src="https://www.zhihu.com/equation?tex=n\times m" alt="n\times m" class="ee_img tr_noresize" eeimg="1"> ，这里的意思也就是  <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1">  的元素都是实数， <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1">  的形状是  <img src="https://www.zhihu.com/equation?tex=n\times m" alt="n\times m" class="ee_img tr_noresize" eeimg="1">  的一个矩阵），那么接下来就可以定义一个函数 <img src="https://www.zhihu.com/equation?tex= f: R^{m\times n} → R" alt=" f: R^{m\times n} → R" class="ee_img tr_noresize" eeimg="1">  ，对应这里的就是  <img src="https://www.zhihu.com/equation?tex=f(A) = tr(AB)" alt="f(A) = tr(AB)" class="ee_img tr_noresize" eeimg="1"> 。这里要注意，这个矩阵是有意义的，因为如果  <img src="https://www.zhihu.com/equation?tex=A \in R^{m\times n} " alt="A \in R^{m\times n} " class="ee_img tr_noresize" eeimg="1"> ，那么  <img src="https://www.zhihu.com/equation?tex=AB" alt="AB" class="ee_img tr_noresize" eeimg="1">  就是一个方阵，是方阵就可以应用  <img src="https://www.zhihu.com/equation?tex=trace" alt="trace" class="ee_img tr_noresize" eeimg="1">  求迹运算；因此，实际上  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1">  映射的是从  <img src="https://www.zhihu.com/equation?tex=R^{m\times n} " alt="R^{m\times n} " class="ee_img tr_noresize" eeimg="1">  到实数域  <img src="https://www.zhihu.com/equation?tex=R" alt="R" class="ee_img tr_noresize" eeimg="1"> 。这样接下来就可以使用矩阵导数来找到  <img src="https://www.zhihu.com/equation?tex=\nabla_Af(A)" alt="\nabla_Af(A)" class="ee_img tr_noresize" eeimg="1">  ，这个导函数本身也是一个  <img src="https://www.zhihu.com/equation?tex=m \times n " alt="m \times n " class="ee_img tr_noresize" eeimg="1"> 的矩阵。上面的等式 <img src="https://www.zhihu.com/equation?tex=(1)" alt="(1)" class="ee_img tr_noresize" eeimg="1">  表明了这个导数矩阵的第  <img src="https://www.zhihu.com/equation?tex=(i,j)" alt="(i,j)" class="ee_img tr_noresize" eeimg="1"> 个元素等同于  <img src="https://www.zhihu.com/equation?tex=B^T" alt="B^T" class="ee_img tr_noresize" eeimg="1">  （ <img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1"> 的转置）的第  <img src="https://www.zhihu.com/equation?tex=(i,j)" alt="(i,j)" class="ee_img tr_noresize" eeimg="1">  个元素，或者更直接表示成  <img src="https://www.zhihu.com/equation?tex=B_{ji}" alt="B_{ji}" class="ee_img tr_noresize" eeimg="1"> 。

上面等式 <img src="https://www.zhihu.com/equation?tex=(1-3)" alt="(1-3)" class="ee_img tr_noresize" eeimg="1">  都很简单，证明就都留给读者做练习了。等式 <img src="https://www.zhihu.com/equation?tex=(4)" alt="(4)" class="ee_img tr_noresize" eeimg="1"> 需要用逆矩阵的伴随矩阵来推导出。 <img src="https://www.zhihu.com/equation?tex=^3" alt="^3" class="ee_img tr_noresize" eeimg="1"> 

>逆矩阵：设 <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1"> 是数域上的一个n阶矩阵，若在相同数域上存在另一个n阶矩阵 <img src="https://www.zhihu.com/equation?tex=A^{-1}" alt="A^{-1}" class="ee_img tr_noresize" eeimg="1"> ，使得 <img src="https://www.zhihu.com/equation?tex=AA^{-1}=I" alt="AA^{-1}=I" class="ee_img tr_noresize" eeimg="1"> ,则称 <img src="https://www.zhihu.com/equation?tex=A^{-1}" alt="A^{-1}" class="ee_img tr_noresize" eeimg="1"> 为 <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1"> 的逆矩阵，而A则被称为[可逆矩阵](https://baike.baidu.com/item/可逆矩阵/11035614)。注： <img src="https://www.zhihu.com/equation?tex=I" alt="I" class="ee_img tr_noresize" eeimg="1"> 为[单位矩阵](https://baike.baidu.com/item/单位矩阵/8540268)。
>
>伴随矩阵：设矩阵 <img src="https://www.zhihu.com/equation?tex=A=(a_{ij})_{n*n}" alt="A=(a_{ij})_{n*n}" class="ee_img tr_noresize" eeimg="1"> ,将矩阵 <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1"> 的元素 <img src="https://www.zhihu.com/equation?tex=a_{ij}" alt="a_{ij}" class="ee_img tr_noresize" eeimg="1"> 所在的第i行第j列去掉，剩余的元素按原有的排列顺序组成的n-1阶矩阵所确定的行列式称为元素 <img src="https://www.zhihu.com/equation?tex=a_{ij}" alt="a_{ij}" class="ee_img tr_noresize" eeimg="1"> 的余子式，记为 <img src="https://www.zhihu.com/equation?tex=M_{ij}" alt="M_{ij}" class="ee_img tr_noresize" eeimg="1"> ，称 <img src="https://www.zhihu.com/equation?tex=A_{ij=}(-1)^{i+j}M_{ij}" alt="A_{ij=}(-1)^{i+j}M_{ij}" class="ee_img tr_noresize" eeimg="1"> 为元素 <img src="https://www.zhihu.com/equation?tex=a_{ij}" alt="a_{ij}" class="ee_img tr_noresize" eeimg="1"> 的代数余子式。
>
>定义一个矩阵  <img src="https://www.zhihu.com/equation?tex=A'" alt="A'" class="ee_img tr_noresize" eeimg="1"> ，它的第  <img src="https://www.zhihu.com/equation?tex=(i,j)" alt="(i,j)" class="ee_img tr_noresize" eeimg="1">  个元素是 <img src="https://www.zhihu.com/equation?tex= (−1)^{i+j}" alt=" (−1)^{i+j}" class="ee_img tr_noresize" eeimg="1">  与矩阵  <img src="https://www.zhihu.com/equation?tex=A " alt="A " class="ee_img tr_noresize" eeimg="1"> 移除 第  <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  行 和 第  <img src="https://www.zhihu.com/equation?tex=j" alt="j" class="ee_img tr_noresize" eeimg="1">  列 之后的行列式的乘积，最后组成一个与矩阵 <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1"> 大小相等的新的矩阵。可以证明有 <img src="https://www.zhihu.com/equation?tex=A^{−1} = (A')^T /|A|" alt="A^{−1} = (A')^T /|A|" class="ee_img tr_noresize" eeimg="1"> 。这也就意味着  <img src="https://www.zhihu.com/equation?tex=A' = |A|(A^{−1})^T " alt="A' = |A|(A^{−1})^T " class="ee_img tr_noresize" eeimg="1"> 。此外，一个矩阵  <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">  的行列式也可以写成  <img src="https://www.zhihu.com/equation?tex=|A| = \sum_j A_{ij}A'_{ij}" alt="|A| = \sum_j A_{ij}A'_{ij}" class="ee_img tr_noresize" eeimg="1">  。因为  <img src="https://www.zhihu.com/equation?tex=(A')_{ij}" alt="(A')_{ij}" class="ee_img tr_noresize" eeimg="1">  不依赖  <img src="https://www.zhihu.com/equation?tex=A_{ij}" alt="A_{ij}" class="ee_img tr_noresize" eeimg="1">  （通过定义也能看出来），这也就意味着 <img src="https://www.zhihu.com/equation?tex=(\frac  \partial {\partial A_{ij}})|A| = A'_{ij} " alt="(\frac  \partial {\partial A_{ij}})|A| = A'_{ij} " class="ee_img tr_noresize" eeimg="1"> ，综合起来也就得到上面的结果。

##### 2.2 最小二乘法回顾（Least squares revisited）

通过刚才的内容，我们大概掌握了矩阵导数这一工具，接下来就继续用逼近模型（closed-form）来找到能让  <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1">  最小的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  值。首先咱们把  <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  用矩阵-向量的记号来重新表述。

给定一个训练集，把**设计矩阵（design matrix）**  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  设置为一个  <img src="https://www.zhihu.com/equation?tex=m\times n" alt="m\times n" class="ee_img tr_noresize" eeimg="1">  矩阵（实际上，如果考虑到截距项，也就是  <img src="https://www.zhihu.com/equation?tex=\theta_0" alt="\theta_0" class="ee_img tr_noresize" eeimg="1">  那一项，就应该是  <img src="https://www.zhihu.com/equation?tex=m\times (n+1)" alt="m\times (n+1)" class="ee_img tr_noresize" eeimg="1">  矩阵），这个矩阵里面包含了训练样本的输入值每一行的 <img src="https://www.zhihu.com/equation?tex=x^i" alt="x^i" class="ee_img tr_noresize" eeimg="1"> 代表一个 <img src="https://www.zhihu.com/equation?tex=n" alt="n" class="ee_img tr_noresize" eeimg="1"> 或 <img src="https://www.zhihu.com/equation?tex=n+1" alt="n+1" class="ee_img tr_noresize" eeimg="1"> 列的行向量：


<img src="https://www.zhihu.com/equation?tex=X =\begin{bmatrix}
-(x^{(1)}) ^T-\\
-(x^{(2)}) ^T-\\
\vdots \\
-(x^{(m)}) ^T-\\
\end{bmatrix}
" alt="X =\begin{bmatrix}
-(x^{(1)}) ^T-\\
-(x^{(2)}) ^T-\\
\vdots \\
-(x^{(m)}) ^T-\\
\end{bmatrix}
" class="ee_img tr_noresize" eeimg="1">

然后，设  <img src="https://www.zhihu.com/equation?tex=\vec{y}" alt="\vec{y}" class="ee_img tr_noresize" eeimg="1">  是一个  <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1">  维向量（m-dimensional vector），其中包含了训练集中的所有目标值：


<img src="https://www.zhihu.com/equation?tex=y =\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
\vdots \\
y^{(m)}\\
\end{bmatrix}
" alt="y =\begin{bmatrix}
y^{(1)}\\
y^{(2)}\\
\vdots \\
y^{(m)}\\
\end{bmatrix}
" class="ee_img tr_noresize" eeimg="1">

因为  <img src="https://www.zhihu.com/equation?tex=h_\theta (x^{(i)}) = (x^{(i)})^T\theta " alt="h_\theta (x^{(i)}) = (x^{(i)})^T\theta " class="ee_img tr_noresize" eeimg="1"> （加入截距项，上文有推导），所以可以证明存在下面这种等量关系：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
X\theta - \vec{y}  &=
\begin{bmatrix}
(x^{(1)})^T\theta \\
\vdots \\
(x^{(m)})^T\theta\\
\end{bmatrix} -
\begin{bmatrix}
y^{(1)}\\
\vdots \\
y^{(m)}\\
\end{bmatrix}\\
& =
\begin{bmatrix}
h_\theta (x^{1}) -y^{(1)}\\
\vdots \\
h_\theta (x^{m})-y^{(m)}\\
\end{bmatrix}\\
\end{aligned}
" alt="\begin{aligned}
X\theta - \vec{y}  &=
\begin{bmatrix}
(x^{(1)})^T\theta \\
\vdots \\
(x^{(m)})^T\theta\\
\end{bmatrix} -
\begin{bmatrix}
y^{(1)}\\
\vdots \\
y^{(m)}\\
\end{bmatrix}\\
& =
\begin{bmatrix}
h_\theta (x^{1}) -y^{(1)}\\
\vdots \\
h_\theta (x^{m})-y^{(m)}\\
\end{bmatrix}\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

对于向量  <img src="https://www.zhihu.com/equation?tex=\vec{z}" alt="\vec{z}" class="ee_img tr_noresize" eeimg="1">  ，则有  <img src="https://www.zhihu.com/equation?tex=z^T z = \sum_i z_i^2" alt="z^T z = \sum_i z_i^2" class="ee_img tr_noresize" eeimg="1">  ，因此利用这个性质，可以推出:


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\frac 12(X\theta - \vec{y})^T (X\theta - \vec{y}) &=\frac 12 \sum^m_{i=1}(h_\theta (x^{(i)})-y^{(i)})^2\\
&= J(\theta)
\end{aligned}
" alt="\begin{aligned}
\frac 12(X\theta - \vec{y})^T (X\theta - \vec{y}) &=\frac 12 \sum^m_{i=1}(h_\theta (x^{(i)})-y^{(i)})^2\\
&= J(\theta)
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

最后，要让  <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  的值最小，就要找到函数对于 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 导数。结合等式 <img src="https://www.zhihu.com/equation?tex=(2)" alt="(2)" class="ee_img tr_noresize" eeimg="1"> 和等式 <img src="https://www.zhihu.com/equation?tex=(3)" alt="(3)" class="ee_img tr_noresize" eeimg="1"> ，就能得到下面这个等式 <img src="https://www.zhihu.com/equation?tex=(5)" alt="(5)" class="ee_img tr_noresize" eeimg="1"> ：


<img src="https://www.zhihu.com/equation?tex=\nabla_{A^T} trABA^TC =B^TA^TC^T+BA^TC \qquad \text{(5)}
" alt="\nabla_{A^T} trABA^TC =B^TA^TC^T+BA^TC \qquad \text{(5)}
" class="ee_img tr_noresize" eeimg="1">

因此就有：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \frac 12 (X\theta - \vec{y})^T (X\theta - \vec{y}) \\
&= \frac  12 \nabla_\theta (\theta ^TX^TX\theta -\theta^T X^T \vec{y} - \vec{y} ^TX\theta +\vec{y}^T \vec{y})\\
&= \frac  12 \nabla_\theta tr(\theta ^TX^TX\theta -\theta^T X^T \vec{y} - \vec{y} ^TX\theta +\vec{y}^T \vec{y})\\
&= \frac  12 \nabla_\theta (tr \theta ^TX^TX\theta - 2tr\vec{y} ^T X\theta)\\
&= \frac  12 (X^TX\theta+X^TX\theta-2X^T\vec{y}) \\
&= X^TX\theta-X^T\vec{y}\\
\end{aligned}
" alt="\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \frac 12 (X\theta - \vec{y})^T (X\theta - \vec{y}) \\
&= \frac  12 \nabla_\theta (\theta ^TX^TX\theta -\theta^T X^T \vec{y} - \vec{y} ^TX\theta +\vec{y}^T \vec{y})\\
&= \frac  12 \nabla_\theta tr(\theta ^TX^TX\theta -\theta^T X^T \vec{y} - \vec{y} ^TX\theta +\vec{y}^T \vec{y})\\
&= \frac  12 \nabla_\theta (tr \theta ^TX^TX\theta - 2tr\vec{y} ^T X\theta)\\
&= \frac  12 (X^TX\theta+X^TX\theta-2X^T\vec{y}) \\
&= X^TX\theta-X^T\vec{y}\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

- 第三步，我们用到了一个定理**：一个实数的迹就是这个实数**；

- 第四步用到了  <img src="https://www.zhihu.com/equation?tex=trA = trA^T" alt="trA = trA^T" class="ee_img tr_noresize" eeimg="1">  这个定理，常数项求导被消去；
- 第五步用到了等式 <img src="https://www.zhihu.com/equation?tex=(5)" alt="(5)" class="ee_img tr_noresize" eeimg="1"> ，其中  <img src="https://www.zhihu.com/equation?tex=A^T =\theta, B=B^T =X^TX, C=I" alt="A^T =\theta, B=B^T =X^TX, C=I" class="ee_img tr_noresize" eeimg="1"> ,还用到了等式  <img src="https://www.zhihu.com/equation?tex=(1)" alt="(1)" class="ee_img tr_noresize" eeimg="1"> 。
- 要让  <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  取得最小值，就设导数为  <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1">  ，然后就得到了下面的**法线方程（normal equations）：**


<img src="https://www.zhihu.com/equation?tex=X^TX\theta =X^T\vec{y}
" alt="X^TX\theta =X^T\vec{y}
" class="ee_img tr_noresize" eeimg="1">

所以让  <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1">  取值最小的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  就是


<img src="https://www.zhihu.com/equation?tex=\theta = (X^TX)^{-1}X^T\vec{y}
" alt="\theta = (X^TX)^{-1}X^T\vec{y}
" class="ee_img tr_noresize" eeimg="1">

如果使用正则方法求 <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1"> 的最小值，根据这个公式就可以直接求得，使用Matlab或者Python可以很快得到结果，是实际应用中比较常规的一种方法。

#### 3 概率解释（Probabilistic interpretation）

在解决回归问题的时候，可能有这样的疑问，那就是为什么选择线性回归，尤其是为什么选择最小二乘法作为成本函数  <img src="https://www.zhihu.com/equation?tex=J" alt="J" class="ee_img tr_noresize" eeimg="1">  ？在本节里，我们会给出一系列的概率基本假设，基于这些假设，就可以推出最小二乘法回归是一种非常自然的算法。

首先假设目标变量和输入值存在下面这种等量关系：


<img src="https://www.zhihu.com/equation?tex=y^{(i)}=\theta^T x^{(i)}+ \epsilon ^{(i)}
" alt="y^{(i)}=\theta^T x^{(i)}+ \epsilon ^{(i)}
" class="ee_img tr_noresize" eeimg="1">

上式中  <img src="https://www.zhihu.com/equation?tex= \epsilon ^{(i)}" alt=" \epsilon ^{(i)}" class="ee_img tr_noresize" eeimg="1">  是误差项，用于表征建模训练结果所忽略的输入变量导致的效果 （比如可能某些特征对于房价的影响很明显，但我们做回归的时候忽略掉了）或者一些随机的噪音信息（random noise）。为方便研究，假设  <img src="https://www.zhihu.com/equation?tex= \epsilon ^{(i)}" alt=" \epsilon ^{(i)}" class="ee_img tr_noresize" eeimg="1"> 是独立同分布的，服从高斯分布（Gaussian distribution ，也叫正态分布 Normal distribution），其均值为  <img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1"> ，方差（variance）为  <img src="https://www.zhihu.com/equation?tex=\sigma ^2" alt="\sigma ^2" class="ee_img tr_noresize" eeimg="1"> 。这样就可以把这个假设写成  <img src="https://www.zhihu.com/equation?tex= \epsilon ^{(i)} ∼ N (0, \sigma ^2)" alt=" \epsilon ^{(i)} ∼ N (0, \sigma ^2)" class="ee_img tr_noresize" eeimg="1">  。然后  <img src="https://www.zhihu.com/equation?tex= \epsilon ^{(i)} " alt=" \epsilon ^{(i)} " class="ee_img tr_noresize" eeimg="1">   的密度函数就是：


<img src="https://www.zhihu.com/equation?tex=p(\epsilon ^{(i)} )= \frac 1{\sqrt{2\pi}\sigma} exp (- \frac  {(\epsilon ^{(i)} )^2}{2\sigma^2})
" alt="p(\epsilon ^{(i)} )= \frac 1{\sqrt{2\pi}\sigma} exp (- \frac  {(\epsilon ^{(i)} )^2}{2\sigma^2})
" class="ee_img tr_noresize" eeimg="1">

这意味着存在下面的等量关系：


<img src="https://www.zhihu.com/equation?tex=p(y ^{(i)} |x^{(i)}; \theta)= \frac 1{\sqrt{2\pi}\sigma} exp (- \frac  {(y^{(i)} -\theta^T x ^{(i)} )^2}{2\sigma^2})
" alt="p(y ^{(i)} |x^{(i)}; \theta)= \frac 1{\sqrt{2\pi}\sigma} exp (- \frac  {(y^{(i)} -\theta^T x ^{(i)} )^2}{2\sigma^2})
" class="ee_img tr_noresize" eeimg="1">

这里的记号  <img src="https://www.zhihu.com/equation?tex=“p(y ^{(i)} |x^{(i)}; \theta)”" alt="“p(y ^{(i)} |x^{(i)}; \theta)”" class="ee_img tr_noresize" eeimg="1">  表示的是这是一个对于给定  <img src="https://www.zhihu.com/equation?tex=x^{(i)}" alt="x^{(i)}" class="ee_img tr_noresize" eeimg="1">  时  <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">  的分布，用 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  代表该分布的参数。 注意这里不能用  <img src="https://www.zhihu.com/equation?tex=\theta(“p(y ^{(i)} |x^{(i)},\theta)”)" alt="\theta(“p(y ^{(i)} |x^{(i)},\theta)”)" class="ee_img tr_noresize" eeimg="1"> 来当做条件，因为  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  并不是一个随机变量。这个  <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">   的分布还可以写成 <img src="https://www.zhihu.com/equation?tex=y^{(i)} | x^{(i)}; \theta ∼ N (\theta ^T x^{(i)}, \sigma^2)" alt="y^{(i)} | x^{(i)}; \theta ∼ N (\theta ^T x^{(i)}, \sigma^2)" class="ee_img tr_noresize" eeimg="1"> 。

给定一个设计矩阵（design matrix） <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1"> ，其包含了所有的 <img src="https://www.zhihu.com/equation?tex=x^{(i)}" alt="x^{(i)}" class="ee_img tr_noresize" eeimg="1"> ，然后再给定  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> ，那么  <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">  的分布是什么？数据的概率以 <img src="https://www.zhihu.com/equation?tex=p (\vec{y}|X;\theta )" alt="p (\vec{y}|X;\theta )" class="ee_img tr_noresize" eeimg="1">  的形式给出。在 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 取某个固定值的情况下，这个等式通常可以看做是一个  <img src="https://www.zhihu.com/equation?tex=\vec{y}" alt="\vec{y}" class="ee_img tr_noresize" eeimg="1">  的函数（也可以看成是  <img src="https://www.zhihu.com/equation?tex=X" alt="X" class="ee_img tr_noresize" eeimg="1">  的函数）。当我们要把它当做  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的函数的时候，就称它为 **似然**函数（likelihood function)


<img src="https://www.zhihu.com/equation?tex=L(\theta) =L(\theta;X,\vec{y})=p(\vec{y}|X;\theta)
" alt="L(\theta) =L(\theta;X,\vec{y})=p(\vec{y}|X;\theta)
" class="ee_img tr_noresize" eeimg="1">

结合之前对  <img src="https://www.zhihu.com/equation?tex=\epsilon^{(i)}" alt="\epsilon^{(i)}" class="ee_img tr_noresize" eeimg="1">  的独立性假设 （这里对 <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">  以及给定的  <img src="https://www.zhihu.com/equation?tex=x^{(i)}" alt="x^{(i)}" class="ee_img tr_noresize" eeimg="1">  也都做同样假设），就可以把上面这个等式改写成下面的形式：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
L(\theta) &=\prod ^m _{i=1}p(y^{(i)}|x^{(i)};\theta)\\
&=\prod ^m _{i=1} \frac  1{\sqrt{2\pi}\sigma} exp(- \frac {(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2})\\
\end{aligned}
" alt="\begin{aligned}
L(\theta) &=\prod ^m _{i=1}p(y^{(i)}|x^{(i)};\theta)\\
&=\prod ^m _{i=1} \frac  1{\sqrt{2\pi}\sigma} exp(- \frac {(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2})\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

现在，给定了 <img src="https://www.zhihu.com/equation?tex=y^{(i)}" alt="y^{(i)}" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=x^{(i)}" alt="x^{(i)}" class="ee_img tr_noresize" eeimg="1"> 之间关系的概率模型了，用什么方法来选择咱们对参数  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的最佳猜测呢？最大似然法（maximum likelihood）告诉我们要选择能让数据的似然函数尽可能大的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 。也就是说，咱们要找的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  能够让函数  <img src="https://www.zhihu.com/equation?tex=L(\theta)" alt="L(\theta)" class="ee_img tr_noresize" eeimg="1">  取到最大值。

除了找到  <img src="https://www.zhihu.com/equation?tex=L(\theta)" alt="L(\theta)" class="ee_img tr_noresize" eeimg="1">  最大值，我们还以对任何严格递增的  <img src="https://www.zhihu.com/equation?tex=L(\theta)" alt="L(\theta)" class="ee_img tr_noresize" eeimg="1">  的函数求最大值。如果我们不直接使用  <img src="https://www.zhihu.com/equation?tex=L(\theta)" alt="L(\theta)" class="ee_img tr_noresize" eeimg="1"> ，而是使用对数函数，来找**对数似然函数  <img src="https://www.zhihu.com/equation?tex=l(\theta)" alt="l(\theta)" class="ee_img tr_noresize" eeimg="1"> ** 的最大值，那这样对于求导来说就简单了一些：


<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
l(\theta) &=\log L(\theta)\\
&=\log \prod ^m _{i=1} \frac  1{\sqrt{2\pi}\sigma} exp(- \frac {(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2})\\
&= \sum ^m _{i=1}log \frac  1{\sqrt{2\pi}\sigma} exp(- \frac {(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2})\\
&= m \log \frac  1{\sqrt{2\pi}\sigma}- \frac 1{\sigma^2}\cdot \frac 12 \sum^m_{i=1} (y^{(i)}-\theta^Tx^{(i)})^2\\
\end{aligned}
" alt="\begin{aligned}
l(\theta) &=\log L(\theta)\\
&=\log \prod ^m _{i=1} \frac  1{\sqrt{2\pi}\sigma} exp(- \frac {(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2})\\
&= \sum ^m _{i=1}log \frac  1{\sqrt{2\pi}\sigma} exp(- \frac {(y^{(i)}-\theta^T x^{(i)})^2}{2\sigma^2})\\
&= m \log \frac  1{\sqrt{2\pi}\sigma}- \frac 1{\sigma^2}\cdot \frac 12 \sum^m_{i=1} (y^{(i)}-\theta^Tx^{(i)})^2\\
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">

因此，对  <img src="https://www.zhihu.com/equation?tex=l(\theta)" alt="l(\theta)" class="ee_img tr_noresize" eeimg="1">  取得最大值也就意味着下面这个子式取到最小值：


<img src="https://www.zhihu.com/equation?tex=\frac 12 \sum^m _{i=1} (y^{(i)}-\theta^Tx^{(i)})^2
" alt="\frac 12 \sum^m _{i=1} (y^{(i)}-\theta^Tx^{(i)})^2
" class="ee_img tr_noresize" eeimg="1">

到这里我们能发现这个子式实际上就是  <img src="https://www.zhihu.com/equation?tex=J(\theta)" alt="J(\theta)" class="ee_img tr_noresize" eeimg="1"> ，也就是最原始的最小二乘成本函数（least-squares cost function）。

总结一下也就是：在对数据进行概率假设的基础上，最小二乘回归得到的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  和最大似然法估计的  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  是一致的。所以这是一系列的假设，其前提是认为最小二乘回归（least-squares regression）能够被判定为一种非常自然的方法，这种方法正好就进行了最大似然估计（maximum likelihood estimation）。（要注意，对于验证最小二乘法是否为一个良好并且合理的过程来说，这些概率假设并不是必须的，此外可能（也确实）有其他的自然假设能够用来评判最小二乘方法。）

另外还要注意，在刚才的讨论中，我们最终对  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的选择并不依赖  <img src="https://www.zhihu.com/equation?tex=\sigma^2" alt="\sigma^2" class="ee_img tr_noresize" eeimg="1"> ，而且也确实在不知道  <img src="https://www.zhihu.com/equation?tex=\sigma^2" alt="\sigma^2" class="ee_img tr_noresize" eeimg="1">  的情况下就已经找到了结果。稍后我们还要对这个情况加以利用，到时候我们会讨论指数族以及广义线性模型。

#### 4 局部加权线性回归（Locally weighted linear regression）

假如问题还是根据从实数域内取值的  <img src="https://www.zhihu.com/equation?tex=x\in R" alt="x\in R" class="ee_img tr_noresize" eeimg="1">  来预测  <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  。左下角的图显示了使用  <img src="https://www.zhihu.com/equation?tex=y = \theta_0 + \theta_1x" alt="y = \theta_0 + \theta_1x" class="ee_img tr_noresize" eeimg="1">  来对一个数据集进行拟合。我们明显能看出来这个数据的趋势并不是一条严格的直线，所以用直线进行的拟合就不是好的方法。

<img src="https://raw.githubusercontent.com/GSYfate/Markdown4Zhihu/master/Data/cs229-notes1/cs229note1f5.png" alt="cs229note1f5" style="zoom:67%;" />

如果不用直线，而增加一个二次项，用 <img src="https://www.zhihu.com/equation?tex=y = \theta_0 + \theta_1x +\theta_2x^2" alt="y = \theta_0 + \theta_1x +\theta_2x^2" class="ee_img tr_noresize" eeimg="1">  来拟合。（中间的图）很明显，我们对特征补充后，拟合效果变好了，因此适度的增加特征可以提高模型的效果。不过，增加太多特征也会造成麻烦：最右边的图就是使用了五次多项式  <img src="https://www.zhihu.com/equation?tex=y = \sum^5_{j=0} \theta_jx^j" alt="y = \sum^5_{j=0} \theta_jx^j" class="ee_img tr_noresize" eeimg="1">  来进行拟合。看图可以发现，虽然这个拟合曲线完美地通过了所有当前数据集中的数据，但我们明显不能认为这个曲线是一个合适的预测工具，这样过拟合的曲线很难作出正确的预测，因为当给出其他的房屋数据时，它对房屋价格的估计会偏差很大。在图中甚至出现了后半段房屋面积越大价格越低的情况。

最左边的图像就是一个**欠拟合(under fitting)** 的例子，比如明显能看出拟合的模型漏掉了数据集中的结构信息；而最右边的图像就是一个**过拟合(over fitting)** 的例子。（在课程的后续部分中，讨论到关于学习理论的时候，会给出这些概念的标准定义，也会给出拟合程度对于一个猜测的好坏检验的意义。）

正如前文谈到的，也正如上面这个例子展示的，一个学习算法要保证能良好运行，特征的选择是非常重要的。在本节，咱们就简要地讲一下局部加权线性回归（locally weighted linear regression ，缩写为LWR），这个方法是假设有足够多的训练数据，对不太重要的特征进行一些筛选。

在原始版本的线性回归算法中，要对一个查询点  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  进行预测，比如要衡量 <img src="https://www.zhihu.com/equation?tex=h(x)" alt="h(x)" class="ee_img tr_noresize" eeimg="1"> ，要经过下面的步骤：

1. 使用参数  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  进行拟合，让数据集中的值与拟合算出的值的差值平方 <img src="https://www.zhihu.com/equation?tex=\sum_i(y^{(i)} − \theta^T x^{(i)} )^2" alt="\sum_i(y^{(i)} − \theta^T x^{(i)} )^2" class="ee_img tr_noresize" eeimg="1"> 最小(最小二乘法的思想)；
2. 输出  <img src="https://www.zhihu.com/equation?tex=\theta^T x" alt="\theta^T x" class="ee_img tr_noresize" eeimg="1">  。

相应地，在 LWR 局部加权线性回归方法中，步骤如下：

1. 使用参数  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  进行拟合，让加权距离 <img src="https://www.zhihu.com/equation?tex=\sum_i w^{(i)}(y^{(i)} − \theta^T x^{(i)} )^2" alt="\sum_i w^{(i)}(y^{(i)} − \theta^T x^{(i)} )^2" class="ee_img tr_noresize" eeimg="1">  最小；
2. 输出  <img src="https://www.zhihu.com/equation?tex=\theta^T x" alt="\theta^T x" class="ee_img tr_noresize" eeimg="1"> 。


上面式子中的  <img src="https://www.zhihu.com/equation?tex=w^{(i)}" alt="w^{(i)}" class="ee_img tr_noresize" eeimg="1">  是非负的权值。直观点说就是，如果对应某个 <img src="https://www.zhihu.com/equation?tex=i" alt="i" class="ee_img tr_noresize" eeimg="1">  的权值  <img src="https://www.zhihu.com/equation?tex=w^{(i)}" alt="w^{(i)}" class="ee_img tr_noresize" eeimg="1">  特别大，那么在选择拟合参数  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的时候，就要尽量让这一点的  <img src="https://www.zhihu.com/equation?tex=(y^{(i)} − \theta^T x^{(i)} )^2" alt="(y^{(i)} − \theta^T x^{(i)} )^2" class="ee_img tr_noresize" eeimg="1">  最小。而如果权值 <img src="https://www.zhihu.com/equation?tex=w^{(i)}" alt="w^{(i)}" class="ee_img tr_noresize" eeimg="1">   特别小，那么这一点对应的 <img src="https://www.zhihu.com/equation?tex=(y^{(i)} − \theta^T x^{(i)} )^2" alt="(y^{(i)} − \theta^T x^{(i)} )^2" class="ee_img tr_noresize" eeimg="1">  就基本在拟合过程中忽略掉了。通俗地讲，利用权重筛选出我们想要的特征。

对于权值的选取可以使用下面这个比较标准的公式： <img src="https://www.zhihu.com/equation?tex=^4" alt="^4" class="ee_img tr_noresize" eeimg="1"> 


<img src="https://www.zhihu.com/equation?tex=w^{(i)} = exp(- \frac {(x^{(i)}-x)^2}{2\tau^2})
" alt="w^{(i)} = exp(- \frac {(x^{(i)}-x)^2}{2\tau^2})
" class="ee_img tr_noresize" eeimg="1">

>4 如果  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  是有值的向量，那就要对上面的式子进行泛化，得到的是 <img src="https://www.zhihu.com/equation?tex=w^{(i)} = exp(− \frac {(x^{(i)}-x)^T(x^{(i)}-x)}{2\tau^2})" alt="w^{(i)} = exp(− \frac {(x^{(i)}-x)^T(x^{(i)}-x)}{2\tau^2})" class="ee_img tr_noresize" eeimg="1"> ，或者: <img src="https://www.zhihu.com/equation?tex=w^{(i)} = exp(− \frac {(x^{(i)}-x)^T\Sigma ^{-1}(x^{(i)}-x)}{2})" alt="w^{(i)} = exp(− \frac {(x^{(i)}-x)^T\Sigma ^{-1}(x^{(i)}-x)}{2})" class="ee_img tr_noresize" eeimg="1"> ，这就看是选择用 <img src="https://www.zhihu.com/equation?tex=\tau" alt="\tau" class="ee_img tr_noresize" eeimg="1">  还是  <img src="https://www.zhihu.com/equation?tex=\Sigma" alt="\Sigma" class="ee_img tr_noresize" eeimg="1"> 。


要注意的是，权值是依赖每个特定的点  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  的，而这些点正是我们要去进行预测评估的点。此外，如果  <img src="https://www.zhihu.com/equation?tex=|x^{(i)} − x|" alt="|x^{(i)} − x|" class="ee_img tr_noresize" eeimg="1">  非常小，那么权值  <img src="https://www.zhihu.com/equation?tex=w^{(i)} " alt="w^{(i)} " class="ee_img tr_noresize" eeimg="1"> 就接近  <img src="https://www.zhihu.com/equation?tex=1" alt="1" class="ee_img tr_noresize" eeimg="1"> ；反之如果  <img src="https://www.zhihu.com/equation?tex=|x^{(i)} − x|" alt="|x^{(i)} − x|" class="ee_img tr_noresize" eeimg="1">  非常大，那么权值  <img src="https://www.zhihu.com/equation?tex=w^{(i)} " alt="w^{(i)} " class="ee_img tr_noresize" eeimg="1"> 就变小。所以可以看出，  <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  的选择过程中，查询点  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  附近的训练样本有更高得多的权值。（ <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> is chosen giving a much higher “weight” to the (errors on) training examples close to the query point x.）（还要注意，当权值的方程的形式跟高斯分布的密度函数比较接近的时候，权值和高斯分布并没有什么直接联系，尤其是当权值不是随机值，且呈现正态分布或者其他形式分布的时候。）随着点 <img src="https://www.zhihu.com/equation?tex=x^{(i)} " alt="x^{(i)} " class="ee_img tr_noresize" eeimg="1">  到查询点  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1">  的距离降低，训练样本的权值的也在降低，参数 <img src="https://www.zhihu.com/equation?tex=\tau" alt="\tau" class="ee_img tr_noresize" eeimg="1">   控制了这个降低的速度； <img src="https://www.zhihu.com/equation?tex=\tau" alt="\tau" class="ee_img tr_noresize" eeimg="1"> 也叫做**带宽参数**，这个也是在你的作业中需要来体验和尝试的一个参数。

局部加权线性回归是咱们接触的第一个**非参数** 算法。而更早之前咱们看到的无权重的线性回归算法就是一种**参数** 学习算法，因为有固定的有限个数的参数（也就是  <img src="https://www.zhihu.com/equation?tex=\theta_i" alt="\theta_i" class="ee_img tr_noresize" eeimg="1">  ），这些参数用来拟合数据。我们对  <img src="https://www.zhihu.com/equation?tex=\theta_i" alt="\theta_i" class="ee_img tr_noresize" eeimg="1">  进行了拟合之后，就把它们存了起来，也就不需要再保留训练数据样本来进行更进一步的预测了。与之相反，如果用局部加权线性回归算法，我们就必须一直保留着整个训练集。这里的非参数算法中的 非参数“non-parametric” 是粗略地指：为了呈现出假设  <img src="https://www.zhihu.com/equation?tex=h" alt="h" class="ee_img tr_noresize" eeimg="1">  随着数据集规模的增长而线性增长，我们需要以一定顺序保存一些数据的规模。（The term “non-parametric” (roughly) refers to the fact that the amount of stuff we need to keep in order to represent the hypothesis h grows linearly with the size of the training set. ）

**未完待续**



## 参考链接

本笔记大量参考大佬在github上已经做好的学习笔记，感谢大佬的分享，特此声明。

**原作者**：[Andrew Ng  吴恩达](http://www.andrewng.org/)

**讲义翻译者**：[CycleUser](https://www.zhihu.com/people/cycleuser/columns)

**Github参考链接：**[Github 地址](https://github.com/Kivy-CN/Stanford-CS-229-CN)

[斯坦福大学 CS229 课程网站](http://cs229.stanford.edu/)

[知乎专栏](https://zhuanlan.zhihu.com/MachineLearn)

