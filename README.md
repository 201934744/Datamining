# Datamining
<br />作业一：测试sklearn中8种聚类算法在给定的两个数据集上的聚类效果
<br />一、实验目的：
聚类分析是研究对事物进行分类的一种多元统计方法，广泛应用于科学研究、生产生活和社会实践中，
<br />本实验主要测试了sklearn中8种聚类算法在给定的数字数据集和新闻数据集上的聚类效果。下面对8种聚类分析算法进行简要介绍：
<br />1、K-Means为聚类分析中最广泛使用的算法，主要通过将样本划分为k个方差齐次的类来实现数据聚类。
<br />2、Affinity propagation聚类算法是基于数据点间的"信息传递"的一种聚类算法。与k-均值算法或k中心点算法不同，AP算法不需要在运行算法之前确定聚类的个数。
<br />3、Mean Shift算法,一般是指一个迭代的步骤,即先算出当前点的偏移均值,移动该点到其偏移均值,然后以此为新的起始点,继续移动,直到满足一定的条件结束。
4、Spectral Clustering（谱聚类）是一种基于图论的聚类方法，它能够识别任意形状的样本空间且收敛于全局最有解，其基本思想是利用样本数据的相似矩阵进行特征分解后得到的特征向量进行聚类。
5、Ward hierarchical clustering层次聚类算法，层次聚类试图在不同的“层次”上对样本数据集进行划分，一层一层地进行聚类。就划分策略可分为自底向上的凝聚方法和自上向下的分裂方法。
6、Agglomerative clustering是一种自底而上的层次聚类方法，它能够根据指定的相似度或距离定义计算出类之间的距离。
7、DBSCAN是一个比较有代表性的基于密度的聚类算法。与划分和层次聚类方法不同，它将簇定义为密度相连的点的最大集合。
8、Gaussian Mixture实现了期望最大化(EM)算法来拟合高斯混合模型。它还可以为多元模型绘制置信椭球体，并计算贝叶斯信息准则来评估数据中的簇数。
二、实验步骤：
1、实验环境搭建：在Windows下下载安装Anaconda3，并进行环境变量设置。开始使对python环境搭建比较陌生，走不少弯路。
2、代码编写：详细代码见Cluster文件夹。主要包括几部分：模块导入、数据集的加载、选择聚类算法、参数调整、结果输出。
三、实验结果及分析：
1、对第一个数据集进行聚类分析，结果输出如下：
alg             time    nmi     homo    compl
k-means         0.16s   0.626   0.602   0.650
Aff             3.79s   0.655   0.932   0.460
MeanShift       5.20s   0.063   0.014   0.281
AggClu          0.12s   0.466   0.239   0.908
DBSCAN          0.31s   0.375   0.000   1.000
2、对第二个数据集进行聚类分析，结果输出如下：
alg             time    nmi     homo    compl
k-means         3.05s   0.523   0.487   0.562
Aff             11.64s  0.411   0.885   0.191
SpectralCluster 1.63s   0.390   0.399   0.380
DBSCAN          0.42s   0.016   0.002   0.168
四、下一步考虑：
1、部分算法未能输出结果，将代码进行查错和原因分析。
2、目前仅实验默认参数，下一步将根据聚类结果进行参数调整。
3、未能将结果进行可视化图形绘制。  
