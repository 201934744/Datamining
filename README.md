#By 杨震201934744 2019.10.09
<br />DataMining作业一：测试sklearn中8种聚类算法在给定的两个数据集上的聚类效果
=

<br />一、实验目的：
-

<br />&emsp;&emsp;聚类分析是研究对事物进行分类的一种多元统计方法，广泛应用于科学研究、生产生活和社会实践中，<br />本实验主要测试了sklearn中8种聚类算法在给定的数字数据集和新闻数据集上的聚类效果。下面对8种聚类分析算法进行简要介绍：
<br />&emsp;&emsp;1、K-Means为聚类分析中最广泛使用的算法，主要通过将样本划分为k个方差齐次的类来实现数据聚类。
<br />&emsp;&emsp;2、Affinity propagation聚类算法是基于数据点间的"信息传递"的一种聚类算法。与k-均值算法或k中心点算法不同，AP算法不需要在运行算法之前确定聚类的个数。
<br />&emsp;&emsp;3、Mean Shift算法,一般是指一个迭代的步骤,即先算出当前点的偏移均值,移动该点到其偏移均值,然后以此为新的起始点,继续移动,直到满足一定的条件结束。
<br />&emsp;&emsp;4、Spectral Clustering（谱聚类）是一种基于图论的聚类方法，它能够识别任意形状的样本空间且收敛于全局最有解，其基本思想是利用样本数据的相似矩阵进行特征分解后得到的特征向量进行聚类。
<br />&emsp;&emsp;5、Ward hierarchical clustering层次聚类算法，层次聚类试图在不同的“层次”上对样本数据集进行划分，一层一层地进行聚类。就划分策略可分为自底向上的凝聚方法和自上向下的分裂方法。
<br />&emsp;&emsp;6、Agglomerative clustering是一种自底而上的层次聚类方法，它能够根据指定的相似度或距离定义计算出类之间的距离。
<br />&emsp;&emsp;7、DBSCAN是一个比较有代表性的基于密度的聚类算法。与划分和层次聚类方法不同，它将簇定义为密度相连的点的最大集合。
<br />&emsp;&emsp;8、Gaussian Mixture实现了期望最大化(EM)算法来拟合高斯混合模型。它还可以为多元模型绘制置信椭球体，并计算贝叶斯信息准则来评估数据中的簇数。

<br />二、实验步骤：
-

<br />&emsp;&emsp;1、实验环境搭建：在Windows10下下载安装Anaconda3，并进行系统环境变量设置。开始使对python环境搭建比较陌生，走了不少弯路。
<br />&emsp;&emsp;2、代码编写：详细代码和注释见Cluster文件夹。主要包括几部分：模块导入、数据集的加载、选择聚类算法、参数调整、结果输出。
<br />&emsp;&emsp;3、存在问题与解决办法：在对第一个数据集进行聚类时，部分算法因缺少新的label标签而无法完成，增加estimator.predict(dataset.data)语句，对不能生成标签的算法进行预估，从而问题解决。在对第二个数据集进行聚类时，因数据矩阵过于稀疏导致部分算法无法完成，对数据集X增加toarray()操作，使聚类完成。

<br />三、实验结果及分析：
-

<br />&emsp;&emsp;1、对第一个数据集进行聚类分析，结果输出如下：
<br /><table>
        <tr>
              <th>Alg</th>
              <th>Time</th>
              <th>NMI</th>
              <th>Homo</th>
              <th>Compl</th>
        </tr>
        <tr>
            <td>k-means</td>
            <td>0.16s </td>
            <td>0.626</td>
            <td>0.602 </td>
            <td>0.650</td>
         </tr>
         <tr>
            <td>AffinityPropagation </td>
            <td>3.79s </td>
            <td>0.655</td>
            <td>0.932 </td>
            <td>0.460</td>
         </tr>
         <tr>
            <td>MeanShift</td>
            <td> 5.20s </td>
            <td>0.063</td>
            <td>0.014 </td>
            <td>0.281</td>
         </tr>
         <tr>
            <td>SpectralClustering</td>
            <td> 0.33s </td>
            <td>0.514  </td>
            <td>0.272  </td>
            <td>0.974  </td>
         </tr>
         <tr>
            <td>Ward </td>
            <td>0.16s </td>
            <td>0.466 </td>
            <td>0.239 </td>
            <td>0.908</td>
         </tr>
         <tr>
            <td>AgglomerativeClustering </td>
            <td>0.12s </td>
            <td>0.466</td>
            <td>0.239 </td>
            <td>0.908</td>
         </tr>
         <tr>
            <td>DBSCAN</td>
            <td>0.31s </td>
            <td>0.375 </td>
            <td>0.000 </td>
            <td>1.000 </td>
         </tr>
         <tr>
            <td>GaussianMixture</td>
            <td>0.11s </td>
            <td>0.455 </td>
            <td>0.300 </td>
            <td>0.691 </td>
         </tr>
      <table>      
<br />&emsp;&emsp;2、对第二个数据集进行聚类分析，结果输出如下：
<br /><table>
        <tr>
              <th>Alg</th>
              <th>Time</th>
              <th>NMI</th>
              <th>Homo</th>
              <th>Compl</th>
        </tr>
        <tr>
            <td>k-means</td>
            <td>3.05s </td>
            <td>0.523</td>
            <td>0.487 </td>
            <td>0.562</td>
         </tr>
         <tr>
            <td>Aff </td>
            <td>11.64s </td>
            <td>0.411</td>
            <td>0.885 </td>
            <td>0.191</td>
         </tr>
         <tr>
            <td>MeanShift </td>
            <td>5.94s </td>
            <td>0.000 </td>
            <td>0.000 </td>
            <td>1.000 </td>
         </tr>
         <tr>
            <td>SpectralCluster</td>
            <td>1.63s </td>
            <td>0.390</td>
            <td>0.399 </td>
            <td>0.380</td>
         </tr>
         <tr>
            <td>Ward </td>
            <td>54.55s </td>
            <td>0.557 </td>
            <td>0.517 </td>
            <td>0.601 </td>
         </tr>
         <tr>
            <td>DBSCAN </td>
            <td>0.42s </td>
            <td>0.016 </td>
            <td>0.002 </td>
            <td>0.168</td>
         </tr>
      <table>   

<br />四、下一步考虑：
-
<br />&emsp;&emsp;1、部分算法未能输出结果，将代码进行查错和原因分析。
<br />&emsp;&emsp;2、目前仅实验默认参数，下一步将根据聚类结果进行参数调整。
<br />&emsp;&emsp;3、未能将结果进行可视化图形绘制。  
