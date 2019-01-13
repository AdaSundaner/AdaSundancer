
# coding: utf-8

# ### 使用梯度上升找到最佳参数

# #### 1、logistic回归梯度上升优化算法

# In[3]:

import numpy as np
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr = open("E:/pythonStudy/MachineLearingInAction/machinelearninginaction/Ch05/testSet.txt")
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# sigmoid函数
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

# 梯度上升算法的主要逻辑
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn) # 转换成Numpy矩阵格式
    labelMat=np.mat(classmethod).transpose() # 对1*100的矩阵做转置
    m,n=np.shape(dataMatrix)
    alpha=0.001 # 梯度上升算法的步长
    maxCycles=500 # 循环次数
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights) # 矩阵相乘        
        error=labelMat - h # 真实类别和预测类别的差值
        weights=weights+alpha*dataMatrix.transpose()*error # 回归系数
    return weights


# In[4]:

# 测试梯度上升算法
dataArr,labelMat=loadDataSet()
gradAscent(dataArr,labelMat)


# #### 2、画出决策边界

# In[ ]:

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,x='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# In[ ]:

weights=gradAscent(dataArr,labelMat)
plotBestFit(weights.getA())

