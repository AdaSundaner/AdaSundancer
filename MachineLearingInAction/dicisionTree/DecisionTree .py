
# coding: utf-8

# In[27]:

from math import log


# ### 创建数据集

# In[28]:

def createDataSet():
    dataSet=[[1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels


# ### 计算给定数据集的香农熵

# In[29]:

def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={} # 创建标签字典
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0 # 如果当前key不存在，则扩展字典将当前标签值加入字典
        labelCounts[currentLabel]+=1
    shannonEnt=0.0 # 默认香农熵为0.0
    for key in labelCounts: # 计算香农熵
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob* log(prob,2)
    return shannonEnt


# ### 划分数据集

# In[30]:

def splitDataSet(dataSet,axis,value):
    retDataSet=[] # 最终处理完成的list对象
    for reatVec in dataSet:
        if featVec ==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec) # 当我们按照某个特征划分数据集时，要将所有符合要求的元素抽取出来
    return retDataSet


# ### 选择最好的数据集划分方式

# In[31]:

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1 # 特征数量值
    baseEntropy=calcShannonEnt(dataSet) # 初始香农熵
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures): # 对每个特征值做如下处理：
        featList=[example[i] for example in dataSet]  # 获取每个特征值的全集
        uniqueVals=set(featList) # 将每个特征值的全集处理成不重复的集合
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value) # 按照特征值的每个取值，对数据集进行划分
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet) # 计算香农熵
        infoGain=baseEntropy-newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature


# ### 若数据集已经处理了所有的属性，但是类标签仍旧不唯一，此时选择概率最大的作为标签

# In[32]:

import operator as op
def majorityCnt(classList):
    calssCount={}
    for vote in classList:
        if vote not in calssCount.keys(): classCount[vote]=0
        classCount[vote] +=1
    sortedClassCount=sorted(classCount.items(),key=op.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


# ### 创建树

# In[33]:

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):  # 类别相同则停止继续划分
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList) # 遍历完所有特征时返回出现次数最多的类别
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

