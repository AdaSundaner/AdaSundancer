
# coding: utf-8

# In[11]:

import numpy as np


# In[12]:

def loadSimpData():
    datMat=np.matrix([[1.,2.1],
                  [2.,1.1],
                  [1.3,1.],
                  [1.,1.],
                  [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels


# In[13]:

dataMat,classLabels=loadSimpData()


# In[14]:

dataMat,classLabels

