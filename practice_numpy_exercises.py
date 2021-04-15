#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np


# In[4]:


#create a custom dtype that describes a color as four unisgned bytes(RGBA)


# In[5]:


color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])


# In[6]:


#given a 1D array, negate all elements which are between 3 and 8, in place.


# In[10]:


Z = np.arange(11)
Z[(3<Z)&(Z <= 8)] *= -1
print(Z)


# In[11]:


#what is the output of the following script?


# In[13]:


print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))


# In[14]:


#Consider an integer vector Z, which of these expressions are legal?


# In[17]:


Z = 14


# In[18]:


Z**Z


# In[19]:


2 << Z >> 2


# In[20]:


Z < - Z


# In[21]:


1j*Z


# In[22]:


Z/1/1


# In[23]:


Z < Z > Z


# In[24]:


#what are the result of the following expressions?


# In[25]:


np.array(0) //  np.array(0)


# In[26]:


np.array(0) // np.array(0.)


# In[27]:


np.array(0) / np.array(0)


# In[28]:


np.array(0)/np.array(0.)


# In[29]:


#How to round away from zero a float array?


# In[30]:


Z = np.random.uniform(-10,+10,10)
print(Z)
print(np.trunc(Z + np.copysign(0.5,Z)))


# In[31]:


#Create a 5x5 matrix with row values ranging from 0 to 4


# In[32]:


Z = np.zeros((5,5))
print(Z)


# In[33]:


Z += np.arange(5)
print(Z)


# In[34]:


#consider a generator function that generates 10 integers and use it to build an array


# In[36]:


def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count = -1)
print(Z)


# In[37]:


#create a vector of size 10 with values ranging from 0 to 1, both excluded


# In[38]:


Z = np.linspace(0,1,12,endpoint= True)[1:-1]
print(Z)


# In[39]:


#Create a random vector of size 10 and sort it


# In[40]:


Z = np.random.random(10)
Z.sort()
print(Z)


# In[41]:


#How to Sum a small array faster than np.sum?


# In[42]:


Z= np.arange(10)
print(Z)
np.add.reduce(Z)


# In[43]:


#Consider two random array A and B, check if they are equal


# In[46]:


A = np.random.randint(0,2,5)
print(A)
B = np.random.randint(0,2,5)
print(B)
equal = np.allclose(A,B)
print(equal)


# In[47]:


#How an array immutable (read-only)
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1


# In[48]:


#Consider a random 10x2 matrix representing cartesian cooradinates, convert them to polar coordinates


# In[49]:


Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)


# In[50]:


#Create random vector of size 10 and replace the maximum value by 0


# In[54]:


Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)


# In[55]:


#Create a structured array with x and y coordinates covering the [0,1]x[0,1]area


# In[57]:


Z = np.zeros((10,10), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
                            np.linspace(0,1,10))
print(Z)


# In[58]:


#Given two arrays, X and Y, construct the Cauchy matrix C (Cij = 1/(xi-yj))


# In[59]:


X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X,Y)
print(np.linalg.det(C))


# In[60]:


#Print the minimum and maximum representable value for each numpy scalar type


# In[61]:


for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)


# In[ ]:




