#!/usr/bin/env python
# coding: utf-8

# # Part 1: ML Assignment
# 
# Mobile price:
# •Data: https://raw.githubusercontent.com/nick-edu/dmmldl/master/MobilePrice.csv
# •Description: https://raw.githubusercontent.com/nick-edu/dmmldl/master/MobilePriceColumns.
# txt
# Answer the following questions for your chosen dataset.
# 
# ## 1.1 Exploratory Data Analysis
# Perform exploratory data analysis on your chosen data set covering the following items.
# 1. How many rows and columns are there in your selected dataset? Are there any missing values in any of the columns?
# 2. Choose few columns from your dataset and describe them using various visualizations e.g. histograms, scatterplots etc.
# 
# 

# In[107]:


#Subquestion 1

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import math

#import data

mobiledf= pd.read_csv('https://raw.githubusercontent.com/nick-edu/dmmldl/master/MobilePrice.csv')

#describe data

mobiledf.describe


# In[36]:


#explore data sets

mobiledf.dtypes


# In[37]:


mobiledf.isnull().sum()


# In[38]:


for column_name in mobiledf.columns:
    column = mobiledf[column_name]
    count = (column == 0).sum()
    print('Count of zeros in column ', column_name, ' is : ', count)


# In[39]:


for column_name in mobiledf.columns:
    column = mobiledf[column_name]
    countof1 = (column == 1).sum()
    countof0 = (column == 0).sum()
    print('Count of zeros in column ', column_name, ' is : ', countof1+countof0)


# In[40]:


#following code will explore the records ouside of our assumptions table outlined in the report


# In[49]:


#(mobiledf['px_height']==0).sum()

#selectedcolumns= ['px_height', 'screen_width']

#mobiledf[mobiledf['primary_camera'].eq(0).all(1)]
pd.set_option("display.max_rows", 15, "display.max_columns", None)

mobiledf.loc[mobiledf['primary_camera']==0]


# In[42]:


sns.pairplot(mobiledf, hue= "price_range")


# In[72]:


#add new columns 
mobiledf2=mobiledf

mobiledf2['screen_area']= mobiledf.screen_height * mobiledf.screen_width


# In[73]:


mobiledf2 = mobiledf2.astype({"depth": int, "front_camera": int, "internal_memory": int,"n_cores": int, "primary_camera": float, "ram": float, "px_height": float, "px_width": float, "talk_time": float, "screen_height": float, "screen_width": float})


# In[74]:


mobiledf2.dtypes


# In[75]:


mobiledf2.rename(columns={"width": "weight"}, inplace= True)


# In[76]:


mobiledf2["px_width2"]=mobiledf2["px_width"]**2
mobiledf2["px_height2"]=mobiledf2["px_height"]**2
mobiledf2["swin"]=mobiledf2["screen_width"]/2.54
mobiledf2["shin"]=mobiledf2["screen_height"]/2.54
mobiledf2["screen_width2"]=mobiledf2["swin"]**2
mobiledf2["screen_height2"]=mobiledf2["shin"]**2
mobiledf2["topf"]=mobiledf2["px_width"]/mobiledf2["px_height"]
mobiledf2["bottomf"]=mobiledf2["screen_width2"]/mobiledf2["screen_height2"]
mobiledf2["PPI"]=mobiledf2["topf"]/mobiledf2["bottomf"]


# In[77]:


mobiledf2.drop(["px_width2","px_height2", "swin", "shin", "screen_width2", "screen_height2", "topf", "bottomf"], axis=1,inplace= True)


# In[78]:


mobiledf2


# In[ ]:


#We created a second set of data where the PPI and screen area was added and any phone with a screen area of 0 was removed from the dataset
#Because the data is stored as int instead of floats this datd has a possibility of being valid BUT the odds of a screen having a 
#dimension less than 1 centimeter is very low. 
#we also changed the column width to weight to match the data description


# In[79]:


mobiledf2.loc[mobiledf2['screen_area']==0]
mobiledf3 = mobiledf2[mobiledf2.screen_area != 0]


# In[96]:


mobiledf3.loc[mobiledf3['PPI']==np.inf]


# In[97]:


mobiledf3 = mobiledf3.drop(labels=1933, axis=0)


# In[99]:


mobiledf3["screen_area"].hist()


# In[105]:


mobiledf3.plot.scatter(x = mobiledf3["screen_width"], y = mobiledf3["screen_height"])


# ## 1.2 Clustering
# As part of this sub-question you will perform clustering on your chosen dataset from the above links. Choose one of the clustering algorithms that were discussed during the lecture for the application on your chosen dataset. Then choose few columns from your dataset that you think suitable for performing your chosen clustering. Describe and reflect on the clustering results and you are free to use the graphs/images and any other sort of visualizations also.

# In[108]:


#Divide up the data set to a test, train, and 
X_train, X_test, y_train, y_test = train_test_split(mobiledf3.data, mobiledf3.target, test_size=0.2, random_state = 42)

#Iøve done something wrong but itøs probably the whole target thing. I need to remove it but I'm tired and will work on it later


# # Question 2
# ## 2.1 Principal Component Analysis
# You need to download Olivetti faces dataset for this question and you can get it in one of the following ways.
# 1. Olivetti faces dataset download links
# •Data: https://cs.nyu.edu/~roweis/data/olivettifaces.mat
# •Images: https://cs.nyu.edu/~roweis/data/olivettifaces.gif
# 2. Dataset using Scikit-learn [1]: Use the sample code shown in listing 1.
# 
# a) Consider Olivetti faces dataset and use a classical dimensionality reduction technique (e.g. PCA) while preserving 99% of the variance. Then compute the reconstruction error for each image.
# 
# b) Next, take some of the images you built using the dimensionality reduction technique and modify/add some noise to some of the images using techniques such as rotate, flip, darken (you can use libraries such as scikit-image [2] etc. to do this) and look at their reconstruction error. You willnotice that how much larger the reconstruction error is.
# 
# c) Finally, plot all the 3 respective reconstructed images side by side (original image, image after PCA, image after PCA + noise) and compare the results.
# 
# 2.2 Singular Value Decomposition
# Find the singular values of the matrix Ashown below. The purpose of this assignment is to understand how to calculate the SVD for a given matrix. Therefore use a pen and paper to do the calculations, otherwise, if you are writing your own Python program, make sure to print/produce all the steps, so that we can verify your workings easily.

# # Question 3
# Write a brief extended abstract (of approximately 2-pages) containing your reflections on the appli-
# cations of one of the following unsupervised machine learning techniques.
# 1. Dimensionality Reduction
# 2. Clustering
# 
# In order to answer the above question, you can also do a tiny literature review to find out how your chosen techniques is used across various domains to address/solve which kinds of problems. Also note that these techniques are old, quite popular and have laid foundation for many other machine learning/applied techniques. For example, recommender systems, customer segmentation, image segmentation etc. use internally/got inspired from the clustering technique. Therefore you have a several choices to write/shape your extended abstract and feel free to choose whatever the direction you want to explore. Furthermore, you can also reflect based on your experience in using the chosen techniques in your assignment, on their capabilities and limitations in terms of applying them on various datasets.
# 
# Finally, we would like to see more of your reflections and critical comments rather than just reproducing/reporting from what you have found in literature review. Your report should confirm to general formatting guidelines and academic standards that is expected for written projects at CBS and therefore also use a proper referencing/citations style (e.g. APA, MLA, Harvard etc. ) for your report.

# In[ ]:




