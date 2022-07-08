#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from PIL import Image
from pathlib import Path
import collections 
from itertools import chain

image =r"C:\Users\Anna MacKenzie\OneDrive\Desktop\Training_ML\train"


# In[2]:


def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    
    bw=img.convert('L')
    color = np.asarray(img)
    BW = np.asarray(bw)
    BW = BW[:,:,np.newaxis]
    photo = np.dstack((BW, color))
    
    flat= np.array(photo).ravel()
    
    it=iter(flat)
    tuple_list= list(zip(it, it, it, it))
    return tuple_list
    
    #zipper=zip(BWflat, RGB_array)
    #image_dictionary=dict(zipper)
    #return image_dictionary


# In[3]:


def dictionary_folder(image_folder):   
    images = Path(image_folder).glob('*.jpg')

    list_of_files = []
    for image in images:
        list_of_files.append(str(image))
        
    list_of_all_tuples = []
        
    for image in list_of_files:
        print("one photo!")
        
        one_photo_tuple = load_image(image)
        list_of_all_tuples.append(one_photo_tuple)
        
    return list_of_all_tuples


# In[19]:


done = dictionary_folder(image)


# In[20]:


done


# In[21]:


import itertools
merge=list(itertools.chain(*done))
merge


# In[22]:


ass=len(set(merge))


# In[23]:


ass


# In[17]:


merge=np.array(merge)
count=len(merge.unique())

