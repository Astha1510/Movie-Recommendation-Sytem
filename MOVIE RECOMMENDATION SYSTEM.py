#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import warnings


# In[4]:


warnings.filterwarnings('ignore')#filter unneccessary warnings that will come


# How to Load Data Set

# In[5]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)#sep-\t this is seperater tab  #df is data frame it's in panda library


# In[6]:


df.head()# head it shows top 5 enteries to show more than 5 pass argument eg df.head(10)


# In[7]:


#timestamp is when that person rated that particular movie  


# In[8]:


df.shape# basically tells us that it is data set of rows and coloumns


# In[9]:


df['user_id']# total user 


# In[10]:



df['user_id'].nunique() # to get how many unique user r there


# In[11]:


df['item_id'].nunique()


# In[12]:


movies_title=pd.read_csv('u.item',sep="\|",header=None,encoding="ISO-8859-1") #loading 2 nd data set


# In[13]:


movies_title.shape


# In[14]:


movies_titles=movies_title[[0,1]]
movies_titles.columns=['item_id','title']
movies_titles.head()


# In[15]:


import pandas as pd
df=pd.merge(df,movies_titles,on="item_id")#merges dataset 1 and 2  


# In[16]:


df


# In[17]:


df.tail()#last 5 enteries


# In[18]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])#mean -average rating 


# In[19]:


ratings.head()


# In[20]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# #Create the Recommendar System

# In[21]:


df.head()


# In[22]:


moviemat=df.pivot_table(index="user_id",columns="title",values="rating")


# In[23]:


moviemat.head()


# In[24]:


starwars_user_ratings=moviemat['Star Wars (1977)'] #creation of movie recommendation of starwars


# In[26]:


starwars_user_ratings.head(10)#rating given by 10 people to star wars


# In[27]:


similar_to_starwars=moviemat.corrwith(starwars_user_ratings)#how closely starwars is related to other movies


# In[28]:


similar_to_starwars


# In[29]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])


# In[31]:


corr_starwars.dropna(inplace=True)# dropna will clear all nan values


# In[32]:


corr_starwars


# In[33]:


corr_starwars.head()


# In[38]:


corr_starwars.sort_values('correlation',ascending=False).head(10)


# In[39]:


ratings


# In[41]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[42]:


corr_starwars


# In[44]:


corr_starwars.head()


# In[52]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False)


# In[89]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return predictions


# In[90]:


predict_my_movies=predict_movies('Titanic (1997)')


# In[92]:


predict_my_movies.head()


# In[93]:


predict_my_movies=predict_movies('Kolya (1996)')


# In[94]:


predict_my_movies.head()


# In[ ]:




