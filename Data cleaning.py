#!/usr/bin/env python
# coding: utf-8

# ### Recommendations with MovieTweetings: Getting to Know The Data
# 
# Throughout this lesson, you will be working with the [MovieTweetings Data](https://github.com/sidooms/MovieTweetings/tree/master/recsyschallenge2014).  To get started, you can read more about this project and the dataset from the [publication here](http://crowdrec2013.noahlab.com.hk/papers/crowdrec2013_Dooms.pdf).
# 
# **Note:** There are solutions to each of the notebooks available by hitting the orange jupyter logo in the top left of this notebook.  Additionally, you can watch me work through the solutions on the screencasts that follow each workbook. 
# 
# To get started, read in the libraries and the two datasets you will be using throughout the lesson using the code below.
# 
#  

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t

get_ipython().run_line_magic('matplotlib', 'inline')

# Read in the datasets
movies = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat', delimiter='::', header=None, names=['movie_id', 'movie', 'genre'], dtype={'movie_id': object}, engine='python',encoding="utf8")
reviews = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat', delimiter='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], dtype={'movie_id': object, 'user_id': object, 'timestamp': object}, engine='python')


# #### 1. Take a Look At The Data 
# 
# Take a look at the data and use your findings to fill in the dictionary below with the correct responses to show your understanding of the data.

# In[11]:


movies.head()


# In[3]:


import re


# In[4]:


movies["year_2"]=movies["movie"].apply(lambda x : re.search(r"[1-9]\w+",x)[0])


# In[8]:


movies["year"]=pd.Series(movies["year_2"],dtype=int)


# In[9]:


def period(x):
    try:
        x=int(x)
        if (x>=1800 and x<1900):
            return "1800's"
        elif(x>=1900 and x<2000):
            return "1900's"
        else:
            return "2000's"
    except ValueError:
        return None


# In[10]:


movies["year"]=movies["year"].apply(period)


# In[11]:


movies=pd.concat([movies,pd.get_dummies(movies.year)],axis=1)


# In[12]:


def genre(x):
    try:
        return x.split("|")
    except:
        return None


# In[13]:


movies["new"]=movies["genre"].apply(genre)


# In[65]:


genres=movies["new"].explode().unique()


# In[68]:


genres=genres[np.where(genres!=None)]


# In[69]:


genres


# In[62]:


k=pd.concat([movies,pd.get_dummies(movies["new"].explode())],axis=1)


# In[71]:


movies=pd.concat([movies,k[genres].groupby(k.index).sum()],axis=1)


# In[ ]:


# Use your findings to match each variable to the correct statement in the dictionary
a = 53968
b = 10
c = 7
d = 31245
e = 15
f = 0
g = 4
h = 712337
i = 28

dict_sol1 = {
'The number of movies in the dataset': 
'The number of ratings in the dataset':
'The number of different genres':
'The number of unique users in the dataset':
'The number missing ratings in the reviews dataset':
'The average rating given across all ratings':
'The minimum rating given across all ratings':
'The maximum rating given across all ratings':
}

# Check your solution
t.q1_check(dict_sol1)


# #### 2. Data Cleaning
# 
# Next, we need to pull some additional relevant information out of the existing columns. 
# 
# For each of the datasets, there are a couple of cleaning steps we need to take care of:
# 
# #### Movies
# * Pull the date from the title and create new column
# * Dummy the date column with 1's and 0's for each century of a movie (1800's, 1900's, and 2000's)
# * Dummy column the genre with 1's and 0's
# 
# #### Reviews
# * Create a date out of time stamp
# * Create month/day/year columns from timestamp as dummies
# 
# You can check your results against the header of my solution by running the cell below with the **show_clean_dataframes** function.
pd.datetime.date(re)
# In[73]:


k=pd.to_datetime(reviews.timestamp,unit="s")


# In[74]:


reviews["month"]=k.apply(lambda x :pd.datetime.date(x).month)
reviews["day"]=k.apply(lambda x:pd.datetime.date(x).day )
reviews["year"]=k.apply(lambda x: pd.datetime.date(x).year )


# In[75]:


reviews=pd.concat([reviews,pd.get_dummies(reviews["day"]),pd.get_dummies(reviews["month"],prefix="month"),                 pd.get_dummies(reviews["year"])],axis=1)


# In[78]:


movies.to_csv("clean_movies.csv")
reviews.to_csv("clean_reviews.csv")


# In[203]:


ranking=pd.DataFrame()

ranking["review_avg"]=reviews["rating"].groupby(reviews.movie_id).apply(test)
ranking["count"]=reviews["rating"].groupby(reviews.movie_id).count()
ranking["recent"]=reviews["timestamp"].groupby(reviews.movie_id).apply(np.max)
ranking=ranking[ranking['count']>=5]


# In[213]:


ranking.sort_values(by=["review_avg","count","recent"],ascending=[False,False,False],inplace=True)


# In[218]:


ranking=pd.merge(ranking,movies,on="movie_id",how="left")
ranking.head(20)


# In[255]:


recommended_movies=ranking[ranking.new.apply(truth,args=["Comedy","Drama"]) & ranking.year_2.apply(truth,args=["2017","2020"])]


# In[246]:


def truth(x,*criteria):
    if type(x)==list:
        return any([i in criteria for i in x])
    else:
        return x in criteria


# In[ ]:


reviews_new, movies_new = t.show_clean_dataframes()

