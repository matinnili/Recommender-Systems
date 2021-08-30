#!/usr/bin/env python
# coding: utf-8

# ### Implementing FunkSVD - Solution
# 
# In this notebook we will take a look at writing our own function that performs FunkSVD, which will follow the steps you saw in the previous video.  If you find that you aren't ready to tackle this task on your own, feel free to skip to the following video where you can watch as I walk through the steps.
# 
# To test our algorithm, we will run it on the subset of the data you worked with earlier.  Run the cell below to get started.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import svd_tests as t
get_ipython().run_line_magic('matplotlib', 'inline')

# Read in the datasets
movies = pd.read_csv('../data/movies_clean.csv')
reviews = pd.read_csv('../data/reviews_clean.csv')

del movies['Unnamed: 0']
del reviews['Unnamed: 0']

# Create user-by-item matrix
user_items = reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()

# Create data subset
user_movie_subset = user_by_movie[[73486, 75314,  68646, 99685]].dropna(axis=0)
ratings_mat = np.matrix(user_movie_subset)
print(ratings_mat)


# `1.` You will use the **user_movie_subset** matrix to show that your FunkSVD algorithm will converge.  In the below cell, use the comments and document string to assist you as you complete writing your own function to complete FunkSVD.  You may also want to try to complete the funtion on your own without the assistance of comments.  You may feel free to remove and add to the function in any way that gets you a working solution! 
# 
# **Notice:** There isn't a sigma matrix in this version of matrix factorization.

# In[2]:


def FunkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization
    
    INPUT:
    ratings_mat - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate 
    iters - (int) the number of iterations
    
    OUTPUT:
    user_mat - (numpy array) a user by latent feature matrix
    movie_mat - (numpy array) a latent feature by movie matrix
    '''
    
    # Set up useful values to be used through the rest of the function
    n_users = ratings_mat.shape[0]
    n_movies = ratings_mat.shape[1]
    num_ratings = np.count_nonzero(~np.isnan(ratings_mat))
    
    # initialize the user and movie matrices with random values
    user_mat = np.random.rand(n_users, latent_features)
    movie_mat = np.random.rand(latent_features, n_movies)
    
    # initialize sse at 0 for first iteration
    sse_accum = 0
    
    # keep track of iteration and MSE
    print("Optimizaiton Statistics")
    print("Iterations | Mean Squared Error ")
    
    # for each iteration
    for iteration in range(iters):

        # update our sse
        old_sse = sse_accum
        sse_accum = 0
        
        # For each user-movie pair
        for i in range(n_users):
            for j in range(n_movies):
                
                # if the rating exists
                if ratings_mat[i, j] > 0:
                    
                    # compute the error as the actual minus the dot product of the user and movie latent features
                    diff = ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])
                    
                    # Keep track of the sum of squared errors for the matrix
                    sse_accum += diff**2
                    
                    # update the values in each matrix in the direction of the gradient
                    for k in range(latent_features):
                        user_mat[i, k] += learning_rate * (2*diff*movie_mat[k, j])
                        movie_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

        # print results
        print("%d \t\t %f" % (iteration+1, sse_accum / num_ratings))
        
    return user_mat, movie_mat 


# `2.` Try out your function on the **user_movie_subset** dataset.  First try 4 latent features, a learning rate of 0.005, and 10 iterations.  When you take the dot product of the resulting **U** and **V** matrices, how does the resulting **user_movie** matrix compare to the original subset of the data?

# In[3]:


user_mat, movie_mat = FunkSVD(ratings_mat, latent_features=4, learning_rate=0.005, iters=10)


# In[4]:


print(np.dot(user_mat, movie_mat))
print(ratings_mat)


# **The predicted ratings from the dot product are already starting to look a lot like the original data values even after only 10 iterations.  You can see some extreme low values that are not captured well yet.  The 5 in the second to last row in the first column is predicted as an 8, and the 4 in the second row and second column is predicted to be a 7.  Clearly the model is not done learning, but things are looking good.**

# `3.` Let's try out the function again on the **user_movie_subset** dataset.  This time we will again use 4 latent features and a learning rate of 0.005.  However, let's bump up the number of iterations to 250.  When you take the dot product of the resulting **U** and **V** matrices, how does the resulting **user_movie** matrix compare to the original subset of the data?  What do you notice about your error at the end of the 250 iterations?

# In[5]:


user_mat, movie_mat = FunkSVD(ratings_mat, latent_features=4, learning_rate=0.005, iters=250)


# In[6]:


print(np.dot(user_mat, movie_mat))
print(ratings_mat)


# **In this case, we were able to completely reconstruct the item-movie matrix to obtain an essentially 0 mean squared error. I obtained 0 MSE on iteration 165.**

# The last time we placed an **np.nan** value into this matrix the entire svd algorithm in python broke.  Let's see if that is still the case using your FunkSVD function.  In the below cell, I have placed a nan into the first cell of your numpy array.  
# 
# `4.` Use 4 latent features, a learning rate of 0.005, and 250 iterations.  Are you able to run your SVD without it breaking (something that was not true about the python built in)?  Do you get a prediction for the nan value?  What is your prediction for the missing value? Use the cells below to answer these questions.

# In[7]:


ratings_mat[0, 0] = np.nan
ratings_mat


# In[8]:


# run SVD on the matrix with the missing value
user_mat, movie_mat = FunkSVD(ratings_mat, latent_features=4, learning_rate=0.005, iters=250)


# In[9]:


preds = np.dot(user_mat, movie_mat)
print("The predicted value for the missing rating is {}:".format(preds[0,0]))
print()
print("The actual value for the missing rating is {}:".format(ratings_mat[0,0]))
print()
print("That's right! You just predicted a rating for a user-movie pair that was never rated!")
print("But if you look in the original matrix, this was actually a value of 10. Not bad!")


# Now let's extend this to a more realistic example. Unfortunately, running this function on your entire user-movie matrix is still not something you likely want to do on your local machine.  However, we can see how well this example extends to 
# 
# `5.` Given the size of this matrix, this will take quite a bit of time.  Consider the following hyperparameters: 4 latent features, 0.005 learning rate, and 20 iterations.  Grab a snack, take a walk, and this should be done running in a bit.

# In[10]:


# Setting up a matrix of the first 1000 users with movie ratings
first_1000_users = np.matrix(user_by_movie.head(1000))

# perform funkSVD on the matrix of the top 1000 users
user_mat, movie_mat = FunkSVD(first_1000_users, latent_features=4, learning_rate=0.005, iters=20)


# `6.` Now that you have a set of predictions for each user-movie pair.  Let's answer a few questions about your results. Provide the correct values to each of the variables below, and check your solutions using the tests below.

# In[11]:


# How many actual ratings exist in first_1000_users
num_ratings = np.count_nonzero(~np.isnan(first_1000_users))
print("The number of actual ratings in the first_1000_users is {}.".format(num_ratings))
print()

# How many ratings did we make for user-movie pairs that didn't have ratings
ratings_for_missing = first_1000_users.shape[0]*first_1000_users.shape[1] - num_ratings
print("The number of ratings made for user-movie pairs that didn't have ratings is {}".format(ratings_for_missing))


# In[12]:


# Test your results against the solution
assert num_ratings == 10852, "Oops!  The number of actual ratings doesn't quite look right."
assert ratings_for_missing == 31234148, "Oops!  The number of movie-user pairs that you made ratings for that didn't actually have ratings doesn't look right."

# Make sure you made predictions on all the missing user-movie pairs
preds = np.dot(user_mat, movie_mat)
assert np.isnan(preds).sum() == 0
print("Nice job!  Looks like you have predictions made for all the missing user-movie pairs! But I still have one question... How good are they?")


# In[ ]:




