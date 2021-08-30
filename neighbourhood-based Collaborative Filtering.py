#!/usr/bin/env python
# coding: utf-8

# ## Recommendations with MovieTweetings: Collaborative Filtering
# 
# One of the most popular methods for making recommendations is **collaborative filtering**.  In collaborative filtering, you are using the collaboration of user-item recommendations to assist in making new recommendations.  
# 
# There are two main methods of performing collaborative filtering:
# 
# 1. **Neighborhood-Based Collaborative Filtering**, which is based on the idea that we can either correlate items that are similar to provide recommendations or we can correlate users to one another to provide recommendations.
# 
# 2. **Model Based Collaborative Filtering**, which is based on the idea that we can use machine learning and other mathematical models to understand the relationships that exist amongst items and users to predict ratings and provide ratings.
# 
# 
# In this notebook, you will be working on performing **neighborhood-based collaborative filtering**.  There are two main methods for performing collaborative filtering:
# 
# 1. **User-based collaborative filtering:** In this type of recommendation, users related to the user you would like to make recommendations for are used to create a recommendation.
# 
# 2. **Item-based collaborative filtering:** In this type of recommendation, first you need to find the items that are most related to each other item (based on similar ratings).  Then you can use the ratings of an individual on those similar items to understand if a user will like the new item.
# 
# In this notebook you will be implementing **user-based collaborative filtering**.  However, it is easy to extend this approach to make recommendations using **item-based collaborative filtering**.  First, let's read in our data and necessary libraries.
# 
# **NOTE**: Because of the size of the datasets, some of your code cells here will take a while to execute, so be patient!

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t
from scipy.sparse import csr_matrix
from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')

# Read in the datasets
movies = pd.read_csv('movies_clean.csv')
reviews = pd.read_csv('reviews_clean.csv')

del movies['Unnamed: 0']
del reviews['Unnamed: 0']

print(reviews.head())


# ### Measures of Similarity
# 
# When using **neighborhood** based collaborative filtering, it is important to understand how to measure the similarity of users or items to one another.  
# 
# There are a number of ways in which we might measure the similarity between two vectors (which might be two users or two items).  In this notebook, we will look specifically at two measures used to compare vectors:
# 
# * **Pearson's correlation coefficient**
# 
# Pearson's correlation coefficient is a measure of the strength and direction of a linear relationship. The value for this coefficient is a value between -1 and 1 where -1 indicates a strong, negative linear relationship and 1 indicates a strong, positive linear relationship. 
# 
# If we have two vectors x and y, we can define the correlation between the vectors as:
# 
# 
# $$CORR(x, y) = \frac{\text{COV}(x, y)}{\text{STDEV}(x)\text{ }\text{STDEV}(y)}$$
# 
# where 
# 
# $$\text{STDEV}(x) = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$
# 
# and 
# 
# $$\text{COV}(x, y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$
# 
# where n is the length of the vector, which must be the same for both x and y and $\bar{x}$ is the mean of the observations in the vector.  
# 
# We can use the correlation coefficient to indicate how alike two vectors are to one another, where the closer to 1 the coefficient, the more alike the vectors are to one another.  There are some potential downsides to using this metric as a measure of similarity.  You will see some of these throughout this workbook.
# 
# 
# * **Euclidean distance**
# 
# Euclidean distance is a measure of the straightline distance from one vector to another.  Because this is a measure of distance, larger values are an indication that two vectors are different from one another (which is different than Pearson's correlation coefficient).
# 
# Specifically, the euclidean distance between two vectors x and y is measured as:
# 
# $$ \text{EUCL}(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$
# 
# Different from the correlation coefficient, no scaling is performed in the denominator.  Therefore, you need to make sure all of your data are on the same scale when using this metric.
# 
# **Note:** Because measuring similarity is often based on looking at the distance between vectors, it is important in these cases to scale your data or to have all data be in the same scale.  In this case, we will not need to scale data because they are all on a 10 point scale, but it is always something to keep in mind!
# 
# ------------
# 
# ### User-Item Matrix
# 
# In order to calculate the similarities, it is common to put values in a matrix.  In this matrix, users are identified by each row, and items are represented by columns.  
# 
# 
# ![alt text](images/userxitem.png "User Item Matrix")
# 

# In the above matrix, you can see that **User 1** and **User 2** both used **Item 1**, and **User 2**, **User 3**, and **User 4** all used **Item 2**.  However, there are also a large number of missing values in the matrix for users who haven't used a particular item.  A matrix with many missing values (like the one above) is considered **sparse**.
# 
# Our first goal for this notebook is to create the above matrix with the **reviews** dataset.  However, instead of 1 values in each cell, you should have the actual rating.  
# 
# The users will indicate the rows, and the movies will exist across the columns. To create the user-item matrix, we only need the first three columns of the **reviews** dataframe, which you can see by running the cell below.

# In[2]:


user_items = reviews[['user_id', 'movie_id', 'rating']]
user_items.head()


# ### Creating the User-Item Matrix
# 
# In order to create the user-items matrix (like the one above), I personally started by using a [pivot table](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html). 
# 
# However, I quickly ran into a memory error (a common theme throughout this notebook).  I will help you navigate around many of the errors I had, and achieve useful collaborative filtering results! 
# 
# _____
# 
# `1.` Create a matrix where the users are the rows, the movies are the columns, and the ratings exist in each cell, or a NaN exists in cells where a user hasn't rated a particular movie. If you get a memory error (like I did), [this link here](https://stackoverflow.com/questions/39648991/pandas-dataframe-pivot-memory-error) might help you!

# In[ ]:


# Create user-by-item matrix
user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()


# Check your results below to make sure your matrix is ready for the upcoming sections.

# In[ ]:


assert movies.shape[0] == user_by_movie.shape[1], "Oh no! Your matrix should have {} columns, and yours has {}!".format(movies.shape[0], user_by_movie.shape[1])
assert reviews.user_id.nunique() == user_by_movie.shape[0], "Oh no! Your matrix should have {} rows, and yours has {}!".format(reviews.user_id.nunique(), user_by_movie.shape[0])
print("Looks like you are all set! Proceed!")
HTML('<img src="images/greatjob.webp">')


# `2.` Now that you have a matrix of users by movies, use this matrix to create a dictionary where the key is each user and the value is an array of the movies each user has rated.

# In[ ]:


# Create a dictionary with users and corresponding movies seen

def movies_watched(user_id):
    '''
    INPUT:
    user_id - the user_id of an individual as int
    OUTPUT:
    movies - an array of movies the user has watched
    '''
    movies = user_by_movie.loc[user_id][user_by_movie.loc[user_id].isnull() == False].index.values

    return movies


def create_user_movie_dict():
    '''
    INPUT: None
    OUTPUT: movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
    
    Creates the movies_seen dictionary
    '''
    n_users = user_by_movie.shape[0]
    movies_seen = dict()

    for user1 in range(1, n_users+1):
        
        # assign list of movies to each user key
        movies_seen[user1] = movies_watched(user1)
    
    return movies_seen
    
movies_seen = create_user_movie_dict()


# `3.` If a user hasn't rated more than 2 movies, we consider these users "too new".  Create a new dictionary that only contains users who have rated more than 2 movies.  This dictionary will be used for all the final steps of this workbook.

# In[ ]:


# Remove individuals who have watched 2 or fewer movies - don't have enough data to make recs

def create_movies_to_analyze(movies_seen, lower_bound=2):
    '''
    INPUT:  
    movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
    lower_bound - (an int) a user must have more movies seen than the lower bound to be added to the movies_to_analyze dictionary

    OUTPUT: 
    movies_to_analyze - a dictionary where each key is a user_id and the value is an array of movie_ids
    
    The movies_seen and movies_to_analyze dictionaries should be the same except that the output dictionary has removed 
    
    '''
    movies_to_analyze = dict()

    for user, movies in movies_seen.items():
        if len(movies) > lower_bound:
            movies_to_analyze[user] = movies
    return movies_to_analyze

movies_to_analyze = create_movies_to_analyze(movies_seen)


# In[ ]:


# Run the tests below to check that your movies_to_analyze matches the solution
assert len(movies_to_analyze) == 23512, "Oops!  It doesn't look like your dictionary has the right number of individuals."
assert len(movies_to_analyze[2]) == 23, "Oops!  User 2 didn't match the number of movies we thought they would have."
assert len(movies_to_analyze[7])  == 3, "Oops!  User 7 didn't match the number of movies we thought they would have."
print("If this is all you see, you are good to go!")


# ### Calculating User Similarities
# 
# Now that you have set up the **movies_to_analyze** dictionary, it is time to take a closer look at the similarities between users. Below is the pseudocode for how I thought about determining the similarity between users:
# 
# ```
# for user1 in movies_to_analyze
#     for user2 in movies_to_analyze
#         see how many movies match between the two users
#         if more than two movies in common
#             pull the overlapping movies
#             compute the distance/similarity metric between ratings on the same movies for the two users
#             store the users and the distance metric
# ```
# 
# However, this took a very long time to run, and other methods of performing these operations did not fit on the workspace memory!
# 
# Therefore, rather than creating a dataframe with all possible pairings of users in our data, your task for this question is to look at a few specific examples of the correlation between ratings given by two users.  For this question consider you want to compute the [correlation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) between users.
# 
# `4.` Using the **movies_to_analyze** dictionary and **user_by_movie** dataframe, create a function that computes the correlation between the ratings of similar movies for two users.  Then use your function to compare your results to ours using the tests below.  

# In[ ]:


def compute_correlation(user1, user2):
    '''
    INPUT
    user1 - int user_id
    user2 - int user_id
    OUTPUT
    the correlation between the matching ratings between the two users
    '''
    # Pull movies for each user
    movies1 = movies_to_analyze[user1]
    movies2 = movies_to_analyze[user2]
    
    
    # Find Similar Movies
    sim_movs = np.intersect1d(movies1, movies2, assume_unique=True)
    
    # Calculate correlation between the users
    df = user_by_movie.loc[(user1, user2), sim_movs]
    corr = df.transpose().corr().iloc[0,1]
    
    return corr #return the correlation


# In[ ]:


# Test your function against the solution
assert compute_correlation(2,2) == 1.0, "Oops!  The correlation between a user and itself should be 1.0."
assert round(compute_correlation(2,66), 2) == 0.76, "Oops!  The correlation between user 2 and 66 should be about 0.76."
assert np.isnan(compute_correlation(2,104)), "Oops!  The correlation between user 2 and 104 should be a NaN."

print("If this is all you see, then it looks like your function passed all of our tests!")


# ### Why the NaN's?
# 
# If the function you wrote passed all of the tests, then you have correctly set up your function to calculate the correlation between any two users.  
# 
# `5.` But one question is, why are we still obtaining **NaN** values?  As you can see in the code cell above, users 2 and 104 have a correlation of **NaN**. Why?

# Think and write your ideas here about why these NaNs exist, and use the cells below to do some coding to validate your thoughts. You can check other pairs of users and see that there are actually many NaNs in our data - 2,526,710 of them in fact. These NaN's ultimately make the correlation coefficient a less than optimal measure of similarity between two users.
# 
# ```
# In the denominator of the correlation coefficient, we calculate the standard deviation for each user's ratings.  The ratings for user 2 are all the same rating on the movies that match with user 104.  Therefore, the standard deviation is 0.  Because a 0 is in the denominator of the correlation coefficient, we end up with a **NaN** correlation coefficient.  Therefore, a different approach is likely better for this particular situation.
# ```

# In[ ]:


# Which movies did both user 2 and user 104 see?
set_2 = set(movies_to_analyze[2])
set_104 = set(movies_to_analyze[104])
set_2.intersection(set_104)


# In[ ]:


# What were the ratings for each user on those movies?
print(user_by_movie.loc[2, set_2.intersection(set_104)])
print(user_by_movie.loc[104, set_2.intersection(set_104)])


# `6.` Because the correlation coefficient proved to be less than optimal for relating user ratings to one another, we could instead calculate the euclidean distance between the ratings.  I found [this post](https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy) particularly helpful when I was setting up my function.  This function should be very similar to your previous function.  When you feel confident with your function, test it against our results.

# In[ ]:


def compute_euclidean_dist(user1, user2):
    '''
    INPUT
    user1 - int user_id
    user2 - int user_id
    OUTPUT
    the euclidean distance between user1 and user2
    '''
    # Pull movies for each user
    movies1 = movies_to_analyze[user1]
    movies2 = movies_to_analyze[user2]
    
    
    # Find Similar Movies
    sim_movs = np.intersect1d(movies1, movies2, assume_unique=True)
    
    # Calculate euclidean distance between the users
    df = user_by_movie.loc[(user1, user2), sim_movs]
    dist = np.linalg.norm(df.loc[user1] - df.loc[user2])
    
    return dist #return the euclidean distance


# In[1]:


# Read in solution euclidean distances"
import pickle
df_dists = pd.read_pickle("data/Term2/recommendations/lesson1/data/dists.p")


# In[ ]:


# Test your function against the solution
assert compute_euclidean_dist(2,2) == df_dists.query("user1 == 2 and user2 == 2")['eucl_dist'][0], "Oops!  The distance between a user and itself should be 0.0."
assert round(compute_euclidean_dist(2,66), 2) == round(df_dists.query("user1 == 2 and user2 == 66")['eucl_dist'][1], 2), "Oops!  The distance between user 2 and 66 should be about 2.24."
assert np.isnan(compute_euclidean_dist(2,104)) == np.isnan(df_dists.query("user1 == 2 and user2 == 104")['eucl_dist'][4]), "Oops!  The distance between user 2 and 104 should be 2."

print("If this is all you see, then it looks like your function passed all of our tests!")


# ### Using the Nearest Neighbors to Make Recommendations
# 
# In the previous question, you read in **df_dists**. Therefore, you have a measure of distance between each user and every other user. This dataframe holds every possible pairing of users, as well as the corresponding euclidean distance.
# 
# Because of the **NaN** values that exist within the correlations of the matching ratings for many pairs of users, as we discussed above, we will proceed using **df_dists**. You will want to find the users that are 'nearest' each user.  Then you will want to find the movies the closest neighbors have liked to recommend to each user.
# 
# I made use of the following objects:
# 
# * df_dists (to obtain the neighbors)
# * user_items (to obtain the movies the neighbors and users have rated)
# * movies (to obtain the names of the movies)
# 
# `7.` Complete the functions below, which allow you to find the recommendations for any user.  There are five functions which you will need:
# 
# * **find_closest_neighbors** - this returns a list of user_ids from closest neighbor to farthest neighbor using euclidean distance
# 
# 
# * **movies_liked** - returns an array of movie_ids
# 
# 
# * **movie_names** - takes the output of movies_liked and returns a list of movie names associated with the movie_ids
# 
# 
# * **make_recommendations** - takes a user id and goes through closest neighbors to return a list of movie names as recommendations
# 
# 
# * **all_recommendations** = loops through every user and returns a dictionary of with the key as a user_id and the value as a list of movie recommendations

# In[ ]:


def find_closest_neighbors(user):
    '''
    INPUT:
        user - (int) the user_id of the individual you want to find the closest users
    OUTPUT:
        closest_neighbors - an array of the id's of the users sorted from closest to farthest away
    '''
    # I treated ties as arbitrary and just kept whichever was easiest to keep using the head method
    # You might choose to do something less hand wavy
    
    closest_users = df_dists[df_dists['user1']==user].sort_values(by='eucl_dist').iloc[1:]['user2']
    closest_neighbors = np.array(closest_users)
    
    return closest_neighbors
    
    
    
def movies_liked(user_id, min_rating=7):
    '''
    INPUT:
    user_id - the user_id of an individual as int
    min_rating - the minimum rating considered while still a movie is still a "like" and not a "dislike"
    OUTPUT:
    movies_liked - an array of movies the user has watched and liked
    '''
    movies_liked = np.array(user_items.query('user_id == @user_id and rating > (@min_rating -1)')['movie_id'])
    
    return movies_liked


def movie_names(movie_ids):
    '''
    INPUT
    movie_ids - a list of movie_ids
    OUTPUT
    movies - a list of movie names associated with the movie_ids
    
    '''
    movie_lst = list(movies[movies['movie_id'].isin(movie_ids)]['movie'])
   
    return movie_lst
    
    
def make_recommendations(user, num_recs=10):
    '''
    INPUT:
        user - (int) a user_id of the individual you want to make recommendations for
        num_recs - (int) number of movies to return
    OUTPUT:
        recommendations - a list of movies - if there are "num_recs" recommendations return this many
                          otherwise return the total number of recommendations available for the "user"
                          which may just be an empty list
    '''
    # I wanted to make recommendations by pulling different movies than the user has already seen
    # Go in order from closest to farthest to find movies you would recommend
    # I also only considered movies where the closest user rated the movie as a 9 or 10
    
    # movies_seen by user (we don't want to recommend these)
    movies_seen = movies_watched(user)
    closest_neighbors = find_closest_neighbors(user)
    
    # Keep the recommended movies here
    recs = np.array([])
    
    # Go through the neighbors and identify movies they like the user hasn't seen
    for neighbor in closest_neighbors:
        neighbs_likes = movies_liked(neighbor)
        
        #Obtain recommendations for each neighbor
        new_recs = np.setdiff1d(neighbs_likes, movies_seen, assume_unique=True)
        
        # Update recs with new recs
        recs = np.unique(np.concatenate([new_recs, recs], axis=0))
        
        # If we have enough recommendations exit the loop
        if len(recs) > num_recs-1:
            break
    
    # Pull movie titles using movie ids
    recommendations = movie_names(recs)
    
    return recommendations

def all_recommendations(num_recs=10):
    '''
    INPUT 
        num_recs (int) the (max) number of recommendations for each user
    OUTPUT
        all_recs - a dictionary where each key is a user_id and the value is an array of recommended movie titles
    '''
    
    # All the users we need to make recommendations for
    users = np.unique(df_dists['user1'])
    n_users = len(users)
    
    #Store all recommendations in this dictionary
    all_recs = dict()
    
    # Make the recommendations for each user
    for user in users:
        all_recs[user] = make_recommendations(user, num_recs)
    
    return all_recs

all_recs = all_recommendations(10)


# In[ ]:


# This loads our solution dictionary so you can compare results - FULL PATH IS "data/Term2/recommendations/lesson1/data/all_recs.p"
all_recs_sol = pd.read_pickle("data/Term2/recommendations/lesson1/data/all_recs.p")


# In[ ]:


assert all_recs[2] == make_recommendations(2), "Oops!  Your recommendations for user 2 didn't match ours."
assert all_recs[26] == make_recommendations(26), "Oops!  It actually wasn't possible to make any recommendations for user 26."
assert all_recs[1503] == make_recommendations(1503), "Oops! Looks like your solution for user 1503 didn't match ours."
print("If you made it here, you now have recommendations for many users using collaborative filtering!")
HTML('<img src="images/greatjob.webp">')


# ### Now What?
# 
# If you made it this far, you have successfully implemented a solution to making recommendations using collaborative filtering. 
# 
# `8.` Let's do a quick recap of the steps taken to obtain recommendations using collaborative filtering.  

# In[ ]:


# Check your understanding of the results by correctly filling in the dictionary below
a = "pearson's correlation and spearman's correlation"
b = 'item based collaborative filtering'
c = "there were too many ratings to get a stable metric"
d = 'user based collaborative filtering'
e = "euclidean distance and pearson's correlation coefficient"
f = "manhattan distance and euclidean distance"
g = "spearman's correlation and euclidean distance"
h = "the spread in some ratings was zero"
i = 'content based recommendation'

sol_dict = {
    'The type of recommendation system implemented here was a ...': d,
    'The two methods used to estimate user similarity were: ': e,
    'There was an issue with using the correlation coefficient.  What was it?': h
}

t.test_recs(sol_dict)


# Additionally, let's take a closer look at some of the results.  There are two solution files that you read in to check your results, and you created these objects
# 
# * **df_dists** - a dataframe of user1, user2, euclidean distance between the two users
# * **all_recs_sol** - a dictionary of all recommendations (key = user, value = list of recommendations)  
# 
# `9.` Use these two objects along with the cells below to correctly fill in the dictionary below and complete this notebook!

# In[ ]:


a = 567
b = 1503
c = 1319
d = 1325
e = 2526710
f = 0
g = 'Use another method to make recommendations - content based, knowledge based, or model based collaborative filtering'

sol_dict2 = {
    'For how many pairs of users were we not able to obtain a measure of similarity using correlation?': e,
    'For how many pairs of users were we not able to obtain a measure of similarity using euclidean distance?': f,
    'For how many users were we unable to make any recommendations for using collaborative filtering?': c,
    'For how many users were we unable to make 10 recommendations for using collaborative filtering?': d,
    'What might be a way for us to get 10 recommendations for every user?': g   
}

t.test_recs2(sol_dict2)


# In[ ]:


# Use the cells below for any work you need to do!


# In[ ]:


# Users without recs
users_without_recs = []
for user, movie_recs in all_recs.items():
    if len(movie_recs) == 0:
        users_without_recs.append(user)
    
len(users_without_recs)


# In[ ]:


# NaN euclidean distance values
df_dists['eucl_dist'].isnull().sum()


# In[ ]:


# Users with fewer than 10 recs
users_with_less_than_10recs = []
for user, movie_recs in all_recs.items():
    if len(movie_recs) < 10:
        users_with_less_than_10recs.append(user)
    
len(users_with_less_than_10recs)


# In[ ]:




