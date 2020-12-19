# -*- coding: utf-8 -*-
"""
@authors: Marcin Grelewicz (s17692), Edyta Bartos (s17699)

Reworked version of alghorithms from the book: 
Artificial Intelligence with Python by Prateek Joshi

This program compares each user ratings of movies stored in CSV file.
There are 3 alternative functions to recommend the best movies 
(Euclidean, Pearson and our own: edmar).
This program by default will compare all users to the declared user1. 

In the main function you can change user and/or function.

"""

import pandas as pd
import numpy as np


## Euclidean:
def euclidean_score(dataset, user1, user2):
""" Compute the Euclidean distance score between user1 and user2 """    
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    ## Movies rated by both user1 and user2
    common_movies = {}
    
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
    #print(common_movies)
    
    ## If there are no common movies between the users, then the score is 0 
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            print(item, " ", dataset[user1][item], ":", dataset[user2][item])
            squared_diff.append(np.square(dataset[user1][item] 
                                          - dataset[user2][item]))
    result = 1 / (1 + np.sqrt(np.sum(squared_diff)))

    return "%.2f" % result 


## Pearson:
def pearson_score(dataset, user1, user2):
""" Compute the Pearson correlation score between user1 and user2 """
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    ## Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
   
    num_ratings = len(common_movies) 

    ## If there are no common movies between the users, then the score is 0  
    if num_ratings == 0:
        return 0

    ## Calculate the sum of ratings of all the common movies 
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    
    ## Calculate the sum of squares of ratings of all the common movies 
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for 
                                item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for 
                                item in common_movies])

    ## Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for 
                              item in common_movies])
    
    for item in dataset[user1]:
        if item in dataset[user2]:
            print(item, " ", dataset[user1][item], ":", dataset[user2][item])
            
    ## Calculate the Pearson correlation score
    Sxy = abs(sum_of_products - (user1_sum * user2_sum / num_ratings))
    Sxx = abs(user1_squared_sum - np.square(user1_sum) / num_ratings)
    Syy = abs(user2_squared_sum - np.square(user2_sum) / num_ratings)
    
    if Sxx * Syy == 0:
        return 0
    return Sxy / np.sqrt(Sxx * Syy)



    ## edmar:
def edmar_score(dataset, user1, user2):
""" 
our version based on Euclidean distance score between user1 and user2
should work better when dataset is small and there is only few commmon items
"""
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    ## Movies rated by both user1 and user2
    common_movies = {}
    
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
    #print(common_movies)
    
    ## If there are no common movies between the users, then the score is 0 
    if len(common_movies) == 0:
        return 0

    diff = []
    
    # For each point of difference between scores of user1 and user2
    # penalty points are counted. Penalty is doubled when comparing
    # movies with highest and lowest scores
    for item in dataset[user1]:
        if item in dataset[user2]:
            print(item, " ", dataset[user1][item], ":", dataset[user2][item])
            if dataset[user1][item] < 3:
                diff.append(40*abs(dataset[user1][item] - dataset[user2][item]))
            elif dataset[user1][item] > 8:
                diff.append(40*abs(dataset[user1][item] - dataset[user2][item]))
            else:    
                diff.append(20*abs(dataset[user1][item] - dataset[user2][item]))
    #print(diff)
    import math
    result = math.log10(np.sum(diff)+100/len(common_movies))
    res = 1/(result)*100
    return "%.2f" % res



if __name__=='__main__':
    
    data = pd.read_csv('03.movs_transposed_clean.csv', delimiter=',', 
                       encoding=('UTF-8'))
    df = pd.DataFrame(data)
    
    #users = [df.iloc[0,i] for i in range(14)]
    #print(users)
    
    for columns in df.columns:
        df[columns] = df[columns].str.lower()
        df[columns] = df[columns].str.title()
 
    
    pairs = [[df.iloc[i,j], df.iloc[i+1,j], df.iloc[0,j]] for i 
             in range(1,73,2) for j in range(14)]
    

    dataset = {}
    for movie, score, user in pairs:
        if len(movie) > 1 and movie != "NaN":
            if score != "NaN":
                dataset.setdefault(user, {})[movie] = int(float(score))
      
    ## User1 is a reference user, 
    ## to create recommendations for other user, change user1 value
    user1 = "Paweł Czapiewski"
    user2 = ''
    score_type = "Pearson" # "Euclidean" # "edmar" # 

  
    fit = []    

    for k, v in dataset.items():
        user2 = k
        if score_type == 'Pearson':
            if user1 != user2:
                pear = pearson_score(dataset, user1, user2)
                if pear == 0:
                    print(" --> ", user1, "-", user2, ": ", "No common movies")
                else:
                    print(" --> ", user1, "-", user2, ": ", pear)
                    fit.append([user2, pear])
            print()        
        elif score_type == 'Euclidean':
            if user1 != user2:
                eucl = euclidean_score(dataset, user1, user2)
                if eucl == 0:
                    print(" --> ", user1, "-", user2, ": ", "No common movies")
                else:
                    print(" --> ", user1, "-", user2, ": ", eucl)
                    fit.append([user2, eucl])
            print()        
        elif score_type == 'edmar':
            if user1 != user2:
                edma = edmar_score(dataset, user1, user2)
                if edma == 0:
                    print(" --> ", user1, "-", user2, ": ", "No common movies")
                else:
                    print(" --> ", user1, "-", user2, ": ", edma,"%")
                    fit.append([user2, edma])
            print()
    
    
    fit_sorted = sorted(fit, key=lambda x: x[1])
    worst_fit = fit_sorted[:5]  #the number means number of users to consider
    fit_sorted.reverse()
    best_fit = fit_sorted[:5]
    
    #print("\nHighest probability of match: \n", best_fit)
    
    ##Recommended movies
    pcz_movies = []
    for movie, score, user in pairs:
        if user == user1:
            pcz_movies.append(movie)        
    
    worst_movies = []
    best_movies = []
    for [k,v] in best_fit:
        #userx = k
        for movie, score, user in pairs:
            if user == k and score == '10':
                if movie not in best_movies and movie not in pcz_movies:
                    best_movies.append(movie)
            elif user == k and score == '1':
                if movie not in worst_movies and movie not in pcz_movies:
                    worst_movies.append(movie)
    
    print("\nRecommended movies:")
    for x in best_movies[:7]:
        print("\t", x)
    
    print("\nNot recommended movies:")
    for y in worst_movies[:7]:
        print("\t", y)
    
    # import random    
    # print("\nRecommended movies:")
    # print(random.sample(best_movies, 7))
    # print("\nNot recommended movies:")
    # print(random.sample(worst_movies, 7))
             
                 
