# -*- coding: utf-8 -*-
"""
@authors: Marcin Grelewicz (s17692), Edyta Bartos (s17699)

Reworked version of alghorithms from the book: 
Artificial Intelligence with Python by Prateek Joshi

This program compares each user ratings of movies stored in CSV file.
There are 3 alternative functions to recommend the best movies.
This program by default will compare all users to the declared user1. 

In the main function you can change user and/or function.

"""

import pandas as pd
import numpy as np


## Compute the Euclidean distance score between user1 and user2 
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    ## Movies rated by both user1 and user2
    common_movies = {}
    
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
    print(common_movies)
    
    ## If there are no common movies between the users, then the score is 0 
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] 
                                          - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff))) 


## Compute the Pearson correlation score between user1 and user2 
def pearson_score(dataset, user1, user2):
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
    #print(user1_sum)
    #print(user2_sum)
    
    ## Calculate the sum of squares of ratings of all the common movies 
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for 
                                item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for 
                                item in common_movies])

    ## Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for 
                              item in common_movies])
    #print(sum_of_products)
    
    ## Calculate the Pearson correlation score
    Sxy = abs(sum_of_products - (user1_sum * user2_sum / num_ratings))
    Sxx = abs(user1_squared_sum - np.square(user1_sum) / num_ratings)
    Syy = abs(user2_squared_sum - np.square(user2_sum) / num_ratings)
    #print(Sxy,Sxx,Syy)
    if Sxx * Syy == 0:
        return 0
    #print(common_movies)

    return Sxy / np.sqrt(Sxx * Syy)


    ## our version based on Euclidean distance score between user1 and user2
    # works better when dataset is small and there is only few commmon items
def edmar_score(dataset, user1, user2):
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

    # movs_tmp = [df.iloc[j,i] for i in range(14) for j in range(73) if j%2!=0]
    # movies = []
    # for item in movs_tmp:
    #     if item not in movies:
    #         movies.append(item)
    # #print(movies)    
    
    pairs = [[df.iloc[i,j], df.iloc[i+1,j], df.iloc[0,j]] for i 
             in range(1,73) if i % 2 != 0 for j in range(14)]
    #print(pairs)
    #print(pairs[0][0])
    

    dataset = {}
    for movie, score, user in pairs:
        if len(movie) > 1 and movie != "NaN":
            if score != "NaN":
                dataset.setdefault(user, {})[movie] = int(float(score))
    #print(dataset)
    # for k, v in dataset.items():
    #     print(k,v,'\n')
      
    
    user1 = "Paweł Czapiewski"
    user2 = ''
    score_type = "edmar" # "Euclidean" # "Pearson" # 

  
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
    
    
    best_fit = max(fit, key=lambda x: x[1])
    print("\nHighest probability of match: ", best_fit[0], best_fit[1],"%\n")
    
    ##Recommended movies
    pcz_movies = []
    best_fit_user_movies = []
    for movie, score, user in pairs:
         if user == user1:
             pcz_movies.append(movie)
         if user == best_fit[0]:
             best_fit_user_movies.append(movie)
    
    for movie, score in dataset.items():
        if user == best_fit[0]:
            if score > 8:
                    print(movie,'\n')
    
    print("Recommended movies: ")                        
    for movie, score, user in pairs:
         if user == best_fit[0]:
             if score == '8' or score == '9' or score == '10':
                 if movie not in pcz_movies: 
                     print(movie)    
                 
                 
                 