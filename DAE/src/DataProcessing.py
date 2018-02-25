
# coding: utf-8

# In[115]:

import csv
import collections
import numpy as np
import pickle
from sets import Set


# In[116]:

def countUsers(filename):
    with open(filename) as csvDataFile:
        count = -1
        userNum = 0
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if count == -1:
                count = count+1
                continue;
            
            userNum = row[0]
    return userNum


# In[117]:

movieNum = 28562 # total number of valid movies in the dataset
userNum = 7941 # num of users who make more than 300 ratings

matrix = np.zeros((userNum, movieNum)) 

colToMovieid = dict()
movieidToCol = dict()

movieOverview = dict()
movieGenre = dict()
movieKeyword = dict()

tmbdidToMovieid = dict()
movieidToTmbdid = dict()

validTmbdid = Set()


# In[118]:

def findValid(metadata, keyword):
    overviews = Set()
    with open(metadata) as csvDataFile:
        count = -1
        userNum = 0
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if count == -1:
                count = count+1
                continue;
            overviews.add(row[5])
            count += 1
    
    keywords = Set()
    with open(keyword) as csvDataFile:
        count = -1
        userNum = 0
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if count == -1:
                count = count+1
                continue;
            keywords.add(row[0])
            count += 1
    
    for overviewid in overviews:
        if overviewid in keywords:
            validTmbdid.add(overviewid)


# In[119]:

findValid('./the-movies-dataset/movies_metadata.csv', './the-movies-dataset/keywords.csv')


# In[120]:

def readLink(filename):
    
        with open(filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            count = -1
            cnt = 0
            appeared = Set()
            for row in csvReader:
                if count == -1:
                    count = count+1
                    continue;
        
                movieid = row[0]
                tmbdid = row[2]
                
                if tmbdid not in validTmbdid:
                    continue
                
                if tmbdid in appeared:
                    cnt += 1
                    continue
                    
                appeared.add(tmbdid)
                colToMovieid[count] =movieid
                
                movieidToCol[movieid] = count
                
                tmbdidToMovieid[tmbdid] = movieid
                movieidToTmbdid[movieid] = tmbdid
                
                count = count+1
        print(count)
        print cnt


# In[121]:

readLink('./the-movies-dataset/links.csv')


# In[122]:

def readMoviemetadata(filename):
 
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        count = -1
        for row in csvReader:
            if count == -1:
                    count = count+1
                    continue;
                    
            genre = row[3]
            tmbdid = row[5]
            overview = row[8]
            appeared = Set()
            
            if tmbdid not in validTmbdid:
                continue
                
            if tmbdid in appeared:
                    continue
                    
            appeared.add(tmbdid)
            movieid = tmbdidToMovieid[tmbdid]
            movieOverview[movieid] = overview
            movieGenre[movieid] = genre
        
            count = count+1


# In[123]:

readMoviemetadata('./the-movies-dataset/movies_metadata.csv')


# In[124]:

len(movieGenre)


# In[125]:

def readKeyword(filename):
 
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        count = -1;
        for row in csvReader:
            if count == -1:
                    count = count+1
                    continue;
            
            tmbdid = row[0]
            keyword = row[1]
            appeared = Set()
            
            if tmbdid not in validTmbdid:
                continue
            
            if tmbdid in appeared:
                continue
                    
            appeared.add(tmbdid)
            movieid = tmbdidToMovieid[tmbdid]
            movieKeyword[movieid] = keyword
        
            count = count+1


# In[126]:

readKeyword('./the-movies-dataset/keywords.csv')


# In[127]:

def readRating(filename):
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        count = -1
        cnt = 0
        curruser = -1
        over50 = 0;
        over100 = 0;
        over200 = 0;
        over300 = 0;
        temp = np.zeros(movieNum)
        userCount = 0;
        for row in csvReader:
            if count == -1:
                count = count+1
                continue;
            
            
            userid = int(row[0])
            movieid = row[1]
            rating = row[2]
            
            if movieid not in movieidToTmbdid:
                continue
                
            if userid != curruser:
                curruser = userid
                if cnt >= 50: over50 += 1
                if cnt >= 100: over100 += 1
                if cnt >= 200: over200 += 1
                if cnt >= 300: over300 += 1
                if cnt >= 50:
                    matrix[userCount] = temp
                    temp = np.zeros(movieNum)
                    userCount += 1
                    
                cnt = 0
                
            else:
                cnt +=1
            
            temp[movieidToCol[movieid]] = rating      
            count = count+1
    
    print count
    print over50
    print over100
    print over200
    print over300


# In[128]:

readRating('./the-movies-dataset/ratings.csv')


# In[134]:

matrix.shape


# In[140]:

selectedMovieOverview = dict()
selectedMovieGenre = []
selectedMovieKeyword = []


# In[141]:

import re
import json

ratings = np.zeros((2095, userNum))
movieCount = 0
over50 = 0;
over100 = 0;
over200 = 0;
over300 = 0;
rateSum = 0;

for i in range(movieNum):
    cnt = 0
    index = 0
    temp = np.zeros(userNum)
    for j in range(userNum):
        temp[index] = matrix[j][i]
        index += 1
        if(matrix[j][i] != 0):
            cnt += 1
            
    if cnt >= 50: over50 += 1
    if cnt >= 100: over100 += 1
    if cnt >= 200: over200 += 1
    if cnt >= 300: over300 += 1
    if cnt >= 50:
        rateSum += cnt
        ratings[movieCount] = temp
        movieid = colToMovieid[i]
        
        overview = movieOverview[movieid]
        genre = movieGenre[movieid]
        keyword = movieKeyword[movieid]
        
        selectedMovieOverview[movieCount]= re.sub("[^\w]", " ",  overview).split()
        selectedMovieGenre.append(genre)
        selectedMovieKeyword.append(keyword)
     
        movieCount += 1
        
print over50
print over100
print over200
print over300
print rateSum


# In[148]:

np.savetxt("ratings.gz", ratings)


# In[147]:

selectedMovieOverview[1]


# In[149]:

with open('movieKeyword', 'wb') as fp:
    pickle.dump(selectedMovieKeyword, fp)
    
with open('movieGenre', 'wb') as fp:
    pickle.dump(selectedMovieGenre, fp)


# In[152]:

with open('overview.pickle', 'w') as file:
     file.write(pickle.dumps(selectedMovieOverview)) 


# In[150]:

len(selectedMovieKeyword)


# In[48]:

26024289*1.0/(270896*45843)


# In[ ]:



