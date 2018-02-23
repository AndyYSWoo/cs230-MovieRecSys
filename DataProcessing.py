
# coding: utf-8

# In[54]:

import csv
import collections
import numpy as np
import pickle


# In[55]:

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


# In[58]:

movieNum = 45843 # total number of movies in the dataset
userNum = 20133 # num of users who make more than 300 ratings

matrix = np.zeros((userNum, movieNum)) 

colToMovieid = np.zeros(movieNum)
movieidToCol = collections.defaultdict(int)

movieOverview = collections.defaultdict(str)
movieGenre = collections.defaultdict(str)
movieKeyword = collections.defaultdict(str)

tmbdidToMovieid = collections.defaultdict(int)
movieidToTmbdid = collections.defaultdict(int)


# In[59]:

def readLink(filename):
    
        with open(filename) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            count = -1
            for row in csvReader:
                if count == -1:
                    count = count+1
                    continue;
            
                movieid = row[0]
                tmbdid = row[1]
            
                colToMovieid[count] = movieid;
                movieidToCol[movieid] = count
                
                tmbdidToMovieid[tmbdid] = movieid
                movieidToTmbdid[movieid] = tmbdid
                
                count = count+1


# In[60]:

def readMoviemetadata(filename):
 
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        count = -1;
        for row in csvReader:
            genre = row[3]
            tmbdid = row[5]
            overview = row[9]
            movieid = tmbdidToMovieid[tmbdid]
            movieOverview[movieid] = overview
            movieGenre[movieid] = genre
        
            count = count+1


# In[66]:

def readKeyword(filename):
 
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        count = -1;
        for row in csvReader:
            
            tmbdid = row[0]
            keyword = row[1]
            movieid = tmbdidToMovieid[tmbdid]
            movieKeyword[movieid] = keyword
        
            count = count+1


# In[62]:

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
            if userid != curruser:
                curruser = userid
                if cnt >= 50: over50 += 1
                if cnt >= 100: over100 += 1
                if cnt >= 200: over200 += 1
                if cnt >= 300: over300 += 1
                if cnt >= 300:
                    matrix[userCount] = temp
                    temp = np.zeros(movieNum)
                    userCount += 1
                    
                cnt = 0
                
            else:
                cnt +=1
            
            movieid = row[1]
            rating = row[2]
            
            temp[movieidToCol[movieid]] = rating
            #matrix[userid-1][movieidToCol[movieid]] = rating
        
            count = count+1
    
    print count
    print over50
    print over100
    print over200
    print over300


# In[63]:

readLink('./the-movies-dataset/links.csv')


# In[64]:

readMoviemetadata('./the-movies-dataset/movies_metadata.csv')


# In[67]:

readKeyword('./the-movies-dataset/keywords.csv')


# In[68]:

readRating('./the-movies-dataset/ratings.csv')


# In[69]:

matrix.shape


# In[76]:

selectedMovieOverview = collections.defaultdict(list)
selectedMovieGenre = []
selectedMovieKeyword = []


# In[74]:

import json
 
json_data = "{'id': 80, 'name': 'Crime'}, {'id': 18, 'name': 'Drama'}, {'id': 10769, 'name': 'Foreign'}"
python_obj = json.loads(json_data)
print python_obj["name"]


# In[78]:

import re
import json

ratings = np.zeros((6672, userNum))
movieCount = 0
over50 = 0;
over100 = 0;
over200 = 0;
over300 = 0;

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
    if cnt >= 300: 
        over300 += 1
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


# In[84]:

np.savetxt("ratings.gz", ratings)


# In[87]:

with open('movieKeyword', 'wb') as fp:
    pickle.dump(selectedMovieKeyword, fp)
    
with open('movieGenre', 'wb') as fp:
    pickle.dump(selectedMovieGenre, fp)


# In[88]:

f = open("overview.pkl","wb")
pickle.dump(selectedMovieOverview,f)
f.close()


# In[48]:

26024289*1.0/(270896*45843)


# In[ ]:



