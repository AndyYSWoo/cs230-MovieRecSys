
# coding: utf-8

# In[26]:

import pickle
import numpy as np


# In[27]:

overviews = pickle.load(open( "overview.pickle", "rb" ))


# In[28]:

tokens = Set()


# In[29]:

for i in range(len(overviews)):
    for string in overviews[i]:
        tokens.add(string.lower())


# In[30]:

DEFAULT_FILE_PATH = "./glove.6B.50d.txt"


# In[31]:

def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors"""
    wordVectors = dict()
    #wordVectors = np.zeros((len(tokens), dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[token] = np.asarray(data)
    return wordVectors


# In[32]:

wordVectors = loadWordVectors(tokens)


# In[33]:

overviewVectors = []
for i in range(len(overviews)):
    vector = np.zeros(50) # dimension = 50
    cnt = 0;
    for string in overviews[i]:
        if string.lower() in wordVectors:
            vector += wordVectors[string.lower()]
            cnt += 1
    if cnt != 0:
        vector = vector*1.0 / cnt
    overviewVectors.append(vector)


# In[35]:

with open('overviewVectors', 'wb') as fp:
    pickle.dump(overviewVectors, fp)


# In[ ]:



