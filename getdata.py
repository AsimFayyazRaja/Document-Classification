import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing, cross_validation
from tqdm import tqdm
import csv
from matplotlib import style
import string
from collections import Counter
import sys
import pickle
from nltk.stem import PorterStemmer 
ps = PorterStemmer()
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

print("Reading all files and saving data..")

#opening file
f = open('Words/sarcasm-words.txt', 'r', encoding="ISO-8859-1")

translator = str.maketrans('', '', string.punctuation)
x=f.readline()

i=0
angry=[]
while(x):       #iterating over file to get phrases and words and making their list
    '''
    if i<=35:           #for those files which have garbage in end
        i+=1
        x=f.readline()
        continue
    '''
    x=str(x)
    x=x.split(',')
    for word in x:
        splits=word.split()
        try:
            if "-" not in splits[0]:        #not removing single hyphen
                newword=str(splits[0]).translate(translator)
            else:
                newword=str(splits[0])
        except:
            splits=str(splits).translate(translator)        #removing punctuation using regex
            angry.append(splits)
            continue
        for k in range(1,len(splits)):
            newword+="-"
            sa=str(splits[k]).translate(translator)
            newword+=sa
        print(newword)

        #print(splits)
        if len(splits)!=0:
            angry.append(splits)
        '''
        for split in splits:
            if split not in stop:
                angry.append(ps.stem(split))
        '''

    x=f.readline()

#print(angry)


#dumping data on disk
with open('sarcasm_words', 'wb') as fp:        #saving list for future use
    pickle.dump(angry, fp)