import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation
import csv

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from collections import Counter
from tqdm import tqdm
import collections, re
from sklearn.naive_bayes import MultinomialNB
import random
from random import randint
from sklearn.metrics import average_precision_score
import gensim
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

import pickle

#getting data
with open('angry_words', 'rb') as fp:
    angry=pickle.load(fp)

with open('funny_words', 'rb') as fp:
    funny=pickle.load(fp)

with open('hate_words', 'rb') as fp:
    hate=pickle.load(fp)

with open('positive_words', 'rb') as fp:
    positive=pickle.load(fp)

with open('negative_words', 'rb') as fp:
    negative=pickle.load(fp)

with open('offensive_words', 'rb') as fp:
    offensive=pickle.load(fp)

with open('sarcasm_words', 'rb') as fp:
    sarcasm=pickle.load(fp)

with open('sad_words', 'rb') as fp:
    sad=pickle.load(fp)

with open('happy_words', 'rb') as fp:
    happy=pickle.load(fp)

f = open('test_data.txt', 'r')

print("--------------WORD2VEC RESULTS-------------")

text=[]
x=f.readline()
translator = str.maketrans('', '', string.punctuation)
while(x):       #iterating over file to get phrases and words and making their list
    '''
    if i<=35:
        i+=1
        x=f.readline()
        continue
    '''
    x=str(x)
    x=x.split()
    for word in x:
        splits=word
        #print(splits)
        if len(splits)!=0:
            text.append(str(splits).translate(translator))
            #text.append(splits)
        '''
        for split in splits:
            if split not in stop:
                text.append(ps.stem(split))
        '''

    x=f.readline()

#initial counts are 0
angry_count=0
funny_count=0
happy_count=0
hate_count=0
negative_count=0
positive_count=0
offensive_count=0
sad_count=0
sarcasm_count=0

#loading w2v model
model = gensim.models.KeyedVectors.load_word2vec_format('w2v.vectors', binary=False)


#comparing UNIGRAMS with Word2vec encodings
for word in text:
    for phrase in angry:
        for p in phrase:
            try:
                angry_count+=model.similarity(word,p)       #comparing similarity
            except:
                pass
    for phrase in funny:
        for p in phrase:
            try:
                funny_count+=model.similarity(word,p)
            except:
                pass
    for phrase in happy:
        for p in phrase:
            try:
                happy_count+=model.similarity(word,p)
            except:
                pass
    for phrase in hate:
        for p in phrase:
            try:
                hate_count+=model.similarity(word,p)
            except:
                pass
    for phrase in negative:
        for p in phrase:
            try:
                negative_count+=model.similarity(word,p)
            except:
                pass
    for phrase in offensive:
        for p in phrase:
            try:
                offensive_count+=model.similarity(word,p)
            except:
                pass
    for phrase in positive:
        for p in phrase:
            try:
                positive_count+=model.similarity(word,p)
            except:
                pass
    for phrase in sad:
        for p in phrase:
            try:
                sad_count+=model.similarity(word,p)
            except:
                pass
    for phrase in sarcasm:
        for p in phrase:
            try:
                sarcasm_count+=model.similarity(word,p)
            except:
                pass

total=angry_count+funny_count+happy_count+hate_count+negative_count+positive_count+offensive_count+sad_count+sarcasm_count

#getting total

print("----------------UNIGRAM RESULTS---------------")

print("Angry count by one word: ", float(angry_count/total))
print("Funny count by one word: ", float(funny_count/total))
print("Happy count by one word: ", float(happy_count/total))
print("Hate count by one word: ", float(hate_count/total))
print("Negative count by one word: ", float(negative_count/total))
print("Positive count by one word: ", float(positive_count/total))
print("Offensive count by one word: ", float(offensive_count/total))
print("Sad count by one word: ", float(sad_count/total))
print("Sarcasm count by one word: ", float(sarcasm_count/total))




scores={}
#making dictionary of whole results to sort
scores['angry']=float(angry_count/total)
scores['funny']=float(funny_count/total)
scores['happy']=float(happy_count/total)
scores['hate']=float(hate_count/total)
scores['negative']=float(negative_count/total)
scores['positive']=float(positive_count/total)
scores['sad']=float(sad_count/total)
scores['sarcasm']=float(sarcasm_count/total)
scores['offensive']=float(offensive_count/total)



#claculating prob for each class
angry_prob=float(angry_count/total)
funny_prob=float(funny_count/total)
happy_prob=float(happy_count/total)
hate_prob=float(hate_count/total)
negative_prob=float(negative_count/total)
positive_prob=float(positive_count/total)
sad_prob=float(sad_count/total)
sarcasm_prob=float(sarcasm_count/total)
offensive_prob=float(offensive_count/total)


#sorting the list
sorted_by_value = sorted(scores.items(), key=lambda kv: kv[1])
class1=sorted_by_value[-1]      #taking maximum prob 
#print("Class of given text according to UNIGRAMS is: ", sorted_by_value[-1])


import matplotlib.pyplot as plt
 
labels = ['Angry', 'Funny', 'Happy', 'Hate', 'Negative', 'Positive', 'Offensive', 'Sad', 'Sarcasm']
sizes = [angry_prob, funny_prob, happy_prob, hate_prob, negative_prob, positive_prob, offensive_prob, sad_prob, sarcasm_prob]
#colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes,shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
#plt.tight_layout()

plt.title("UNIGRAM_RESULTS: "+str(class1))
plt.show()


angry_count=0
funny_count=0
happy_count=0
hate_count=0
negative_count=0
positive_count=0
offensive_count=0
sad_count=0
sarcasm_count=0

#comparing BIGRAMS with Word2vec encodings
i=0
for word in text:
    if i<=len(text)-1:
        for phrase in angry:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        angry_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
                    j+=1
        for phrase in funny:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        funny_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
                    j+=1
        for phrase in happy:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        happy_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
                    j+=1
        for phrase in hate:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        hate_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
                    j+=1
        for phrase in negative:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        negative_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
                    j+=1
        for phrase in offensive:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        offensive_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
                    j+=1
        for phrase in positive:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        positive_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
        for phrase in sad:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        sad_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
                    j+=1
        for phrase in sarcasm:
            j=0
            if j<=len(phrase)-1:
                for p in phrase:
                    try:
                        sarcasm_count+=model.similarity(word[i]+word[i+1],p[j]+p[j+1])
                    except:
                        pass
                    j+=1
    i+=1

total=angry_count+funny_count+happy_count+hate_count+negative_count+positive_count+offensive_count+sad_count+sarcasm_count

print("----------------BIGRAM RESULTS---------------")

print("Angry count by one word: ", float(angry_count/total))
print("Funny count by one word: ", float(funny_count/total))
print("Happy count by one word: ", float(happy_count/total))
print("Hate count by one word: ", float(hate_count/total))
print("Negative count by one word: ", float(negative_count/total))
print("Positive count by one word: ", float(positive_count/total))
print("Offensive count by one word: ", float(offensive_count/total))
print("Sad count by one word: ", float(sad_count/total))
print("Sarcasm count by one word: ", float(sarcasm_count/total))




scores={}

scores['angry']=float(angry_count/total)
scores['funny']=float(funny_count/total)
scores['happy']=float(happy_count/total)
scores['hate']=float(hate_count/total)
scores['negative']=float(negative_count/total)
scores['positive']=float(positive_count/total)
scores['sad']=float(sad_count/total)
scores['sarcasm']=float(sarcasm_count/total)
scores['offensive']=float(offensive_count/total)


angry_prob=float(angry_count/total)
funny_prob=float(funny_count/total)
happy_prob=float(happy_count/total)
hate_prob=float(hate_count/total)
negative_prob=float(negative_count/total)
positive_prob=float(positive_count/total)
sad_prob=float(sad_count/total)
sarcasm_prob=float(sarcasm_count/total)
offensive_prob=float(offensive_count/total)

#sorting the list
sorted_by_value = sorted(scores.items(), key=lambda kv: kv[1])
class1=sorted_by_value[-1]      #taking maximum prob 
#print("Class of given text according to BIGRAMS is: ", sorted_by_value[-1])

import matplotlib.pyplot as plt
 
labels = ['Angry', 'Funny', 'Happy', 'Hate', 'Negative', 'Positive', 'Offensive', 'Sad', 'Sarcasm']
sizes = [angry_prob, funny_prob, happy_prob, hate_prob, negative_prob, positive_prob, offensive_prob, sad_prob, sarcasm_prob]
#colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes,shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
#plt.tight_layout()

plt.title("BIGRAM_RESULTS: "+ str(class1))
plt.show()



angry_count=0
funny_count=0
happy_count=0
hate_count=0
negative_count=0
positive_count=0
offensive_count=0
sad_count=0
sarcasm_count=0


##comparing TRIGRAMS with Word2vec encodings
i=0
for word in text:
    if i<=len(text)-2:
        for phrase in angry:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        angry_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1

        for phrase in funny:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        funny_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1

        for phrase in happy:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        happy_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1

        for phrase in hate:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        hate_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1
        for phrase in negative:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        negative_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1

        for phrase in offensive:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        offensive_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1

        for phrase in positive:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        positive_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1

        for phrase in sad:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        sad_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1

        for phrase in sarcasm:
            j=0
            if j<=len(phrase)-2:
                for p in phrase:
                    try:
                        sarcasm_count+=model.similarity(word[i]+word[i+1]+word[i+2],p[j]+p[j+1]+p[i+2])
                    except:
                        pass
                    j+=1
    i+=1


total=angry_count+funny_count+happy_count+hate_count+negative_count+positive_count+offensive_count+sad_count+sarcasm_count

print("----------------TRIGRAM RESULTS---------------")

print("Angry count by one word: ", float(angry_count/total))
print("Funny count by one word: ", float(funny_count/total))
print("Happy count by one word: ", float(happy_count/total))
print("Hate count by one word: ", float(hate_count/total))
print("Negative count by one word: ", float(negative_count/total))
print("Positive count by one word: ", float(positive_count/total))
print("Offensive count by one word: ", float(offensive_count/total))
print("Sad count by one word: ", float(sad_count/total))
print("Sarcasm count by one word: ", float(sarcasm_count/total))

scores={}

scores['angry']=float(angry_count/total)
scores['funny']=float(funny_count/total)
scores['happy']=float(happy_count/total)
scores['hate']=float(hate_count/total)
scores['negative']=float(negative_count/total)
scores['positive']=float(positive_count/total)
scores['sad']=float(sad_count/total)
scores['sarcasm']=float(sarcasm_count/total)
scores['offensive']=float(offensive_count/total)


#making probabilities by dividing score by total

angry_prob=float(angry_count/total)
funny_prob=float(funny_count/total)
happy_prob=float(happy_count/total)
hate_prob=float(hate_count/total)
negative_prob=float(negative_count/total)
positive_prob=float(positive_count/total)
sad_prob=float(sad_count/total)
sarcasm_prob=float(sarcasm_count/total)
offensive_prob=float(offensive_count/total)

#sorting the list
sorted_by_value = sorted(scores.items(), key=lambda kv: kv[1])
class1=sorted_by_value[-1]      #taking maximum prob 

#print("Class of given text according to TRIGRAMS is: ", sorted_by_value[-1])


import matplotlib.pyplot as plt
 
labels = ['Angry', 'Funny', 'Happy', 'Hate', 'Negative', 'Positive', 'Offensive', 'Sad', 'Sarcasm']
sizes = [angry_prob, funny_prob, happy_prob, hate_prob, negative_prob, positive_prob, offensive_prob, sad_prob, sarcasm_prob]
#colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes,shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
#plt.tight_layout()

plt.title("TRIGRAM_RESULTS: "+ str(class1))
plt.show()