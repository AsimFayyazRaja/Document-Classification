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


#getting all dumped files i.e. given words and phrases

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
'''
for word in text:
    if any(word in x for x in angry):
        angry_count+=1
    elif any(word in x for x in funny):
        funny_count+=1
    elif any(word in x for x in happy):
        happy_count+=1
    elif any(word in x for x in hate):
        hate_count+=1
    elif any(word in x for x in negative):
        negative_count+=1
    elif any(word in x for x in positive):
        positive_count+=1
    elif any(word in x for x in sad):
        sad_count+=1
    elif any(word in x for x in sarcasm):
        sarcasm_count+=1
    elif any(word in x for x in offensive):
        offensive_count+=1
'''
import copy
i=0
print(text)


#method is iterated 9 times fo

#count is updated for respected category if phrases match

for word in text:
    flag=False
    for phrase in sad:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                #comparing real word and stemmed word, if any is same increment count
                if w[i+j]==sp or ps.stem(w[i+j])==ps.stem(sp):
                    count+=1
                else:
                    break
            except:
                if w[i] in phrase:
                    count+=1
                    break
            j+=1
        if count>=len(phrase) and len(phrase)>=1:       #comparing for words or phrases
            sad_count+=1
    i+=1
i=0
for word in text:
    flag=False
    for phrase in angry:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                if w[i+j]==sp or ps.stem(w[i+j])==ps.stem(sp):
                    count+=1
                else:
                    break
            except:
                if w[i] in phrase:
                    count+=1
                    break
            j+=1
        if count>=len(phrase) and len(phrase)>=1:
            angry_count+=1
    i+=1
i=0
for word in text:
    flag=False
    for phrase in funny:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                if w[i+j]==sp or ps.stem(w[i+j])==ps.stem(sp):
                    count+=1
                else:
                    break
            except:
                if w[i] in phrase:
                    count+=1
                    break
            j+=1
        if count>=len(phrase) and len(phrase)>=1:
            funny_count+=1
    i+=1
i=0
for word in text:
    flag=False
    for phrase in happy:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                if w[i+j]==sp or ps.stem(w[i+j])==ps.stem(sp):
                    count+=1
                else:
                    break
            except:
                if w[i] in phrase:
                    count+=1
                    break
            j+=1
        if count>=len(phrase) and len(phrase)>=1:
            happy_count+=1
    i+=1
i=0
for word in text:
    flag=False
    for phrase in hate:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                if w[i+j]==sp or ps.stem(w[i+j])==ps.stem(sp):
                    count+=1
                else:
                    break
            except:
                if w[i] in phrase:
                    count+=1
                    break
            j+=1
        if count>=len(phrase) and len(phrase)>=1:
            hate_count+=1
    i+=1
i=0
for word in text:
    flag=False
    for phrase in negative:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                if w[i+j]==sp or ps.stem(w[i+j])==ps.stem(sp):
                    count+=1
                else:
                    break
            except:
                if w[i] in phrase:
                    count+=1
                    break
            j+=1
        if count>=len(phrase) and len(phrase)>=1:
            negative_count+=1
    i+=1
i=0
for word in text:
    flag=False
    for phrase in offensive:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                if w[i+j]==sp or ps.stem(w[i+j])==ps.stem(sp):
                    count+=1
                else:
                    break
            except:
                if w[i] in phrase:
                    count+=1
                    break
            j+=1
        if count>=len(phrase) and len(phrase)>=1:
            offensive_count+=1
    i+=1
r=0
i=0
for word in text:
    flag=False
    for phrase in positive:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                if w[i+j]==sp or (ps.stem(w[i+j])==ps.stem(sp)):
                    count+=1
                else:
                    break
            except Exception as e:
                if w[i] in phrase:
                    count+=1
                    break
                r+=1
                break

            j+=1
        if count>=len(phrase) and len(phrase)>=1:
            positive_count+=1
    i+=1
i=0
for word in text:
    flag=False
    for phrase in sad:
        count=0
        w=copy.deepcopy(text)
        j=0
        for sp in phrase:
            try:
                if w[i+j]==sp or ps.stem(w[i+j])==ps.stem(sp):
                    print(w[i+j])
                    count+=1
                else:
                    break
            except:
                if w[i] in phrase:
                    count+=1
                    break
            j+=1
        if count>=len(phrase) and len(phrase)>=1:
            sarcasm_count+=1
    i+=1


print("Angry count: ", angry_count)
print("Funny count: ", funny_count)
print("Happy count: ", happy_count)
print("Hate count: ", hate_count)
print("Negative count: ", negative_count)
print("Positive count: ", positive_count)
print("Offensive count: ", offensive_count)
print("Sad count: ", sad_count)
print("Sarcasm count: ", sarcasm_count)


angry_prob=0
funny_prob=0
happy_prob=0
hate_prob=0
hate_prob=0
negative_prob=0
positive_prob=0
offensive_prob=0
sad_prob=0
sarcasm_prob=0

total=len(text)

#making probability by dividing with total
angry_prob=float(angry_count/total)
funny_prob=float(funny_count/total)
happy_prob=float(happy_count/total)
hate_prob=float(hate_count/total)
negative_prob=float(negative_count/total)
offensive_prob=float(offensive_count/total)
positive_prob=float(positive_count/total)
sad_prob=float(sad_count/total)
sarcasm_prob=float(sarcasm_count/total)

print("Angry probability: ", angry_prob)
print("Funny probability: ", funny_prob)
print("Happy probability: ", happy_prob)
print("Hate probability: ", hate_prob)
print("Negative probability: ", negative_prob)
print("Positive probability: ", positive_prob)
print("Offensive probability: ", offensive_prob)
print("Sad probability: ", sad_prob)
print("Sarcasm probability: ", sarcasm_prob)


scores={}

#making disctionary of each result to sort
scores['angry']=float(angry_count/total)
scores['funny']=float(funny_count/total)
scores['happy']=float(happy_count/total)
scores['hate']=float(hate_count/total)
scores['negative']=float(negative_count/total)
scores['positive']=float(positive_count/total)
scores['sad']=float(sad_count/total)
scores['sarcasm']=float(sarcasm_count/total)
scores['offensive']=float(offensive_count/total)



import matplotlib.pyplot as plt
 
labels = ['Angry', 'Funny', 'Happy', 'Hate', 'Negative', 'Positive', 'Offensive', 'Sad', 'Sarcasm']
sizes = [angry_prob, funny_prob, happy_prob, hate_prob, negative_prob, positive_prob, offensive_prob, sad_prob, sarcasm_prob]
#colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes,shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()



sorted_by_value = sorted(scores.items(), key=lambda kv: kv[1])      #sorting
class1=sorted_by_value[-1]      #chosing maximum value of probability and class
print("Class of given text according to HANDENGINEERING NLP METHOD is: ", sorted_by_value[-1])

titl=list(scores.keys())[-1]
plt.title(class1)
plt.show()