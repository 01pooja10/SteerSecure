import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import json
from google.colab import files
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense, Flatten
import pickle
from nltk import punkt
import random

#print(pickle.format_version)
nltk.download('punkt')
nltk.download('wordnet')

#uploaded=files.upload()

with open('chatbot_intents.json') as file:
    data=json.load(file,strict=False)

print(data['intents'])

lemm=WordNetLemmatizer()

words=[]
labels=[]
x=[]
y=[]

for intent in data['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        x.append((w,intent['tag']))

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [lemm.lemmatize(i.lower()) for i in words if i != '?']
words=sorted(list(set(words)))

labels=sorted(list(set(labels)))

print(len(words))

print(len(labels))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(labels,open('labels.pkl','wb'))

train=[]
output=[0]*len(labels)

print(len(x))

for doc in x:
    bag=[]
    pattern_w=doc[0]
    pattern_w=[lemm.lemmatize(w.lower()) for w in pattern_w]
    for w in words:
        if w in pattern_w:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output)
    output_row[labels.index(doc[1])]=1
    train.append((bag,output_row))

random.shuffle(train)
train=np.array(train)
train_x=list(train[:,0])
train_y=list(train[:,1])

train=np.array(train)
output=np.array(output)

model=Sequential()
model.add(Dense(64,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(len(train_y[0]),activation='softmax'))

model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

train_x=np.asarray(train_x)
train_y=np.asarray(train_y)

#print(len(train_x))

model.fit(train_x,train_y,epochs=100,verbose=1,batch_size=5)

model.save('chatbot_model.h5')
