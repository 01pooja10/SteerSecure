import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array
from google.colab import drive
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
#import keras.backend as K
import shutil

#drive.mount('/content/drive')
#!unzip '/content/drive/My Drive/distracted_drivers.zip'

#check image
img=cv2.imread('/content/train/c9/img_99692.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

#img.shape
os.listdir('/content/train')

tags={'c0':'Safe driving','c1': 'Texting using phone(right)','c2': 'Talking on the phone(right)','c3': 'Texting using phone(left)','c4': 'Talking on the phone(left)',
      'c5': 'Operating the radio','c6': 'Drinking a beverage','c7': 'Reaching behind','c8': 'Adjusting hair or makeup','c9': 'Talking to a passenger'}
list1=[]
for k in tags.keys():
    list1.append(tags[k])
print(list1)

for i in range(10):
    os.rename('/content/train/c'+str(i),list1[i])

os.listdir('/content')

for i in range(10):
    shutil.move('/content/'+list1[i],'/content/train')

os.listdir('/content/train')

#print(list1)
#os.listdir()

#img=cv2.imread('/content/train/Talking to a passenger/img_99692.jpg')
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.show()

datagen=ImageDataGenerator(rescale=1./255.0,validation_split=0.2)

data=datagen.flow_from_directory('/content/train',target_size=(224,224),class_mode='categorical',shuffle=False,batch_size=64,color_mode='rgb')

train=datagen.flow_from_directory('/content/train',target_size=(224,224),class_mode='categorical',shuffle=False,batch_size=64,color_mode='rgb',subset='training')
val=datagen.flow_from_directory('/content/train',target_size=(224,224),class_mode='categorical',shuffle=False,batch_size=64,color_mode='rgb',subset='validation')

train.class_indices

l=[]
for x in train.class_indices:
    l.append(x)
print(l)

#plt.imshow(train[0][0][6])
#print(train[0][1][0])

#K.clear_session()

model1 = Sequential()


model1.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model1.add(BatchNormalization())
model1.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(BatchNormalization(axis=3))
model1.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model1.add(Dropout(0.3))


model1.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(BatchNormalization())
model1.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(BatchNormalization(axis=3))
model1.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model1.add(Dropout(0.3))


model1.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(BatchNormalization())
model1.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model1.add(BatchNormalization(axis=3))
model1.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model1.add(Dropout(0.5))


model1.add(Flatten())
model1.add(Dense(512,activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.5))
model1.add(Dense(128,activation='relu'))
model1.add(Dropout(0.25))
model1.add(Dense(10,activation='softmax'))

model1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model1.fit(train,validation_data=val,epochs=10,batch_size=64)

#save model
model1.save('model_driver.h5')

#model1.predict()
