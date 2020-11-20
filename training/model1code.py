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
from keras.models import Model
from keras.layers import Input
from keras.applications.vgg16 import VGG16,preprocess_input

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

model_vgg=VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))

model_vgg.summary()

model=Sequential()
model.add(model_vgg)
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))

model.summary()

for i in model_vgg.layers:
    i.trainable=False

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

hist=model.fit(train,val,batch_size=64,epochs=5)

model.save('vgg_model.h5')

def results_vgg(link):
    img1=cv2.imread('{}'.format(link))
    img2=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    re=cv2.resize(img2,(224,224))/255.0
    plt.imshow(re)
    re=img_to_array(re)
    mm=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
    mm[0]=re
    pred=model.predict(mm)
    x=np.argmax(pred)
    print(l[x])

#from google.colab import files
#up=files.upload()

#def get_link(up):
    #for x in up.keys():
        #link=x
        #return link

#link=get_link(up)
get_pred=results_vgg(link)
