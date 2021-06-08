#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Traffic Sign Classification
#There are several different types of traffic signs like speed limits, no entry, traffic signals, turn left or right, children crossing, no passing of heavy vehicles, etc. Traffic signs classification is the process of identifying which class a traffic sign belongs to.
#In this task, you have to build a deep neural network model that can classify traffic signs present in the image into different categories.


# In[31]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
os.chdir('C:\\Users\\aksha\\Downloads\\archive')
from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


# In[32]:



data = []
labels = []
# We have 43 Classes
classes = 43
cur_path = os.getcwd()


# In[33]:


cur_path


# In[34]:


for i in range(classes): #the 43 classes in train folder of archive
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a) #first loop will go to train folder 0TH folder 
            image = image.resize((30,30)) #it will choose the first image and resize into 30X30
            image = np.array(image) # then it will change the image into array
            data.append(image)# it will then append that array into a list 'data' which we have defined upar
            labels.append(i) #this process will continue for all images and all folders 43
        except Exception as e:
            print(e)


# In[35]:


data = np.array(data) # converting into array

labels = np.array(labels)


# In[ ]:





# In[36]:


#os.mkdir('training')

np.save('./training/data',data) #created a training folder in the archive folder in archive
np.save('./training/target',labels) #for future use


# In[37]:


data=np.load('./training/data.npy')
labels=np.load('./training/target.npy')


# In[38]:


print(data.shape, labels.shape) #will print te shape of data and label


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
# tarining the data


# In[40]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) #printing the shape


# In[41]:


# CONVERTING LABELS TO ONEHOT ENCODING
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, 43) # tarin the data and to_categoricslly method helps
y_test = to_categorical(y_test, 43)


# In[20]:


#BUILD THE MODEL
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))

# We have 43 classes that's why we have defined 43 in the dense
model.add(Dense(43, activation='softmax'))
#outcome should be in 43 classes


# In[43]:


#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[44]:


epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))


# In[45]:


# accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[46]:



# Loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[47]:


# TESTING ON TEST DATA
def testing(testcsv):
    y_test = pd.read_csv(testcsv)
    label = y_test["ClassId"].values
    imgs = y_test["Path"].values
    data=[]
    for img in imgs:
        image = Image.open(img)
        image = image.resize((30,30))
        data.append(np.array(image))
    X_test=np.array(data)
    return X_test,label


# In[48]:


X_test, label = testing('Test.csv')


# In[49]:


Y_pred = model.predict_classes(X_test)# it will store in y_pred
Y_pred


# In[50]:


# ACCURACY WITH TEST DATA
from sklearn.metrics import accuracy_score
print(accuracy_score(label, Y_pred))


# In[51]:


# SAVING THE MODEL
model.save("./training/TSR.h5")


# In[2]:


# LOAD THE MODEL
import os
os.chdir(r'C:\\Users\\aksha\\Downloads\\archive')
from keras.models import load_model
model = load_model('./training/TSR.h5') #LOAD THE ORIGINAL TRAINING DATA
# can access that model anytime


# In[3]:


# testing on random image
# Classes of trafic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


# In[4]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def test_on_img(img): #create  a function
    data=[] #making an empty list
    image = Image.open(img)  #it will open iamhe 
    image = image.resize((30,30)) # resize
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = model.predict_classes(X_test) # prediction on image
    return image,Y_pred # and will return prediction and image


# In[18]:


plot,prediction = test_on_img(r'C:\Users\aksha\Downloads\sign10.jpg')

s = [str(i) for i in prediction] 
a = int("".join(s)) 
print("Predicted traffic sign is: ",classes[a])
plt.imshow(plot)
plt.show()
print(a)


# In[ ]:




