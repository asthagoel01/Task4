#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from keras.applications import MobileNet


# In[2]:


img_rows=224
img_cols=224
model= MobileNet(weights='imagenet',include_top=False,input_shape=(224,224,3))#it doesn't mean that we are doing transfer learning
model


# In[3]:


for layer in model.layers:
    layer.trainable=False#Tranfer Learning step1 by freezing layers
model.layers


# In[4]:


def addTopModel(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output#pre trained model output act as input;making a new model
    top_model = GlobalAveragePooling2D()(top_model)#Defined new model
   # top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(256,activation='relu')(top_model)
    #top_model = Dense(256,activation='relu')(top_model)
    top_model = Dense(128,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model
model.input


# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
num_classes = 2 #Output classes are 2
FC_Head = addTopModel(model, num_classes)#New model created

new_model = Model(inputs=model.input, outputs=FC_Head)#total(model)prepared by combine old+new

print(new_model.summary())


# In[6]:


from keras.preprocessing.image import ImageDataGenerator

train_data_dir ='train'#training data
validation_data_dir ='test'#testing data

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size =17
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


# In[7]:


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("trained_model.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

callbacks = [earlystop, checkpoint]
new_model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.0001),
              metrics = ['accuracy'])

nb_train_samples =282
nb_validation_samples =4
epochs =10
batch_size =50

history = new_model.fit_generator(
   generator= train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


# In[8]:


import cv2
import numpy as np
import time

Display = True

from keras.models import load_model#prediction-phase starts
classifier = load_model('trained_model.h5')


# In[ ]:


from os import listdir
from os.path import isfile, join,isdir
import os 

person_dict = {"[0]": "astha", 
               "[1]": "mummy" }
person_dict_n = {"astha": "astha", 
                 "mummy":"mummy" }

def draw_test(name, pred, im):
    BLACK = [0,0,0]
    print(im.shape[0])
    expanded_image = cv2.copyMakeBorder(im, 0, 0, 0, input_im.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, str(pred), (52, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage():
    path = './test/'
    folders = list(filter(lambda x: isdir(join(path, x)), listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + person_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + str(path_class))
    file_path = path + path_class

    file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    print(file_path+"/"+image_name)
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage()
    input_original = input_im.copy()
    input_im = cv2.resize( input_original, (224, 224), interpolation = cv2.INTER_LINEAR)
    cv2.imshow("Test Image", input_im)
    input_im = input_im / 255.#rescaling
    input_im = input_im.reshape(1,224,224,3) 
    
    ## Get Prediction
    print(classifier.predict(input_im, 1, verbose = 0))
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)

    draw_test("Prediction", res, input_original) 
    cv2.waitKey()

cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




