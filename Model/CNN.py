import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input,Conv2D,Flatten,Dense,Dropout,MaxPool2D
from keras.preprocessing.image import load_img,img_to_array
from keras.callbacks import ModelCheckpoint,EarlyStopping4
from sklearn.metrics import confusion_matrix



traindatagen=ImageDataGenerator(rescale=1.0/255,rotation_range=0.3,zoom_range=0.4,horizontal_flip=True,vertical_flip=True)

train_directory="Dataset/Train"

train_data= traindatagen.flow_from_directory(directory=train_directory,class_mode="binary",batch_size=2)
print(train_data.class_indices)



test_directory="Dataset/Test"
validdata_gen= ImageDataGenerator(rescale=1.0/255,rotation_range=0.3,zoom_range=0.4,horizontal_flip=True,vertical_flip=True)
valid_data= validdata_gen.flow_from_directory(directory=test_directory,class_mode="binary",batch_size=2) 


model= Sequential()

model.add(Conv2D(filters=16,kernel_size=(2,2),activation="relu",input_shape=(224,224,3)))
model.add(Conv2D(filters=32,kernel_size=(2,2),activation="relu"))
model.add(MaxPool2D(pool_size=(4,4)))

model.add(Conv2D(filters=64,kernel_size=(2,2),activation="relu"))
model.add(MaxPool2D(pool_size=(4,4)))

model.add(Conv2D(filters=128,kernel_size=(2,2),activation="relu"))
model.add(MaxPool2D(pool_size=(4,4)))

model.add(Flatten())

model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(2,activation="softmax"))


print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])



es= EarlyStopping(monitor="val_accuracy",min_delta=0.01,patience=5,verbose=1)
model_cp= ModelCheckpoint(filepath="best_model.h5",monitor="val_accuracy",save_best_only=True,verbose=1)


history= model.fit(train_data,
                   validation_data=valid_data,
                   validation_steps=16,
                   steps_per_epoch=16,
                   epochs=30,
                   verbose=1,
                   callbacks=[es,model_cp])



cm = confusion_matrix(y_test, y_pred)
print(cm)



