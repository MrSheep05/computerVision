import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import random
import pandas as pd
tf.random.set_seed(1)
def toInt(path):
    array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(array, (128,128))
    return x
X=[]
Y=[]
error=[]
path=os.path.join(os.path.dirname(__file__),"handData")
for img in os.listdir(path):
    try:
        X.append(toInt(os.path.join(path,img)))
        Y.append([float(element) for element in img.split("_")[1:-1]])
    except:
        error.append(img)
X=tf.constant(X)
Y=tf.constant(Y)
length = len(Y)-5
Xt=X[:length]
Yt=Y[:length]
Xp=X[length:]
Yp=Y[length:]
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(128,128,1), padding="same",dilation_rate=2, activation='tanh'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(64,(5,5),activation='relu'),
    tf.keras.layers.MaxPool2D((2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D((2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(99,activation='sigmoid'),
    tf.keras.layers.Dense(5)
])
# model2 = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(10,3,input_shape=(128,128,1),activation='relu'),
#     tf.keras.layers.Conv2D(10,3,activation='relu'),
#     tf.keras.layers.MaxPool2D(2),
#     tf.keras.layers.Conv2D(20,3,activation='relu'),
#     tf.keras.layers.MaxPool2D(2),
#     tf.keras.layers.Conv2D(20,3,activation='relu'),
#     tf.keras.layers.MaxPool2D(2),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(100,activation='tanh'),
#     tf.keras.layers.Dense(5)

# ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=6.28e-4,decay=6.28e-4/100),
loss=tf.keras.losses.MeanAbsoluteError(),metrics=['mae'])
his=model.fit(Xt,Yt,epochs=25,batch_size=8,validation_data=(Xp,Yp))
print(model.evaluate(Xp,Yp))
# pd.DataFrame(his.history).plot()
# plt.show()