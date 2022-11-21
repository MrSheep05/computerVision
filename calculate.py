import tensorflow as tf
import numpy
import cv2
model = tf.keras.models.load_model('firstModel0909.h5')
def calcPhoto(image):
    #predicts fingers position (angle of bend)
    img=numpy.array(image)
    img=cv2.resize(img,(128,128))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=tf.constant(img,dtype=tf.int32)
    pred=model.predict(tf.reshape(img,(-1,128,128,1)))
    return pred