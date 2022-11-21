# computerVision
*[Tensorflow](https://www.tensorflow.org) project for predicting bend of fingers*


This project contains 74 learning data photos and 10 testing data photos *(that is relatively few)*


## Model performance
>**mea** - [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error) of learning data

>**val_mea** - mean absolute error of testing data

X axis presents [epochs](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning) while Y axis presents differnce between desired output and predicted value

*Angle of 180 degrees is represented by 1, so when difference is 0.2 that means the model might predict angle within 36 degrees error
![image](https://user-images.githubusercontent.com/91011923/203149055-5573570a-61cb-4a69-916d-c947a3c65337.png)
