# TensorFlow-Development

#### [_Netflix Stock Price Predictor_](https://github.com/ShivaSaiV/TensorFlow-Development/blob/main/Netflix%20Stock%20Price%20Predictor.ipynb): In progress
Analyzed Netflix stock data from 2002-2020. Creating a model to forecast future stock price.



## _**TensorFlow for Deep Learning**_:

#### - [_Clothing Type_](https://github.com/ShivaSaiV/TensorFlow-Development/blob/main/TensorFlow%20for%20Deep%20Learning/CNNClothingType.ipynb): Convolutional Neural Network 
Developed a machine learning model using Convolutional Neural Networks (CNN) to classify images of clothing, optimizing the Fashion MNIST dataset, which comprises 70,000 greyscale images, 60,000 for training and 10,000 for testing. The model consisted of: 
- An initial Convolutional layer creating 64 convoluted images, using padding for maintaining image dimensions.
- MaxPooling layers for down-sampling, reducing the size of output images from the previous layers while retaining important features.
- Another Convolutional layer generating 128 feature maps, enhancing the model's capability to capture complex patterns.
- MaxPooling again for down-sampling.
- Flattening layer to transform 2D feature maps into 1D feature vectors, preparing the data for input into Dense layers.
- Fully-connected Dense layers with 128 and 512 neurons respectively, employing ReLU activation function to manage the non-linear relationships in the dataset.
- An output layer with 10 units corresponding to the 10 classes of clothing items, utilizing the softmax activation function to provide a probability distribution for the predicted labels.

Achieved an accuracy of over 99% on training data and over 92% on testing data

#### - [_Flowers Classifier with augmentation_](https://github.com/ShivaSaiV/TensorFlow-Development/blob/main/TensorFlow%20for%20Deep%20Learning/FlowersClassification_augmentation.ipynb)
Task: Now is your turn to apply everything you learned in this lesson to create your own CNN to classify images of flowers. 
Developed a machine learning model using a convolutional neural network to classify colored images of flowers. However, there was a problem: overfitting. As a result, this uses data augmentation techniques such as rotation, zoom, horizontal flips, width shift, height shift, and dropout to prevent overfitting. 

#### - [_Clothing Type_](https://github.com/ShivaSaiV/TensorFlow-Development/blob/main/TensorFlow%20for%20Deep%20Learning/(Neural%20Network)%20ClothingType.ipynb): Regular neural network
Developed and trained a neural network model with 60,000 training examples and 10,000 testing examples from fashion MNIST dataset for classifying images of clothing. Flattened images (28 * 28 pixels) into 1d vectors with 784 elements. Added a layer with 128 neurons, another layer with 512 neurons, and an output layer with 10 units (10 labels) that includes softmax activation function for probability distribution. This model achieved over 91% accuracy on training data and over 88% accuracy on testing data. 

#### - [_Celsius to Fahrenheit_](https://github.com/ShivaSaiV/TensorFlow-Development/blob/main/TensorFlow%20for%20Deep%20Learning/CelsiustoFahrenheit.ipynb): 
Very simple model (Dense network) with single layer (first model) and 3 layers (new model) to convert values from Celsius to Fahrenheit. 



## _**ML with Python Course**_:
    
#### - [_Flower Classifier_](https://github.com/ShivaSaiV/TensorFlow-Development/blob/main/Machine%20Learning%20with%20Python/FlowersClassifier.py): 
DNN-based Iris Species classification algorithm to distinguish flowers based on measurements
  
#### - [_Titanic Survival_](https://github.com/ShivaSaiV/TensorFlow-Development/blob/main/Machine%20Learning%20with%20Python/TitanicSurvival.py): 
Used Linear Regression with estimators to predict survival rates
