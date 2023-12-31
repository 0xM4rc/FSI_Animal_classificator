# FSI Convolutional Neural Networks

A Convolutional Neural Network (CNN) is a type of Deep Learning neural network architecture designed for processing structured grid-like data. This neuronal network is powerful for tasks related to image and video analysis, but it can be applied to other types of data like audio recognition, natural language processing...

## CNN Architecture

![Simple CNN Architecture](./assets/imgLayers.png)


- The **Convolutional** layer utilizes filters on the input image to extract distinct features
- **RELU**(Rectified Linear Unit) is an activation function known for its computational efficiency and simple training process, as it doesn't suffer from saturation issues.
- The **Pooling** layer reduces the image size for computational efficiency.
- The **Fully connected** layer generates the ultimate prediction. 
- The **Flatten** operation transforms multidimensional arrays into a one-dimensional array.
- **Softmax** is an activation function that normalizes the output of a neural network into a probability distribution over multiple classes.

The network refines its filters using backpropagation and gradient descent to achieve optimal learning.

![Simple CNN Architecture](./assets/imgHorseCnn.png)
Source: http://cs231n.stanford.edu/

## 🐴 Animal Classifier

An animal categorizer denotes a structure or pattern that has the ability to organize or group animals depending on specific traits or attributes. It generally involves employing machine learning or recognizing patterns, allowing the structure to acquire knowledge from information to differentiate among diverse animal categories or types.

