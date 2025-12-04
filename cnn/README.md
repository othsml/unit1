# Week 5 - Convolutional Neural Network

this ai model is trained on a predefined data set with images, then it uses the torchvision library and a cnn to classify the image into 6 different classes

## key takeaway

The primary lesson from training this simple image classifier is mastering the four-step loop that underpins most Convolutional Neural Network (CNN) models: data transformation, forward pass, backpropagation, and optimization. First, you learn to transform the image data by resizing the image and normalizing its pixel values, converting them into a numerical PyTorch Tensor . Next, the forward pass feeds this Tensor through the network's convolutional and pooling layers to extract features, finally yielding a classification prediction score for categories like 'car' or 'plane'. Crucially, the backpropagation step then calculates the error gradient, which the optimizer uses to adjust the model's internal weights, iteratively improving its ability to correctly map visual features to their true category until the loss is minimized, effectively transforming raw pixel data into a decision-making AI system.

The primary lesson from training this simple sentiment analyzer is mastering the four-step loop that underpins most NLP models: quantification, forward pass, backpropagation, and optimization. First, you learn to quantify text by mapping words to numerical IDs and then converting those IDs into rich word vectors via an embedding layer. Next, the forward pass feeds these vectors through the network, where they are averaged into a single sentence vector to yield a sentiment prediction score. Crucially, the backpropagation step then calculates the error gradient, which the optimizer uses to adjust the model's internal weights, iteratively improving its ability to correctly map text to sentiment until the loss is minimized, effectively transforming raw data into a decision-making AI system.The primary lesson from training this simple image classifier is mastering the four-step loop that underpins most Convolutional Neural Network (CNN) models: data transformation, forward pass, backpropagation, and optimization. First, you learn to transform the image data by resizing the image and normalizing its pixel values, converting them into a numerical PyTorch Tensor . Next, the forward pass feeds this Tensor through the network's convolutional and pooling layers to extract features, finally yielding a classification prediction score for categories like 'car' or 'plane'. Crucially, the backpropagation step then calculates the error gradient, which the optimizer uses to adjust the model's internal weights, iteratively improving its ability to correctly map visual features to their true category until the loss is minimized, effectively transforming raw pixel data into a decision-making AI system.


## libraries used

- pytorch
- torchvision
- pillow (for image processing)
- io and request (to download image from the web)
