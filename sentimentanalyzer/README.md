# Week 4 - Sentiment Analyzer

this ai model is trained on a set of user data, then it takes a user input and predicts if the sentiment is either positive, negative, or neutral.

## key takeaway

The primary lesson from training this simple sentiment analyzer is mastering the four-step loop that underpins most NLP models: quantification, forward pass, backpropagation, and optimization. First, you learn to quantify text by mapping words to numerical IDs and then converting those IDs into rich word vectors via an embedding layer. Next, the forward pass feeds these vectors through the network, where they are averaged into a single sentence vector to yield a sentiment prediction score. Crucially, the backpropagation step then calculates the error gradient, which the optimizer uses to adjust the model's internal weights, iteratively improving its ability to correctly map text to sentiment until the loss is minimized, effectively transforming raw data into a decision-making AI system.

## libraries used

- pytorch

