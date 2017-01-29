
# Classification

## Examples

### How to detect people in an image

A way to do it is by using a binary classifier.

The classes would be "pedestrian" or "no pedestrian"

And then slide it off possible locations in the image.

### Webpage ranking

Imagine we want to find all the websites available that match a keyword search

One way of doing this is to classify pairs of (QUERY,WEB_PAGE) as relevant/not relevant.

But if you run the classifier with all the websites in the world that might be too much time...

## Logistic Classifier

A logistic classifier is a linear classifier:
> WX + b = Y

It takes the input (ie pixels of image) and applies linear function to predict.

It's just a giant matrix multiply - takes all inputs as a vector, multiplied by a matrix, and outputs the predictions - one per output class.

The weights and bias is where ML comes in.

We want to turn the scores to probabilities.

The way to take scores to probabilities is by using a **softmax function**.

Scores in logistic regression are called **logits** (this is the output of the neurons)

