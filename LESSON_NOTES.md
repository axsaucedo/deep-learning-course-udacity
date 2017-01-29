
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

### Variation of confidence

If we multiply the inputs by 10, our confidence becomes higher, so one element approaches to 1 whilst the rest approach to 0

When we divide by 10, our confidence ends in almost an equal distribution of probability with the classes.

### One hot encoding

We want a vector that has a 1 for the correct class and 0 for the rest.

One hot encoding works well until you get into problems that has thousands or millions of classes.

In this case, vector is too large, and has mostly ZEROS.

We can use embeddings for this.

We can now measure how well we're doing with vectors - labels x predictions.

To measure the distance between two probability vectors is called **cross entropy**.

You need to make sure labels and distriutions are in the right place because of the LOG.

#### Recap

* Input is turned into logits using a linear model
* Then feed logits into softmax to turn into probabilities
* Then compare probabilities with one-hot labels using cross entropy function
* This is called multinomial-logistic classification

## Minimizing Cross entropy

How do we find the biases and the weights?

* Low distance for correct class
* High distance from incorrect class

We can measure that distance averaged over the entire training set for all the inputs and all the labels.

This is called the **TRAINING LOSS**

Loss is a function of several weights - a function that will be large in some areas and small in others - we want to find where the loss is the smallest.

We will use gradient descent - gradient of loss with respect of params, and follow the derivative by taking a step backwards.

## Numerical Stability

We need to be careful with numbers that are too large or too small. Adding very small values to large value can introduce a lot of errors.

We want our variables to have 0 mean, and equal variance when possible.

There are also good mathematical reasons to keep the values you compute roughly with a mean of zero and equal variance when doing optimization.

If you're dealing with images it's simpler, as you only take the values, substract 128 and divide by 128.

### INitializing Weights and Biases

Simple general method - draw weights randomly from gaussian distr. with mean 0 and stdev SIGMA.

## RECAP

* Training data is normalized to have 0 mean and equial variance 
* Multiply by weight matrix initialized with gaussian distr
* We apply the softmax
* Then apply cross entropy loss
* Finally calculate the average of this loss over entire training data

Then...

* Compute derivative of loss in respect to weights
* Derivative of loss in respect to biases
* Take a step back in direction opposite to derivative

**THEN REPEAT! Until we reach the minimum**






