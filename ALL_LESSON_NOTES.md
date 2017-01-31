
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


During the exercise we had sets for:

* Training
* Validation
* Testing

Measuring performance is subtle.

If we go to our classificaiton task, and try to predict some images and check how many we get right - that is the error measure.

Now we measure with images that we've never seen but performance gets worse.

Classifier doesn't do as well...

Imagine we compare a classifier that just compares the new image to any of the oher images we've seen and just returns the label.

It is a great classifier - but as soon as it sees a new image, it's lost.

It has **memorized examples**, so now it has overtrained...

Job is to **generalize to new data**.

### How to quantify generalization?

We take a small subset of training set, not using in training, adn measure the error on test data.

This way classifier can't cheat.

But there are some times where you tweak to your test data, but you might still have bad accuracy... so your classifier can see your test data **Through your own eyes** as you make choices of how you want to tweak your classifier

So how to solve it?

### Validation + Test Data

To solve it, we take another chunk of out training set and **hide it under a rock**


# Overfitting  Datasets

We need to be careful about overfitting our dataset - we need to use a validation set.

## Kaggle Challenge

Classification platform for ML challenges

A lot of the competitors submit models that work really well with them, but when they submit it, it works much worse.

## Validation and Test Size

Imagine we have a model with 66% accuracy

And you make it go up to 83% 

But this is not reliable... this might only be a change of label for a single example

**The more data, the more accurate**

A change that affects 30 labels, typcally can be trusted. 

### The Rule of 30

If we had 3000 examples, we would have to see a minimum of 1% improvement in order for it to be trusted.

This is 1/100 * 3000 = 30.


### Validation Set Size

Holding less than 30,000 examples, show changes in 0.1 in accuracy.

Only 30k examples can be a lot of data with a small training set

Cross validation can be a good way, but it's often slow

But getting more data is often the best.


### Optimizing a logistic classifier

Training logistic regression using gradient descent is great.
For one thing you're directly optimizing for what we care about.
In practice, a lot of ML is about designing the right loss function to optimize.

**Big problem**
It's very difficult to scale

#### Stochastic Gradient Descent

The problem with scaling gradient descent.

If computing your loss takes N floating point operations, computing the gradient takes **3 times that amount**.

The loss function is often quite large, which requires a lot of computation.

Plus it needs to train in a lot of data.

Because **gradient descent is iterative** we need to do this multiple times.

#### Optimizing (Cheating)

We instead can optimize by computing a **"terrible estimate"**.

That estimate is so bad we'll wonder why it's right.

We basically compute the average loss for a very small fraction of the data.

Between 1 and 1000 samples each time.

**It needs to be random**.

We will need to take many more smaller steps.

This is **Stochastic Gradient Descent**.

It comes with a lot of issues in practice, but it works...

### Helping Stochastic Gradient Descent

* Inputs
    - Mean needs to be close to ZERO
    - Have equal variance (small)
* Initial weights
    - They have to be random
    - Mean close to ZERO
    - Equal variance (small too)

We need some rules:

#### Momentum

We are now taking small steps in almost random directions.

We can take advantage of the knowledge we've gathered from previous gradients:
> We can keep a running average of the gradients
> And use running average instead of direction for current batch of data

#### Learning Rate Decay

When we switched to SGD, we said we were gonna take smaller steps

**HOW SMALL SHOULD THE STEPS BE?**

It's beneficial to take smaller and smaller steps

Lowering gradient over time is key!

#### Learning Rate Tuning

Having a higher learning rate does not mean we are learning faster

You can take a model, lower the learning rate, and **get to a better model faster**

We might be tempted to have a look at a graph of the loss/steps:
> A higher learning rate might start faster, but plateu faster
> Whilst a lower learning rate might reach better optimization

**Never trust how fast you learn, as that is not correlated with model quality!!**

#### SGD Black Magic

Many parameters to play with:

* Initial learning rate
* Learning rate decay 
* Momentum
* Batch Size
* Weight initialization

If we have to remember 1 thing, always **try to lower learning rate first**

One aproach: ADAGRAD

##### ADAGRAD

This is an alternative to SGD that does momentum and learning rate decay automatically for us.

This makes learning less sensitive to hyperparameters

But it tends to be less accurate

However it does the job

### RECAP

* We have a simple linear model
* It emits probabilities
* We can use probabilities for classification
* We know how to optimize parameters
    - We can use SGD and its variance to optimize

# Intro to Deep Neural Networks

We are now going to turn the linear regression model into a deep net

## Linear Regression Model

How many parameters did the previous model have? I think 10

* The input is 28x28 = 784
* But also the number of weights are 10
* So the total number of parameters **is 7850**

That is often the case: If you have N inputs and k outputs, you have (N+1) *k to use.

This is also linear, so the amount of flexibility is low.

* If two inputs interact in addition, then it would work
* If two inputs interact as products, then it won't work

Linear operations are really good - GPUs are made for linear computations.

Linear operations are very stable.

The derivatives are also constant - so it's stable.

**We want to keep our parameters in linear models, but we would also want the entire model to be non-linear.**

### Non-linearities

The favourite function is **ReLU** - Rectified Linear Unit

ReLUs are the simplest non-linear function that exist:

* They are y>0 linearly when x > 0
* And y=0 if x is less than 0

## Network of ReLUs

Instead of having a single matrix multiply as our classifier, we'll insert a ReLU in the middle, allowing us to have 2 matrixes.

Going from the inputs to the ReLUs - and anotherone connecting the ReLU.

Now we have a new node we can tune. We can make it as big as we want.

This is the biggest Neural Network.

### The Chain Rule

Stacking up simple operations

One reason to chain up operations is because it makes math very simple

A deep learning framework manages it for us

The chain rule: Two functions composed, one applied to ouptut of other. You can compute the derivative by taking the product of the compoents.

This is powerful - as long as you know how to write the derivatives of the individual functions, it's possible to bind them together and compute the whole function.

There is a way to write the chain rule with very efficient computation and that looks like a very simple data pipeline.

### Backprop

* Imagine your net is a stack of operations - linear transforms, relus, etc.
* Some have parameters, some don't like relus
* To compute the derivatives
    - the data of the graph flows backwards
    - gets combined using the chain rule
    - and it produces gradients
    - That graph can be derived completely from individual operations in network

This is called backpropagation

Running model up to predictions is called forward propagation.

### RECAP

* To run stochastic gradient descent
    - For every single batch of data
    - We run the forward prop
    - Then the backward prop
    - That will give us gradients for each of the weights in our models
    - Then we apply the gradients with the learning rates with our original weights
* Repeat that all over again.

This is how it's optimized

## Training a Deep Network

We managed to put together a 2-layer neural network, we can make it more complex by increasing the size of the layer in the middle.

But increasing the H is not particulary efficient.

You would have to make it very big but it would be hard to train.

Instead you can add more layers and make the net deeper.

### Deep networks

You want to make it deeper because of:

* Parameter efficiency
    - More performance with less parameters going deeper
* TEnds to have  a hierarchical structure  
    - This is very good for things like images

### History of deep net

Deep models only shine if you have enough data to train them

We know better today how to train very big models using regularization techniques.

There is a general issue called the **skinny jeans problem** - Skinny jeans look good, but it's hard to get into them so you just wear slightly bigger. Same with Deep Networks - You can't know the exact right size, so we try nets that are too big for data and then **prevent them from overfitting.**


## Overfitting

### Early termination

We stop training as soon as the model stops improving

This is still the best way to avoid the network from overfitting

### Regularization

To apply regularization is applying artificial  constraints on network that implicitly reduce number of free parameters.

L_2 regularization is an example - adding another term to the loss that penalizes the formula.

### L2 Regularization

The structure of the network doesn't have to change because you just add it to the loss.

L2 norm is the sum of the squares of the weights.

The derivative of 1/2 x**2 is x

### Dropout

* Imagine you have one layer connected to the other
* The outputs are called activations
* Take the activations, and for a random number of neurons set them to zero

Net can never rely on any specific activaiton as it might be squashed, so it prevents overfitting

#### Training & Evaluation

We want something deterministic when measuring a network with dropout

We want to take the conscensus of these redundant models

You get consensus by averaging activations

During training, not only use 0 activations, but also scale the remaining activations by 2

The result is a number of these activations that is properly scaled

# Convolutional Neural Network

## Statistical Invariance

You can reduce complexity of specific datasets. In images for example, if color doesn't matter you can reduce complexity by combining colour channels into a single monochromatic channel.

In another image with a cat, you want to identify that it has cats, no matter where.

If the network has to learn the same picture with the cat in multiple positions it would reduce complexity

But if you can teach it that objects are the same no matter where in the screen

This is called translation invariance - different positions, same kitten.

Another example, having a long text , the meaning of the word "kitten" doesn't change no matter where it is.

The way you achieve  this in Neural Networks is using **weight sharing.**

And you **train the weights jointly** for those inputs.

It's a very important idea - **statistical invariants** - things that don't change with time or space are everywhere!

This is where Convolutional Neural Networks kicks in for images!

For text - this is where Recurrent Neural Networks come in.

## Convolutional Neural Networks

Also called convnets 

Convnets are Neural Networks that share params across space.

Images can be represented as a 2d squere - because you have RGB channels, you have a depth.

We'll take  a small patch of the image and run a Neural Network.

Then we'll go through the image scanning patches, as if we were painting it with a brush  

Now you have an output that has multiple channels.

We have less weights, and shared across space.

Instead of having stacks of multiplayed layers, we have convolutional layers.

At first we have the image, then we apply convolutions that compress the symetric dyensions.

At the top we put the classifier.

If we are going to implement this, there is a lot of things to get right.

### Patches

Patches are often called **kernels**

If you don't go outside the edge = **valid padding**
If you have zero padding = **same padding**

### Feature Map Sizes

Imagine we have a 28x28 image
We run a 3x3 convolution on it
With an input depth = 3
and output depth = 8

## More about convolutions

* Pooling
* 1x1 convolutions
* inception

### Pooling

We take all the convolutions and combine them.

most common is max pooling: We take all the numbers from a specific stride, and just choose the largest.

* This is good becuase it's parameter free
* It's more accurate
* But it's more expensive
* It has more hyperparameters
    - Pooling size
    - Pooling stride

A very typical architecture is one altrernating convolutions and pooling layers, followed by fully connected layers and finally the classifier.

A common CNN is the Alexnet

### Average pooling

Instead of taking the max pooling, we can take the average pooling 

Another idea is the 1x1 convolutions - but why should we use it?

If we add a 1x1 patch, we would basically have a Neural Network before the convolutions, making the net deeper by default and have more parameters without changing the structure.

## Inception Modules

The idea is that at each layer of the convnet we can have a pooling op, convolution, etc.

We ened to decide the size of the convolution

Inception model requires you to have a composition of avg pooling foolowed by 5z5, and then concatenate the output

You can choose the parameters in a way that the Neural Network performs better




























