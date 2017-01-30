
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

Going from the inputs to the ReLUs

