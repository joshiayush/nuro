# Logistic Regression

Instead of predicting *exactly* 0 or 1, **logistic regression** generates a probability—a value between 0 and 1, exclusive. For example, consider a logistic regression model for spam detection. If the model infers a value of 0.932 on a particular email message, it implies a 93.2% probability that the email message is spam. More precisely, it means that in the limit of *infinite* training examples, the set of examples for which the model predicts 0.932 will actually be spam 93.2% of the time and the remaining 6.8% will not.

## Calculating a Probability

Many problems require a probability estimate as output. Logistic regression is an extremely efficient mechanism for calculating probabilities. Practically speaking, you can use the returned probability in either of the following two ways:

* "As is"
* Converted to a binary category.

Let's consider how we might use the probability "as is." Suppose we create a logistic regression model to predict the probability that a dog will bark during the middle of the night. We'll call that probability:

$$\mathrm{p}(\mathrm{bark} \vert \mathrm{night})$$

If the logistic regression model predicts $\mathrm{p}(\mathrm{bark} \vert \mathrm{night}) = 0.05$, then over a year, the dog's owners should be startled awake approximately 18 times:

$$\mathrm{startled} = \mathrm{p}(\mathrm{bark} \vert \mathrm{night}) \cdot \mathrm{nights}$$
$$= 0.05 \cdot 365$$
$$= 18$$

In many cases, you'll map the logistic regression output into the solution to a binary classification problem, in which the goal is to correctly predict one of two possible labels (e.g., "spam" or "not spam").

You might be wondering how a logistic regression model can ensure output that always falls between 0 and 1. As it happens, a **sigmoid function**, defined as follows, produces output having those same characteristics:

$$y = \dfrac{1}{1 + e^{-z}}$$

The sigmoid function yields the following plot:

<div align='center'>
  <img src='https://developers.google.com/machine-learning/crash-course/images/SigmoidFunction.png' />

  <strong>Figure 1: Sigmoid function.</strong>
</div>

If $z$ represents the output of the linear layer of a model trained with logistic regression, then $sigmoid(z)$ will yield a value (a probability) between 0 and 1. In mathematical terms:

$$y^ \prime = \dfrac{1}{1 + e^{-z}}$$

where:

* $y^ \prime$ is the output of the logistic regression model for a particular example.

For $z$:

<div class="math-jax-block">

$$z = b + w_{1}x_{1} + w_{2}x_{2} + ... + w_{N}x_{N}$$

</div>

where:

  * The $w$ values are the model's learned weights, and $b$ is the bias.
  * The $x$ values are the feature values for a particular example.

Note that $z$ is also referred to as the *log-odds* because the inverse of the sigmoid states that $z$ can be defined as the log of the probability of the 1 label (eg., "dog barks") divided by the probability of the 0 label (eg., "dog doesn't bark"):

$$z = \mathrm{log} \left( \dfrac{y}{1- y} \right)$$

Here is the sigmoid function with ML labels:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LogisticRegressionOutput.svg' />

  <strong>Figure 2: Logistic regression output.</strong>
</div>

## Loss and Regularization

### Loss function for Logistic Regression

The loss function for linear regression is squared loss. The loss function for logistic regression is **Log Loss**, which is defined as follows:

<div class="math-jax-block">

$$\mathrm{Log\ Loss} = \sum_{(x,y) \in D} -ylog(y^ \prime) - (1 - y)log(1 - y^ \prime)$$

</div>

where:

* $(x,y) \in D$ is the data set containing many labeled examples, which are $(x,y)$ pairs.
* $y$ is the label in a labeled example. Since this is logistic regression, every value of $y$ must either be 0 or 1.
* $y^ \prime$ is the predicted value (somewhere between 0 and 1), given the set of features in $x$.

### Regularization in Logistic Regression

[Regularization](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/video-lecture) is extremely important in logistic regression modeling. Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions. Consequently, most logistic regression models use one of the following two strategies to dampen model complexity:

* $L_{2}$ regularization
* Early stopping, that is, limiting the number of training steps or the learning rate.

Imagine that you assign a unique id to each example, and map each id to its own feature. If you don't specify a regularization function, the model will become completely overfit. That's because the model would try to drive loss to zero on all examples and never get there, driving the weights for each indicator feature to +infinity or -infinity. This can happen in high dimensional data with feature crosses, when there’s a huge mass of rare crosses that happen only on one example each.

Fortunately, using L2 or early stopping will prevent this problem.

