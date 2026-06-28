# Reducing Loss

To train a model, we need a good way to reduce the model’s loss. An iterative approach is one widely used method for reducing loss, and is as easy and efficient as walking down a hill.

## An iterative approach

Iterative learning might remind you of the "[Hot and Cold](http://www.howcast.com/videos/258352-how-to-play-hot-and-cold/)" kid's game for finding a hidden object like a thimble. In this game, the "hidden object" is the best possible model. You'll start with a wild guess ("The value of $w_{1}$ is $0$.") and wait for the system to tell you what the loss is. Then, you'll try another guess ("The value of $w_{1}$ is $0.5$.") and see what the loss is. Aah, you're getting warmer. Actually, if you play this game right, you'll usually be getting warmer. The real trick to the game is trying to find the best possible model as efficiently as possible.

The following figure suggests the iterative trial-and-error process that machine learning algorithms use to train a model:

<div align='center'>
  <img src="https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentDiagram.svg" />

  <strong>Figure 1. An iterative approach to training a model.</strong>
</div>

Iterative strategies are prevalent in machine learning, primarily because they scale so well to large data sets.

The "model" takes one or more features as input and returns one prediction ($y′$) as output. To simplify, consider a model that takes one feature and returns one prediction:

$$y′=b+w_{1}x_{1}$$

What initial values should we set for $b$ and $w_{1}$? For linear regression problems, it turns out that the starting values aren't important. We could pick random values, but we'll just take the following trivial values instead:

* $b=0$
* $w_{1}=0$

Suppose that the first feature value is 10. Plugging that feature value into the prediction function yields:

$$y′=0+0⋅10=0$$

The "Compute Loss" part of the diagram is the [loss function](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss) that the model will use. Suppose we use the squared loss function. The loss function takes in two input values:

* $y′$: The model's prediction for features $x$
* $y$: The correct label corresponding to features $x$

At last, we've reached the "Compute parameter updates" part of the diagram. It is here that the machine learning system examines the value of the loss function and generates new values for $b$ and $w_{1}$. For now, just assume that this mysterious box devises new values and then the machine learning system re-evaluates all those features against all those labels, yielding a new value for the loss function, which yields new parameter values. And the learning continues iterating until the algorithm discovers the model parameters with the lowest possible loss. Usually, you iterate until overall loss stops changing or at least changes extremely slowly. When that happens, we say that the model has [**converged**](https://developers.google.com/machine-learning/glossary#convergence).

## Gradient Descent

The iterative approach diagram contained a green hand-wavy box entitled "Compute parameter updates." We'll now replace that algorithmic fairy dust with something more substantial.

Suppose we had the time and the computing resources to calculate the loss for all possible values of $w_{1}$. For the kind of regression problems we've been examining, the resulting plot of loss vs. $w_{1}$ will always be convex. In other words, the plot will always be bowl-shaped, kind of like this:

<div align='center'>
  <img src="https://developers.google.com/static/machine-learning/crash-course/images/convex.svg" />

  <strong>Figure 2. Regression problems yield convex loss vs. weight plots.</strong>
</div>

Convex problems have only one minimum; that is, only one place where the slope is exactly 0. That minimum is where the loss function converges.

Calculating the loss function for every conceivable value of $w_{1}$ over the entire data set would be an inefficient way of finding the convergence point. Let's examine a better mechanism—very popular in machine learning—called **gradient descent**.

The first stage in gradient descent is to pick a starting value (a starting point) for $w_{1}$. The starting point doesn't matter much; therefore, many algorithms simply set $w_{1}$ to $0$ or pick a random value. The following figure shows that we've picked a starting point slightly greater than $0$:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentStartingPoint.svg' />
  
  <strong>Figure 3. A starting point for gradient descent.</strong>
</div>

The gradient descent algorithm then calculates the gradient of the loss curve at the starting point. Here in this Figure, the gradient of the loss is equal to the [derivative](https://wikipedia.org/wiki/Differential_calculus#The_derivative) (slope) of the curve, and tells you which way is "warmer" or "colder." When there are multiple weights, the gradient is a vector of partial derivatives with respect to the weights.

### Partial derivatives

A **multivariable function** is a function with more than one argument, such as:

$$f(x,y)=e^{2y}\mathrm{sin}(x)$$

The **partial derivate** $f$ **with respect to** $x$, denoted as follows:

$$\dfrac{∂f}{∂x}$$

is a derivative of $f$ considered as a function of $x$ alone. To find the following:

$$\dfrac{∂f}{∂x}$$

so must hold $y$ constant (so $f$ is now a function of one variable $x$), and take the regular derivative of $f$ with respect to $x$. For example, when $y$ is fixed at $1$, the preceding function becomes:

$$f(x)=e^{2}\mathrm{sin}(x)$$

This is just a function of one variable $x$, whose derivative is:

$$e^{2}\mathrm{cos}(x)$$

In general, thinking of $y$ as fixed, the partial derivative of $f$ with respect to $x$ is calculated as follows:

$$\dfrac{∂f}{∂x}(x,y)=e^{2y}\mathrm{sin}(x)$$

Similarly, if we hold $x$ fixed instead, the partial derivative of $f$ with respect to $y$ is:

$$\dfrac{∂f}{∂x}(x,y)=2e^{2y}\mathrm{sin}(x)$$

Intuitively, a partial derivative tells how much the function changes when you perturb one variable a bit. In the preceding example:

$$\dfrac{\partial f}{\partial x}(0,1)=e^2 \approx 7.4$$

So when you start at $(0,1)$, hold $y$ constant, and move $x$ a little, $f$ changes by about $7.4$ times the amount that you changed $x$.

In machine learning, partial derivatives are mostly used in conjunction with the gradient of a function.

### Gradients

The **gradient** of a function, denoted as follows, is the vector of partial derivatives with respect to all of the independent variables:

$$∇f$$

For instance, if:

$$f(x,y)=e^{2y}\mathrm{sin}(x)$$

then:

<div class="math-jax-block">

$$∇f(x,y)= \left( \dfrac{\partial f}{\partial x}(x,y), \dfrac{\partial f}{\partial y}(x,y) \right)=(e^{2y}\mathrm{cos}(x), 2e^{2y}\mathrm{sin}(x))$$

</div>

Note the following:

* $∇f$ - Points in the direction of greatest increase of the function.
* $−∇f$ - Points in the direction of greatest decrease of the function.

The number of dimensions in a vector is equal to the number of variables in the formula for $f$; in other words, the vector falls within the domain space of the function. For instance, the graph of the following function $f(x,y)$:

<div class="math-jax-block">

$$f(x,y)=4+(x−2)^2+2y^2$$

</div>

when viewed in three dimensions with $z=f(x,y)$ looks like a valley with a minimum at $(2,0,4)$:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/ThreeDimensionalPlot.svg' />
</div>

The gradient of $f(x,y)$ is a two-dimensional vector that tells you in which $(x,y)$ direction to move for the maximum increase in height. Thus, the negative of the gradient moves you in the direction of maximum decrease in height. In other words, the negative of the gradient vector points into the valley.

In machine learning, gradients are used in gradient descent. We often have a loss function of many variables that we are trying to minimize, and we try to do this by following the negative of the gradient of the function.

Okay, so comming back to our gradient descent. We've found that a gradient is a vector that has both of the following characterstics:

* A direction
* A magnitude

The gradient always points in the direction of steepest increase in the loss function. The gradient descent algorithm takes a step in the direction of the negative gradient in order to reduce loss as quickly as possible.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentNegativeGradient.svg' />

  <strong>Figure 4. Gradient descent relies on negative gradients.</strong>
</div>

To determine the next point along the loss function curve, the gradient descent algorithm adds some fraction of the gradient's magnitude to the starting point as shown in the following figure:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/GradientDescentGradientStep.svg' />

  <strong>Figure 5. A gradient step moves us to the next point on the loss curve.</strong>
</div>

The gradient descent then repeats this process, edging ever closer to the minimum.

## Learning Rate

As noted, the gradient vector has both a direction and a magnitude. Gradient descent algorithms multiply the gradient by a scalar known as the **learning rate** (also sometimes called **step size**) to determine the next point. For example, if the gradient magnitude is $2.5$ and the learning rate is $0.01$, then the gradient descent algorithm will pick the next point $0.025$ away from the previous point.

**Hyperparameters** are the knobs that programmers tweak in machine learning algorithms. Most machine learning programmers spend a fair amount of time tuning the learning rate. If you pick a learning rate that is too small, learning will take too long:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LearningRateTooSmall.svg' />

  <strong>Figure 6. Learning rate is too small.</strong>
</div>

Conversely, if you specify a learning rate that is too large, the next point will perpetually bounce haphazardly across the bottom of the well like a quantum mechanics experiment gone horribly wrong:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LearningRateTooLarge.svg' />

  <strong>Figure 7. Learning rate is too large.</strong>
</div>

There's a [Goldilocks](https://wikipedia.org/wiki/Goldilocks_principle) learning rate for every regression problem. The Goldilocks value is related to how flat the loss function is. If you know the gradient of the loss function is small then you can safely try a larger learning rate, which compensates for the small gradient and results in a larger step size.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LearningRateJustRight.svg' />

  <strong>Figure 8. Learning rate is just right.</strong>
</div>

The ideal learning rate in one-dimension is $\dfrac{1}{f(x)″}$ (the inverse of the second derivative of $f(x)$ at $x$).

The ideal learning rate for $2$ or more dimensions is the inverse of the [Hessian](https://wikipedia.org/wiki/Hessian_matrix) (matrix of second partial derivatives).

The story for general convex functions is more complex.

## Stochastic Gradient Descent

**Stochastic Gradient Descent (SGD)** is an optimization algorithm used to find the minimum of a function. It is a type of gradient descent algorithm that is often used in machine learning and deep learning. It is called "stochastic" because it uses random samples of the data to estimate the gradient of the objective function, rather than using the entire dataset.

Here's how it works:

* The algorithm starts with an initial guess of the parameters.
* It then selects a random sample of the data and calculates the gradient of the objective function with respect to the parameters using that sample.
* The parameters are then updated in the direction opposite to the gradient.
* This process is repeated until the parameters converge to a minimum of the objective function.

One of the main advantage of stochastic gradient descent is that it is computationally efficient. Because it uses a random sample of the data at each step, it can be much faster than batch gradient descent, which uses the entire dataset to calculate the gradient. Additionally, because it uses random samples, it can "escape" from local minima and converge to a global minimum.

An example of an application of **SGD** is Linear Regression, where the objective function is the mean squared error and the parameters are the weights of the model. Another example is Logistic Regression, where the objective function is the cross-entropy loss and the parameters are the weights of the model.

It should be noted that the optimization speed of **SGD** can be affected by the choice of the learning rate and the shuffling of the data during each iteration. Also it is not guaranteed to find the global minimum because of the randomness but it can be useful in practice.

In summary, **Stochastic Gradient Descent** is an optimization algorithm that is efficient and can help to find the global minimum of a function. It has been widely used in machine learning and deep learning tasks.

