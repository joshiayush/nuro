# Regularization for Simplicity

**Regularization** means penalizing the complexity of a model to reduce overfitting.

## L2 Regularization

Consider the following **generalization curve**, which shows the loss for both the training set and validation set against the number of training iterations.

<div align='center'>

  <img src='https://developers.google.com/static/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg' />

  <strong>Figure 1. Loss on training set and validation set.</strong>
</div>

Figure 1 shows a model in which training loss gradually decreases, but validation loss eventually goes up. In other words, this generalization curve shows that the model is [overfitting](https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting) to the data in the training set. Channeling our inner [Ockham](https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting#ockham), perhaps we could prevent overfitting by penalizing complex models, a principle called **regularization**.

In other words, instead of simply aiming to minimize loss (empirical risk minimization):

<div class="math-jax-block">

$$\mathrm{minimize(Loss(Data \vert Model))}$$

</div>

we'll now minimize loss+complexity, which is called **structural risk minimization**.

<div class="math-jax-block">

$$\mathrm{minimize(Loss(Data \vert Model) + \mathrm{complexity(Model)})}$$

</div>

Our training optimization algorithm is now a function of two terms: the **loss term**, which measures how well the model fits the data, and the **regularization term**, which measures model complexity.

There are two commom ways to think of model complexity:

  * Model complexity as a function of the *weights* of all the features in the model.
  * Model complexity as a function of the *total number of features* with nonzero weights.

If model complexity is a function of weights, a feature weight with a high absolute value is more complex that a feature weight with a low absolute value.

We can quantify complexity using the $L_{2}$ **regularization** formula, which defines the regularization term as the sum of the squares of all the feature weights:

<div class="math-jax-block">

$$\mathrm{L_{2}\ regularization\ term} = \mathrm{\lvert \lvert w \rvert \rvert}^2_{2} = w_{1}^2 + w_{2}^2 + ... + w_{n}^2$$

</div>

In this formula, weights close to zero have little effect on model complexity, while outlier weights can have a huge impact.

For example, a linear model with the following weights:

<div class="math-jax-block">

$$\{w_{1} = 0.2, w_{2} = 0.5, w_{3} = 5, w_{4} = 1, w_{5} = 0.25, w_{6} = 0.75\}$$

</div>

Has an $L_{2}$ regularization term of 26.915:

<div class="math-jax-block">

$= w_{1}^2 + w_{2}^2 + w_{3}^2 + w_{4}^2 + w_{5}^2 + w_{6}^2$

</div>

<div class="math-jax-block">

$= 0.2^2 + 0.5^2 + 5^2 + 1^2 + 0.25^2 + 0.75^2$

</div>

<div class="math-jax-block">

$= 0.04 + 0.24 + 25 + 1 + 0.0625 + 0.5625$

</div>

<div class="math-jax-block">

$= 26.915$

</div>

But $w_{3}$, with a squared value of 25, contributes nearly all the complexity. The sum of the squares of all five other weights adds just 1.915 to the $L_{2}$ regularization term.

## Lambda

Model developers tune the overall impact of the regularization term by multiplying its value by a scalar known as **lambda** (also called the **regularization rate**). That is, model devvelopers aim to do the following:

<div class="math-jax-block">

$$\mathrm{minimize(Loss(Data \vert Model) + \lambda \ complexity(Model))}$$

</div>

Performing $L_{2}$ regularization has the following effect on the model:

  * Encourages weight values toward 0 (but not exactly 0).
  * Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution.

Increasing the lambda value strengthens the regularization effect. For example, the histogram of weights for a high value of lambda might look as shown in Figure 2.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/HighLambda.svg' height='400px' />

  <strong>Figure 2. Histogram of weights.</strong>
</div>

Lowering the value of lambda tends to yield a flatter histogram, as shown in Figure 3.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LowLambda.svg' />

  <strong>Figure 3. Histogram of weights produced by a lower lambda value.</strong>
</div>

When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:

* If your lambda value is too high, your model will be simple, but you run the risk of underfitting your data. Your model won't learn enough about the training data to make useful predictions.
* If your lambda value is too low, your model will be more complex, and you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.

The ideal value of lambda produces a model that generalizes well to new, previously unseen data. Unfortunately, that ideal value of lambda is data-dependent, so you'll need to do some tuning.

### L2 regularization and Learning rate

There's a close connection between learning rate and lambda. Strong L2 regularization values tend to drive feature weights closer to 0. Lower learning rates (with early stopping) often produce the same effect because the steps away from 0 aren't as large. Consequently, tweaking learning rate and lambda simultaneously may have confounding effects.

**Early stopping** means ending training before the model fully reaches convergence. In practice, we often end up with some amount of implicit early stopping when training in an [online](https://developers.google.com/machine-learning/crash-course/production-ml-systems) (continuous) fashion. That is, some new trends just haven't had enough data yet to converge.

As noted, the effects from changes to regularization parameters can be confounded with the effects from changes in learning rate or number of iterations. One useful practice (when training across a fixed batch of data) is to give yourself a high enough number of iterations that early stopping doesn't play into things.

