{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "5f28c1b2-d2ed-4e53-bfe9-916c65184a15",
   "metadata": {},
   "source": [
    "# Linear Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2dac1672-00c7-4dc5-9e33-718f1ce9105b",
   "metadata": {},
   "source": [
    "Linear regression is a <span class=\"global-text-highlight\">statistical technique used to find the relationship between variables</span>. In an ML context, linear regression finds the relationship between features and a label.\n",
    "\n",
    "For example, suppose we want to predict a car's fuel efficiency in miles per gallon based on how heavy the car is, and we have the following dataset:\n",
    "\n",
    "| Pounds in 1000s (feature) |\tMiles per gallon (label) |\n",
    "|---|---|\n",
    "| 3.5 | 18 |\n",
    "| 3.69 | 15 |\n",
    "| 3.44 | 18 |\n",
    "| 3.43 | 16 |\n",
    "| 4.34 | 15 |\n",
    "| 4.42 | 14 |\n",
    "| 2.37 | 24 |"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40277977-e412-4ae7-967d-ba9f6c3e871f",
   "metadata": {},
   "source": [
    "If we plotted these points, we'd get the following graph:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e4ae054-e5ef-41cb-a28f-ae951a55acfb",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/car-data-points.png\" alt=\"car-data-points\" />\n",
    "    <span><b>Figure 1.</b> Car heaviness (in pounds) versus miles per gallon rating. As a car gets heavier, its miles per gallon rating generally decreases.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15cf838a-319c-4552-90d0-57923ab69bb6",
   "metadata": {},
   "source": [
    "We could create our own model by drawing a best fit line through the points:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b743d82-f318-4cfa-8d6a-220b8be169e2",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/car-data-points-with-model.png\" alt=\"car-data-points-with-model\" />\n",
    "    <span><b>Figure 2.</b> A best fit line drawn through the data from the previous figure.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aa200c9a-e7f8-405c-bdba-8a7ba660459a",
   "metadata": {},
   "source": [
    "## Equation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7da030ec-446c-41ec-b78e-74dcdd897197",
   "metadata": {},
   "source": [
    "In algebraic terms, the model would be defined as $y=mx+c$, where\n",
    "\n",
    "- $y$ is miles per gallon—the value we want to predict.\n",
    "- $m$ is the slope of the line.\n",
    "- $x$ is pounds—our input value.\n",
    "- $b$ is the y-intercept.\n",
    "\n",
    "In ML, we write the equation for a linear regression model as follows:\n",
    "\n",
    "$$y^`=b+w_{1}x_{1}$$\n",
    "\n",
    "where:\n",
    "\n",
    "- $y$ is the predicted label—the output.\n",
    "- $b$ is the bias of the model. <span class=\"global-text-highlight mjx-chtml\">Bias is the same concept as the y-intercept in the algebraic equation for a line. In ML, bias is sometimes referred to as $w_{0}$. Bias is a parameter of the model and is calculated during training.</span>\n",
    "- $w_{1}$ is the weight of the feature. <span class=\"global-text-highlight\">Weight is the same concept as the slope  in the algebraic equation for a line. Weight is a parameter of the model and is calculated during training.</span>\n",
    "- $x_{1}$ is a feature—the input."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b459073-21f9-4d64-ab7b-ae4a7ddb59d4",
   "metadata": {},
   "source": [
    "During training, the model calculates the weight and bias that produce the best model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50ad34dd-2a81-47f9-9520-6ab98b2358af",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/linear-regression-equation.png\" alt=\"linear regression equation\" />\n",
    "    <span><b>Figure 3.</b> Mathematical representation of a linear model.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "67e820e4-d7a6-4ff7-89b0-d775b78f6519",
   "metadata": {},
   "source": [
    "In our example, we'd calculate the weight and bias from the line we drew. The bias is 34 (where the line intersects the y-axis), and the weight is –4.6 (the slope of the line). The model would be defined as $y^`=34+(-4.6)(x_{1})$, and we could use it to make predictions. For instance, using this model, a 4,000-pound car would have a predicted fuel efficiency of 15.6 miles per gallon."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bc969916-384c-475b-b3a6-5f77e91deed5",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/linear-regression-model-prediction.png\" alt=\"linear-regression-model-prediction\" />\n",
    "    <span><b>Figure 4.</b> Using the model, a 4,000-pound car has a predicted fuel efficiency of 15.6 miles per gallon.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "60c26777-5130-452e-b93e-f470be4fb8e0",
   "metadata": {},
   "source": [
    "### Models with multiple features\n",
    "\n",
    "Although the example in this section uses only one feature—the heaviness of the car—a more sophisticated model might rely on multiple features, each having a separate weight ($w_{1}$, $w_{2}$, etc.). For example, a model that relies on five features would be written as follows:\n",
    "\n",
    "$$y^`=b+w_{1}x_{1}+w_{2}x_{2}+w_{3}x_{3}+w_{4}x_{4}+w_{5}x_{5}$$\n",
    "\n",
    "For example, a model that predicts gas mileage could additionally use features such as the following:\n",
    "\n",
    "- Engine displacement\n",
    "- Acceleration\n",
    "- Number of cylinders\n",
    "- Horsepower\n",
    "\n",
    "This model would be written as follows:\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/linear-regression-equation-multiple-features.png\" alt=\"linear-regression-equation-multiple-features\" />\n",
    "    <span><b>Figure 5.</b> A model with five features to predict a car's miles per gallon rating.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c650436-dd19-4afc-ba57-7afa98a20ca8",
   "metadata": {},
   "source": [
    "By graphing a couple of these additional features, we can see that they also have a linear relationship to the label, miles per gallon:\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/linear-regression-displacement-feature.png\" alt=\"linear-regression-displacement-feature\" />\n",
    "    <span><b>Figure 6.</b> A car's displacement in cubic centimeters and its miles per gallon rating. As a car's engine gets bigger, its miles per gallon rating generally decreases.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90797f31-9777-457e-ab1d-efb1a6f8eb91",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/linear-regression-acceleration-feature.png\" alt=\"linear-regression-acceleration-feature\" />\n",
    "    <span><b>Figure 7.</b> A car's acceleration and its miles per gallon rating. As a car's acceleration takes longer, the miles per gallon rating generally increases.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd9912f3-2e7b-4b3b-9ebb-bbe036dd945c",
   "metadata": {},
   "source": [
    "## Loss\n",
    "\n",
    "<span class=\"global-text-highlight\">Loss is a numerical metric that describes how wrong a model's predictions are.</span> Loss measures the distance between the model's predictions and the actual labels. The goal of training a model is to minimize the loss, reducing it to its lowest possible value.\n",
    "\n",
    "In the following image, you can visualize loss as arrows drawn from the data points to the model. The arrows show how far the model's predictions are from the actual values.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/linear-regression-loss-lines.png\" alt=\"linear-regression-loss-lines\" />\n",
    "    <span><b>Figure 8.</b> Loss is measured from the actual value to the predicted value.<span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "51c2a6df-d449-4746-81b1-ec389a06df33",
   "metadata": {},
   "source": [
    "### Distance of loss\n",
    "\n",
    "In statistics and machine learning, loss measures the difference between the predicted and actual values. Loss focuses on the distance between the values, not the direction. For example, if a model predicts 2, but the actual value is 5, we don't care that the loss is negative (2 – 5= –3). Instead, we care that the distance between the values is 3. Thus, all methods for calculating loss remove the sign.\n",
    "\n",
    "The two most common methods to remove the sign are the following:\n",
    "\n",
    "- Take the absolute value of the difference between the actual value and the prediction.\n",
    "- Square the difference between the actual value and the prediction."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cdea063b-40da-46ee-b23e-37c3a2676efc",
   "metadata": {},
   "source": [
    "## Types of loss\n",
    "\n",
    "In linear regression, there are <span class=\"global-text-highlight\">five main types of loss</span>, which are outlined in the following table.\n",
    "\n",
    "| Loss type | Definition | Equation |\n",
    "| :--- | :--- | :--- |\n",
    "| $\\text{L}_1$ loss | The sum of the absolute values of the difference between the predicted values and the actual values. | $\\sum \\|\\textit{actual value} - \\textit{predicted value}\\|$ |\n",
    "| Mean absolute error (MAE) | The average of $\\text{L}_1$ losses across a set of $N$ examples. | $\\frac{1}{N} \\sum \\|\\textit{actual value} - \\textit{predicted value}\\|$ |\n",
    "| $\\text{L}_2$ loss | The sum of the squared difference between the predicted values and the actual values. | $\\sum (\\textit{actual value} - \\textit{predicted value})^2$ |\n",
    "| Mean squared error (MSE) | The average of $\\text{L}_2$ losses across a set of $N$ examples. | $\\frac{1}{N} \\sum (\\textit{actual value} - \\textit{predicted value})^2$ |\n",
    "| Root mean squared error (RMSE) | The square root of the mean squared error (MSE). | $\\sqrt{\\frac{1}{N} \\sum (\\textit{actual value} - \\textit{predicted value})^2}$ |"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "beec0825-fb48-4e92-a570-fe760ca999a2",
   "metadata": {},
   "source": [
    "The functional difference between $L_{1}$ loss and $L_{2}$ loss (or between MAE/RMSE and MSE) is squaring. When the difference between the prediction and label is large, squaring makes the loss even larger. When the difference is small (less than 1), squaring makes the loss even smaller.\n",
    "\n",
    "Loss metrics like MAE and RMSE may be preferable to $L_{2}$ loss or MSE in some use cases because they tend to be more human-interpretable, as they measure error using the same scale as the model's predicted value.\n",
    "\n",
    "When processing multiple examples at once, we recommend averaging the losses across all the examples, whether using MAE, MSE, or RMSE."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b52f403-359d-4c81-82a2-06112849af04",
   "metadata": {},
   "source": [
    "### Calculating loss example\n",
    "\n",
    "In the previous section, we created the following model to <span class=\"global-text-highlight\">predict fuel efficiency based on car heaviness</span>:\n",
    "\n",
    "* Model: $y' = 34 + (-4.6)(x_1)$\n",
    "    * Weight: -4.6\n",
    "    * Bias: 34\n",
    "\n",
    "If the model predicts that a 2,370-pound car gets 23.1 miles per gallon, but it actually gets 24 miles per gallon, we would calculate the $L_2$ loss as follows:\n",
    "\n",
    "> **Note:** The formula uses 2.37 because the graphs are scaled to 1000s of pounds."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6343126d-6daf-45d4-827b-22cc1be51538",
   "metadata": {},
   "source": [
    "| Value | Equation | Result |\n",
    "| :--- | :--- | :--- |\n",
    "| Prediction | $\\textit{bias} + (\\textit{weight} * \\textit{feature value})$<br><br>$34 + (-4.6 * 2.37)$ | 23.1 |\n",
    "| Actual value | $\\textit{label}$ | 24 |\n",
    "| $\\text{L}_2$ loss | $(\\textit{actual value} - \\textit{predicted value})^2$<br><br>$(24 - 23.1)^2$ | 0.81 |\n",
    "\n",
    "In this example, the <span class=\"global-text-highlight\">$\\text{L}_2$ loss for that single data point is 0.81</span>."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f04b8698-3116-4459-ac30-0343c93b4a27",
   "metadata": {},
   "source": [
    "## Choosing a loss\n",
    "\n",
    "Deciding whether to use MAE or MSE can depend on the dataset and the way you want to handle certain predictions. Most feature values in a dataset typically fall within a distinct range. For example, cars are normally between 2000 and 5000 pounds and get between 8 to 50 miles per gallon. An 8,000-pound car, or a car that gets 100 miles per gallon, is outside the typical range and would be considered an outlier.\n",
    "\n",
    "An outlier can also refer to how far off a model's predictions are from the real values. For instance, 3,000 pounds is within the typical car-weight range, and 40 miles per gallon is within the typical fuel-efficiency range. However, a 3,000-pound car that gets 40 miles per gallon would be an outlier in terms of the model's prediction because the model would predict that a 3,000-pound car would get around 20 miles per gallon.\n",
    "\n",
    "When choosing the best loss function, consider how you want the model to treat outliers. For instance, <span class=\"global-text-highlight\">MSE moves the model more toward the outliers, while MAE doesn't. $L_2$ loss incurs a much higher penalty for an outlier than $L_1$ loss</span>. For example, the following images show a model trained using MAE and a model trained using MSE. The red line represents a fully trained model that will be used to make predictions. The outliers are closer to the model trained with MSE than to the model trained with MAE."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "24739676-1c1c-4bf9-8121-03f31aa3413b",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/model-mse.png\" alt=\"model-mse\" />\n",
    "    <span><b>Figure 9.</b> MSE loss moves the model closer to the outliers.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e2dbbfd-5a13-47bd-8892-89d44911f6c4",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/model-mae.png\" alt=\"model-mae\" />\n",
    "    <span><b>Figure 10.</b> MAE loss keeps the model farther from the outliers.</span>\n",
    "</div>\n",
    "\n",
    "Note the relationship between the model and the data:\n",
    "\n",
    "* **MSE.** <span class=\"global-text-highlight\">The model is closer to the outliers but further away from most of the other data points</span>.\n",
    "* **MAE.** <span class=\"global-text-highlight\">The model is further away from the outliers but closer to most of the other data points</span>."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f76c31d3-a6f1-4587-b1b3-eb1d529d3f20",
   "metadata": {},
   "source": [
    "## Hyperparameters\n",
    "\n",
    "<span class=\"global-text-highlight\">**Hyperparameters** are variables that control different aspects of training</span>. Three common hyperparameters are:\n",
    "\n",
    "* **Learning rate**\n",
    "* **Batch size**\n",
    "* **Epochs**\n",
    "\n",
    "In contrast, <span class=\"global-text-highlight\">**parameters** are the variables, like the weights and bias, that are part of the model itself</span>. In other words, hyperparameters are values that you control; parameters are values that the model calculates during training.\n",
    "\n",
    "### Learning rate\n",
    "\n",
    "<span class=\"global-text-highlight\">**Learning rate** is a floating point number you set that influences how quickly the model converges</span>. If the learning rate is too low, the model can take a long time to converge. However, if the learning rate is too high, the model never converges, but instead bounces around the weights and bias that minimize the loss. The goal is to pick a learning rate that's not too high nor too low so that the model converges quickly.\n",
    "\n",
    "The learning rate determines the magnitude of the changes to make to the weights and bias during each step of the gradient descent process. The model multiplies the gradient by the learning rate to determine the model's parameters (weight and bias values) for the next iteration. In the third step of gradient descent, the \"small amount\" to move in the direction of negative slope refers to the learning rate.\n",
    "\n",
    "The difference between the old model parameters and the new model parameters is proportional to the slope of the loss function. For example, if the slope is large, the model takes a large step. If small, it takes a small step. For example, if the gradient's magnitude is 2.5 and the learning rate is 0.01, then the model will change the parameter by 0.025.\n",
    "\n",
    "The ideal learning rate helps the model to converge within a reasonable number of iterations. In Figure 20, the loss curve shows the model significantly improving during the first 20 iterations before beginning to converge:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "88ced2b8-ae59-4aa9-b149-ab33b3ed3fb9",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/hyperparameters-correct-lr.png\" alt=\"correct-lr\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 20.</b> Loss graph showing a model trained with a learning rate that converges quickly.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c5de2c35-fe85-40f8-a6e4-6636fa92ff2b",
   "metadata": {},
   "source": [
    "In contrast, <span class=\"global-text-highlight\">a learning rate that's too small can take too many iterations to converge</span>. In Figure 21, the loss curve shows the model making only minor improvements after each iteration:\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/hyperparameters-small-lr.png\" alt=\"small-lr\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 21.</b> Loss graph showing a model trained with a small learning rate.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eee514ca-3042-41d0-9ec2-07f15b2c9c3d",
   "metadata": {},
   "source": [
    "A learning rate that's too large never converges because each iteration either causes the loss to bounce around or continually increase. In **Figure 22**, the loss curve shows the model decreasing and then increasing loss after each iteration, and in Figure 23 the loss increases at later iterations:\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/hyperparameters-high-lr.png\" alt=\"higher-lr\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 22.</b> Loss graph showing a model trained with a learning rate that's too big, where the loss curve fluctuates wildly, going up and down as the iterations increase.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5e3c6bd7-157e-430b-85cc-9c3f52cf519e",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/hyperparameters-increasing-loss.png\" alt=\"Loss curve exploding due to large learning rate\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 23.</b> Loss graph showing a model trained with a <span class=\"global-text-highlight\">learning rate that's too big, where the loss curve drastically increases in later iterations</span>.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df7bc9a1-577f-4a6a-a8b7-d15b1bb783ea",
   "metadata": {},
   "source": [
    "### Batch size\n",
    "\n",
    "**Batch size** is a <span class=\"global-text-highlight\">hyperparameter that refers to the number of examples the model processes before updating its weights and bias</span>. You might think that the model should calculate the loss for *every* example in the dataset before updating the weights and bias. However, when a dataset contains hundreds of thousands or even millions of examples, using the full batch isn't practical.\n",
    "\n",
    "Two common techniques to get the right gradient on *average* without needing to look at every example in the dataset before updating the weights and bias are **stochastic gradient descent** and **mini-batch stochastic gradient descent**:\n",
    "\n",
    "* **Stochastic gradient descent (SGD)**: Stochastic gradient descent uses only a <span class=\"global-text-highlight\">single example (a batch size of one) per iteration</span>. Given enough iterations, SGD works but is very noisy. \"Noise\" refers to variations during training that cause the loss to increase rather than decrease during an iteration. The term \"stochastic\" indicates that the one example comprising each batch is chosen at random.\n",
    "\n",
    "Notice in the following image how loss slightly fluctuates as the model updates its weights and bias using SGD, which can lead to noise in the loss graph:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "57ae694d-a672-4da6-bd50-99a97b988792",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/noisy-gradient.png\" alt=\"Model trained with stochastic gradient descent (SGD) showing noise in the loss curve\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 24.</b> Model trained with stochastic gradient descent (SGD) showing noise in the loss curve.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c06fd3d-b0f5-4d54-b3d0-1b60b56781f0",
   "metadata": {},
   "source": [
    "Note that using stochastic gradient descent can <span class=\"global-text-highlight\">produce noise throughout the entire loss curve, not just near convergence</span>."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a6a9059-538b-4141-91ba-a952966601e2",
   "metadata": {},
   "source": [
    "* **Mini-batch stochastic gradient descent (mini-batch SGD)**: Mini-batch stochastic gradient descent is a <span class=\"global-text-highlight\">compromise between full-batch and SGD</span>. For *N* number of data points, the batch size can be any number greater than 1 and less than *N*. The model chooses the examples included in each batch at random, averages their gradients, and then updates the weights and bias once per iteration.\n",
    "\n",
    "Determining the number of examples for each batch depends on the dataset and the available compute resources. <span class=\"global-text-highlight\">In general, small batch sizes behaves like SGD, and larger batch sizes behaves like full-batch gradient descent.</span>\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/mini-batch-sgd.png\" alt=\"Model trained with mini-batch SGD\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 25.</b> Model trained with mini-batch SGD.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22bf4147-6798-4baa-8994-067295a35187",
   "metadata": {},
   "source": [
    "When training a model, you might think that noise is an undesirable characteristic that should be eliminated. However, <span class=\"global-text-highlight\">a certain amount of noise can be a good thing</span>. In later modules, you'll learn how <span class=\"global-text-highlight\">noise can help a model generalize better</span> and find the optimal weights and bias in a neural network."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7a97e449-ca9d-48c8-b5c6-72aacedcafc7",
   "metadata": {},
   "source": [
    "### Epochs\n",
    "\n",
    "During training, an **epoch** means that the model has processed <span class=\"global-text-highlight\">every example in the training set *once*</span>. For example, given a training set with 1,000 examples and a mini-batch size of 100 examples, it will take the model 10 iterations to complete one epoch.\n",
    "\n",
    "Training typically requires many epochs. That is, the system needs to process every example in the training set multiple times.\n",
    "\n",
    "The number of epochs is a hyperparameter you set before the model begins training. In many cases, you'll need to experiment with how many epochs it takes for the model to converge. <span class=\"global-text-highlight\">In general, more epochs produces a better model, but also takes more time to train.</span>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e9f7d8a-4b22-4fc2-8cef-cc426967d0d7",
   "metadata": {},
   "source": [
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/batch-size.png\" alt=\"Full batch versus mini batch comparison\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 26.</b> <span class=\"global-text-highlight\">Full batch versus mini batch</span>.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23cfb342-dc01-4092-a2ee-4a86e03d7193",
   "metadata": {},
   "source": [
    "The following table describes how batch size and epochs relate to the number of times a model updates its parameters.\n",
    "\n",
    "| Batch type | When weights and bias updates occur |\n",
    "| :--- | :--- |\n",
    "| Full batch | <span class=\"global-text-highlight\">After the model looks at all the examples in the dataset.</span> For instance, if a dataset contains 1,000 examples and the model trains for 20 epochs, the model updates the weights and bias 20 times, once per epoch. |\n",
    "| Stochastic gradient descent | <span class=\"global-text-highlight\">After the model looks at a single example from the dataset.</span> For instance, if a dataset contains 1,000 examples and trains for 20 epochs, the model updates the weights and bias 20,000 times. |\n",
    "| Mini-batch stochastic gradient descent | <span class=\"global-text-highlight\">After the model looks at the examples in each batch.</span> For instance, if a dataset contains 1,000 examples, and the batch size is 100, and the model trains for 20 epochs, the model updates the weights and bias 200 times. |"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
