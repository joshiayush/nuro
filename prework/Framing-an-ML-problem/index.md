# Framing an ML problem

After verifying that your problem is best solved using ML and that you have access to the data you'll need, you're ready to frame your problem in ML terms. You frame a problem in ML terms by completing the following tasks:

* Define the ideal outcome and the model's goal.
* Identify the models output.
* Define success metrics.

## Define the ideal outcome and the model's goal

Independent of the ML model, what's the ideal outcome? In other words, what is the exact task you want your product or feature to perform? This is the same statement you previously defined in the State the goal section.

Connect the model's goal to the ideal outcome by explicitly defining what you want the model to do. The following table states the ideal outcomes and the model's goal for hypothetical apps:

App	| Ideal outcome |	Model's goal
:--|:--|:--
Weather app |	Calculate precipitation in six hour increments for a geographic region. |	Predict six-hour precipitation amounts for specific geographic regions.
Video app |	Recommend useful videos. | Predict whether a user will click on a video.
Mail app | Detect spam. |	Warn the user if the email appears to be spam.
Map app |	Calculate travel time. | Predict how long it will take to travel between two points.
Banking app | Identify fraudulent transactions.	| Predict if a transaction was made by the card holder.
Dining app | Identify cuisine by a restaurant's menu.	| Predict the type of restaurant.

## Choose the right kind of model

Your choice of model type depends upon the specific context and constraints of your problem.

A [classification model](https://developers.google.com/machine-learning/glossary#classification-model) predicts what category the input data should belong, for example, whether an input should be classified as A, B, or C.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/problem-framing/images/classification-model.png' />

  <strong>Figure 1.</strong> A classification model making predictions.
</div>

Based on the model's prediction, your app might make a decision. For example, if the prediction is category A, then do X; if the prediction is category B, then do, Y; if the prediction is category C, then do Z. In some cases, the prediction is the app's output.

<div align='center'>
  <img = src='https://developers.google.com/static/machine-learning/problem-framing/images/class-product-code.png' />

  <strong>Figure 2.</strong> A classification model's output being used in the product code to make a decision.
</div>

A [regression model](https://developers.google.com/machine-learning/glossary#regression-model) predicts where to place the input data on a number line.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/problem-framing/images/regression-model.png' />

  <strong>Figure 3.</strong> A regression model making a numeric prediction.
</div>

Based on the model's prediction, your app might make a decision. For example, if the prediction falls within range A, do X; if the prediction falls within range B, do Y; if the prediction falls within range C, do Z. In some cases, the prediction is the app's output.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/problem-framing/images/regression-decision.png' />

  <strong>Figure 4.</strong> A regression model's output being used in the product code to make a decision.
</div>

Consider the following scenario:

You want to [cache](https://wikipedia.org/wiki/Cache_(computing)) videos based on their predicted popularity. In other words, if your model predicts that a video will be popular, you want to quickly serve it to users. To do so, you'll use the more effective and expensive cache. For other videos, you'll use a different cache. Your caching criteria is the following:

* If a video is predicted to get 50 or more views, you'll use the expensive cache.
* If a video is predicted to get between 30 and 50 views, you'll use the cheap cache.
* If the video is predicted to get less than 30 views, you won't cache the video.

You think a regression model is the right approach because you'll be predicting a numeric value — the number of views. However, when training the regression model, you realize that it produces the same loss for a prediction of 28 and 32 for videos that have 30 views. In other words, although your app will have very different behavior if the prediction is 28 versus 32, the model considers both predictions equally good.

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/problem-framing/images/training.png' />

  <strong>Figure 5.</strong> Training a regression model.
</div>

Regression models are unaware of product-defined thresholds. Therefore, if your app's behavior changes significantly because of small differences in a regression model's predictions, you should consider implementing a classification model instead.

In this scenario, a classification model would produce the correct behavior because a classification model would produce a higher loss for a prediction of 28 than 32. In a sense, classification models produce thresholds by default.

This scenario highlights two important points:

* __Predict the decision.__ When possible, predict the decision your app will take. In the video example, a classification model would predict the decision if the categories it classified videos into were "no cache", "cheap cache", and "expensive cache." Hiding your app's behavior from the model can cause your app to produce the wrong behavior.

* __Understand the problem's constraints.__ If your app takes different actions based on different thresholds, determine if those thresholds are fixed or dynamic.

  * __Dynamic thresholds:__ If thresholds are dynamic, use a regression model and set the thresholds limits in your app's code. This lets you easily update the thresholds while still having the model make reasonable predictions.
  * __Fixed thresholds:__ If thresholds are fixed, use a classification model and label your datasets based on the threshold limits.

In general, most cache provisioning is dynamic and the thresholds change over time. Therefore, because this is specifically a caching problem, a regression model is the best choice. However, for many problems, the thresholds will be fixed, making a classification model the best solution.

## Identify the model's output

The model's output should accomplish the task defined in the ideal outcome. If you're using a regression model, the numeric prediction should provide the data needed to accomplish the ideal outcome; if you're using a classification model, the categorical prediction should provide the data needed to accomplish the ideal outcome.

There are several subtypes of classification and regression models. Use the corresponding flowcharts to identify which subtype you are using.

__Classification flowchart__

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/problem-framing/images/classification-flowchart.png' />

  <strong>Figure 6.</strong> Diagram of a classification flowchart.
</div>

__Regression flowchart__

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/problem-framing/images/regression-flowchart.png' />

  <strong>Figure 7.</strong> Diagram of a regression flowchart.
</div>

In the weather app, the ideal outcome is to tell users how much it will rain in the next six hours. We could use a regression model that predicts the label `precipitation_amount`.

__Ideal outcome__ |	__Ideal label__
:--|:--
Tell users how much it will rain in their area in the next six hours. | `precipitation_amount`

In the weather app example, the label directly addresses the ideal outcome. In some cases, a one-to-one relationship isn't apparent between the ideal outcome and the label. For example, in the video app, the ideal outcome is to recommend useful videos. However, there's no label in the dataset called `useful_to_user`.

__Ideal outcome__ |	__Ideal label__
:--|:--
_Recommend useful videos._ | ?

Therefore, we'll need to find a proxy label.

## Proxy labels

[Proxy labels](https://developers.google.com/machine-learning/glossary#proxy-labels) substitute for labels that aren't in the dataset. Proxy labels are necessary when you can't directly measure what you want to predict. In the video app, we can't directly measure whether or not a user will find a video useful. It would be great if the dataset had a `useful` feature, and users marked all the videos that they found useful, but because the dataset doesn't, we'll need a proxy label that substitutes for usefulness.

A proxy label for usefulness might be whether or not the user will share or like the video.

__Ideal outcome__ |	__Proxy label__
:--|:--
Recommend useful videos. | __`shared`__ OR __`liked`__

Be cautious with proxy labels because they don't directly measure what you want to predict. For example, the following table outlines issues with potential proxy labels for _Recommend useful videos_:

__Proxy label__ |	__Issue__
:--|:--
Predict whether the user will click the “like” button. | Most users never click “like.”
Predict whether a video will be popular. | Not personalized. Some users might not like popular videos.
Predict whether the user will share the video. | Some users don't share videos. Sometimes, people share videos because they don't like them.
Predict whether the user will click play. |	Maximizes clickbait.
Predict how long they watch the video. | Favors long videos differentially over short videos.
Predict how many times the user will rewatch the video. |	Favors "rewatchable" videos over video genres that aren't rewatchable.

No proxy label can be a perfect substitute for your ideal outcome. All will have potential problems. Pick the one that has the least problems for your use case.

### Check your understanding

__Q1. A company wants to use ML in their health and well-being app to help people feel better. Do you think they'll need to use proxy labels to accomplish their goals?__

> Yes, the company will need to find proxy labels. Categories like happiness and well-being can’t be measured directly. Instead, they need to be approximated with respect to some other feature, like hours spent exercising per week, or time spent engaged in hobbies or with friends.

## Define the success metrics

Define the metrics you'll use to determine whether or not the ML implementation is successful. Success metrics define what you care about, like engagement or helping users take appropriate action, such as watching videos that they'll find useful. Success metrics differ from the model's evaluation metrics, like [accuracy](https://developers.google.com/machine-learning/glossary#accuracy), [precision](https://developers.google.com/machine-learning/glossary#precision), [recall](https://developers.google.com/machine-learning/glossary#recall), or [AUC](https://developers.google.com/machine-learning/glossary#auc-area-under-the-roc-curve).

For example, the weather app's success and failure metrics might be defined as the following:

__Success__	| __Failure__
:--|:--
__Users open the "Will it rain?" feature 50 percent more often than they did before.__ | __Users open the "Will it rain?" feature no more often than before.__

The video app metrics might be defined as the following:

__Success__ | __Failure__
:--|:--
__Users spend on average 20 percent more time on the site.__ | __Users spend on average no more time on site than before.__

We recommend defining ambitious success metrics. High ambitions can cause gaps between success and failure though. For example, users spending on average 10 percent more time on the site than before is neither success nor failure. The undefined gap is not what's important.

What's important is your model's capacity to move closer — or exceed — the definition of success. For instance, when analyzing the model's performance, consider the following question: Would improving the model get you closer to your defined success criteria? For example, a model might have great evaluation metrics, but not move you closer to your success criteria, indicating that even with a perfect model, you would not meet the success criteria you defined. On the other hand, a model might have poor evaluation metrics, but get you closer to your success criteria, indicating that improving the model would get you closer to success.

The following are dimensions to consider when determining if the model is worth improving:

* __Not good enough, but continue.__ The model shouldn't be used in a production environment, but over time it might be significantly improved.
* __Good enough, and continue.__ The model could be used in a production environment, and it might be further improved.
* __Good enough, but can't be made better.__ The model is in a production environment, but it is probably as good as it can be.
* __Not good enough, and never will be.__ The model should not be used in a production environment and no amount of training will probably get it there.

When deciding to improve the model, re-evaluate if the increase in resources, like engineering time and compute costs, justify the predicted improvement of the model.

After defining the success and failure metrics, you need to determine how often you'll measure them. For instance, you could measure your success metrics six days, six weeks, or six months after implementing the system.

When analyzing failure metrics, try to determine why the system failed. For example, the model might be predicting which videos users will click, but the model might start recommending clickbait titles that cause user engagement to drop off. In the weather app example, the model might accurately predict when it will rain but for too large of a geographic region.

### Check your understanding

__A fashion firm wants to sell more clothes. Someone suggests using ML to determine which clothes the firm should manufacture. They think they can train a model to determine which type of clothes are in fashion. After they train the model, they want to apply it to their catalog to decide which clothes to make.__

__Q1. How should they frame their problem in ML terms?__

> __Ideal outcome:__ Determine which products to manufacture. <br>
> __Model’s goal:__ Predict which articles of clothing are in fashion. <br>
> __Model output:__ Binary classification, in_fashion, not_in_fashion <br>
> __Success metrics:__ Sell seventy percent or more of the clothes made. <br>

## Implementing a model

When implementing a model, start simple. Most of the work in ML is on the data side, so getting a full pipeline running for a complex model is harder than iterating on the model itself. After setting up your data pipeline and implementing a simple model that uses a few features, you can iterate on creating a better model.

Simple models provide a good baseline, even if you don't end up launching them. In fact, using a model is probably better than you think. Starting simple helps you determine whether or not a complex model is even justified.

### Train your own model versus using a pre-trained model

Many pre-trained models exist for a variety of use cases and offer many advantages. However, pre-trained models only really work when the label and features match your dataset exactly. For example, if a pre-trained model uses 25 features and your dataset only includes 24 of them, the pre-trained model will most likely make bad predictions.

Commonly, ML practitioners use matching subsections of inputs from a pre-trained model for fine-tuning or transfer learning. If a pre-trained model doesn't exist for your particular use case, consider using subsections from a pre-trained model when training your own.

For information on pre-trained models, see [pre-trained models from TensorFlow Hub](https://www.tensorflow.org/hub).

## Monitoring

During problem framing, consider the monitoring and alerting infrastructure your ML solution needs.

### Model deployment

In some cases, a newly trained model might be worse than the model currently in production. If it is, you'll want to prevent it from being released into production and get an alert that your automated deployment has failed.

### Training-serving skew

If any of the incoming features used for inference have values that fall outside the distribution range of the data used in training, you'll want to be alerted because it's likely the model will make poor predictions. For example, if your model was trained to predict temperatures for equatorial cities at sea level, then your serving system should alert you of incoming data with latitudes and longitudes, and/or altitudes outside the range the model was trained on. Conversely, the serving system should alert you if the model is making predictions that are outside the distribution range that was seen during training.

### Inference server

If you're providing inferences through an RPC system, you'll want to monitor the RPC server itself and get an alert if it stops providing inferences.