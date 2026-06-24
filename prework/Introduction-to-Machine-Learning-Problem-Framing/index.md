# Introduction to Machine Learning Problem Framing

_Introduction to Machine Learning Problem Framing_ teaches you how to determine if ML is a good approach for a problem and explains how to outline an ML solution.

## Overview

__Problem framing__ is the process of analyzing a problem to isolate the individual elements that need to be addressed to solve it. Problem framing helps determine your project's technical feasibility and provides a clear set of goals and success criteria. When considering an ML solution, effective problem framing can determine whether or not your product ultimately succeeds.

> Formal problem framing is the critical beginning for solving an ML problem, as it forces us to better understand both the problem and the data in order to design and build a bridge between them. - _TensorFlow engineer_

At a high level, ML problem framing consists of two distinct steps:

1. Determining whether ML is a right approach for solving a problem.
2. Framing the problem in ML terms.

### Questions

__Q1. Why is problem framing important?__

> Problem framing ensures that an ML approach is a good solution to the problem before beginning to work with data and train a model.

## Understand the problem

To understand the problem, perform the following tasks:

* State the goal for the product you are developing or refactoring.
* Determine whether the goal is best solved using ML.
* Verify you have the data required to train a model.

### State the goal

Begin by stating your goal in non-ML terms. The goal is the answer to the question "What am I trying to accomplish?"

The following table clearly states goals for hypothetical apps:

__Application__ |	__Goal__
:--|:--
Weather app	| Calculate precipitation in six-hour increments for a geographic region.
Video app |	Recommend useful videos.
Mail app |	Detect spam.
Map app	| Calculate travel time.
Banking app |	Identify fraudulent transactions.
Dining app	| Identify cuisine by a restaurant's menu.

### Clear use case for ML

Some view ML as a universal tool that can be applied to all problems. In reality, ML is a specialized tool suitable only for particular problems. You don't want to implement a complex ML solution when a simpler non-ML solution will work.

To confirm that ML is the right approach, first verify that your current non-ML solution is optimized. If you don't have a non-ML solution implemented, try solving the problem manually using a [__heuristic__](https://developers.google.com/machine-learning/glossary#heuristic).

The non-ML solution is the benchmark you'll use to determine whether ML is a good use case for your problem. Consider the following questions when comparing a non-ML approach to an ML one:

* __Quality:__ How much better do you think an ML solution can be? If you think an ML solution might be only a small improvement, that might indicate the current solution is the best one.
* __Cost and maintenance:__ How expensive is the ML solution in both the short- and long-term? In some cases, it costs significantly more in terms of compute resources and time to implement ML. Consider the following questions:
  * Can the ML solution justify the increase in cost? Note that small improvements in large systems can easily justify the cost and maintenance of implementing an ML solution.
  * How much maintenance will the solution require? In many cases, ML implementations need dedicated long-term maintenance.
  * Does your product have the resources to support training or hiring people with ML expertise?

### Questions

__Q1. Why is it important to have a non-ML solution or heuristic in place before analyzing an ML solution?__

> A non-ML solution is the benchmark to measure an ML solution against.

### Data

Data is the driving force of ML. To make good [predictions](https://developers.google.com/machine-learning/glossary#prediction), you need data that contains [features](https://developers.google.com/machine-learning/glossary#feature) with predictive power. Your data should have the following characteristics:

* __Abundant.__ The more relevant and useful examples in your [dataset](https://developers.google.com/machine-learning/glossary#data-set-or-dataset), the better your model will be.
* __Consistent and reliable.__ Having data that's consistently and reliably collected will produce a better model. For example, an ML-based weather model will benefit from data gathered over many years from the same reliable instruments.
* __Trusted.__ Understand where your data will come from. Will the data be from trusted sources you control, like logs from your product, or will it be from sources you don't have much insight into, like the output from another ML system?
* __Available.__ Make sure all inputs are available at prediction time in the correct format. If it will be difficult to obtain certain feature values at prediction time, omit those features from your datasets.
* __Correct.__ In large datasets, it's inevitable that some [labels](https://developers.google.com/machine-learning/glossary#label) will have incorrect values, but if more than a small percentage of [labels](https://developers.google.com/machine-learning/glossary#label) are incorrect, the model will produce poor predictions.
* __Representative.__ The datasets should be as representative of the real world as possible. In other words, the datasets should accurately reflect the events, user behaviors, and/or the phenomena of the real world being modeled. Training on unrepresentative datasets can cause poor performance when the model is asked to make real-world predictions.

If you can't get the data you need in the required format, your model will make poor predictions.

### Predictive power

For a model to make good predictions, the features in your dataset should have predictive power. The more correlated a feature is with a label, the more likely it is to predict it.

Some features will have more predictive power than others. For example, in a weather dataset, features such as `cloud_coverage`, `temperature`, and `dew_point` would be better predictors of rain than `moon_phase` or `day_of_week`. For the video app example, you could hypothesize that features such as `video_description`, `length` and `views` might be good predictors for which videos a user would want to watch.

Be aware that a feature's predictive power can change because the context or domain changes. For example, in the video app, a feature like `upload_date` might—in general—be weakly correlated with the label. However, in the sub-domain of gaming videos, `upload_date` might be strongly correlated with the label.

Determining which features have predictive power can be a time consuming process. You can manually explore a feature's predictive power by removing and adding it while training a model. You can automate finding a feature's predictive power by using algorithms such as [Pearson correlation](https://wikipedia.org/wiki/Pearson_correlation_coefficient), [Adjusted mutual information (AMI)](https://wikipedia.org/wiki/Adjusted_mutual_information), and [Shapley value](https://wikipedia.org/wiki/Shapley_value#In_machine_learning), which provide a numerical assessment for analyzing the predictive power of a feature.

### Check your understanding

__Q1. When analyzing your datasets, what are three key attributes you should look for?__

* > Features have predictive power for the label.
* > Representative of the real world.
* > Contains correct values.

For more guidance on analyzing and preparing your datasets, see [Data Preparation and Feature Engineering for Machine Learning](https://developers.google.com/machine-learning/data-prep).

## Predictions vs. actions

There's no value in predicting something if you can't turn the prediction into an action that helps users. That is, your product should take action from the model's output.

For example, a model that predicts whether a user will find a video useful should feed into an app that recommends useful videos. A model that predicts whether it will rain should feed into a weather app.

### Check your understanding

__Based on the following scenario, determine if using ML is the best approach to the problem.__

__An engineering team at a large organization is responsible for managing incoming phone calls.__

__The goal: To inform callers how long they'll wait on hold given the current call volume.__

__They don't have any solution in place, but they think a heuristic would be to divide the number of employees answering phones by the current number of customers on hold, and then multiply by 10 minutes. However, they know that some customers have their issues resolved in two minutes, while others can take up to 45 minutes or longer.__

__Their heuristic probably won't get them a precise enough number. They can create a dataset with the following columns: `number_of_callcenter_phones`, `user_issue`, `time_to_resolve`, `call_time`, `time_on_hold`.__

> __Use ML.__ The engineering team has a clearly defined goal. Their heuristic won't be good enough for their use case. The dataset appears to have predictive features for the label, time_on_hold.

