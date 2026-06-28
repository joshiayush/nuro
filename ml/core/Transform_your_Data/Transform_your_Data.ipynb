{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "18e68764-bcef-4ca8-ad6c-ce6695e78c82",
   "metadata": {},
   "source": [
    "# Transform Your Data\n",
    "\n",
    "## Introduction to Transforming Data\n",
    "\n",
    "<span class=\"global-text-highlight\">**Feature engineering** is the process of determining which features might be useful in training a model, and then creating those features by transforming raw data found in log files and other sources.</span> In this section, we focus on when and how to transform numeric and categorical data, and the tradeoffs of different approaches.\n",
    "\n",
    "### Reasons for Data Transformation\n",
    "We transform features primarily for the following reasons:\n",
    "\n",
    "* **Mandatory transformations** for data compatibility. Examples include:\n",
    "    * *Converting non-numeric features into numeric:* You can’t do matrix multiplication on a string, so we must convert the string to some numeric representation.\n",
    "    * *Resizing inputs to a fixed size:* Linear models and feed-forward neural networks have a fixed number of input nodes, so your input data must always have the same size. For example, image models need to reshape the images in their dataset to a fixed size.\n",
    "* **Optional quality transformations** that may help the model perform better. Examples include:\n",
    "    * Tokenization or lower-casing of text features.\n",
    "    * Normalized numeric features (most models perform better afterwards).\n",
    "    * Allowing linear models to introduce non-linearities into the feature space."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90472364-02f8-4720-94c6-8781b08bccfa",
   "metadata": {},
   "source": [
    "### Where to Transform?\n",
    "\n",
    "You can apply transformations either while generating the data on disk, or within the model.\n",
    "\n",
    "#### Transforming prior to training\n",
    "In this approach, we perform the transformation before training. This code lives separate from your machine learning model.\n",
    "\n",
    "**Pros**\n",
    "* Computation is performed only once.\n",
    "* Computation can look at entire dataset to determine the transformation.\n",
    "\n",
    "**Cons**\n",
    "* <span class=\"global-text-highlight\">Transformations need to be reproduced at prediction time. Beware of skew!</span>\n",
    "* Any transformation changes require rerunning data generation, leading to slower iterations.\n",
    "\n",
    "Skew is more dangerous for cases involving online serving. In offline serving, you might be able to reuse the code that generates your training data. In online serving, the code that creates your dataset and the code used to handle live traffic are almost necessarily different, which makes it easy to introduce skew.\n",
    "\n",
    "#### Transforming within the model\n",
    "In this approach, the transformation is part of the model code. The model takes in untransformed data as input and will transform it within the model.\n",
    "\n",
    "**Pros**\n",
    "* Easy iterations.\n",
    "* If you change the transformations, you can still use the same data files.\n",
    "* <span class=\"global-text-highlight\">You're guaranteed the same transformations at training and prediction time.</span>\n",
    "\n",
    "**Cons**\n",
    "* Expensive transforms can increase model latency.\n",
    "* Transformations are per batch.\n",
    "\n",
    "There are many considerations for transforming per batch. Suppose you want to **normalize** a feature by its average value—that is, you want to change the feature values to have mean 0 and standard deviation 1. When transforming inside the model, this normalization will have access to only one batch of data, not the full dataset. You can either normalize by the average value within a batch (dangerous if batches are highly variant), or precompute the average and fix it as a constant in the model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8ea1dfa-cece-4925-b914-7da1f13a8c1f",
   "metadata": {},
   "source": [
    "### Explore, Clean, and Visualize Your Data\n",
    "\n",
    "Explore and clean up your data before performing any transformations on it. You may have done some of the following tasks as you collected and constructed your dataset:\n",
    "\n",
    "- Examine several rows of data.\n",
    "- Check basic statistics.\n",
    "- Fix missing numerical entries.\n",
    "\n",
    "**Visualize your data frequently:** <span class=\"global-text-highlight\">Graphs can help find anomalies or patterns that aren't clear from numerical statistics. Therefore, before getting too far into analysis, look at your data graphically, either via scatter plots or histograms</span>. View graphs not only at the beginning of the pipeline, but also throughout transformation. Visualizations will help you continually check your assumptions and see the effects of any major changes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0027cc48-b656-4e75-bc8b-3afbd2b780ab",
   "metadata": {},
   "source": [
    "## Transforming Numeric Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b449c89b-b6cd-4c11-bff9-6fea9d0d7b33",
   "metadata": {},
   "source": [
    "You may need to apply two kinds of transformations to numeric data:\n",
    "\n",
    "- **Normalizing** - transforming numeric data to the same scale as other numeric data.\n",
    "- **Bucketing** - transforming numeric (usually continuous) data to categorical data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b834c89d-869d-453e-a1ef-8a0177016e79",
   "metadata": {},
   "source": [
    "### Why Normalize Numeric Features?\n",
    "\n",
    "We strongly recommend normalizing a data set that has numeric features covering distinctly different ranges (for example, age and income). <span class=\"global-text-highlight\">When different features have different ranges, gradient descent can \"bounce\" and slow down convergence</span>. Optimizers like <a href=\"https://developers.google.com/machine-learning/glossary#adagrad\">Adagrad</a> and <a href=\"https://arxiv.org/abs/1412.6980\">Adam</a> protect against this problem by creating a separate effective learning rate for each feature.\n",
    "\n",
    "We also recommend normalizing a single numeric feature that covers a wide range, such as \"city population.\" If you don't normalize the \"city population\" feature, training the model might generate `NaN` errors. Unfortunately, optimizers like Adagrad and Adam can't prevent `NaN` errors when there is a wide range of values within a single feature.\n",
    "\n",
    "> **Warning:** When normalizing, ensure that the same normalizations are applied at serving time to avoid skew."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d42db21-f6ac-4f95-bff2-269caefcc719",
   "metadata": {},
   "source": [
    "#### Normalization\n",
    "\n",
    "The goal of normalization is to transform features to be on a similar scale. This improves the performance and training stability of the model.\n",
    "\n",
    "Four common normalization techniques may be useful:\n",
    "\n",
    "- scaling to a range\n",
    "- clipping\n",
    "- log scaling\n",
    "- z-score\n",
    "\n",
    "The following charts show the effect of each normalization technique on the distribution of the raw feature (price) on the left. The charts are based on the data set from 1985 Ward's Automotive Yearbook that is part of the <a href=\"https://archive.ics.uci.edu/ml/datasets/automobile\">UCI Machine Learning Repository under Automobile Data Set</a>.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/normalizations-at-a-glance-v2.svg\" alt=\"Normalization techniques\">\n",
    "    <span><b>Figure 1.</b> Summary of normalization techniques.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2f69c7f-39ad-4977-97cb-98757517bc6f",
   "metadata": {},
   "source": [
    "#### Scaling to a range\n",
    "\n",
    "Scaling means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range—usually 0 and 1 (or sometimes -1 to +1). Use the following simple formula to scale to a range:\n",
    "\n",
    "$$x' = \\frac{(x - x_{min})}{(x_{max} - x_{min})}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "115f9019-546d-42ea-9fa5-676cb87168f2",
   "metadata": {},
   "source": [
    "Scaling to a range is a good choice when both of the following conditions are met:\n",
    "\n",
    "- You know the approximate upper and lower bounds on your data with few or no outliers.\n",
    "- Your data is approximately uniformly distributed across that range.\n",
    "\n",
    "A good example is age. Most age values falls between 0 and 90, and every part of the range has a substantial number of people.\n",
    "\n",
    "In contrast, you would not use scaling on income, because only a few people have very high incomes. The upper bound of the linear scale for income would be very high, and most people would be squeezed into a small part of the scale."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a35c1fb-04b5-4588-9665-bdad2726a343",
   "metadata": {},
   "source": [
    "#### Feature Clipping\n",
    "\n",
    "<span class=\"global-text-highlight\">**Feature clipping** caps all feature values above (or below) a certain value to a fixed value if your data set contains extreme outliers.</span> For example, you could clip all temperature values above 40 to be exactly 40.\n",
    "\n",
    "* You may apply feature clipping before or after other normalizations.\n",
    "* **Formula:** Set min/max values to avoid outliers.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/norm-clipping-outliers.svg\" alt=\"Two-panel distribution comparison of roomsPerPerson. The left plot shows a raw distribution with a long tail of outliers stretching towards 55. The right plot shows the same feature capped to a maximum of 4.0, where outliers are accumulated at 4.0.\" />\n",
    "    <span><b>Figure 2.</b> Comparing a raw distribution and its clipped version.</span>\n",
    "</div>\n",
    "\n",
    "Another simple clipping strategy is to clip by z-score to `+-Nσ` (for example, limit to `+-3σ`). Note that `σ` is the standard deviation."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6dfc72da-4cb8-41bd-8ada-3148ee4f8473",
   "metadata": {},
   "source": [
    "#### Log Scaling\n",
    "\n",
    "<span class=\"global-text-highlight\">**Log scaling** computes the log of your values to compress a wide range to a narrow range.</span>\n",
    "\n",
    "$$x' = log(x)$$\n",
    "\n",
    "* Log scaling is helpful when a handful of your values have many points, while most other values have few points. This data distribution is known as the `power law` distribution.\n",
    "* **Example:** Movie ratings are a good example. Most movies have very few ratings (the data in the tail), while a few have lots of ratings (the data in the head).\n",
    "* Log scaling changes the distribution, helping to improve linear model performance.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/norm-log-scaling-movie-ratings.svg\" alt=\"Two-panel chart comparing ratings per movie. The left plot shows a raw power law distribution with a huge peak near 0 and a long tail stretching to 3500. The right plot shows the log-transformed ratings per movie, compressing the wide range into a narrow, more balanced distribution from 0 to 9.\" />\n",
    "    <span><b>Figure 3.</b> Comparing a raw distribution to its log.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b733c4a6-c3be-4f1d-8843-b5fe556b58a7",
   "metadata": {},
   "source": [
    "#### Z-Score\n",
    "\n",
    "<span class=\"global-text-highlight\">**Z-score** is a variation of scaling that represents the number of standard deviations away from the mean.</span>\n",
    "\n",
    "* You would use z-score to ensure your feature distributions have `mean = 0` and `std = 1`.\n",
    "* It’s useful when there are a few outliers, but not so extreme that you need clipping.\n",
    "\n",
    "$$x' = \\frac{x - \\mu}{\\sigma}$$\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/norm-z-score.svg\" alt=\"Two-panel histogram comparing raw price to its normalized z-score. The left plot displays the raw price feature extending up to a range of 45000. The right plot shows the distribution squeezed down into a range from roughly -1 to +4.\" />\n",
    "    <span><b>Figure 4.</b> Comparing a raw distribution to its z-score distribution.</span>\n",
    "</div>\n",
    "\n",
    "Notice that z-score squeezes raw values that have a range of ~40000 down into a range from roughly -1 to +4.\n",
    "\n",
    "Suppose you're not sure whether the outliers truly are extreme. In this case, start with z-score unless you have feature values that you don't want the model to learn; for example, the values are the result of measurement error or a quirk."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d4cbea3-d4db-49e5-9472-1799735378c1",
   "metadata": {},
   "source": [
    "## Binning (Bucketing)\n",
    "\n",
    "<span class=\"global-text-highlight\">**Binning** (also called **bucketing**) is a feature engineering technique that groups different numerical subranges into bins or buckets.</span> In many cases, binning turns numerical data into categorical data. For example, consider a feature named `X` whose lowest value is 15 and highest value is 425. Using binning, you could represent `X` with the following five bins:\n",
    "\n",
    "* Bin 1: 15 to 34\n",
    "* Bin 2: 35 to 117\n",
    "* Bin 3: 118 to 279\n",
    "* Bin 4: 280 to 392\n",
    "* Bin 5: 393 to 425\n",
    "\n",
    "Bin 1 spans the range 15 to 34, so every value of `X` between 15 and 34 ends up in Bin 1. A model trained on these bins will react no differently to `X` values of 17 and 29 since both values are in Bin 1.\n",
    "\n",
    "The feature vector represents the five bins as follows:\n",
    "\n",
    "| Bin number | Range | Feature vector |\n",
    "| :--- | :--- | :--- |\n",
    "| 1 | 15-34 | `[1.0, 0.0, 0.0, 0.0, 0.0]` |\n",
    "| 2 | 35-117 | `[0.0, 1.0, 0.0, 0.0, 0.0]` |\n",
    "| 3 | 118-279 | `[0.0, 0.0, 1.0, 0.0, 0.0]` |\n",
    "| 4 | 280-392 | `[0.0, 0.0, 0.0, 1.0, 0.0]` |\n",
    "| 5 | 393-425 | `[0.0, 0.0, 0.0, 0.0, 1.0]` |\n",
    "\n",
    "Even though `X` is a single column in the dataset, binning causes a model to treat `X` as *five* separate features. Therefore, the model learns separate weights for each bin."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85415a2a-6f92-4b6d-8fef-1ee23fb6e607",
   "metadata": {},
   "source": [
    "Binning is a good alternative to scaling or clipping when either of the following conditions is met:\n",
    "\n",
    "* The overall *linear* relationship between the feature and the label is weak or nonexistent.\n",
    "* When the feature values are clustered.\n",
    "\n",
    "Binning can feel counterintuitive, given that the model in the previous example treats the values 37 and 115 identically. But when a feature appears more *clumpy* than linear, binning is a much better way to represent the data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8bbeb5fb-d524-43ac-91cd-ed9fd377f94e",
   "metadata": {},
   "source": [
    "### Binning example: number of shoppers versus temperature\n",
    "\n",
    "Suppose you are creating a model that predicts the number of shoppers by the outside temperature for that day. Here's a plot of the temperature versus the number of shoppers:\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/binning_temperature_vs_shoppers.png\" alt=\"A scatter plot of 45 points.\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 9.</b> A scatter plot of 45 points.</span>\n",
    "</div>\n",
    "\n",
    "The plot shows, not surprisingly, that the number of shoppers was highest when the temperature was most comfortable.\n",
    "\n",
    "You could represent the feature as raw values: a temperature of 35.0 in the dataset would be 35.0 in the feature vector. Is that the best idea?\n",
    "\n",
    "During training, a linear regression model learns a single weight for each feature. Therefore, if temperature is represented as a single feature, then a temperature of 35.0 would have five times the influence (or one-fifth the influence) in a prediction as a temperature of 7.0. However, the plot doesn't really show any sort of linear relationship between the label and the feature value.\n",
    "\n",
    "The graph suggests three clusters in the following subranges:\n",
    "\n",
    "* Bin 1 is the temperature range 4-11.\n",
    "* Bin 2 is the temperature range 12-26.\n",
    "* Bin 3 is the temperature range 27-36.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/binning_temperature_vs_shoppers_divided_into_3_bins.png\" alt=\"The scatter plot divided into three bins.\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 10.</b> The scatter plot divided into three bins.</span>\n",
    "</div>\n",
    "\n",
    "The model learns separate weights for each bin.\n",
    "\n",
    "While it's possible to create more than three bins, even a separate bin for each temperature reading, this is often a bad idea for the following reasons:\n",
    "\n",
    "* A model can only learn the association between a bin and a label if there are enough examples in that bin. In the given example, each of the 3 bins contains at least 10 examples, which *might* be sufficient for training. With 33 separate bins, none of the bins would contain enough examples for the model to train on.\n",
    "* A separate bin for each temperature results in 33 separate temperature features. However, you typically should *minimize* the number of features in a model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9fe9e931-b927-46bc-a4a5-3a310a8efaa3",
   "metadata": {},
   "source": [
    "### Quantile Bucketing\n",
    "\n",
    "**Quantile bucketing** creates bucketing boundaries such that the number of examples in each bucket is exactly or nearly equal. Quantile bucketing mostly hides the outliers.\n",
    "\n",
    "To illustrate the problem that quantile bucketing solves, consider the equally spaced buckets shown in the following figure, where each of the ten buckets represents a span of exactly 10,000 dollars. Notice that the bucket from 0 to 10,000 contains dozens of examples but the bucket from 50,000 to 60,000 contains only 5 examples. Consequently, the model has enough examples to train on the 0 to 10,000 bucket but not enough examples to train on for the 50,000 to 60,000 bucket.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/NeedsQuantileBucketing.png\" alt=\"Histogram of car prices showing equally spaced bucket boundaries.\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 13.</b> Some buckets contain a lot of cars; other buckets contain very few cars.</span>\n",
    "</div>\n",
    "\n",
    "In contrast, the following figure uses quantile bucketing to divide car prices into bins with approximately the same number of examples in each bucket. Notice that some of the bins encompass a narrow price span while others encompass a very wide price span.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/QuantileBucketing.png\" alt=\"Histogram of car prices using quantile bucketing boundaries.\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 14.</b> Quantile bucketing gives each bucket about the same number of cars.</span>\n",
    "</div>"
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
