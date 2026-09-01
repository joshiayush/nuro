{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4hFPbXBKaU3B"
   },
   "source": [
    "# Transform Your Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wcuCqivBacpe"
   },
   "source": [
    "## Introduction to Transforming Data\n",
    "\n",
    "[**Feature engineering**](https://developers.google.com/machine-learning/glossary#feature_engineering) is the process of determining which features might be useful in training a model, and then creating those features by transforming raw data found in log files and other sources. In this section, we focus on when and how to transform numeric and categorical data, and the tradeoffs of different approaches."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mldjna5gazUD"
   },
   "source": [
    "### Reasons for Data Transformation\n",
    "\n",
    "We transform features primarily for the following reasons:\n",
    "\n",
    "1. **Mandatory transformations** for data compatibility. Examples include:\n",
    "\n",
    "  * Converting non-numeric features into numeric. You can’t do matrix multiplication on a string, so we must convert the string to some numeric representation.\n",
    "\n",
    "  * Resizing inputs to a fixed size. Linear models and feed-forward neural networks have a fixed number of input nodes, so your input data must always have the same size. For example, image models need to reshape the images in their dataset to a fixed size.\n",
    "\n",
    "2. **Optional quality transformations** that may help the model perform better. Examples include:\n",
    "\n",
    "  * Tokenization or lower-casing of text features.\n",
    "\n",
    "  * Normalized numeric features (most models perform better afterwards).\n",
    "  \n",
    "  * Allowing linear models to introduce non-linearities into the feature space."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "valnRU4kbPOt"
   },
   "source": [
    "### Where to Transform?\n",
    "\n",
    "You can apply transformations either while generating the data on disk, or within the model.\n",
    "\n",
    "**Transforming prior to training**\n",
    "\n",
    "In this approach, we perform the transformation before training. This code lives separate from your machine learning model.\n",
    "\n",
    "**Pros**\n",
    "\n",
    "* Computation is performed only once.\n",
    "* Computation can look at entire dataset to determine the transformation.\n",
    "\n",
    "**Cons**\n",
    "\n",
    "* Transformations need to be reproduced at prediction time. Beware of skew!\n",
    "* Any transformation changes require rerunning data generation, leading to slower iterations.\n",
    "\n",
    "Skew is more dangerous for cases involving online serving. In offline serving, you might be able to reuse the code that generates your training data. In online serving, the code that creates your dataset and the code used to handle live traffic are almost necessarily different, which makes it easy to introduce skew.\n",
    "\n",
    "**Transforming within the model**\n",
    "\n",
    "In this approach, the transformation is part of the model code. The model takes in untransformed data as input and will transform it within the model.\n",
    "\n",
    "**Pros**\n",
    "\n",
    "* Easy iterations. If you change the transformations, you can still use the same data files.\n",
    "* You're guaranteed the same transformations at training and prediction time.\n",
    "\n",
    "**Cons**\n",
    "\n",
    "* Expensive transforms can increase model latency.\n",
    "* Transformations are per batch.\n",
    "\n",
    "There are many considerations for transforming per batch. Suppose you want to [**normalize**](https://developers.google.com/machine-learning/glossary#normalization) a feature by its average value--that is, you want to change the feature values to have mean `0` and standard deviation `1`. When transforming inside the model, this normalization will have access to only one batch of data, not the full dataset. You can either normalize by the average value within a batch (dangerous if batches are highly variant), or precompute the average and fix it as a constant in the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yvKV_aJ6clVI"
   },
   "source": [
    "### Explore, Clean, and Visualize Your Data\n",
    "\n",
    "Explore and clean up your data before performing any transformations on it. You may have done some of the following tasks as you collected and constructed your dataset:\n",
    "\n",
    "* Examine several rows of data.\n",
    "* Check basic statistics.\n",
    "* Fix missing numerical entries.\n",
    "\n",
    "**Visualize your data frequently.** Graphs can help find anomalies or patterns that aren't clear from numerical statistics. Therefore, before getting too far into analysis, look at your data graphically, either via scatter plots or histograms. View graphs not only at the beginning of the pipeline, but also throughout transformation. Visualizations will help you continually check your assumptions and see the effects of any major changes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wZ7CrRdZc8pX"
   },
   "source": [
    "## Transforming Numeric Data\n",
    "\n",
    "You may need to apply two kinds of transformations to numeric data:\n",
    "\n",
    "* **Normalizing** - transforming numeric data to the same scale as other numeric data.\n",
    "* **Bucketing** - transforming numeric (usually continuous) data to categorical data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "XylRN07DdMB5"
   },
   "source": [
    "### Why Normalize Numeric Features?\n",
    "\n",
    "We strongly recommend normalizing a data set that has numeric features covering distinctly different ranges (for example, age and income). When different features have different ranges, gradient descent can \"bounce\" and slow down convergence. Optimizers like [Adagrad](https://developers.google.com/machine-learning/glossary#adagrad) and [Adam](https://arxiv.org/abs/1412.6980) protect against this problem by creating a separate effective learning rate for each feature.\n",
    "\n",
    "We also recommend normalizing a single numeric feature that covers a wide range, such as \"city population.\" If you don't normalize the \"city population\" feature, training the model might generate NaN errors. Unfortunately, optimizers like Adagrad and Adam can't prevent NaN errors when there is a wide range of values within a single feature.\n",
    "\n",
    "###### Warning: When normalizing, ensure that the same normalizations are applied at serving time to avoid skew."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bxaSsZEsdq1J"
   },
   "source": [
    "### Normalization\n",
    "\n",
    "The goal of normalization is to transform features to be on a similar scale. This improves the performance and training stability of the model.\n",
    "\n",
    "Four common normalization techniques may be useful:\n",
    "\n",
    "* scaling to a range\n",
    "* clipping\n",
    "* log scaling\n",
    "* z-score\n",
    "\n",
    "The following charts show the effect of each normalization technique on the distribution of the raw feature (price) on the left. The charts are based on the data set from 1985 Ward's Automotive Yearbook that is part of the [UCI Machine Learning Repository under Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/automobile)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8T2dy5KXeKIZ"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/normalizations-at-a-glance-v2.svg\" />\n",
    "\n",
    "<strong>Figure 1. Summary of normalization techniques.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5weZCDPieZtA"
   },
   "source": [
    "#### Scaling to a range\n",
    "\n",
    "[Recall from MLCC](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data) that [**scaling**](https://developers.google.com/machine-learning/glossary#scaling) means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range—usually 0 and 1 (or sometimes -1 to +1). Use the following simple formula to scale to a range:\n",
    "\n",
    "$$x' = \\dfrac{(x - x_{min})}{(x_{max} - x_{min})}$$\n",
    "\n",
    "Scaling to a range is a good choice when both of the following conditions are met:\n",
    "\n",
    "* You know the approximate upper and lower bounds on your data with few or no outliers.\n",
    "* Your data is approximately uniformly distributed across that range.\n",
    "\n",
    "A good example is age. Most age values falls between 0 and 90, and every part of the range has a substantial number of people.\n",
    "\n",
    "In contrast, you would not use scaling on income, because only a few people have very high incomes. The upper bound of the linear scale for income would be very high, and most people would be squeezed into a small part of the scale."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FAAQzLr3fGNU"
   },
   "source": [
    "#### Feature Clipping\n",
    "\n",
    "If your data set contains extreme outliers, you might try feature clipping, which caps all feature values above (or below) a certain value to fixed value. For example, you could clip all temperature values above 40 to be exactly 40.\n",
    "\n",
    "You may apply feature clipping before or after other normalizations.\n",
    "\n",
    "**Formula: Set min/max values to avoid outliers.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2-Nj5gEIfZx4"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/norm-clipping-outliers.svg\" />\n",
    "\n",
    "<strong>Figure 2. Comparing a raw distribution and its clipped version.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bU4jD5Gkfk1q"
   },
   "source": [
    "Another simple clipping strategy is to clip by z-score to +-Nσ (for example, limit to +-3σ). Note that σ is the standard deviation."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BiguCTUnfwBb"
   },
   "source": [
    "#### Log Scaling\n",
    "\n",
    "Log scaling computes the log of your values to compress a wide range to a narrow range.\n",
    "\n",
    "$$x' = log(x)$$\n",
    "\n",
    "Log scaling is helpful when a handful of your values have many points, while most other values have few points. This data distribution is known as the *power law* distribution. Movie ratings are a good example. In the chart below, most movies have very few ratings (the data in the tail), while a few have lots of ratings (the data in the head). Log scaling changes the distribution, helping to improve linear model performance."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LxRYZgihgSYx"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/norm-log-scaling-movie-ratings.svg\" />\n",
    "\n",
    "<strong>Figure 3. Comparing a raw distribution to its log.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "IKNuciLhgh0N"
   },
   "source": [
    "#### Z-Score\n",
    "\n",
    "Z-score is a variation of scaling that represents the number of standard deviations away from the mean. You would use z-score to ensure your feature distributions have mean = 0 and std = 1. It’s useful when there are a few outliers, but not so extreme that you need clipping.\n",
    "\n",
    "The formula for calculating the z-score of a point, *x*, is as follows:\n",
    "\n",
    "$$x' = \\dfrac{x - μ}{σ}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fCLbugcBg-KO"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/norm-z-score.svg\" />\n",
    "\n",
    "<strong>Figure 4. Comparing a raw distribution to its z-score distribution.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3Pm2pOyghLMH"
   },
   "source": [
    "Notice that z-score squeezes raw values that have a range of ~40000 down into a range from roughly -1 to +4.\n",
    "\n",
    "Suppose you're not sure whether the outliers truly are extreme. In this case, start with z-score unless you have feature values that you don't want the model to learn; for example, the values are the result of measurement error or a quirk."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "jonjRJ5Whxe9"
   },
   "source": [
    "### Bucketing\n",
    "\n",
    "Let's revisit the latitude plot from above."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "tkb4LTpLiqBP"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/ScalingBinningPart1.svg\" />\n",
    "\n",
    "<strong>Figure 1: House prices versus latitude.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "pnQLVZzNi4QD"
   },
   "source": [
    "In cases like the latitude example when there's no linear relationship between the feature and the result, you need to divide the latitudes into buckets to learn something different about housing values for each bucket. This transformation of numeric features into categorical features, using a set of thresholds, is called [**bucketing**](https://developers.google.com/machine-learning/glossary#bucketing) (or binning). In this bucketing example, the boundaries are equally spaced."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Cxz9MWQbjFnD"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/ScalingBinningPart2.svg\" />\n",
    "\n",
    "<strong>Figure 2: House prices versus latitude, now divided into buckets.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "pKItWrEcjWSC"
   },
   "source": [
    "### Quantile Bucketing\n",
    "\n",
    "With one feature per bucket, the model uses as much capacity for a single example in the >45000 range as for all the examples in the 5000-10000 range. This seems wasteful. How might we improve this situation?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "C_5LiJ6RjlVJ"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/bucketizing-needed.svg\" />\n",
    "\n",
    "<strong>Figure 3: Number of cars sold at different prices.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "OkPXSeafjzrP"
   },
   "source": [
    "The problem is that equally spaced buckets don’t capture this distribution well. The solution lies in creating buckets that each have the same number of points. This technique is called [**quantile bucketing**](https://developers.google.com/machine-learning/glossary#quantile_bucketing). For example, the following figure divides car prices into quantile buckets. In order to get the same number of examples in each bucket, some of the buckets encompass a narrow price span while others encompass a very wide price span."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1-kFDZh6j-rl"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/bucketizing-applied.svg\" />\n",
    "\n",
    "<strong>Figure 4: Quantile bucketing gives each bucket about the same number of cars.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SsZFa-i6kW-a"
   },
   "source": [
    "### Summary\n",
    "\n",
    "If you choose to bucketize your numerical features, be clear about how you are setting the boundaries and which type of bucketing you’re applying:\n",
    "\n",
    "* **Buckets with equally spaced boundaries:** the boundaries are fixed and encompass the same range (for example, 0-4 degrees, 5-9 degrees, and 10-14 degrees, or $5,000-$9,999, $10,000-$14,999, and $15,000-$19,999). Some buckets could contain many points, while others could have few or none.\n",
    "* **Buckets with quantile boundaries:** each bucket has the same number of points. The boundaries are not fixed and could encompass a narrow or wide span of values."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3idMSp-9rm0H"
   },
   "source": [
    "## Transforming Categorical Data\n",
    "\n",
    "Some of your features may be discrete values that aren’t in an ordered relationship. Examples include breeds of dogs, words, or postal codes. These features are known as [**categorical**](https://developers.google.com/machine-learning/glossary#categorical_data) and each value is called a category. You can represent categorical values as strings or even numbers, but you won't be able to compare these numbers or subtract them from each other.\n",
    "\n",
    "Oftentimes, you should represent features that contain integer values as categorical data instead of as numerical data. For example, consider a postal code feature in which the values are integers. If you mistakenly represent this feature numerically, then you're asking the model to find a numeric relationship between different postal codes; for example, you're expecting the model to determine that postal code 20004 is twice (or half) the signal as postal code 10002. By representing postal codes as categorical data, you enable the model to find separate signals for each individual postal code.\n",
    "\n",
    "If the number of categories of a data field is small, such as the day of the week or a limited palette of colors, you can make a unique feature for each category. For example:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JJnPY55asNwO"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/categorical-netview.svg\" />\n",
    "\n",
    "<strong>Figure 1: A unique feature for each category.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "rKoivhWqsUff"
   },
   "source": [
    "A model can then learn a separate weight for each color. For example, perhaps the model could learn that red cars are more expensive than green cars.\n",
    "\n",
    "The features can then be indexed."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7sCMLoRhswx6"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/categorical-netview-indexed.svg\" />\n",
    "\n",
    "<strong>Figure 2: Indexed features.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "v8eYjIRIs5G-"
   },
   "source": [
    "This sort of mapping is called a **vocabulary**."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "_iConiGKtAir"
   },
   "source": [
    "### Vocabulary\n",
    "\n",
    "In a vocabulary, each value represents a unique feature.\n",
    "\n",
    "| Index Number | Category |\n",
    "|--|--|\n",
    "| 0 | Red |\n",
    "| 1 | Orange |\n",
    "| 2 | Blue |\n",
    "| ... | ... |\n",
    "\n",
    "The model looks up the index from the string, assigning 1.0 to the corresponding slot in the feature vector and 0.0 to all the other slots in the feature vector."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FP1ODR7cth4N"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/vocabulary-index-sparse-feature.svg\" />\n",
    "\n",
    "<strong>Figure 3: The end-to-end process to map categories to feature vectors.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "iUpuzTMmt9a-"
   },
   "source": [
    "**Note about sparse representation**\n",
    "\n",
    "If your categories are the days of the week, you might, for example, end up representing Friday with the feature vector [0, 0, 0, 0, 1, 0, 0]. However, most implementations of ML systems will represent this vector in memory with a sparse representation. A common representation is a list of non-empty values and their corresponding indices—for example, 1.0 for the value and [4] for the index. This allows you to spend less memory storing a huge amount of 0s and allows more efficient matrix multiplication. In terms of the underlying math, the [4] is equivalent to [0, 0, 0, 0, 1, 0, 0]."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lA-tyg9WuL5K"
   },
   "source": [
    "### Out of Vocab (OOV)\n",
    "\n",
    "Just as numerical data contains outliers, categorical data does, as well. For example, consider a data set containing descriptions of cars. One of the features of this data set could be the car's color. Suppose the common car colors (black, white, gray, and so on) are well represented in this data set and you make each one of them into a category so you can learn how these different colors affect value. However, suppose this data set contains a small number of cars with eccentric colors (mauve, puce, avocado). Rather than giving each of these colors a separate category, you could lump them into a catch-all category called **Out of Vocab (OOV)**. By using OOV, the system won't waste time training on each of those rare colors."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "leUJHKe4ujOe"
   },
   "source": [
    "#### Hashing\n",
    "\n",
    "Another option is to hash every string (category) into your available index space. Hashing often causes collisions, but you rely on the model learning some shared representation of the categories in the same index that works well for the given problem.\n",
    "\n",
    "For important terms, hashing can be worse than selecting a vocabulary, because of collisions. On the other hand, hashing doesn't require you to assemble a vocabulary, which is advantageous if the feature distribution changes heavily over time."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vOYu_7I0uwFy"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/vocab-hash-string.svg\" />\n",
    "\n",
    "<strong>Figure 4: Mapping items to a vocabulary.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HL223zIQu5Rm"
   },
   "source": [
    "#### Hybrid of Hashing and Vocabulary\n",
    "\n",
    "You can take a hybrid approach and combine hashing with a vocabulary. Use a vocabulary for the most important categories in your data, but replace the OOV bucket with multiple OOV buckets, and use hashing to assign categories to buckets.\n",
    "\n",
    "The categories in the hash buckets must share an index, and the model likely won't make good predictions, but we have allocated some amount of memory to attempt to learn the categories outside of our vocabulary."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "K2j7OR-QvN4e"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/vocab-hybrid.svg\" />\n",
    "\n",
    "<strong>Figure 5: Hybrid approach combining vocabulary and hashing.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "rtIHJfrPvsVq"
   },
   "source": [
    "### Note about Embeddings\n",
    "\n",
    "An [**embedding**](https://developers.google.com/machine-learning/crash-course/glossary#embeddings) is a categorical feature represented as a continuous-valued feature. Deep models frequently convert the indices from an index to an embedding."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "AIxmsE2Lv0il"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/data-prep/images/vocabulary-index-sparse-feature-embedding.svg\" />\n",
    "\n",
    "<strong>Figure 6: Sparse feature vectors via embedding</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zsLoQkq6v-Z8"
   },
   "source": [
    "The other transformations we've discussed could be stored on disk, but embeddings are different. Since embeddings are trained, they're not a typical data transformation—they are part of the model. They're trained with other model weights, and functionally are equivalent to a layer of weights.\n",
    "\n",
    "What about pretrained embeddings? Pretrained embeddings are still typically modifiable during training, so they're still conceptually part of the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "jnbYFpdo-kJq"
   },
   "source": [
    "# Regularization for Simplicity\n",
    "\n",
    "**Regularization** means penalizing the complexity of a model to reduce overfitting."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yASiukdf-kJr"
   },
   "source": [
    "## L2 Regularization\n",
    "\n",
    "Consider the following **generalization curve**, which shows the loss for both the training set and validation set against the number of training iterations."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7ZWsH37n-kJu"
   },
   "source": [
    "<div align='center'>\n",
    "\n",
    "  <img src='https://developers.google.com/static/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg' />\n",
    "\n",
    "  <strong>Figure 1. Loss on training set and validation set.</strong>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1M3_31Zt-kJx"
   },
   "source": [
    "Figure 1 shows a model in which training loss gradually decreases, but validation loss eventually goes up. In other words, this generalization curve shows that the model is [overfitting](https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting) to the data in the training set. Channeling our inner [Ockham](https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting#ockham), perhaps we could prevent overfitting by penalizing complex models, a principle called **regularization**.\n",
    "\n",
    "In other words, instead of simply aiming to minimize loss (empirical risk minimization):\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$$\\mathrm{minimize(Loss(Data \\vert Model))}$$\n",
    "\n",
    "</div>\n",
    "\n",
    "we'll now minimize loss+complexity, which is called **structural risk minimization**.\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$$\\mathrm{minimize(Loss(Data \\vert Model) + \\mathrm{complexity(Model)})}$$\n",
    "\n",
    "</div>\n",
    "\n",
    "Our training optimization algorithm is now a function of two terms: the **loss term**, which measures how well the model fits the data, and the **regularization term**, which measures model complexity.\n",
    "\n",
    "There are two commom ways to think of model complexity:\n",
    "\n",
    "  * Model complexity as a function of the *weights* of all the features in the model.\n",
    "  * Model complexity as a function of the *total number of features* with nonzero weights.\n",
    "\n",
    "If model complexity is a function of weights, a feature weight with a high absolute value is more complex that a feature weight with a low absolute value.\n",
    "\n",
    "We can quantify complexity using the $L_{2}$ **regularization** formula, which defines the regularization term as the sum of the squares of all the feature weights:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$$\\mathrm{L_{2}\\ regularization\\ term} = \\mathrm{\\lvert \\lvert w \\rvert \\rvert}^2_{2} = w_{1}^2 + w_{2}^2 + ... + w_{n}^2$$\n",
    "\n",
    "</div>\n",
    "\n",
    "In this formula, weights close to zero have little effect on model complexity, while outlier weights can have a huge impact.\n",
    "\n",
    "For example, a linear model with the following weights:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$$\\{w_{1} = 0.2, w_{2} = 0.5, w_{3} = 5, w_{4} = 1, w_{5} = 0.25, w_{6} = 0.75\\}$$\n",
    "\n",
    "</div>\n",
    "\n",
    "Has an $L_{2}$ regularization term of 26.915:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$= w_{1}^2 + w_{2}^2 + w_{3}^2 + w_{4}^2 + w_{5}^2 + w_{6}^2$\n",
    "\n",
    "</div>\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$= 0.2^2 + 0.5^2 + 5^2 + 1^2 + 0.25^2 + 0.75^2$\n",
    "\n",
    "</div>\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$= 0.04 + 0.24 + 25 + 1 + 0.0625 + 0.5625$\n",
    "\n",
    "</div>\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$= 26.915$\n",
    "\n",
    "</div>\n",
    "\n",
    "But $w_{3}$, with a squared value of 25, contributes nearly all the complexity. The sum of the squares of all five other weights adds just 1.915 to the $L_{2}$ regularization term."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BmTbtZ7G-kJz"
   },
   "source": [
    "## Lambda\n",
    "\n",
    "Model developers tune the overall impact of the regularization term by multiplying its value by a scalar known as **lambda** (also called the **regularization rate**). That is, model devvelopers aim to do the following:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$$\\mathrm{minimize(Loss(Data \\vert Model) + \\lambda \\ complexity(Model))}$$\n",
    "\n",
    "</div>\n",
    "\n",
    "Performing $L_{2}$ regularization has the following effect on the model:\n",
    "\n",
    "  * Encourages weight values toward 0 (but not exactly 0).\n",
    "  * Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution.\n",
    "\n",
    "Increasing the lambda value strengthens the regularization effect. For example, the histogram of weights for a high value of lambda might look as shown in Figure 2."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "EFHnE9N8-kJ0"
   },
   "source": [
    "<div align='center'>\n",
    "  <img src='https://developers.google.com/static/machine-learning/crash-course/images/HighLambda.svg' height='400px' />\n",
    "\n",
    "  <strong>Figure 2. Histogram of weights.</strong>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ELxQpp_t-kJ0"
   },
   "source": [
    "Lowering the value of lambda tends to yield a flatter histogram, as shown in Figure 3."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MBbyUPoU-kJ0"
   },
   "source": [
    "<div align='center'>\n",
    "  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LowLambda.svg' />\n",
    "\n",
    "  <strong>Figure 3. Histogram of weights produced by a lower lambda value.</strong>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "GykIF2-s-kJ2"
   },
   "source": [
    "When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:\n",
    "\n",
    "* If your lambda value is too high, your model will be simple, but you run the risk of underfitting your data. Your model won't learn enough about the training data to make useful predictions.\n",
    "* If your lambda value is too low, your model will be more complex, and you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.\n",
    "\n",
    "The ideal value of lambda produces a model that generalizes well to new, previously unseen data. Unfortunately, that ideal value of lambda is data-dependent, so you'll need to do some tuning."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Q4MvaZEL-kJ4"
   },
   "source": [
    "### L2 regularization and Learning rate\n",
    "\n",
    "There's a close connection between learning rate and lambda. Strong L2 regularization values tend to drive feature weights closer to 0. Lower learning rates (with early stopping) often produce the same effect because the steps away from 0 aren't as large. Consequently, tweaking learning rate and lambda simultaneously may have confounding effects.\n",
    "\n",
    "**Early stopping** means ending training before the model fully reaches convergence. In practice, we often end up with some amount of implicit early stopping when training in an [online](https://developers.google.com/machine-learning/crash-course/production-ml-systems) (continuous) fashion. That is, some new trends just haven't had enough data yet to converge.\n",
    "\n",
    "As noted, the effects from changes to regularization parameters can be confounded with the effects from changes in learning rate or number of iterations. One useful practice (when training across a fixed batch of data) is to give yourself a high enough number of iterations that early stopping doesn't play into things."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LRASWCaz-kJ8"
   },
   "source": [
    "# Logistic Regression\n",
    "\n",
    "Instead of predicting *exactly* 0 or 1, **logistic regression** generates a probability—a value between 0 and 1, exclusive. For example, consider a logistic regression model for spam detection. If the model infers a value of 0.932 on a particular email message, it implies a 93.2% probability that the email message is spam. More precisely, it means that in the limit of *infinite* training examples, the set of examples for which the model predicts 0.932 will actually be spam 93.2% of the time and the remaining 6.8% will not."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5876rS6q-kJ-"
   },
   "source": [
    "## Calculating a Probability\n",
    "\n",
    "Many problems require a probability estimate as output. Logistic regression is an extremely efficient mechanism for calculating probabilities. Practically speaking, you can use the returned probability in either of the following two ways:\n",
    "\n",
    "* \"As is\"\n",
    "* Converted to a binary category.\n",
    "\n",
    "Let's consider how we might use the probability \"as is.\" Suppose we create a logistic regression model to predict the probability that a dog will bark during the middle of the night. We'll call that probability:\n",
    "\n",
    "$$\\mathrm{p}(\\mathrm{bark} \\vert \\mathrm{night})$$\n",
    "\n",
    "If the logistic regression model predicts $\\mathrm{p}(\\mathrm{bark} \\vert \\mathrm{night}) = 0.05$, then over a year, the dog's owners should be startled awake approximately 18 times:\n",
    "\n",
    "$$\\mathrm{startled} = \\mathrm{p}(\\mathrm{bark} \\vert \\mathrm{night}) \\cdot \\mathrm{nights}$$\n",
    "$$= 0.05 \\cdot 365$$\n",
    "$$= 18$$\n",
    "\n",
    "In many cases, you'll map the logistic regression output into the solution to a binary classification problem, in which the goal is to correctly predict one of two possible labels (e.g., \"spam\" or \"not spam\").\n",
    "\n",
    "You might be wondering how a logistic regression model can ensure output that always falls between 0 and 1. As it happens, a **sigmoid function**, defined as follows, produces output having those same characteristics:\n",
    "\n",
    "$$y = \\dfrac{1}{1 + e^{-z}}$$\n",
    "\n",
    "The sigmoid function yields the following plot:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4k1V-y_q-kKB"
   },
   "source": [
    "<div align='center'>\n",
    "  <img src='https://developers.google.com/machine-learning/crash-course/images/SigmoidFunction.png' />\n",
    "\n",
    "  <strong>Figure 1: Sigmoid function.</strong>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dNENTk_X-kKD"
   },
   "source": [
    "If $z$ represents the output of the linear layer of a model trained with logistic regression, then $sigmoid(z)$ will yield a value (a probability) between 0 and 1. In mathematical terms:\n",
    "\n",
    "$$y^ \\prime = \\dfrac{1}{1 + e^{-z}}$$\n",
    "\n",
    "where:\n",
    "\n",
    "* $y^ \\prime$ is the output of the logistic regression model for a particular example.\n",
    "\n",
    "For $z$:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$$z = b + w_{1}x_{1} + w_{2}x_{2} + ... + w_{N}x_{N}$$\n",
    "\n",
    "</div>\n",
    "\n",
    "where:\n",
    "\n",
    "  * The $w$ values are the model's learned weights, and $b$ is the bias.\n",
    "  * The $x$ values are the feature values for a particular example.\n",
    "\n",
    "Note that $z$ is also referred to as the *log-odds* because the inverse of the sigmoid states that $z$ can be defined as the log of the probability of the 1 label (eg., \"dog barks\") divided by the probability of the 0 label (eg., \"dog doesn't bark\"):\n",
    "\n",
    "$$z = \\mathrm{log} \\left( \\dfrac{y}{1- y} \\right)$$\n",
    "\n",
    "Here is the sigmoid function with ML labels:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "GtE7QiwP-kKF"
   },
   "source": [
    "<div align='center'>\n",
    "  <img src='https://developers.google.com/static/machine-learning/crash-course/images/LogisticRegressionOutput.svg' />\n",
    "\n",
    "  <strong>Figure 2: Logistic regression output.</strong>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "KpkYFUZN-kKI"
   },
   "source": [
    "## Loss and Regularization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "eXpmNjFf-kKJ"
   },
   "source": [
    "### Loss function for Logistic Regression\n",
    "\n",
    "The loss function for linear regression is squared loss. The loss function for logistic regression is **Log Loss**, which is defined as follows:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$$\\mathrm{Log\\ Loss} = \\sum_{(x,y) \\in D} -ylog(y^ \\prime) - (1 - y)log(1 - y^ \\prime)$$\n",
    "\n",
    "</div>\n",
    "\n",
    "where:\n",
    "\n",
    "* $(x,y) \\in D$ is the data set containing many labeled examples, which are $(x,y)$ pairs.\n",
    "* $y$ is the label in a labeled example. Since this is logistic regression, every value of $y$ must either be 0 or 1.\n",
    "* $y^ \\prime$ is the predicted value (somewhere between 0 and 1), given the set of features in $x$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5dp3T_Oy-kKL"
   },
   "source": [
    "### Regularization in Logistic Regression\n",
    "\n",
    "[Regularization](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/video-lecture) is extremely important in logistic regression modeling. Without regularization, the asymptotic nature of logistic regression would keep driving loss towards 0 in high dimensions. Consequently, most logistic regression models use one of the following two strategies to dampen model complexity:\n",
    "\n",
    "* $L_{2}$ regularization\n",
    "* Early stopping, that is, limiting the number of training steps or the learning rate.\n",
    "\n",
    "Imagine that you assign a unique id to each example, and map each id to its own feature. If you don't specify a regularization function, the model will become completely overfit. That's because the model would try to drive loss to zero on all examples and never get there, driving the weights for each indicator feature to +infinity or -infinity. This can happen in high dimensional data with feature crosses, when there’s a huge mass of rare crosses that happen only on one example each.\n",
    "\n",
    "Fortunately, using L2 or early stopping will prevent this problem."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gacojX8e-kKM"
   },
   "source": [
    "# Classification"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "VApDu0fp-kKN"
   },
   "source": [
    "## Thresholding\n",
    "\n",
    "Logistic regression returns a probability. You can use the returned probability \"as is\" (for example, the probability that the user will click on this ad is 0.00023) or convert the returned probability to a binary value (for example, this email is spam).\n",
    "\n",
    "A logistic regression model that returns 0.9995 for a particular email message is predicting that it is very likely to be spam. Conversely, another email message with a prediction score of 0.0003 on that same logistic regression model is very likely not spam. However, what about an email message with a prediction score of 0.6? In order to map a logistic regression value to a binary category, you must define a **classification threshold** (also called the **decision threshold**). A value above that threshold indicates \"spam\"; a value below indicates \"not spam.\" It is tempting to assume that the classification threshold should always be 0.5, but thresholds are problem-dependent, and are therefore values that you must tune."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bIigMT3Q-kKP"
   },
   "source": [
    "## True vs. False and Positive vs. Negative\n",
    "\n",
    "In this section, we'll define the primary building blocks of the metrics we'll use to evaluate classification models. But first, a fable:\n",
    "\n",
    "**An Aesop's Fable: The Boy Who Cried Wolf (compressed)**\n",
    "  \n",
    "A shepherd boy gets bored tending the town's flock. To have some fun, he cries out, \"Wolf!\" even though no wolf is in sight. The villagers run to protect the flock, but then get really mad when they realize the boy was playing a joke on them.\n",
    "\n",
    "[Iterate previous paragraph N times.]\n",
    "\n",
    "One night, the shepherd boy sees a real wolf approaching the flock and calls out, \"Wolf!\" The villagers refuse to be fooled again and stay in their houses. The hungry wolf turns the flock into lamb chops. The town goes hungry. Panic ensues.\n",
    "\n",
    "Let's make the following definitions:\n",
    "\n",
    "* \"Wolf\" is a **positive class**.\n",
    "* \"No wolf\" is a **negative class**.\n",
    "\n",
    "We can summarize our \"wolf-prediction\" model using a 2x2 [confusion matrix](https://developers.google.com/machine-learning/glossary#confusion_matrix) that depicts all four possible outcomes:\n",
    "\n",
    "* **True Positive (TP):**\n",
    "  * Reality: A wolf threatened.\n",
    "  * Shepherd said: \"Wolf.\"\n",
    "  * Outcome: Shepherd is a hero.\n",
    "\n",
    "* **False Positive (FP):**\n",
    "  * Reality: No wolf threatened.\n",
    "  * Shepherd said: \"Wolf.\"\n",
    "  * Outcome: Villagers are angry at shepherd for waking them up.\n",
    "\n",
    "* **True Negative (TN):**\n",
    "  * Reality: No wolf threatened.\n",
    "  * Shepherd said: \"No wolf.\"\n",
    "  * Outcome: Everyone is fine.\n",
    "\n",
    "* **False Negative (FN):**\n",
    "  * Reality: A wolf threatened.\n",
    "  * Shepherd said: \"No wolf.\"\n",
    "  * Outcome: The wolf ate all the sheep.\n",
    "\n",
    "A **true positive** is an outcome where the model correctly predicts the positive class. Similarly, a **true negative** is an outcome where the model correctly predicts the negative class.\n",
    "\n",
    "A **false positive** is an outcome where the model incorrectly predicts the positive class. And a **false negative** is an outcome where the model incorrectly predicts the negative class."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CEJxpSnTDTQo"
   },
   "source": [
    "## Accuracy\n",
    "\n",
    "Accuracy is one metric for evaluating classification models. Informally, **accuracy** is the fraction of predictions our model got right. Formally, accuracy has the following definition:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Accuracy} = \\dfrac{\\mathrm{Number\\ of\\ correct\\ predictions}}{\\mathrm{Total\\ number\\ of\\ predictions}}$\n",
    "\n",
    "</div>\n",
    "\n",
    "For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Accuracy} = \\dfrac{TP + TN}{TP + TN + FP + FN}$\n",
    "\n",
    "</div>\n",
    "\n",
    "Where *TP* = True Positives, *TN* = True Negatives, *FP* = False Positives, and *FN* = False Negatives."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NHxGOa5xEfI-"
   },
   "source": [
    "Let's try calculating accuracy for the following model that classified 100 tumors as [malignant](https://wikipedia.org/wiki/Malignancy) (the positive class) or [benign](https://wikipedia.org/wiki/Benign_tumor) (the negative class):\n",
    "\n",
    "* **True Positive (TP):**\n",
    "  * Reality: Malignant\n",
    "  * ML model predicted: Malignant\n",
    "  * **Number of TP results**: 1\n",
    "\n",
    "* **False Positive (FP):**\n",
    "  * Reality: Benign\n",
    "  * ML model predicted: Malignant\n",
    "  * **Number of TP results**: 1\n",
    "\n",
    "* **True Negative (TN):**\n",
    "  * Reality: Benign\n",
    "  * ML model predicted: Benign\n",
    "  * **Number of TP results**: 90\n",
    "\n",
    "* **False Negative (FN):**\n",
    "  * Reality: Malignant\n",
    "  * ML model predicted: Benign\n",
    "  * **Number of TP results**: 8"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ILz5mWZ_FS6X"
   },
   "source": [
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Accuracy} = \\dfrac{TP + TN}{TP + TN + FP + FN} = \\dfrac{1 + 90}{1 + 90 + 1 + 8} = 0.91$\n",
    "\n",
    "</div>\n",
    "\n",
    "---\n",
    "\n",
    "\n",
    "\n",
    "Accuracy comes out to 0.91, or 91% (91 correct predictions out of 100 total examples). That means our tumor classifier is doing a great job of identifying malignancies, right?\n",
    "\n",
    "Actually, let's do a closer analysis of positives and negatives to gain more insight into our model's performance.\n",
    "\n",
    "Of the 100 tumor examples, 91 are benign (90 TNs and 1 FP) and 9 are malignant (1 TP and 8 FNs).\n",
    "\n",
    "Of the 91 benign tumors, the model correctly identifies 90 as benign. That's good. However, of the 9 malignant tumors, the model only correctly identifies 1 as malignant—a terrible outcome, as 8 out of 9 malignancies go undiagnosed!\n",
    "\n",
    "While 91% accuracy may seem good at first glance, another tumor-classifier model that always predicts benign would achieve the exact same accuracy (91/100 correct predictions) on our examples. In other words, our model is no better than one that has zero predictive ability to distinguish malignant tumors from benign tumors.\n",
    "\n",
    "Accuracy alone doesn't tell the full story when you're working with a **class-imbalanced data set**, like this one, where there is a significant disparity between the number of positive and negative labels."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "bKMaum3yIz9J"
   },
   "source": [
    "## Precision and Recall"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FQdzXztXI4Y9"
   },
   "source": [
    "### Precision\n",
    "\n",
    "Precision attempts to answer the following question:\n",
    "\n",
    "```\n",
    "What proportion of positive identifications was actually correct?\n",
    "```\n",
    "\n",
    "Precision is defined as follows:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Precision} = \\dfrac{TP}{TP + FP}$\n",
    "\n",
    "</div>\n",
    "\n",
    "Let's calculate precision for our ML model from the previous section that analyzes tumors:\n",
    "\n",
    "* **True Positives (TPs): 1**\n",
    "* **False Positives (FPs): 1**\n",
    "* **False Negatives (FNs): 8**\n",
    "* **True Negatives (TNs): 90**\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Precision} = \\dfrac{TP}{TP + FP} = \\dfrac{1}{1 + 1} = 0.5$\n",
    "\n",
    "</div>\n",
    "\n",
    "Our model has a precision of 0.5—in other words, when it predicts a tumor is malignant, it is correct 50% of the time."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Gl_VaBwGJyVC"
   },
   "source": [
    "### Recall\n",
    "\n",
    "**Recall** attempts to answer the following question:\n",
    "\n",
    "```\n",
    "What proportion of actual positives was identified correctly?\n",
    "```\n",
    "\n",
    "Mathematically, recall is defined as follows:\n",
    "\n",
    "$\\mathrm{Recall} = \\dfrac{TP}{TP + FN}$\n",
    "\n",
    "Let's calculate recall for our tumor classifier:\n",
    "\n",
    "* **True Positives (TPs): 1**\n",
    "* **False Positives (FPs): 1**\n",
    "* **False Negatives (FNs): 8**\n",
    "* **True Negatives (TNs): 90**\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Recall} = \\dfrac{TP}{TP + FN} = \\dfrac{1}{1 + 8} = 0.11$\n",
    "\n",
    "</div>\n",
    "\n",
    "Our model has a recall of 0.11—in other words, it correctly identifies 11% of all malignant tumors."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HY8NyJsLKs0B"
   },
   "source": [
    "### Precision and Recall: A Tug of War\n",
    "\n",
    "To fully evaluate the effectiveness of a model, you must examine both precision and recall. Unfortunately, precision and recall are often in tension. That is, improving precision typically reduces recall and vice versa. Explore this notion by looking at the following figure, which shows 30 predictions made by an email classification model. Those to the right of the classification threshold are classified as \"spam\", while those to the left are classified as \"not spam.\"\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/PrecisionVsRecallBase.svg\" />\n",
    "\n",
    "<strong>Figure 1. Classifying email messages as spam or not spam.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lafOngnuLHoe"
   },
   "source": [
    "Let's calculate precision and recall based on the results shown in Figure 1:\n",
    "\n",
    "* **True Positives (TP): 8**\n",
    "* **False Positives (FP): 2**\n",
    "* **False Negatives (FN): 3**\n",
    "* **True Negatives (TN): 17**\n",
    "\n",
    "Precision measures the percentage of **emails flagged as spam** that were correctly classified—that is, the percentage of dots to the right of the threshold line that are green in Figure 1:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Precision} = \\dfrac{TP}{TP + FP} = \\dfrac{8}{8 + 2} = 0.8$\n",
    "\n",
    "</div>\n",
    "\n",
    "Recall measures the percentage of **actual spam emails** that were correctly classified—that is, the percentage of green dots that are to the right of the threshold line in Figure 1:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Precision} = \\dfrac{TP}{TP + FN} = \\dfrac{8}{8 + 3} = 0.73$\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1M-x4CE8MKmj"
   },
   "source": [
    "Figure 2 illustrates the effect of increasing the classification threshold.\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/PrecisionVsRecallRaiseThreshold.svg\" />\n",
    "\n",
    "<strong>Figure 2. Increasing classification threshold.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "T4S7z1PbMVP9"
   },
   "source": [
    "The number of false positives decreases, but false negatives increase. As a result, precision increases, while recall decreases:\n",
    "\n",
    "* **True Positives (TP): 7**\n",
    "* **False Positives (FP): 1**\n",
    "* **False Negatives (FN): 4**\n",
    "* **True Negatives (TN): 18**\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Precision} = \\dfrac{TP}{TP + FP} = \\dfrac{7}{7 + 1} = 0.88$\n",
    "\n",
    "</div>\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Precision} = \\dfrac{TP}{TP + FN} = \\dfrac{7}{7 + 4} = 0.64$\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HxGD4rILMrYp"
   },
   "source": [
    "Conversely, Figure 3 illustrates the effect of decreasing the classification threshold (from its original position in Figure 1).\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/PrecisionVsRecallLowerThreshold.svg\" />\n",
    "\n",
    "<strong>Figure 3. Decreasing classification threshold.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1rW7LFjjM39P"
   },
   "source": [
    "False positives increase, and false negatives decrease. As a result, this time, precision decreases and recall increases:\n",
    "\n",
    "* **True Positives (TP): 9**\n",
    "* **False Positives (FP): 3**\n",
    "* **False Negatives (FN): 2**\n",
    "* **True Negatives (TN): 16**\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Precision} = \\dfrac{TP}{TP + FP} = \\dfrac{9}{9 + 3} = 0.75$\n",
    "\n",
    "</div>\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{Precision} = \\dfrac{TP}{TP + FN} = \\dfrac{9}{9 + 2} = 0.82$\n",
    "\n",
    "</div>\n",
    "\n",
    "Various metrics have been developed that rely on both precision and recall. For example, see [F1 score](https://wikipedia.org/wiki/F1_score)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2VdE-uz2RK6V"
   },
   "source": [
    "## ROC Curve and AUC"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MNvPzfLWRXP-"
   },
   "source": [
    "### ROC Curve\n",
    "\n",
    "An **ROC curve (receiver operating characteristic curve)** is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:\n",
    "\n",
    "* True positive rate\n",
    "* False positive rate\n",
    "\n",
    "**True Positive Rate (TPR)** is a synonym for recall and is therefore defined as follows:\n",
    "\n",
    "$\\mathrm{TPR} = \\dfrac{TP}{TP + FN}$\n",
    "\n",
    "**False Positive Rate (FPR)** is defined as follows:\n",
    "\n",
    "$\\mathrm{FPR} = \\dfrac{FP}{FP + TN}$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ap-mrLWdSI6z"
   },
   "source": [
    "An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg\" />\n",
    "\n",
    "<strong>Figure 4. TP vs. FP rate at different classification thresholds.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "13PGBuFwSSxy"
   },
   "source": [
    "To compute the points in an ROC curve, we could evaluate a logistic regression model many times with different classification thresholds, but this would be inefficient. Fortunately, there's an efficient, sorting-based algorithm that can provide this information for us, called AUC."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QVUsqcpbSrag"
   },
   "source": [
    "### AUC: Area Under the ROC Curve\n",
    "\n",
    "**AUC** stands for \"Area under the ROC Curve.\" That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/AUC.svg\" />\n",
    "\n",
    "<strong>Figure 5. AUC (Area under the ROC Curve).</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "oAyf1QqgTJe9"
   },
   "source": [
    "AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. For example, given the following examples, which are arranged from left to right in ascending order of logistic regression predictions:\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/AUCPredictionsRanked.svg\" />\n",
    "\n",
    "<strong>Figure 6. Predictions ranked in ascending order of logistic regression score.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kRtN5DlETSCg"
   },
   "source": [
    "AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.\n",
    "\n",
    "AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.\n",
    "\n",
    "AUC is desirable for the following two reasons:\n",
    "\n",
    "* AUC is **scale-invariant**. It measures how well predictions are ranked, rather than their absolute values.\n",
    "* AUC is **classification-threshold-invariant**. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.\n",
    "\n",
    "However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:\n",
    "\n",
    "* **Scale invariance is not always desirable**. For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that.\n",
    "\n",
    "* **Classification-threshold invariance is not always desirable**. In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Ad-HCIAB0sso"
   },
   "source": [
    "## Prediction Bias\n",
    "\n",
    "Logistic regression predictions should be unbiased. That is:\n",
    "\n",
    "```\n",
    "\"average of predictions\" should ≈ \"average of observations\"\n",
    "```\n",
    "\n",
    "**Prediction bias** is a quantity that measures how far apart those two averages are. That is:\n",
    "\n",
    "<div class=\"math-jax-block\">\n",
    "\n",
    "$\\mathrm{prediction\\ bias} = \\mathrm{average\\ of\\ predictions} - \\mathrm{average\\ of\\ labels\\ in\\ data\\ set}$\n",
    "\n",
    "</div>\n",
    "\n",
    "<small>**Note:** \"Prediction bias\" is a different quantity than [bias](https://developers.google.com/machine-learning/crash-course/descending-into-ml) (the b in wx + b).</small>\n",
    "\n",
    "A significant nonzero prediction bias tells you there is a bug somewhere in your model, as it indicates that the model is wrong about how frequently positive labels occur."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "on8h_k3l1y7w"
   },
   "source": [
    "For example, let's say we know that on average, 1% of all emails are spam. If we don't know anything at all about a given email, we should predict that it's 1% likely to be spam. Similarly, a good spam model should predict on average that emails are 1% likely to be spam. (In other words, if we average the predicted likelihoods of each individual email being spam, the result should be 1%.) If instead, the model's average prediction is 20% likelihood of being spam, we can conclude that it exhibits prediction bias.\n",
    "\n",
    "Possible root causes of prediction bias are:\n",
    "\n",
    "* Incomplete feature set\n",
    "* Noisy data set\n",
    "* Buggy pipeline\n",
    "* Biased training sample\n",
    "* Overly strong regularization\n",
    "\n",
    "You might be tempted to correct prediction bias by post-processing the learned model—that is, by adding a **calibration layer** that adjusts your model's output to reduce the prediction bias. For example, if your model has +3% bias, you could add a calibration layer that lowers the mean prediction by 3%. However, adding a calibration layer is a bad idea for the following reasons:\n",
    "\n",
    "* You're fixing the symptom rather than the cause.\n",
    "* You've built a more brittle system that you must now keep up to date.\n",
    "\n",
    "If possible, avoid calibration layers. Projects that use calibration layers tend to become reliant on them—using calibration layers to fix all their model's sins. Ultimately, maintaining the calibration layers can become a nightmare.\n",
    "\n",
    "<small>**Note:** A good model will usually have near-zero bias. That said, a low prediction bias does not prove that your model is good. A really terrible model could have a zero prediction bias. For example, a model that just predicts the mean value for all examples would be a bad model, despite having zero bias.</small>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5mB773Um2QPw"
   },
   "source": [
    "### Bucketing and Prediction Bias\n",
    "\n",
    "Logistic regression predicts a value between 0 and 1. However, all labeled examples are either exactly 0 (meaning, for example, \"not spam\") or exactly 1 (meaning, for example, \"spam\"). Therefore, when examining prediction bias, you cannot accurately determine the prediction bias based on only one example; you must examine the prediction bias on a \"bucket\" of examples. That is, prediction bias for logistic regression only makes sense when grouping enough examples together to be able to compare a predicted value (for example, 0.392) to observed values (for example, 0.394).\n",
    "\n",
    "You can form buckets in the following ways:\n",
    "\n",
    "* Linearly breaking up the target predictions.\n",
    "* Forming quantiles.\n",
    "\n",
    "Consider the following calibration plot from a particular model. Each dot represents a bucket of 1,000 values. The axes have the following meanings:\n",
    "\n",
    "* The x-axis represents the average of values the model predicted for that bucket.\n",
    "* The y-axis represents the actual average of values in the data set for that bucket.\n",
    "\n",
    "Both axes are logarithmic scales."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PR54j06J27Bc"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/BucketingBias.svg\" />\n",
    "\n",
    "<strong>Figure 8. Prediction bias curve (logarithmic scales)</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9Md-rJL93NOp"
   },
   "source": [
    "Why are the predictions so poor for only part of the model? Here are a few possibilities:\n",
    "\n",
    "* The training set doesn't adequately represent certain subsets of the data space.\n",
    "* Some subsets of the data set are noisier than others.\n",
    "* The model is overly [regularized](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/video-lecture). (Consider reducing the value of [lambda](https://developers.google.com/machine-learning/glossary#lambda).)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-5wkzdxjHGq1"
   },
   "source": [
    "# Regularization for Sparsity"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "L_fZufC4HRF9"
   },
   "source": [
    "## L₁ Regularization\n",
    "\n",
    "Sparse vectors often contain many dimensions. Creating a [feature cross](https://developers.google.com/machine-learning/crash-course/feature-crosses/video-lecture) results in even more dimensions. Given such high-dimensional feature vectors, model size may become huge and require huge amounts of RAM.\n",
    "\n",
    "In a high-dimensional sparse vector, it would be nice to encourage weights to drop to exactly 0 where possible. A weight of exactly 0 essentially removes the corresponding feature from the model. Zeroing out features will save RAM and may reduce noise in the model.\n",
    "\n",
    "For example, consider a housing data set that covers not just California but the entire globe. Bucketing global latitude at the minute level (60 minutes per degree) gives about 10,000 dimensions in a sparse encoding; global longitude at the minute level gives about 20,000 dimensions. A feature cross of these two features would result in roughly 200,000,000 dimensions. Many of those 200,000,000 dimensions represent areas of such limited residence (for example, the middle of the ocean) that it would be difficult to use that data to generalize effectively. It would be silly to pay the RAM cost of storing these unneeded dimensions. Therefore, it would be nice to encourage the weights for the meaningless dimensions to drop to exactly 0, which would allow us to avoid paying for the storage cost of these model coefficients at inference time.\n",
    "\n",
    "We might be able to encode this idea into the optimization problem done at training time, by adding an appropriately chosen regularization term.\n",
    "\n",
    "Would $L_{2}$ regularization accomplish this task? Unfortunately not. $L_{2}$ regularization encourages weights to be small, but doesn't force them to exactly 0.0.\n",
    "\n",
    "An alternative idea would be to try and create a regularization term that penalizes the count of non-zero coefficient values in a model. Increasing this count would only be justified if there was a sufficient gain in the model's ability to fit the data. Unfortunately, while this count-based approach is intuitively appealing, it would turn our convex optimization problem into a non-convex optimization problem. So this idea, known as $L_{0}$ regularization isn't something we can use effectively in practice.\n",
    "\n",
    "However, there is a regularization term called $L_{1}$ regularization that serves as an approximation to $L_{0}$, but has the advantage of being convex and thus efficient to compute. So we can use $L_{1}$ regularization to encourage many of the uninformative coefficients in our model to be exactly 0, and thus reap RAM savings at inference time."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "F5mVYgKiISr2"
   },
   "source": [
    "### $L_{1}$ vs. $L_{2}$ regularization\n",
    "\n",
    "$L_{2}$ and $L_{1}$ penalize weights differently:\n",
    "\n",
    "* $L_{2}$ penalizes $weight^2$.\n",
    "* $L_{1}$ penalizes $|weight|$.\n",
    "\n",
    "Consequently, $L_{2}$ and $L_{1}$ have different derivatives:\n",
    "\n",
    "* The derivative of $L_{2}$ is 2 * weight.\n",
    "* The derivative of $L_{1}$ is k (a constant, whose value is independent of weight).\n",
    "\n",
    "You can think of the derivative of $L_{2}$ as a force that removes x% of the weight every time. As [Zeno](https://en.wikipedia.org/wiki/Zeno's_paradoxes#Dichotomy_paradox) knew, even if you remove x percent of a number billions of times, the diminished number will still never quite reach zero. (Zeno was less familiar with floating-point precision limitations, which could possibly produce exactly zero.) At any rate, $L_{2}$ does not normally drive weights to zero.\n",
    "\n",
    "You can think of the derivative of $L_{1}$ as a force that subtracts some constant from the weight every time. However, thanks to absolute values, $L_{1}$ has a discontinuity at 0, which causes subtraction results that cross 0 to become zeroed out. For example, if subtraction would have forced a weight from +0.1 to -0.2, $L_{1}$ will set the weight to exactly 0. Eureka, $L_{1}$ zeroed out the weight.\n",
    "\n",
    "$L_{1}$ regularization—penalizing the absolute value of all the weights—turns out to be quite efficient for wide models.\n",
    "\n",
    "Note that this description is true for a one-dimensional model.\n",
    "\n",
    "To compare the results of $L_{1}$ and $L_{2}$ regularization on a network of weights, click the play button on the [playground](https://developers.google.com/machine-learning/crash-course/regularization-for-sparsity/l1-regularization#l1-vs.-l2-regularization.)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4NqypZIFSGTh"
   },
   "source": [
    "# Neural Networks\n",
    "\n",
    "Neural networks are a more sophisticated version of feature crosses. In essence, neural networks learn the appropriate feature crosses for you."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "97rlCF6ISRmJ"
   },
   "source": [
    "## Structure\n",
    "\n",
    "If you recall from the Feature Crosses unit, the following classification problem is nonlinear:\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/machine-learning/crash-course/images/FeatureCrosses1.png\" />\n",
    "\n",
    "<strong>Figure 1. Nonlinear classification problem.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FKSI7D1_SfLx"
   },
   "source": [
    "\"Nonlinear\" means that you can't accurately predict a label with a model of the form $b + w_{1}x_{1} + w_{2}x_{2}$ In other words, the \"decision surface\" is not a line. Previously, we looked at feature crosses as one possible approach to modeling nonlinear problems.\n",
    "\n",
    "Now consider the following data set:\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/machine-learning/crash-course/images/NonLinearSpiral.png\" />\n",
    "\n",
    "<strong>Figure 2. A more difficult nonlinear classification problem.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Uut9fir2S45v"
   },
   "source": [
    "The data set shown in Figure 2 can't be solved with a linear model.\n",
    "\n",
    "To see how neural networks might help with nonlinear problems, let's start by representing a linear model as a graph:\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/linear_net.svg\" />\n",
    "\n",
    "<strong>Figure 3. Linear model as graph.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "qoqUz62BTB1q"
   },
   "source": [
    "Each blue circle represents an input feature, and the green circle represents the weighted sum of the inputs."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Y4beOJzTP6MS"
   },
   "source": [
    "### The Linear Unit\n",
    "\n",
    "So let's begin with the fundamental component of a neural network: the individual neuron. As a diagram, a **neuron** (or **unit**) with one input looks like:\n",
    "\n",
    "<div align='center'>\n",
    "\n",
    "<img src=\"https://storage.googleapis.com/kaggle-media/learn/images/mfOlDR6.png\" />\n",
    "\n",
    "<strong><i>The Linear Unit:</i> y = wx + b</strong>\n",
    "\n",
    "</div>\n",
    "\n",
    "The input is $x$. Its connection to the neuron has a **weight** which is $w$. Whenever a value flows through a connection, you multiply the value by the connection's weight. For the input $x$, what reaches the neuron is $w * x$. A neural network \"learns\" by modifying its weights.\n",
    "\n",
    "The $b$ is a special kind of weight we call it **bias**. The bias doesn't have any input data associated with it; instead, we put a $1$ in the diagram so that the value that reaches the neuron is just $b$ (since $1 * b = b$). The bias enables the neuron to modify the output independently of its inputs.\n",
    "\n",
    "The $y$ is the value the neuron ultimately outputs. To get the output, the neuron sums up all the values it receives through its connections. This neuron's activation is $y = w * x + b$, or as a formula $y=wx+b$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WYYOQ9BSRN3F"
   },
   "source": [
    "### Multiple Inputs\n",
    "\n",
    "In the previous section we saw how can we handle a single input using *The Linear Unit*, but what if we wanted to expand our model to include more inputs? That's easy enough. We can just add more input connections to the neuron, one for each additional feature. To find the output, we would multiply each input to its connection weight and then add them all together."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "AWTrr3PiR6et"
   },
   "source": [
    "<div align='center'>\n",
    "\n",
    "<img src=\"https://storage.googleapis.com/kaggle-media/learn/images/vyXSnlZ.png\" />\n",
    "\n",
    "<strong>A linear unit with three inputs.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DXYdbGAFR9EM"
   },
   "source": [
    "The formula for this neuron would be $y=w0x0+w1x1+w2x2+b$. A linear unit with two inputs will fit a plane, and a unit with more inputs than that will fit a hyperplane."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fppT1ZWPSUqu"
   },
   "source": [
    "### Layers\n",
    "\n",
    "Neural networks typically organize their neurons into **layers**. When we collect together linear units having a common set of inputs we get a **dense** layer."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yWc04mriSfTQ"
   },
   "source": [
    "<div align='center'>\n",
    "\n",
    "<img src=\"https://storage.googleapis.com/kaggle-media/learn/images/2MA4iMV.png\" />\n",
    "\n",
    "<strong>A dense layer of two linear units receiving two inputs and a bias.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b1NS6SG5SreL"
   },
   "source": [
    "You could think of each layer in a neural network as performing some kind of relatively simple transformation. Through a deep stack of layers, a neural network can transform its inputs in more and more complex ways. In a well-trained neural network, each layer is a transformation getting us a little bit closer to a solution."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "e-3bvwkeTKi-"
   },
   "source": [
    "### Hidden Layers\n",
    "\n",
    "In the model represented by the following graph, we've added a \"hidden layer\" of intermediary values. Each yellow node in the hidden layer is a weighted sum of the blue input node values. The output is a weighted sum of the yellow nodes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uKhff6YTTQiS"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/1hidden.svg\" height='400px' />\n",
    "<strong>Figure 4. Graph of two-layer model.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7HBebf58TWas"
   },
   "source": [
    "Is this model linear? Yes—its output is still a linear combination of its inputs.\n",
    "\n",
    "In the model represented by the following graph, we've added a second hidden layer of weighted sums."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "f47cwNz-Tihy"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/2hidden.svg\" height=\"400px\" />\n",
    "\n",
    "<strong>Figure 5. Graph of three-layer model.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SK9615nDTpG9"
   },
   "source": [
    "Is this model still linear? Yes, it is. When you express the output as a function of the input and simplify, you get just another weighted sum of the inputs. This sum won't effectively model the nonlinear problem in Figure 2."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dQixfYTwTdur"
   },
   "source": [
    "### Activation Functions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lrCQyVaQTjHc"
   },
   "source": [
    "<div align='center'>\n",
    "\n",
    "<img src=\"https://storage.googleapis.com/kaggle-media/learn/images/OLSUEYT.png\" />\n",
    "\n",
    "<i>Without activation functions, neural networks can only learn linear relationships. In order to fit curves, we'll need to use activation functions.</i>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1uIQpA6xT4c8"
   },
   "source": [
    "To model a nonlinear problem, we can directly introduce a nonlinearity. We can pipe each hidden layer node through a nonlinear function.\n",
    "\n",
    "In the model represented by the following graph, the value of each node in Hidden Layer 1 is transformed by a nonlinear function before being passed on to the weighted sums of the next layer. This nonlinear function is called the activation function."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CQkM3x0hUARX"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/activation.svg\" />\n",
    "\n",
    "<strong>Figure 6. Graph of three-layer model with activation function.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TPZQiL1iUPNO"
   },
   "source": [
    "An **activation function** is simply some function we apply to each of a layer's outputs (its activations). The most common is the rectifier function  $max(0,x)$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sObsNBnKUXvx"
   },
   "source": [
    "<div align='center'>\n",
    "\n",
    "<img src=\"https://storage.googleapis.com/kaggle-media/learn/images/aeIyAlF.png\" />\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3Gr42j0xUdx5"
   },
   "source": [
    "The rectifier function has a graph that's a line with the negative part \"rectified\" to zero. Applying the function to the outputs of a neuron will put a bend in the data, moving us away from simple lines.\n",
    "\n",
    "When we attach the rectifier to a linear unit, we get a **rectified linear unit** or **ReLU**. (For this reason, it's common to call the rectifier function the \"ReLU function\".) Applying a ReLU activation to a linear unit means the output becomes $max(0, w * x + b)$, which we might draw in a diagram like:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LT_El95dUthu"
   },
   "source": [
    "<div align='center'>\n",
    "\n",
    "<img src=\"https://storage.googleapis.com/kaggle-media/learn/images/eFry7Yu.png\" />\n",
    "\n",
    "<i>A rectified linear unit.</i>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Avc7ghTeUIEY"
   },
   "source": [
    "Now that we've added an activation function, adding layers has more impact. Stacking nonlinearities on nonlinearities lets us model very complicated relationships between the inputs and the predicted outputs. In brief, each layer is effectively learning a more complex, higher-level function over the raw inputs. If you'd like to develop more intuition on how this works, see [Chris Olah's excellent blog post](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "__iv1hztUUgp"
   },
   "source": [
    "### Common Activation Functions\n",
    "\n",
    "The following **sigmoid** activation function converts the weighted sum to a value between 0 and 1.\n",
    "\n",
    "$$F(x)=\\dfrac{1}{1 + e^{-x}}$$\n",
    "\n",
    "Here's a plot:\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/sigmoid.svg\" />\n",
    "\n",
    "<strong>Figure 7. Sigmoid activation function.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NNFYZdi1UsRx"
   },
   "source": [
    "The following **rectified linear unit** activation function (or **ReLU**, for short) often works a little better than a smooth function like the sigmoid, while also being significantly easier to compute.\n",
    "\n",
    "$$F(x) = max(0, x)$$\n",
    "\n",
    "The superiority of ReLU is based on empirical findings, probably driven by ReLU having a more useful range of responsiveness. A sigmoid's responsiveness falls off relatively quickly on both sides.\n",
    "\n",
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://developers.google.com/static/machine-learning/crash-course/images/relu.svg\" />\n",
    "\n",
    "<strong>Figure 8. ReLU activation function.</strong>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cxUJAbvBVQLR"
   },
   "source": [
    "In fact, any mathematical function can serve as an activation function. Suppose that $\\sigma$ represents our activation function (Relu, Sigmoid, or whatever). Consequently, the value of a node in the network is given by the following formula:\n",
    "\n",
    "$$\\sigma(w.x + b)$$\n",
    "\n",
    "TensorFlow provides out-of-the-box support for many activation functions. You can find these activation functions within TensorFlow's [list of wrappers for primitive neural network operations](https://www.tensorflow.org/api_docs/python/tf/nn). That said, we still recommend starting with ReLU."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "M6WFk9TTVCb8"
   },
   "source": [
    "### Stacking Dense Layers\n",
    "\n",
    "Now that we have some nonlinearity, let's see how we can stack layers to get complex data transformations."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7ZPkfHYbVGcn"
   },
   "source": [
    "<div align='center'>\n",
    "\n",
    "<img src=\"https://storage.googleapis.com/kaggle-media/learn/images/Y5iwFQZ.png\" />\n",
    "\n",
    "<i>A stack of dense layers makes a \"fully-connected\" network.</i>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "wJvqRos5VOlu"
   },
   "source": [
    "The layers before the output layer are sometimes called **hidden** since we never see their outputs directly.\n",
    "\n",
    "Now, notice that the final (output) layer is a linear unit (meaning, no activation function). That makes this network appropriate to a regression task, where we are trying to predict some arbitrary numeric value. Other tasks (like classification) might require an activation function on the output."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "swQ8zofkVxUd"
   },
   "source": [
    "### Summary\n",
    "\n",
    "Now our model has all the standard components of what people usually mean when they say \"neural network\":\n",
    "\n",
    "* A set of nodes, analogous to neurons, organized in layers.\n",
    "* A set of weights representing the connections between each neural network layer and the layer beneath it. The layer beneath may be another neural network layer, or some other kind of layer.\n",
    "* A set of biases, one for each node.\n",
    "* An activation function that transforms the output of each node in a layer. Different layers may have different activation functions."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "pz_VyGGXipWg"
   },
   "source": [
    "# Training Neural Networks\n",
    "\n",
    "**Backpropagation** is the most common training algorithm for neural networks. It makes gradient descent feasible for multi-layer neural networks."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FliCK2_Njxd2"
   },
   "source": [
    "## Best Practices\n",
    "\n",
    "This section explains backpropagation's failure cases and the most common way to regularize a neural network."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ryqeNBaDj34M"
   },
   "source": [
    "### Failure Cases\n",
    "\n",
    "There are a number of common ways for backpropagation to go wrong.\n",
    "\n",
    "#### Vanishing Gradients\n",
    "\n",
    "The gradients for the lower layers (closer to the input) can become very small. In deep networks, computing these gradients can involve taking the product of many small terms.\n",
    "\n",
    "When the gradients vanish toward 0 for the lower layers, these layers train very slowly, or not at all.\n",
    "\n",
    "The ReLU activation function can help prevent vanishing gradients.\n",
    "\n",
    "#### Exploding Gradients\n",
    "\n",
    "If the weights in a network are very large, then the gradients for the lower layers involve products of many large terms. In this case you can have exploding gradients: gradients that get too large to converge.\n",
    "\n",
    "Batch normalization can help prevent exploding gradients, as can lowering the learning rate.\n",
    "\n",
    "#### Dead ReLU Units\n",
    "\n",
    "Once the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck. It outputs 0 activation, contributing nothing to the network's output, and gradients can no longer flow through it during backpropagation. With a source of gradients cut off, the input to the ReLU may not ever change enough to bring the weighted sum back above 0.\n",
    "\n",
    "Lowering the learning rate can help keep ReLU units from dying."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZNVls-ewkXoE"
   },
   "source": [
    "### Dropout Regularization\n",
    "\n",
    "Yet another form of regularization, called **Dropout**, is useful for neural networks. It works by randomly \"dropping out\" unit activations in a network for a single gradient step. The more you drop out, the stronger the regularization:\n",
    "\n",
    "* 0.0 = No dropout regularization.\n",
    "* 1.0 = Drop out everything. The model learns nothing.\n",
    "* Values between 0.0 and 1.0 = More useful."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "9AHUCy0GWXlj"
   },
   "source": [
    "<div align=\"center\">\n",
    "\n",
    "<img src=\"https://storage.googleapis.com/kaggle-media/learn/images/a86utxY.gif\" />\n",
    "\n",
    "<i>Here, 50% dropout has been added between the two hidden layers.</i>\n",
    "\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BjBWVeQaW3ow"
   },
   "source": [
    "### Batch Normalization (batchnorm)\n",
    "\n",
    "With neural networks, it's generally a good idea to put all of your data on a common scale. The reason is that SGD will shift the network weights in proportion to how large an activation the data produces. Features that tend to produce activations of very different sizes can make for unstable training behavior.\n",
    "\n",
    "Now, if it's good to normalize the data before it goes into the network, maybe also normalizing inside the network would be better! In fact, we have a special kind of layer that can do this, the **batch normalization layer**. A batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation, and then also putting the data on a new scale with two trainable rescaling parameters. Batchnorm, in effect, performs a kind of coordinated rescaling of its inputs.\n",
    "\n",
    "Most often, batchnorm is added as an aid to the optimization process (though it can sometimes also help prediction performance). Models with batchnorm tend to need fewer epochs to complete training. Moreover, batchnorm can also fix various problems that can cause the training to get \"stuck\". Consider adding batch normalization to your models, especially if you're having trouble during training."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "include_colab_link": true,
   "provenance": [],
   "toc_visible": true
  },
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
  },
  "vscode": {
   "interpreter": {
    "hash": "d6f2081d9e0b7d32ecec298930d2c84146e44d12c2a59792b9d51f0a2535e83c"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
