# Transform Your Data

## Introduction to Transforming Data

[**Feature engineering**](https://developers.google.com/machine-learning/glossary#feature_engineering) is the process of determining which features might be useful in training a model, and then creating those features by transforming raw data found in log files and other sources. In this section, we focus on when and how to transform numeric and categorical data, and the tradeoffs of different approaches.

### Reasons for Data Transformation

We transform features primarily for the following reasons:

1. **Mandatory transformations** for data compatibility. Examples include:

  * Converting non-numeric features into numeric. You can’t do matrix multiplication on a string, so we must convert the string to some numeric representation.

  * Resizing inputs to a fixed size. Linear models and feed-forward neural networks have a fixed number of input nodes, so your input data must always have the same size. For example, image models need to reshape the images in their dataset to a fixed size.

2. **Optional quality transformations** that may help the model perform better. Examples include:

  * Tokenization or lower-casing of text features.

  * Normalized numeric features (most models perform better afterwards).
  
  * Allowing linear models to introduce non-linearities into the feature space.

### Where to Transform?

You can apply transformations either while generating the data on disk, or within the model.

**Transforming prior to training**

In this approach, we perform the transformation before training. This code lives separate from your machine learning model.

**Pros**

* Computation is performed only once.
* Computation can look at entire dataset to determine the transformation.

**Cons**

* Transformations need to be reproduced at prediction time. Beware of skew!
* Any transformation changes require rerunning data generation, leading to slower iterations.

Skew is more dangerous for cases involving online serving. In offline serving, you might be able to reuse the code that generates your training data. In online serving, the code that creates your dataset and the code used to handle live traffic are almost necessarily different, which makes it easy to introduce skew.

**Transforming within the model**

In this approach, the transformation is part of the model code. The model takes in untransformed data as input and will transform it within the model.

**Pros**

* Easy iterations. If you change the transformations, you can still use the same data files.
* You're guaranteed the same transformations at training and prediction time.

**Cons**

* Expensive transforms can increase model latency.
* Transformations are per batch.

There are many considerations for transforming per batch. Suppose you want to [**normalize**](https://developers.google.com/machine-learning/glossary#normalization) a feature by its average value--that is, you want to change the feature values to have mean `0` and standard deviation `1`. When transforming inside the model, this normalization will have access to only one batch of data, not the full dataset. You can either normalize by the average value within a batch (dangerous if batches are highly variant), or precompute the average and fix it as a constant in the model.

### Explore, Clean, and Visualize Your Data

Explore and clean up your data before performing any transformations on it. You may have done some of the following tasks as you collected and constructed your dataset:

* Examine several rows of data.
* Check basic statistics.
* Fix missing numerical entries.

**Visualize your data frequently.** Graphs can help find anomalies or patterns that aren't clear from numerical statistics. Therefore, before getting too far into analysis, look at your data graphically, either via scatter plots or histograms. View graphs not only at the beginning of the pipeline, but also throughout transformation. Visualizations will help you continually check your assumptions and see the effects of any major changes.

## Transforming Numeric Data

You may need to apply two kinds of transformations to numeric data:

* **Normalizing** - transforming numeric data to the same scale as other numeric data.
* **Bucketing** - transforming numeric (usually continuous) data to categorical data.

### Why Normalize Numeric Features?

We strongly recommend normalizing a data set that has numeric features covering distinctly different ranges (for example, age and income). When different features have different ranges, gradient descent can "bounce" and slow down convergence. Optimizers like [Adagrad](https://developers.google.com/machine-learning/glossary#adagrad) and [Adam](https://arxiv.org/abs/1412.6980) protect against this problem by creating a separate effective learning rate for each feature.

We also recommend normalizing a single numeric feature that covers a wide range, such as "city population." If you don't normalize the "city population" feature, training the model might generate NaN errors. Unfortunately, optimizers like Adagrad and Adam can't prevent NaN errors when there is a wide range of values within a single feature.

###### Warning: When normalizing, ensure that the same normalizations are applied at serving time to avoid skew.

### Normalization

The goal of normalization is to transform features to be on a similar scale. This improves the performance and training stability of the model.

Four common normalization techniques may be useful:

* scaling to a range
* clipping
* log scaling
* z-score

The following charts show the effect of each normalization technique on the distribution of the raw feature (price) on the left. The charts are based on the data set from 1985 Ward's Automotive Yearbook that is part of the [UCI Machine Learning Repository under Automobile Data Set](https://archive.ics.uci.edu/ml/datasets/automobile).

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/normalizations-at-a-glance-v2.svg" />

<strong>Figure 1. Summary of normalization techniques.</strong>

</div>

#### Scaling to a range

[Recall from MLCC](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data) that [**scaling**](https://developers.google.com/machine-learning/glossary#scaling) means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range—usually 0 and 1 (or sometimes -1 to +1). Use the following simple formula to scale to a range:

$$x' = \dfrac{(x - x_{min})}{(x_{max} - x_{min})}$$

Scaling to a range is a good choice when both of the following conditions are met:

* You know the approximate upper and lower bounds on your data with few or no outliers.
* Your data is approximately uniformly distributed across that range.

A good example is age. Most age values falls between 0 and 90, and every part of the range has a substantial number of people.

In contrast, you would not use scaling on income, because only a few people have very high incomes. The upper bound of the linear scale for income would be very high, and most people would be squeezed into a small part of the scale.

#### Feature Clipping

If your data set contains extreme outliers, you might try feature clipping, which caps all feature values above (or below) a certain value to fixed value. For example, you could clip all temperature values above 40 to be exactly 40.

You may apply feature clipping before or after other normalizations.

**Formula: Set min/max values to avoid outliers.**

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/norm-clipping-outliers.svg" />

<strong>Figure 2. Comparing a raw distribution and its clipped version.</strong>

</div>

Another simple clipping strategy is to clip by z-score to +-Nσ (for example, limit to +-3σ). Note that σ is the standard deviation.

#### Log Scaling

Log scaling computes the log of your values to compress a wide range to a narrow range.

$$x' = log(x)$$

Log scaling is helpful when a handful of your values have many points, while most other values have few points. This data distribution is known as the *power law* distribution. Movie ratings are a good example. In the chart below, most movies have very few ratings (the data in the tail), while a few have lots of ratings (the data in the head). Log scaling changes the distribution, helping to improve linear model performance.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/norm-log-scaling-movie-ratings.svg" />

<strong>Figure 3. Comparing a raw distribution to its log.</strong>

</div>

#### Z-Score

Z-score is a variation of scaling that represents the number of standard deviations away from the mean. You would use z-score to ensure your feature distributions have mean = 0 and std = 1. It’s useful when there are a few outliers, but not so extreme that you need clipping.

The formula for calculating the z-score of a point, *x*, is as follows:

$$x' = \dfrac{x - μ}{σ}$$

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/norm-z-score.svg" />

<strong>Figure 4. Comparing a raw distribution to its z-score distribution.</strong>

</div>

Notice that z-score squeezes raw values that have a range of ~40000 down into a range from roughly -1 to +4.

Suppose you're not sure whether the outliers truly are extreme. In this case, start with z-score unless you have feature values that you don't want the model to learn; for example, the values are the result of measurement error or a quirk.

### Bucketing

Let's revisit the latitude plot from above.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/ScalingBinningPart1.svg" />

<strong>Figure 1: House prices versus latitude.</strong>

</div>

In cases like the latitude example when there's no linear relationship between the feature and the result, you need to divide the latitudes into buckets to learn something different about housing values for each bucket. This transformation of numeric features into categorical features, using a set of thresholds, is called [**bucketing**](https://developers.google.com/machine-learning/glossary#bucketing) (or binning). In this bucketing example, the boundaries are equally spaced.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/ScalingBinningPart2.svg" />

<strong>Figure 2: House prices versus latitude, now divided into buckets.</strong>

</div>

### Quantile Bucketing

With one feature per bucket, the model uses as much capacity for a single example in the >45000 range as for all the examples in the 5000-10000 range. This seems wasteful. How might we improve this situation?

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/bucketizing-needed.svg" />

<strong>Figure 3: Number of cars sold at different prices.</strong>

</div>

The problem is that equally spaced buckets don’t capture this distribution well. The solution lies in creating buckets that each have the same number of points. This technique is called [**quantile bucketing**](https://developers.google.com/machine-learning/glossary#quantile_bucketing). For example, the following figure divides car prices into quantile buckets. In order to get the same number of examples in each bucket, some of the buckets encompass a narrow price span while others encompass a very wide price span.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/bucketizing-applied.svg" />

<strong>Figure 4: Quantile bucketing gives each bucket about the same number of cars.</strong>

</div>

### Summary

If you choose to bucketize your numerical features, be clear about how you are setting the boundaries and which type of bucketing you’re applying:

* **Buckets with equally spaced boundaries:** the boundaries are fixed and encompass the same range (for example, 0-4 degrees, 5-9 degrees, and 10-14 degrees, or $5,000-$9,999, $10,000-$14,999, and $15,000-$19,999). Some buckets could contain many points, while others could have few or none.
* **Buckets with quantile boundaries:** each bucket has the same number of points. The boundaries are not fixed and could encompass a narrow or wide span of values.

## Transforming Categorical Data

Some of your features may be discrete values that aren’t in an ordered relationship. Examples include breeds of dogs, words, or postal codes. These features are known as [**categorical**](https://developers.google.com/machine-learning/glossary#categorical_data) and each value is called a category. You can represent categorical values as strings or even numbers, but you won't be able to compare these numbers or subtract them from each other.

Oftentimes, you should represent features that contain integer values as categorical data instead of as numerical data. For example, consider a postal code feature in which the values are integers. If you mistakenly represent this feature numerically, then you're asking the model to find a numeric relationship between different postal codes; for example, you're expecting the model to determine that postal code 20004 is twice (or half) the signal as postal code 10002. By representing postal codes as categorical data, you enable the model to find separate signals for each individual postal code.

If the number of categories of a data field is small, such as the day of the week or a limited palette of colors, you can make a unique feature for each category. For example:

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/categorical-netview.svg" />

<strong>Figure 1: A unique feature for each category.</strong>

</div>

A model can then learn a separate weight for each color. For example, perhaps the model could learn that red cars are more expensive than green cars.

The features can then be indexed.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/categorical-netview-indexed.svg" />

<strong>Figure 2: Indexed features.</strong>

</div>

This sort of mapping is called a **vocabulary**.

### Vocabulary

In a vocabulary, each value represents a unique feature.

| Index Number | Category |
|--|--|
| 0 | Red |
| 1 | Orange |
| 2 | Blue |
| ... | ... |

The model looks up the index from the string, assigning 1.0 to the corresponding slot in the feature vector and 0.0 to all the other slots in the feature vector.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/vocabulary-index-sparse-feature.svg" />

<strong>Figure 3: The end-to-end process to map categories to feature vectors.</strong>

</div>

**Note about sparse representation**

If your categories are the days of the week, you might, for example, end up representing Friday with the feature vector [0, 0, 0, 0, 1, 0, 0]. However, most implementations of ML systems will represent this vector in memory with a sparse representation. A common representation is a list of non-empty values and their corresponding indices—for example, 1.0 for the value and [4] for the index. This allows you to spend less memory storing a huge amount of 0s and allows more efficient matrix multiplication. In terms of the underlying math, the [4] is equivalent to [0, 0, 0, 0, 1, 0, 0].

### Out of Vocab (OOV)

Just as numerical data contains outliers, categorical data does, as well. For example, consider a data set containing descriptions of cars. One of the features of this data set could be the car's color. Suppose the common car colors (black, white, gray, and so on) are well represented in this data set and you make each one of them into a category so you can learn how these different colors affect value. However, suppose this data set contains a small number of cars with eccentric colors (mauve, puce, avocado). Rather than giving each of these colors a separate category, you could lump them into a catch-all category called **Out of Vocab (OOV)**. By using OOV, the system won't waste time training on each of those rare colors.

#### Hashing

Another option is to hash every string (category) into your available index space. Hashing often causes collisions, but you rely on the model learning some shared representation of the categories in the same index that works well for the given problem.

For important terms, hashing can be worse than selecting a vocabulary, because of collisions. On the other hand, hashing doesn't require you to assemble a vocabulary, which is advantageous if the feature distribution changes heavily over time.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/vocab-hash-string.svg" />

<strong>Figure 4: Mapping items to a vocabulary.</strong>

</div>

#### Hybrid of Hashing and Vocabulary

You can take a hybrid approach and combine hashing with a vocabulary. Use a vocabulary for the most important categories in your data, but replace the OOV bucket with multiple OOV buckets, and use hashing to assign categories to buckets.

The categories in the hash buckets must share an index, and the model likely won't make good predictions, but we have allocated some amount of memory to attempt to learn the categories outside of our vocabulary.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/vocab-hybrid.svg" />

<strong>Figure 5: Hybrid approach combining vocabulary and hashing.</strong>

</div>

### Note about Embeddings

An [**embedding**](https://developers.google.com/machine-learning/crash-course/glossary#embeddings) is a categorical feature represented as a continuous-valued feature. Deep models frequently convert the indices from an index to an embedding.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/data-prep/images/vocabulary-index-sparse-feature-embedding.svg" />

<strong>Figure 6: Sparse feature vectors via embedding</strong>

</div>

The other transformations we've discussed could be stored on disk, but embeddings are different. Since embeddings are trained, they're not a typical data transformation—they are part of the model. They're trained with other model weights, and functionally are equivalent to a layer of weights.

What about pretrained embeddings? Pretrained embeddings are still typically modifiable during training, so they're still conceptually part of the model.

