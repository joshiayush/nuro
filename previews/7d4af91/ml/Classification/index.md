# Classification

## Thresholding

Logistic regression returns a probability. You can use the returned probability "as is" (for example, the probability that the user will click on this ad is 0.00023) or convert the returned probability to a binary value (for example, this email is spam).

A logistic regression model that returns 0.9995 for a particular email message is predicting that it is very likely to be spam. Conversely, another email message with a prediction score of 0.0003 on that same logistic regression model is very likely not spam. However, what about an email message with a prediction score of 0.6? In order to map a logistic regression value to a binary category, you must define a **classification threshold** (also called the **decision threshold**). A value above that threshold indicates "spam"; a value below indicates "not spam." It is tempting to assume that the classification threshold should always be 0.5, but thresholds are problem-dependent, and are therefore values that you must tune.

## True vs. False and Positive vs. Negative

In this section, we'll define the primary building blocks of the metrics we'll use to evaluate classification models. But first, a fable:

**An Aesop's Fable: The Boy Who Cried Wolf (compressed)**
  
A shepherd boy gets bored tending the town's flock. To have some fun, he cries out, "Wolf!" even though no wolf is in sight. The villagers run to protect the flock, but then get really mad when they realize the boy was playing a joke on them.

[Iterate previous paragraph N times.]

One night, the shepherd boy sees a real wolf approaching the flock and calls out, "Wolf!" The villagers refuse to be fooled again and stay in their houses. The hungry wolf turns the flock into lamb chops. The town goes hungry. Panic ensues.

Let's make the following definitions:

* "Wolf" is a **positive class**.
* "No wolf" is a **negative class**.

We can summarize our "wolf-prediction" model using a 2x2 [confusion matrix](https://developers.google.com/machine-learning/glossary#confusion_matrix) that depicts all four possible outcomes:

* **True Positive (TP):**
  * Reality: A wolf threatened.
  * Shepherd said: "Wolf."
  * Outcome: Shepherd is a hero.

* **False Positive (FP):**
  * Reality: No wolf threatened.
  * Shepherd said: "Wolf."
  * Outcome: Villagers are angry at shepherd for waking them up.

* **True Negative (TN):**
  * Reality: No wolf threatened.
  * Shepherd said: "No wolf."
  * Outcome: Everyone is fine.

* **False Negative (FN):**
  * Reality: A wolf threatened.
  * Shepherd said: "No wolf."
  * Outcome: The wolf ate all the sheep.

A **true positive** is an outcome where the model correctly predicts the positive class. Similarly, a **true negative** is an outcome where the model correctly predicts the negative class.

A **false positive** is an outcome where the model incorrectly predicts the positive class. And a **false negative** is an outcome where the model incorrectly predicts the negative class.

## Accuracy

Accuracy is one metric for evaluating classification models. Informally, **accuracy** is the fraction of predictions our model got right. Formally, accuracy has the following definition:

<div class="math-jax-block">

$\mathrm{Accuracy} = \dfrac{\mathrm{Number\ of\ correct\ predictions}}{\mathrm{Total\ number\ of\ predictions}}$

</div>

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

<div class="math-jax-block">

$\mathrm{Accuracy} = \dfrac{TP + TN}{TP + TN + FP + FN}$

</div>

Where *TP* = True Positives, *TN* = True Negatives, *FP* = False Positives, and *FN* = False Negatives.

Let's try calculating accuracy for the following model that classified 100 tumors as [malignant](https://wikipedia.org/wiki/Malignancy) (the positive class) or [benign](https://wikipedia.org/wiki/Benign_tumor) (the negative class):

* **True Positive (TP):**
  * Reality: Malignant
  * ML model predicted: Malignant
  * **Number of TP results**: 1

* **False Positive (FP):**
  * Reality: Benign
  * ML model predicted: Malignant
  * **Number of TP results**: 1

* **True Negative (TN):**
  * Reality: Benign
  * ML model predicted: Benign
  * **Number of TP results**: 90

* **False Negative (FN):**
  * Reality: Malignant
  * ML model predicted: Benign
  * **Number of TP results**: 8

<div class="math-jax-block">

$\mathrm{Accuracy} = \dfrac{TP + TN}{TP + TN + FP + FN} = \dfrac{1 + 90}{1 + 90 + 1 + 8} = 0.91$

</div>

---



Accuracy comes out to 0.91, or 91% (91 correct predictions out of 100 total examples). That means our tumor classifier is doing a great job of identifying malignancies, right?

Actually, let's do a closer analysis of positives and negatives to gain more insight into our model's performance.

Of the 100 tumor examples, 91 are benign (90 TNs and 1 FP) and 9 are malignant (1 TP and 8 FNs).

Of the 91 benign tumors, the model correctly identifies 90 as benign. That's good. However, of the 9 malignant tumors, the model only correctly identifies 1 as malignant—a terrible outcome, as 8 out of 9 malignancies go undiagnosed!

While 91% accuracy may seem good at first glance, another tumor-classifier model that always predicts benign would achieve the exact same accuracy (91/100 correct predictions) on our examples. In other words, our model is no better than one that has zero predictive ability to distinguish malignant tumors from benign tumors.

Accuracy alone doesn't tell the full story when you're working with a **class-imbalanced data set**, like this one, where there is a significant disparity between the number of positive and negative labels.

## Precision and Recall

### Precision

Precision attempts to answer the following question:

```
What proportion of positive identifications was actually correct?
```

Precision is defined as follows:

<div class="math-jax-block">

$\mathrm{Precision} = \dfrac{TP}{TP + FP}$

</div>

Let's calculate precision for our ML model from the previous section that analyzes tumors:

* **True Positives (TPs): 1**
* **False Positives (FPs): 1**
* **False Negatives (FNs): 8**
* **True Negatives (TNs): 90**

<div class="math-jax-block">

$\mathrm{Precision} = \dfrac{TP}{TP + FP} = \dfrac{1}{1 + 1} = 0.5$

</div>

Our model has a precision of 0.5—in other words, when it predicts a tumor is malignant, it is correct 50% of the time.

### Recall

**Recall** attempts to answer the following question:

```
What proportion of actual positives was identified correctly?
```

Mathematically, recall is defined as follows:

$\mathrm{Recall} = \dfrac{TP}{TP + FN}$

Let's calculate recall for our tumor classifier:

* **True Positives (TPs): 1**
* **False Positives (FPs): 1**
* **False Negatives (FNs): 8**
* **True Negatives (TNs): 90**

<div class="math-jax-block">

$\mathrm{Recall} = \dfrac{TP}{TP + FN} = \dfrac{1}{1 + 8} = 0.11$

</div>

Our model has a recall of 0.11—in other words, it correctly identifies 11% of all malignant tumors.

### Precision and Recall: A Tug of War

To fully evaluate the effectiveness of a model, you must examine both precision and recall. Unfortunately, precision and recall are often in tension. That is, improving precision typically reduces recall and vice versa. Explore this notion by looking at the following figure, which shows 30 predictions made by an email classification model. Those to the right of the classification threshold are classified as "spam", while those to the left are classified as "not spam."

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/PrecisionVsRecallBase.svg" />

<strong>Figure 1. Classifying email messages as spam or not spam.</strong>

</div>

Let's calculate precision and recall based on the results shown in Figure 1:

* **True Positives (TP): 8**
* **False Positives (FP): 2**
* **False Negatives (FN): 3**
* **True Negatives (TN): 17**

Precision measures the percentage of **emails flagged as spam** that were correctly classified—that is, the percentage of dots to the right of the threshold line that are green in Figure 1:

<div class="math-jax-block">

$\mathrm{Precision} = \dfrac{TP}{TP + FP} = \dfrac{8}{8 + 2} = 0.8$

</div>

Recall measures the percentage of **actual spam emails** that were correctly classified—that is, the percentage of green dots that are to the right of the threshold line in Figure 1:

<div class="math-jax-block">

$\mathrm{Precision} = \dfrac{TP}{TP + FN} = \dfrac{8}{8 + 3} = 0.73$

</div>

Figure 2 illustrates the effect of increasing the classification threshold.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/PrecisionVsRecallRaiseThreshold.svg" />

<strong>Figure 2. Increasing classification threshold.</strong>

</div>

The number of false positives decreases, but false negatives increase. As a result, precision increases, while recall decreases:

* **True Positives (TP): 7**
* **False Positives (FP): 1**
* **False Negatives (FN): 4**
* **True Negatives (TN): 18**

<div class="math-jax-block">

$\mathrm{Precision} = \dfrac{TP}{TP + FP} = \dfrac{7}{7 + 1} = 0.88$

</div>

<div class="math-jax-block">

$\mathrm{Precision} = \dfrac{TP}{TP + FN} = \dfrac{7}{7 + 4} = 0.64$

</div>

Conversely, Figure 3 illustrates the effect of decreasing the classification threshold (from its original position in Figure 1).

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/PrecisionVsRecallLowerThreshold.svg" />

<strong>Figure 3. Decreasing classification threshold.</strong>

</div>

False positives increase, and false negatives decrease. As a result, this time, precision decreases and recall increases:

* **True Positives (TP): 9**
* **False Positives (FP): 3**
* **False Negatives (FN): 2**
* **True Negatives (TN): 16**

<div class="math-jax-block">

$\mathrm{Precision} = \dfrac{TP}{TP + FP} = \dfrac{9}{9 + 3} = 0.75$

</div>

<div class="math-jax-block">

$\mathrm{Precision} = \dfrac{TP}{TP + FN} = \dfrac{9}{9 + 2} = 0.82$

</div>

Various metrics have been developed that rely on both precision and recall. For example, see [F1 score](https://wikipedia.org/wiki/F1_score).

## ROC Curve and AUC

### ROC Curve

An **ROC curve (receiver operating characteristic curve)** is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

* True positive rate
* False positive rate

**True Positive Rate (TPR)** is a synonym for recall and is therefore defined as follows:

$\mathrm{TPR} = \dfrac{TP}{TP + FN}$

**False Positive Rate (FPR)** is defined as follows:

$\mathrm{FPR} = \dfrac{FP}{FP + TN}$

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/ROCCurve.svg" />

<strong>Figure 4. TP vs. FP rate at different classification thresholds.</strong>

</div>

To compute the points in an ROC curve, we could evaluate a logistic regression model many times with different classification thresholds, but this would be inefficient. Fortunately, there's an efficient, sorting-based algorithm that can provide this information for us, called AUC.

### AUC: Area Under the ROC Curve

**AUC** stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/AUC.svg" />

<strong>Figure 5. AUC (Area under the ROC Curve).</strong>

</div>

AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. For example, given the following examples, which are arranged from left to right in ascending order of logistic regression predictions:

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/AUCPredictionsRanked.svg" />

<strong>Figure 6. Predictions ranked in ascending order of logistic regression score.</strong>

</div>

AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.

AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

AUC is desirable for the following two reasons:

* AUC is **scale-invariant**. It measures how well predictions are ranked, rather than their absolute values.
* AUC is **classification-threshold-invariant**. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:

* **Scale invariance is not always desirable**. For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that.

* **Classification-threshold invariance is not always desirable**. In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.

## Prediction Bias

Logistic regression predictions should be unbiased. That is:

```
"average of predictions" should ≈ "average of observations"
```

**Prediction bias** is a quantity that measures how far apart those two averages are. That is:

<div class="math-jax-block">

$\mathrm{prediction\ bias} = \mathrm{average\ of\ predictions} - \mathrm{average\ of\ labels\ in\ data\ set}$

</div>

<small>**Note:** "Prediction bias" is a different quantity than [bias](https://developers.google.com/machine-learning/crash-course/descending-into-ml) (the b in wx + b).</small>

A significant nonzero prediction bias tells you there is a bug somewhere in your model, as it indicates that the model is wrong about how frequently positive labels occur.

For example, let's say we know that on average, 1% of all emails are spam. If we don't know anything at all about a given email, we should predict that it's 1% likely to be spam. Similarly, a good spam model should predict on average that emails are 1% likely to be spam. (In other words, if we average the predicted likelihoods of each individual email being spam, the result should be 1%.) If instead, the model's average prediction is 20% likelihood of being spam, we can conclude that it exhibits prediction bias.

Possible root causes of prediction bias are:

* Incomplete feature set
* Noisy data set
* Buggy pipeline
* Biased training sample
* Overly strong regularization

You might be tempted to correct prediction bias by post-processing the learned model—that is, by adding a **calibration layer** that adjusts your model's output to reduce the prediction bias. For example, if your model has +3% bias, you could add a calibration layer that lowers the mean prediction by 3%. However, adding a calibration layer is a bad idea for the following reasons:

* You're fixing the symptom rather than the cause.
* You've built a more brittle system that you must now keep up to date.

If possible, avoid calibration layers. Projects that use calibration layers tend to become reliant on them—using calibration layers to fix all their model's sins. Ultimately, maintaining the calibration layers can become a nightmare.

<small>**Note:** A good model will usually have near-zero bias. That said, a low prediction bias does not prove that your model is good. A really terrible model could have a zero prediction bias. For example, a model that just predicts the mean value for all examples would be a bad model, despite having zero bias.</small>

### Bucketing and Prediction Bias

Logistic regression predicts a value between 0 and 1. However, all labeled examples are either exactly 0 (meaning, for example, "not spam") or exactly 1 (meaning, for example, "spam"). Therefore, when examining prediction bias, you cannot accurately determine the prediction bias based on only one example; you must examine the prediction bias on a "bucket" of examples. That is, prediction bias for logistic regression only makes sense when grouping enough examples together to be able to compare a predicted value (for example, 0.392) to observed values (for example, 0.394).

You can form buckets in the following ways:

* Linearly breaking up the target predictions.
* Forming quantiles.

Consider the following calibration plot from a particular model. Each dot represents a bucket of 1,000 values. The axes have the following meanings:

* The x-axis represents the average of values the model predicted for that bucket.
* The y-axis represents the actual average of values in the data set for that bucket.

Both axes are logarithmic scales.

<div align="center">

<img src="https://developers.google.com/static/machine-learning/crash-course/images/BucketingBias.svg" />

<strong>Figure 8. Prediction bias curve (logarithmic scales)</strong>

</div>

Why are the predictions so poor for only part of the model? Here are a few possibilities:

* The training set doesn't adequately represent certain subsets of the data space.
* Some subsets of the data set are noisier than others.
* The model is overly [regularized](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/video-lecture). (Consider reducing the value of [lambda](https://developers.google.com/machine-learning/glossary#lambda).)

