# Validation Set

Partitioning a data set into a training set and test set lets you judge whether a given model will generalize well to new data. However, using only two partitions may be insufficient when doing many rounds of hyperparameter tuning.

## Another Partition

Previously, we worked on two partitions of our data set and with those two partitions, the workflow could look as follows:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/WorkflowWithTestSet.svg' />

  <strong>Figure 1. A possible workflow?</strong>
</div>

In the figure, "Tweak model" means adjusting anything about the model you can dream up—from changing the learning rate, to adding or removing features, to designing a completely new model from scratch. At the end of this workflow, you pick the model that does best on the *test set*.

Dividing the data set into two sets is a good idea, but not a panacea. You can greatly reduce your chances of overfitting by partitioning the data set into the three subsets shown in the following figure:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/PartitionThreeSets.svg' />

  <strong>Figure 2. Slicing a single data set into three subsets.</strong>
</div>

Use the **validation set** to evaluate results from the training set. Then, use the test set to double-check your evaluation after the model has "passed" the validation set. The following figure shows this new workflow:

<div align='center'>
  <img src='https://developers.google.com/static/machine-learning/crash-course/images/WorkflowWithValidationSet.svg' />

  <strong>Figure 3. A better workflow.</strong>
</div>

In this improved workflow:

1. Pick the model that does best on the validation set.
2. Double-check that model against the test set.

This is a better workflow because it creates fewer exposures to the test set.

