{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a98bd101-ce95-4e93-904d-345dbcf97e34",
   "metadata": {},
   "source": [
    "# Training, Validation, and Test Sets\n",
    "\n",
    "You should test a model against a *different* set of examples than those used to train the model. As you'll learn a little later, <span class=\"global-text-highlight\">testing on different examples is stronger proof of your model's fitness</span> than testing on the same set of examples. Where do you get those different examples? Traditionally in machine learning, you get those different examples by <span class=\"global-text-highlight\">splitting the original dataset</span>. You might assume, therefore, that you should split the original dataset into two subsets:\n",
    "\n",
    "* A **training set** <span class=\"global-text-highlight\">that the model trains on</span>.\n",
    "* A **test set** <span class=\"global-text-highlight\">for evaluation of the trained model</span>.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/dataset_split_training_test.png\" alt=\"Figure 8 diagram showing a horizontal bar representing a dataset split into a large training set and a smaller test set.\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 8.</b> Not an optimal split.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "241f81d4-9879-4ec8-b143-961a0206b52d",
   "metadata": {},
   "source": [
    "Dividing the dataset into two sets is a decent idea, but a better approach is to <span class=\"global-text-highlight\">divide the dataset into *three* subsets</span>. In addition to the training set and the test set, the third subset is:\n",
    "\n",
    "* A **validation set** <span class=\"global-text-highlight\">performs the initial testing on the model as it is being trained</span>.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/three_subset_split.png\" alt=\"Figure 9 horizontal bar chart showing a dataset split into a large training set, a smaller validation set, and an equally small test set.\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 9.</b> A much better split.</span>\n",
    "</div>\n",
    "\n",
    "Use the **validation set** to <span class=\"global-text-highlight\">evaluate results from the training set</span>. After repeated use of the validation set suggests that your model is making good predictions, <span class=\"global-text-highlight\">use the test set to double-check your model</span>.\n",
    "\n",
    "The following figure suggests this workflow. In the figure, \"Tweak model\" means adjusting anything about the model — from changing the learning rate, to adding or removing features, to designing a completely new model from scratch.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/development_testing_workflow.svg\" alt=\"Figure 10 flowchart illustrating the iterative cycle of training on a training set, evaluating on a validation set, and tweaking the model, followed by picking the best model and confirming results on a test set.\" />\n",
    "    <br/>\n",
    "    <span><b>Figure 10.</b> A good workflow for development and testing.</span>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2076951-86b3-499e-b3b7-b8a8f3251fca",
   "metadata": {},
   "source": [
    "The workflow shown in Figure 10 is optimal, but even with that workflow, <span class=\"global-text-highlight\">test sets and validation sets still \"wear out\" with repeated use</span>. That is, <span class=\"global-text-highlight\">the more you use the same data to make decisions about hyperparameter settings or other model improvements, the less confidence that the model will make good predictions on new data</span>. For this reason, it's a good idea to <span class=\"global-text-highlight\">collect more data to \"refresh\" the test set and validation set</span>. Starting anew is a great reset."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "131afe6d-634a-400f-9563-abdec2cdada3",
   "metadata": {},
   "source": [
    "## Additional problems with test sets\n",
    "\n",
    "After splitting a dataset into training, validation, and test sets, <span class=\"global-text-highlight\">delete any examples in the validation set or test set that are duplicates of examples in the training set</span>. <span class=\"global-text-highlight\">The only fair test of a model is against new examples, not duplicates</span>.\n",
    "\n",
    "For example, consider a model that predicts whether an email is spam, using the subject line, email body, and sender's email address as features. Suppose you divide the data into training and test sets, with an 80-20 split. After training, the model achieves 99% precision on both the training set and the test set. You'd probably expect a lower precision on the test set, so you take another look at the data and discover that many of the examples in the test set are duplicates of examples in the training set. The problem is that you neglected to scrub duplicate entries for the same spam email from your input database before splitting the data. You've <span class=\"global-text-highlight\">inadvertently trained on some of your test data</span>.\n",
    "\n",
    "In summary, a good test set or validation set meets all of the following criteria:\n",
    "\n",
    "* <span class=\"global-text-highlight\">Large enough to yield statistically significant testing results</span>.\n",
    "* <span class=\"global-text-highlight\">Representative of the dataset as a whole</span>. In other words, don't pick a test set with different characteristics than the training set.\n",
    "* <span class=\"global-text-highlight\">Representative of the real-world data</span> that the model will encounter as part of its business purpose.\n",
    "* <span class=\"global-text-highlight\">Zero examples duplicated in the training set</span>."
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
