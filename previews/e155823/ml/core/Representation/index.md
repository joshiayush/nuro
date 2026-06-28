{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6f3992ff-2b33-4afb-b92a-2160d3899757",
   "metadata": {},
   "source": [
    "# Representation\n",
    "\n",
    "A machine learning model can't directly see, hear, or sense input examples. Instead, you must create a representation of the data to provide the model with a useful vantage point into the data's key qualities. That is, in order to train a model, you must <span class=\"global-text-highlight\">choose the set of features that best represent the data</span>."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5dd56f10-974f-428d-a9e8-7ab35af1285f",
   "metadata": {},
   "source": [
    "## Feature Engineering\n",
    "\n",
    "In traditional programming, the focus is on code. In machine learning projects, the <span class=\"global-text-highlight\">focus shifts to representation</span>. That is, one way developers hone a model is by adding and improving its features.\n",
    "\n",
    "### Mapping Raw Data to Features\n",
    "\n",
    "The left side of *Figure 1* illustrates raw data from an input data source; the right side illustrates a **feature vector**, which is the <span class=\"global-text-highlight\">set of floating-point values comprising the examples in your data set</span>. **Feature engineering** means <span class=\"global-text-highlight\">transforming raw data into a feature vector</span>. Expect to spend significant time doing feature engineering.\n",
    "\n",
    "Many machine learning models must represent the features as <span class=\"global-text-highlight\">real-numbered vectors since the feature values must be multiplied by the model weights</span>.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/mapping_raw_data_feature_vector.png\" alt=\"Figure 1 diagram illustrating the process of mapping multi-modal raw data (a mug image, text review, and metadata log) to a structured feature vector (a stack of floating-point values, including text embeddings and image features) via a central feature engineering step.\" />\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3722a404-8f95-402b-ae1b-943b64ab3567",
   "metadata": {},
   "source": [
    "### Mapping numeric values\n",
    "\n",
    "<span class=\"global-text-highlight\">Integer and floating-point data don't need a special encoding because they can be multiplied by a numeric weight.</span> As suggested in *Figure 2*, converting the raw integer value 6 to the feature value 6.0 is trivial:\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/mapping_numeric_values.png\" alt=\"Mapping integer values to floating-point values\" />\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34d3252a-81ea-4fdc-ba57-347dc4615dc0",
   "metadata": {},
   "source": [
    "### Mapping categorical values\n",
    "\n",
    "<span class=\"global-text-highlight\">Categorical features have a discrete set of possible values.</span> For example, there might be a feature called `street_name` with options that include:\n",
    "\n",
    "```python\n",
    "{'Charleston Road', 'North Shoreline Boulevard', 'Shorebird Way', 'Rengstorff Avenue'}\n",
    "```\n",
    "\n",
    "<span class=\"global-text-highlight\">Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values.</span>\n",
    "\n",
    "We can accomplish this by defining a mapping from the feature values, which we'll refer to as the **vocabulary** of possible values, to integers. Since not every street in the world will appear in our dataset, we can group all other streets into a catch-all \"other\" category, known as an **OOV (out-of-vocabulary) bucket**.\n",
    "\n",
    "Using this approach, here's how we can map our street names to numbers:\n",
    "\n",
    "* map Charleston Road to 0\n",
    "* map North Shoreline Boulevard to 1\n",
    "* map Shorebird Way to 2\n",
    "* map Rengstorff Avenue to 3\n",
    "* map everything else (OOV) to 4\n",
    "\n",
    "However, if we incorporate these index numbers directly into our model, it will impose some constraints that might be problematic:\n",
    "\n",
    "* We'll be learning a single weight that applies to all streets. For example, if we learn a weight of 6 for `street_name`, then we will multiply it by 0 for Charleston Road, by 1 for North Shoreline Boulevard, 2 for Shorebird Way and so on. Consider a model that predicts house prices using `street_name` as a feature. <span class=\"global-text-highlight\">It is unlikely that there is a linear adjustment of price based on the street name, and furthermore this would assume you have ordered the streets based on their average house price.</span> Our model needs the flexibility of learning different weights for each street that will be added to the price estimated using the other features.\n",
    "* We aren't accounting for cases where `street_name` may take multiple values. For example, many houses are located at the corner of two streets, and there's no way to encode that information in the `street_name` value if it contains a single index.\n",
    "\n",
    "To remove both these constraints, we can instead create a binary vector for each categorical feature in our model that represents values as follows:\n",
    "\n",
    "* For values that apply to the example, set corresponding vector elements to `1`.\n",
    "* Set all other elements to `0`.\n",
    "\n",
    "The length of this vector is equal to the number of elements in the vocabulary. <span class=\"global-text-highlight\">This representation is called a **one-hot** encoding when a single value is 1, and a **multi-hot** encoding when multiple values are 1.</span>\n",
    "\n",
    "*Figure 3* illustrates a one-hot encoding of a particular street: Shorebird Way. The element in the binary vector for Shorebird Way has a value of `1`, while the elements for all other streets have values of `0`.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/one_hot_street_mapping.png\" alt=\"Mapping street address via one-hot encoding\" />\n",
    "</div>\n",
    "\n",
    "<span class=\"global-text-highlight\">This approach effectively creates a Boolean variable for every feature value (e.g., street name).</span> Here, if a house is on Shorebird Way then the binary value is 1 only for Shorebird Way. Thus, the model uses only the weight for Shorebird Way.\n",
    "\n",
    "Similarly, if a house is at the corner of two streets, then two binary values are set to 1, and the model uses both their respective weights."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ea586fb-4320-40fc-96b0-879e6c49bcc7",
   "metadata": {},
   "source": [
    "### Sparse Representation\n",
    "\n",
    "Suppose that you had 1,000,000 different street names in your data set that you wanted to include as values for `street_name`. <span class=\"global-text-highlight\">Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a very inefficient representation in terms of both storage and computation time when processing these vectors.</span> In this situation, a common approach is to use a <span class=\"global-text-highlight\">sparse representation in which only nonzero values are stored</span>. In sparse representations, an independent model weight is still learned for each feature value, as described above."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79e38e33-93b1-4b8f-a306-136371ac12a8",
   "metadata": {},
   "source": [
    "## Cleaning Data\n",
    "\n",
    "Apple trees produce some mixture of great fruit and wormy messes. Yet the apples in high-end grocery stores display 100% perfect fruit. Between orchard and grocery, someone spends significant time removing the bad apples or throwing a little wax on the salvageable ones. <span class=\"global-text-highlight\">As an ML engineer, you'll spend enormous amounts of your time tossing out bad examples and cleaning up the salvageable ones.</span> Even a few \"bad apples\" can spoil a large data set.\n",
    "\n",
    "## Scaling feature values\n",
    "\n",
    "<span class=\"global-text-highlight\">**Scaling** means converting floating-point feature values from their natural range (for example, 100 to 900) into a standard range (for example, 0 to 1 or -1 to +1).</span> If a feature set consists of only a single feature, then scaling provides little to no practical benefit. If, however, a feature set consists of multiple features, then feature scaling provides the following benefits:\n",
    "\n",
    "* Helps gradient descent converge more quickly.\n",
    "* Helps avoid the \"NaN trap,\" in which one number in the model becomes a NaN (e.g., when a value exceeds the floating-point precision limit during training), and—due to math operations—every other number in the model also eventually becomes a NaN.\n",
    "* Helps the model learn appropriate weights for each feature. <span class=\"global-text-highlight\">Without feature scaling, the model will pay too much attention to the features having a wider range.</span>\n",
    "\n",
    "You don't have to give every floating-point feature exactly the same scale. Nothing terrible will happen if Feature A is scaled from -1 to +1 while Feature B is scaled from -3 to +3. However, your model will react poorly if Feature B is scaled from 5000 to 100000.\n",
    "\n",
    "One obvious way to scale numerical data is to linearly map [min value, max value] to a small scale, such as [-1, +1].\n",
    "\n",
    "Another popular scaling tactic is to calculate the Z score of each value. <span class=\"global-text-highlight\">The Z score relates the number of standard deviations away from the mean.</span> In other words:\n",
    "\n",
    "$$\\text{scaled value} = (\\text{value} - \\text{mean})/\\text{stddev}$$\n",
    "\n",
    "For example, given:\n",
    "\n",
    "* mean = 100\n",
    "* standard deviation = 20\n",
    "* original value = 130\n",
    "\n",
    "then:\n",
    "\n",
    "```python\n",
    "scaled_value = (130 - 100) / 20\n",
    "scaled_value = 1.5\n",
    "```\n",
    "\n",
    "<span class=\"global-text-highlight\">Scaling with Z scores means that most scaled values will be between -3 and +3</span>, but a few values will be a little higher or lower than that range."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04ca1aaa-18ee-4a82-83b4-5cd4a2ac511a",
   "metadata": {},
   "source": [
    "### Handling extreme outliers\n",
    "\n",
    "The following plot represents a feature called `roomsPerPerson` from the California Housing data set. The value of `roomsPerPerson` was calculated by dividing the total number of rooms for an area by the population for that area. The plot shows that the vast majority of areas in California have one or two rooms per person. But take a look along the x-axis.\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/outliers_long_tail.png\" alt=\"A distribution plot showing a very long tail of extreme outlier values for roomsPerPerson\" />\n",
    "</div>\n",
    "\n",
    "<span class=\"global-text-highlight\">How could we minimize the influence of those extreme outliers? Well, one way would be to take the log of every value:</span>\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/log_scaling_tail.png\" alt=\"Distribution plot after applying logarithmic scaling, showing a compressed but still visible tail\" />\n",
    "</div>\n",
    "\n",
    "Log scaling does a slightly better job, but there's still a significant tail of outlier values. Let's pick yet another approach. <span class=\"global-text-highlight\">What if we simply \"cap\" or \"clip\" the maximum value of `roomsPerPerson` at an arbitrary value, say 4.0?</span>\n",
    "\n",
    "<div align=\"center\">\n",
    "    <img src=\"../../../static/images/clipping_at_4.png\" alt=\"Distribution plot where all values greater than 4.0 are clipped, creating a prominent peak/hill at 4.0\" />\n",
    "</div>\n",
    "\n",
    "<span class=\"global-text-highlight\">Clipping the feature value at 4.0 doesn't mean that we ignore all values greater than 4.0. Rather, it means that all values that were greater than 4.0 now become 4.0.</span> This explains the funny hill at 4.0. Despite that hill, the scaled feature set is now more useful than the original data."
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
