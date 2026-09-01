{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ec69cd67-332f-4e75-bd2f-19140c9a23f2",
   "metadata": {},
   "source": [
    "# Linear Regression\n",
    "\n",
    "In this notebook, we will train a **linear regression** model using **PyTorch** to predict the median house value in California. We will start from a simple model, then improve its accuracy by cleaning our data and engineering new features.\n",
    "\n",
    "To learn the theoretical concepts involved in **Linear Regression**, read this [doc](https://joshiayush.github.io/nuro/ml/models/Linear_Regression/)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "929b1fda-fa1b-455a-9564-bd6e5acc30e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.datasets import fetch_california_housing\n",
    "from sklearn.metrics import r2_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader\n",
    "from torch.utils.data import TensorDataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e829f657-dcc8-4337-9c39-8099c1c8b470",
   "metadata": {},
   "source": [
    "## What Are We Importing?\n",
    "\n",
    "We need three groups of libraries:\n",
    "\n",
    "- **Data handling**: `pandas` and `numpy` for loading and manipulating the dataset.\n",
    "- **Scikit-learn utilities**: `fetch_california_housing` to download the data, `train_test_split` to split it, `StandardScaler` to normalize features, and `r2_score` to evaluate our model.\n",
    "- **PyTorch**: `torch` for tensors, `nn` for the linear layer and loss function, `optim` for the Adam optimizer, and `DataLoader`/`TensorDataset` for feeding the data to the model in small batches."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b08c6dfd-427a-4e52-bd0c-be6b48986344",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing = fetch_california_housing()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9e2a8eb8-7665-4007-9cf8-9d352a84bdb9",
   "metadata": {},
   "source": [
    "## The California Housing Dataset\n",
    "\n",
    "`fetch_california_housing()` returns a `Bunch` object containing:\n",
    "\n",
    "- `housing[\"data\"]`: the **features** — 8 numerical attributes about each neighborhood, such as median income (`MedInc`), house age, and geographic coordinates.\n",
    "- `housing[\"target\"]`: the **label** — the median house value (`MedHouseVal`) we want to predict.\n",
    "- `housing[\"feature_names\"]` and `housing[\"target_names\"]`: the names of the columns.\n",
    "\n",
    "We combine everything into a single `pandas` DataFrame so we can inspect and clean it easily. <span class=\"global-text-highlight\">A DataFrame is a tabular structure — rows are examples, columns are features and the label.</span>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "72dcec11-29ef-4a83-8cb0-a3a338e496b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame(data=housing[\"data\"], columns=housing[\"feature_names\"])\n",
    "df[housing[\"target_names\"][0]] = housing[\"target\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "983e61b4-43b1-45bf-8c2b-828359e0a18a",
   "metadata": {},
   "source": [
    "### Why Clean the Data First?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "04dfea8e-3dc3-42c9-91d0-bfad1d22f904",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MedInc</th>\n",
       "      <th>HouseAge</th>\n",
       "      <th>AveRooms</th>\n",
       "      <th>AveBedrms</th>\n",
       "      <th>Population</th>\n",
       "      <th>AveOccup</th>\n",
       "      <th>Latitude</th>\n",
       "      <th>Longitude</th>\n",
       "      <th>MedHouseVal</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>3.870671</td>\n",
       "      <td>28.639486</td>\n",
       "      <td>5.429000</td>\n",
       "      <td>1.096675</td>\n",
       "      <td>1425.476744</td>\n",
       "      <td>3.070655</td>\n",
       "      <td>35.631861</td>\n",
       "      <td>-119.569704</td>\n",
       "      <td>2.068558</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.899822</td>\n",
       "      <td>12.585558</td>\n",
       "      <td>2.474173</td>\n",
       "      <td>0.473911</td>\n",
       "      <td>1132.462122</td>\n",
       "      <td>10.386050</td>\n",
       "      <td>2.135952</td>\n",
       "      <td>2.003532</td>\n",
       "      <td>1.153956</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.499900</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.846154</td>\n",
       "      <td>0.333333</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.692308</td>\n",
       "      <td>32.540000</td>\n",
       "      <td>-124.350000</td>\n",
       "      <td>0.149990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.563400</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>4.440716</td>\n",
       "      <td>1.006079</td>\n",
       "      <td>787.000000</td>\n",
       "      <td>2.429741</td>\n",
       "      <td>33.930000</td>\n",
       "      <td>-121.800000</td>\n",
       "      <td>1.196000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.534800</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>5.229129</td>\n",
       "      <td>1.048780</td>\n",
       "      <td>1166.000000</td>\n",
       "      <td>2.818116</td>\n",
       "      <td>34.260000</td>\n",
       "      <td>-118.490000</td>\n",
       "      <td>1.797000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>4.743250</td>\n",
       "      <td>37.000000</td>\n",
       "      <td>6.052381</td>\n",
       "      <td>1.099526</td>\n",
       "      <td>1725.000000</td>\n",
       "      <td>3.282261</td>\n",
       "      <td>37.710000</td>\n",
       "      <td>-118.010000</td>\n",
       "      <td>2.647250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>15.000100</td>\n",
       "      <td>52.000000</td>\n",
       "      <td>141.909091</td>\n",
       "      <td>34.066667</td>\n",
       "      <td>35682.000000</td>\n",
       "      <td>1243.333333</td>\n",
       "      <td>41.950000</td>\n",
       "      <td>-114.310000</td>\n",
       "      <td>5.000010</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             MedInc      HouseAge      AveRooms     AveBedrms    Population  \\\n",
       "count  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000   \n",
       "mean       3.870671     28.639486      5.429000      1.096675   1425.476744   \n",
       "std        1.899822     12.585558      2.474173      0.473911   1132.462122   \n",
       "min        0.499900      1.000000      0.846154      0.333333      3.000000   \n",
       "25%        2.563400     18.000000      4.440716      1.006079    787.000000   \n",
       "50%        3.534800     29.000000      5.229129      1.048780   1166.000000   \n",
       "75%        4.743250     37.000000      6.052381      1.099526   1725.000000   \n",
       "max       15.000100     52.000000    141.909091     34.066667  35682.000000   \n",
       "\n",
       "           AveOccup      Latitude     Longitude   MedHouseVal  \n",
       "count  20640.000000  20640.000000  20640.000000  20640.000000  \n",
       "mean       3.070655     35.631861   -119.569704      2.068558  \n",
       "std       10.386050      2.135952      2.003532      1.153956  \n",
       "min        0.692308     32.540000   -124.350000      0.149990  \n",
       "25%        2.429741     33.930000   -121.800000      1.196000  \n",
       "50%        2.818116     34.260000   -118.490000      1.797000  \n",
       "75%        3.282261     37.710000   -118.010000      2.647250  \n",
       "max     1243.333333     41.950000   -114.310000      5.000010  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50ee7aec-fd78-4024-a875-141cbb1255a0",
   "metadata": {},
   "source": [
    "If we look closely at `df.describe()`, the max values tell an important story:\n",
    "\n",
    "| Feature | 75th percentile | Max |\n",
    "|---|---|---|\n",
    "| `AveRooms` | 6.05 | 141.9 |\n",
    "| `AveOccup` | 3.28 | 1243.3 |\n",
    "\n",
    "The vast majority of neighborhoods have a handful of rooms and occupants, yet a few extreme values reach into the hundreds or thousands. <span class=\"global-text-highlight\">These are **outliers** — extreme values that sit far outside the normal range.</span> We will handle them before training."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "094ed675-2eda-43f1-b68d-4b7aaf3be9d4",
   "metadata": {},
   "source": [
    "## Cleaning the Data: Clipping Outliers\n",
    "\n",
    "A linear model is trained using **Mean Squared Error**, which squares the error for every example. A single neighborhood with 141 rooms would produce an enormous squared error, pulling the model's weights in a weird direction just to accommodate that one row.\n",
    "\n",
    "**The fix**: we *clip* each column at its 99th percentile. `quantile(0.99)` returns the value below which 99% of the data falls, and `clip(upper=...)` caps anything above that value. <span class=\"global-text-highlight\">This tames the extreme tail without removing any rows.</span>\n",
    "\n",
    "To see why this helps, imagine every neighborhood sorted by room count, from smallest to largest. The 99th percentile is the line that separates the normal range from the extreme tail:\n",
    "\n",
    "```text\n",
    "Sorted houses:  [ 1.1, 2.5, 4.0, ... , 13.1, 13.2 ]   [ 45.2, 89.0, 141.9 ]\n",
    "                \\_________________________________/    \\_________________/\n",
    "                   99% of houses (normal range)         Top 1% (outliers)\n",
    "                                                    ^\n",
    "                                99th percentile (quantile 0.99)\n",
    "```\n",
    "\n",
    "The problem: without clipping, those few outliers stretch the number line so far that all the normal houses get squished together:\n",
    "\n",
    "```text\n",
    "               Normal houses                          Outliers\n",
    "             [||||||||||||||||||||||||||]                    |         |\n",
    "+------------+---------------------------+-------------------+---------+-----------> rooms\n",
    "0           10                          30                  60       100         150\n",
    "```\n",
    "\n",
    "After clipping at the 99th percentile, the outliers are pulled back to the cap, so the model can actually tell normal houses apart:\n",
    "\n",
    "```text\n",
    "                Normal houses\n",
    "              [||||||||||||||||||||||||||]   <-- outliers are now capped here\n",
    "+------------+---------------------------+-------------------+---------+-----------> rooms\n",
    "0           10                   13.2 (cap)                  60       100         150\n",
    "```\n",
    "\n",
    "We also engineer two **ratio features**:\n",
    "- `RoomsPerHousehold` = `AveRooms / AveOccup` — how spacious a home is per person.\n",
    "- `BedroomsPerRoom` = `AveBedrms / AveRooms` — how much of the home is bedrooms.\n",
    "\n",
    "These give the model information it cannot derive from the raw columns alone. Finally, we split the data into train and test sets using a fixed `random_state` so our results are reproducible."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "cae43756-d18f-465d-aaca-942bd2413d42",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean = df.copy()\n",
    "for c in [\"AveRooms\", \"AveBedrms\", \"Population\", \"AveOccup\"]:\n",
    "  upper = df_clean[c].quantile(0.99)\n",
    "  df_clean[c] = df_clean[c].clip(upper=upper)\n",
    "\n",
    "df_clean[\"RoomsPerHousehold\"] = df_clean[\"AveRooms\"] / df_clean[\"AveOccup\"]\n",
    "df_clean[\"BedroomsPerRoom\"] = df_clean[\"AveBedrms\"] / df_clean[\"AveRooms\"]\n",
    "\n",
    "feature_names_updated = housing[\"feature_names\"] + [\n",
    "    \"RoomsPerHousehold\", \"BedroomsPerRoom\"\n",
    "]\n",
    "\n",
    "X_numpy = df_clean[feature_names_updated].to_numpy()\n",
    "y_numpy = df_clean[housing[\"target_names\"][0]].to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "0ce3ea0f-ddb4-44a3-87bd-cd78b4dca33d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MedInc</th>\n",
       "      <th>HouseAge</th>\n",
       "      <th>AveRooms</th>\n",
       "      <th>AveBedrms</th>\n",
       "      <th>Population</th>\n",
       "      <th>AveOccup</th>\n",
       "      <th>Latitude</th>\n",
       "      <th>Longitude</th>\n",
       "      <th>MedHouseVal</th>\n",
       "      <th>RoomsPerHousehold</th>\n",
       "      <th>BedroomsPerRoom</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "      <td>20640.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>3.870671</td>\n",
       "      <td>28.639486</td>\n",
       "      <td>5.330588</td>\n",
       "      <td>1.076287</td>\n",
       "      <td>1403.613896</td>\n",
       "      <td>2.915167</td>\n",
       "      <td>35.631861</td>\n",
       "      <td>-119.569704</td>\n",
       "      <td>2.068558</td>\n",
       "      <td>1.936423</td>\n",
       "      <td>0.213026</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.899822</td>\n",
       "      <td>12.585558</td>\n",
       "      <td>1.330038</td>\n",
       "      <td>0.160058</td>\n",
       "      <td>973.476399</td>\n",
       "      <td>0.734751</td>\n",
       "      <td>2.135952</td>\n",
       "      <td>2.003532</td>\n",
       "      <td>1.153956</td>\n",
       "      <td>0.644395</td>\n",
       "      <td>0.057784</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.499900</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.846154</td>\n",
       "      <td>0.333333</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.692308</td>\n",
       "      <td>32.540000</td>\n",
       "      <td>-124.350000</td>\n",
       "      <td>0.149990</td>\n",
       "      <td>0.311943</td>\n",
       "      <td>0.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.563400</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>4.440716</td>\n",
       "      <td>1.006079</td>\n",
       "      <td>787.000000</td>\n",
       "      <td>2.429741</td>\n",
       "      <td>33.930000</td>\n",
       "      <td>-121.800000</td>\n",
       "      <td>1.196000</td>\n",
       "      <td>1.523082</td>\n",
       "      <td>0.175662</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.534800</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>5.229129</td>\n",
       "      <td>1.048780</td>\n",
       "      <td>1166.000000</td>\n",
       "      <td>2.818116</td>\n",
       "      <td>34.260000</td>\n",
       "      <td>-118.490000</td>\n",
       "      <td>1.797000</td>\n",
       "      <td>1.937936</td>\n",
       "      <td>0.203690</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>4.743250</td>\n",
       "      <td>37.000000</td>\n",
       "      <td>6.052381</td>\n",
       "      <td>1.099526</td>\n",
       "      <td>1725.000000</td>\n",
       "      <td>3.282261</td>\n",
       "      <td>37.710000</td>\n",
       "      <td>-118.010000</td>\n",
       "      <td>2.647250</td>\n",
       "      <td>2.296090</td>\n",
       "      <td>0.239466</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>15.000100</td>\n",
       "      <td>52.000000</td>\n",
       "      <td>10.357033</td>\n",
       "      <td>2.127541</td>\n",
       "      <td>5805.830000</td>\n",
       "      <td>5.394812</td>\n",
       "      <td>41.950000</td>\n",
       "      <td>-114.310000</td>\n",
       "      <td>5.000010</td>\n",
       "      <td>14.960159</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             MedInc      HouseAge      AveRooms     AveBedrms    Population  \\\n",
       "count  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000   \n",
       "mean       3.870671     28.639486      5.330588      1.076287   1403.613896   \n",
       "std        1.899822     12.585558      1.330038      0.160058    973.476399   \n",
       "min        0.499900      1.000000      0.846154      0.333333      3.000000   \n",
       "25%        2.563400     18.000000      4.440716      1.006079    787.000000   \n",
       "50%        3.534800     29.000000      5.229129      1.048780   1166.000000   \n",
       "75%        4.743250     37.000000      6.052381      1.099526   1725.000000   \n",
       "max       15.000100     52.000000     10.357033      2.127541   5805.830000   \n",
       "\n",
       "           AveOccup      Latitude     Longitude   MedHouseVal  \\\n",
       "count  20640.000000  20640.000000  20640.000000  20640.000000   \n",
       "mean       2.915167     35.631861   -119.569704      2.068558   \n",
       "std        0.734751      2.135952      2.003532      1.153956   \n",
       "min        0.692308     32.540000   -124.350000      0.149990   \n",
       "25%        2.429741     33.930000   -121.800000      1.196000   \n",
       "50%        2.818116     34.260000   -118.490000      1.797000   \n",
       "75%        3.282261     37.710000   -118.010000      2.647250   \n",
       "max        5.394812     41.950000   -114.310000      5.000010   \n",
       "\n",
       "       RoomsPerHousehold  BedroomsPerRoom  \n",
       "count       20640.000000     20640.000000  \n",
       "mean            1.936423         0.213026  \n",
       "std             0.644395         0.057784  \n",
       "min             0.311943         0.100000  \n",
       "25%             1.523082         0.175662  \n",
       "50%             1.937936         0.203690  \n",
       "75%             2.296090         0.239466  \n",
       "max            14.960159         1.000000  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_clean.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8da0d1e-265e-496d-96d6-62129c5791b2",
   "metadata": {},
   "source": [
    "### The Train / Test Split\n",
    "\n",
    "`train_test_split` shuffles the data and reserves **33.3%** of it as the test set. The model will only ever see the training portion during training. We keep the test set completely hidden until the very end, so the final score tells us how well the model generalizes to data it has never seen."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "bde30b79-e509-4507-8286-dd5340130f7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_numpy,\n",
    "                                                    y_numpy,\n",
    "                                                    test_size=0.333,\n",
    "                                                    random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "92b98450-b75a-4f9f-a28a-1de72a8bddce",
   "metadata": {},
   "source": [
    "## Scaling the Features\n",
    "\n",
    "Our features are on very different scales — `MedInc` is in the single digits, while `Population` is in the thousands. Without scaling, a feature with larger numbers would dominate the loss purely because of its magnitude. <span class=\"global-text-highlight\">This is called **feature dominance**.</span>\n",
    "\n",
    "`StandardScaler` subtracts the mean and divides by the standard deviation of each column, so every feature is centered around 0 with unit variance. We call `fit_transform` on the training data only, then `transform` the test data using the same statistics — never fitting on the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e6c329a2-d9c3-4c92-a3be-5f2098f4f2c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d07f00b-6793-4371-8dcd-1e2f2197cf62",
   "metadata": {},
   "source": [
    "### From NumPy to Tensors\n",
    "\n",
    "PyTorch cannot work directly with NumPy arrays, so we convert them into **tensors** — PyTorch's n-dimensional arrays. We cast to `float32` because that is the default precision PyTorch expects, and we reshape `y` into a column vector of shape `(n_samples, 1)` to match the model's output shape. `y.view(-1, 1)` is shorthand for `y.view(y.shape[0], 1)`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "40e79cf3-96f6-42a3-a618-4ac401ccdf09",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = torch.from_numpy(X_train.astype(np.float32))\n",
    "y = torch.from_numpy(y_train.astype(np.float32))\n",
    "y = y.view(-1, 1)  # equivalent to `y.view(y.shape[0], 1)`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "59fd8e91-6ba5-4cb0-8ce3-6eb22b71e2b5",
   "metadata": {},
   "source": [
    "## Batching the Data: The DataLoader\n",
    "\n",
    "Computing the gradient over all 13,766 training examples at once is slow and noisy. Instead, we use **mini-batches**. `TensorDataset` pairs each `X` sample with its `y` label, and `DataLoader` iterates over shuffled chunks of `batch_size=64`. <span class=\"global-text-highlight\">This gives the optimizer many small weight updates per epoch instead of one giant one, which converges faster and more stably.</span>\n",
    "\n",
    "We keep the test tensors separate — they are not batched because we only need one forward pass through them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e076de17-64bc-4726-b45a-7dfa86a11dd2",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dataset = TensorDataset(X, y)\n",
    "train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "b744f908-d784-43c3-9d67-d75e0eb92dfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = torch.from_numpy(X_test.astype(np.float32))\n",
    "y_test = torch.from_numpy(y_test.astype(np.float32))\n",
    "y_test = y_test.view(-1, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "fc89b36a-83e2-4589-9eb9-75f3ce3cad29",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(13766, 10)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n_samples, n_features = X.shape\n",
    "n_samples, n_features"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f36d730b-f54f-4242-88e3-0ae77beed188",
   "metadata": {},
   "source": [
    "## Building the Model\n",
    "\n",
    "`nn.Linear(n_features, 1)` creates a single layer that computes $y = XW + b$, where $W$ is a weight matrix of shape `(n_features, 1)` and $b$ is a scalar bias. Because we engineered two extra features, `n_features` is **10** (not 8) — notice the shape `(13766, 10)` from the previous cell. The model therefore has 10 weights plus a bias, for 11 learnable parameters in total."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b328c4f5-93d6-413d-bae9-33e4c0ac4e25",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = nn.Linear(n_features, 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8bf7ce88-f218-4b8f-9a5a-13634daade36",
   "metadata": {},
   "source": [
    "## Loss Function and Optimizer\n",
    "\n",
    "- **`nn.MSELoss()`** — the <span class=\"global-text-highlight\">**Mean Squared Error**</span> loss. It measures the average squared difference between predictions and true values. Perfect predictions give a loss of 0; larger errors are punished quadratically.\n",
    "- **`optim.Adam(...)`** — an optimizer that updates the weights using gradient descent. Adam adapts the learning rate for each parameter individually, which usually converges faster than plain SGD. The `lr=0.01` controls the step size of each update."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "75ef3b82-9366-4258-88ff-5fe20718e2bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion = nn.MSELoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=0.01)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "903177f0-6398-456e-9561-7d2c411d01a2",
   "metadata": {},
   "source": [
    "## The Training Loop\n",
    "\n",
    "Each epoch does the following:\n",
    "\n",
    "1. **Forward pass** — predict house values for every batch with `model(X_batch)`.\n",
    "2. **Compute loss** — compare predictions to the true labels using MSE.\n",
    "3. **Zero the gradients** — `optimizer.zero_grad()` clears the accumulated gradients from the previous step, otherwise they would add up across batches.\n",
    "4. **Backward pass** — `loss.backward()` computes the gradient of the loss with respect to every parameter.\n",
    "5. **Update** — `optimizer.step()` nudges the weights in the direction that reduces the loss.\n",
    "\n",
    "We track the average loss per epoch so we can watch the model improve. Notice that the loss drops sharply in the first few epochs and then plateaus around 0.44 — the model has converged."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8adce7b6-3388-4e8c-b441-723d9c2d36c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  5 | Train Loss: 0.4368\n",
      "Epoch 10 | Train Loss: 0.4415\n",
      "Epoch 15 | Train Loss: 0.4376\n",
      "Epoch 20 | Train Loss: 0.4381\n",
      "Epoch 25 | Train Loss: 0.4386\n",
      "Epoch 30 | Train Loss: 0.4381\n"
     ]
    }
   ],
   "source": [
    "n_epochs = 30\n",
    "for epoch in range(n_epochs):\n",
    "  model.train()\n",
    "  total_loss = 0.0\n",
    "\n",
    "  for X_batch, y_batch in train_loader:\n",
    "    y_pred = model(X_batch)\n",
    "    loss = criterion(y_pred, y_batch)\n",
    "    optimizer.zero_grad()\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "\n",
    "    total_loss += loss.item() * len(X_batch)\n",
    "\n",
    "  avg_loss = total_loss / len(X)\n",
    "  if (epoch + 1) % 5 == 0:\n",
    "    print(f\"Epoch {epoch+1:2d} | Train Loss: {avg_loss:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ef778c0-9c77-4d9f-b1a3-89d6beaf2c41",
   "metadata": {},
   "source": [
    "## Evaluating the Model\n",
    "\n",
    "After training, we run the model on the *held-out* test set and measure performance with the **coefficient of determination** $R^2$.\n",
    "\n",
    "$R^2$ tells us what fraction of the variance in house prices our model explains, on a scale from negative values up to 1.0 (a perfect fit). We detach the predictions from the computation graph (`.detach()`) and convert them back to NumPy before scoring."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "dd82ee4f-2a93-4c5d-a6f9-5a5cfc03e8ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model(X_test).detach().numpy()\n",
    "y_pred = y_pred.flatten()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d0850acc-81bc-40ab-875c-041ceaadc75b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6684893369674683"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r2_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "255e29e7-a26b-4b4a-81e4-be3d078e6eb4",
   "metadata": {},
   "source": [
    "## What Did We Achieve?\n",
    "\n",
    "Our test $R^2$ score is around **0.67**, up from roughly 0.59 on the raw features. The two changes that mattered:\n",
    "\n",
    "1. **Clipping outliers** at the 99th percentile stopped a handful of extreme neighborhoods from distorting the loss.\n",
    "2. **Engineering ratio features** gave the model relationships it could not discover from the raw columns alone.\n",
    "\n",
    "A purely linear model has its limits though. To push higher, we could add **polynomial features** to capture non-linear interactions — a natural next experiment."
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
