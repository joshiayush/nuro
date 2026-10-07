{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1CungjeV2iePpeFFCKq9r9BcoabLyMWvo",
      "authorship_tag": "ABX9TyPwXKN+OHlofimS+ffT/IYp",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/joshiayush/ai/blob/master/ai/projects/machine-learning/algorithms/kNN.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# K-Nearest Neighbors Algorithm\n",
        "\n",
        "\n",
        "The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another."
      ],
      "metadata": {
        "id": "x6ZQLz1wUqFm"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "For classification problems, a class label is assigned on the basis of a majority vote—i.e. the label that is most frequently represented around a given data point is used. While this is technically considered “plurality voting”, the term, “majority vote” is more commonly used in literature. The distinction between these terminologies is that “majority voting” technically requires a majority of greater than 50%, which primarily works when there are only two categories. When you have multiple classes—e.g. four categories, you don’t necessarily need 50% of the vote to make a conclusion about a class; you could assign a class label with a vote of greater than 25%."
      ],
      "metadata": {
        "id": "m7vuO9VPOg57"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "<div align=\"left\">\n",
        "\n",
        "<img height=\"400px\" src=\"https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/ef/3a/KNN.component.complex-narrative-xl-retina.ts=1653407890466.png/content/adobe-cms/us/en/topics/knn/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/complex_narrative/items/content_group/image\" />\n",
        "\n",
        "</div>"
      ],
      "metadata": {
        "id": "YaCLUDrIOssN"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Regression problems use a similar concept as classification problem, but in this case, the average the k nearest neighbors is taken to make a prediction about a classification. The main distinction here is that classification is used for discrete values, whereas regression is used with continuous ones. However, before a classification can be made, the distance must be defined. Euclidean distance is most commonly used, which we’ll delve into more below.\n",
        "\n",
        "It's also worth noting that the KNN algorithm is also part of a family of “lazy learning” models, meaning that it only stores a training dataset versus undergoing a training stage. This also means that all the computation occurs when a classification or prediction is being made. Since it heavily relies on memory to store all its training data, it is also referred to as an instance-based or memory-based learning method."
      ],
      "metadata": {
        "id": "x-00ArnIPJ6d"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Compute KNN: distance metrics\n",
        "\n",
        "To recap, the goal of the k-nearest neighbor algorithm is to identify the nearest neighbors of a given query point, so that we can assign a class label to that point. In order to do this, KNN has a few requirements:\n",
        "\n",
        "__Determine your distance metrics__\n",
        "\n",
        "In order to determine which data points are closest to a given query point, the distance between the query point and the other data points will need to be calculated. These distance metrics help to form decision boundaries, which partitions query points into different regions. You commonly will see decision boundaries visualized with Voronoi diagrams.\n",
        "\n",
        "__Euclidean distance (p=2):__ This is the most commonly used distance measure, and it is limited to real-valued vectors. Using the below formula, it measures a straight line between the query point and the other point being measured.\n",
        "\n",
        "$\\mathrm{Eculidean\\ Distance} = d(x, y) = \\sqrt{\\sum_{i=1}^{n}(y_{i} - x_{i})^{2}}$"
      ],
      "metadata": {
        "id": "5dl-Iii_P5yV"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "__Manhattan distance (p=1):__ This is also another popular distance metric, which measures the absolute value between two points. It is also referred to as taxicab distance or city block distance as it is commonly visualized with a grid, illustrating how one might navigate from one address to another via city streets.\n",
        "\n",
        "$\\mathrm{Manhattan\\ Distance} = d(x, y) = \\left( \\sum_{i=1}^{m}|x_{i} - y_{i}| \\right)$"
      ],
      "metadata": {
        "id": "JL4Q3qFQRXni"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "__Minkowski distance:__ This distance measure is the generalized form of Euclidean and Manhattan distance metrics. The parameter, p, in the formula below, allows for the creation of other distance metrics. Euclidean distance is represented by this formula when p is equal to two, and Manhattan distance is denoted with p equal to one.\n",
        "\n",
        "$\\mathrm{Minkowski\\ Distance} = d(x, y) = \\left( \\sum_{i=1}^{n}|x_{i} - y_{i}| \\right)^{\\dfrac{1}{p}}$"
      ],
      "metadata": {
        "id": "dKPfBFJ1SRg0"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "__Hamming distance:__ This technique is used typically used with Boolean or string vectors, identifying the points where the vectors do not match. As a result, it has also been referred to as the overlap metric. This can be represented with the following formula:\n",
        "\n",
        "$\\mathrm{Hamming\\ Distance} = D_{H} = \\left( \\sum_{i=1}^{k}|x_{i} - y_{i}| \\right)$\n",
        "\n",
        "$x = y, D = 0$\n",
        "\n",
        "$x \\neq y, D \\neq 1$\n",
        "\n",
        "As an example, if you had the following strings, the hamming distance would be 2 since only two of the values differ."
      ],
      "metadata": {
        "id": "nTgo_d8WSgeo"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "<div align=\"center\">\n",
        "\n",
        "<img width=\"800px\" height=\"300px\" src=\"https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/e8/5a/HammingDistanceExample.component.complex-narrative-xl-retina.ts=1653407891607.png/content/adobe-cms/us/en/topics/knn/jcr:content/root/table_of_contents/body/content_section_styled/content-section-body/complex_narrative0/items/content_group_1457609298/image\" />\n",
        "\n",
        "</div>"
      ],
      "metadata": {
        "id": "gDwRIbNOS_lK"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Compute KNN: defining k\n",
        "\n",
        "\n",
        "The k value in the k-NN algorithm defines how many neighbors will be checked to determine the classification of a specific query point. For example, if k=1, the instance will be assigned to the same class as its single nearest neighbor. Defining k can be a balancing act as different values can lead to overfitting or underfitting. Lower values of k can have high variance, but low bias, and larger values of k may lead to high bias and lower variance. The choice of k will largely depend on the input data as data with more outliers or noise will likely perform better with higher values of k. Overall, it is recommended to have an odd number for k to avoid ties in classification, and cross-validation tactics can help you choose the optimal k for your dataset."
      ],
      "metadata": {
        "id": "VezPPSu9TspK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import joblib\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from sklearn.metrics import classification_report\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from imblearn.over_sampling import RandomOverSampler"
      ],
      "metadata": {
        "id": "9-dOJPd8XCcP"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "_MAGIC_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/ai/projects/datasets/magic+gamma+telescope/magic04.data'"
      ],
      "metadata": {
        "id": "OCC-itCxWeBr"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "We need to explicitly define names since the dataset does not come with a column field."
      ],
      "metadata": {
        "id": "AgT-WtTEYFYJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv(\n",
        "    _MAGIC_DATASET_PATH,\n",
        "    names=[\n",
        "        \"fLength\",\n",
        "        \"fWidth\",\n",
        "        \"fSize\",\n",
        "        \"fConc\",\n",
        "        \"fConc1\",\n",
        "        \"fAsym\",\n",
        "        \"fM3Long\",\n",
        "        \"fM3Trans\",\n",
        "        \"fAlpha\",\n",
        "        \"fDist\",\n",
        "        \"class\",\n",
        "    ],\n",
        ")"
      ],
      "metadata": {
        "id": "rXk6-h8pXHFv"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "BijtH3-gYJMf",
        "outputId": "145f2dd1-5bfe-4e8c-9959-ede901a6e398"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        fLength    fWidth   fSize   fConc  fConc1     fAsym   fM3Long  \\\n",
              "0       28.7967   16.0021  2.6449  0.3918  0.1982   27.7004   22.0110   \n",
              "1       31.6036   11.7235  2.5185  0.5303  0.3773   26.2722   23.8238   \n",
              "2      162.0520  136.0310  4.0612  0.0374  0.0187  116.7410  -64.8580   \n",
              "3       23.8172    9.5728  2.3385  0.6147  0.3922   27.2107   -6.4633   \n",
              "4       75.1362   30.9205  3.1611  0.3168  0.1832   -5.5277   28.5525   \n",
              "...         ...       ...     ...     ...     ...       ...       ...   \n",
              "19015   21.3846   10.9170  2.6161  0.5857  0.3934   15.2618   11.5245   \n",
              "19016   28.9452    6.7020  2.2672  0.5351  0.2784   37.0816   13.1853   \n",
              "19017   75.4455   47.5305  3.4483  0.1417  0.0549   -9.3561   41.0562   \n",
              "19018  120.5135   76.9018  3.9939  0.0944  0.0683    5.8043  -93.5224   \n",
              "19019  187.1814   53.0014  3.2093  0.2876  0.1539 -167.3125 -168.4558   \n",
              "\n",
              "       fM3Trans   fAlpha     fDist class  \n",
              "0       -8.2027  40.0920   81.8828     g  \n",
              "1       -9.9574   6.3609  205.2610     g  \n",
              "2      -45.2160  76.9600  256.7880     g  \n",
              "3       -7.1513  10.4490  116.7370     g  \n",
              "4       21.8393   4.6480  356.4620     g  \n",
              "...         ...      ...       ...   ...  \n",
              "19015    2.8766   2.4229  106.8258     h  \n",
              "19016   -2.9632  86.7975  247.4560     h  \n",
              "19017   -9.4662  30.2987  256.5166     h  \n",
              "19018  -63.8389  84.6874  408.3166     h  \n",
              "19019   31.4755  52.7310  272.3174     h  \n",
              "\n",
              "[19020 rows x 11 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-a8b049e9-85b7-47c5-ae4e-8c6e70494852\" class=\"colab-df-container\">\n",
              "    <div>\n",
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
              "      <th>fLength</th>\n",
              "      <th>fWidth</th>\n",
              "      <th>fSize</th>\n",
              "      <th>fConc</th>\n",
              "      <th>fConc1</th>\n",
              "      <th>fAsym</th>\n",
              "      <th>fM3Long</th>\n",
              "      <th>fM3Trans</th>\n",
              "      <th>fAlpha</th>\n",
              "      <th>fDist</th>\n",
              "      <th>class</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>28.7967</td>\n",
              "      <td>16.0021</td>\n",
              "      <td>2.6449</td>\n",
              "      <td>0.3918</td>\n",
              "      <td>0.1982</td>\n",
              "      <td>27.7004</td>\n",
              "      <td>22.0110</td>\n",
              "      <td>-8.2027</td>\n",
              "      <td>40.0920</td>\n",
              "      <td>81.8828</td>\n",
              "      <td>g</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>31.6036</td>\n",
              "      <td>11.7235</td>\n",
              "      <td>2.5185</td>\n",
              "      <td>0.5303</td>\n",
              "      <td>0.3773</td>\n",
              "      <td>26.2722</td>\n",
              "      <td>23.8238</td>\n",
              "      <td>-9.9574</td>\n",
              "      <td>6.3609</td>\n",
              "      <td>205.2610</td>\n",
              "      <td>g</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>162.0520</td>\n",
              "      <td>136.0310</td>\n",
              "      <td>4.0612</td>\n",
              "      <td>0.0374</td>\n",
              "      <td>0.0187</td>\n",
              "      <td>116.7410</td>\n",
              "      <td>-64.8580</td>\n",
              "      <td>-45.2160</td>\n",
              "      <td>76.9600</td>\n",
              "      <td>256.7880</td>\n",
              "      <td>g</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>23.8172</td>\n",
              "      <td>9.5728</td>\n",
              "      <td>2.3385</td>\n",
              "      <td>0.6147</td>\n",
              "      <td>0.3922</td>\n",
              "      <td>27.2107</td>\n",
              "      <td>-6.4633</td>\n",
              "      <td>-7.1513</td>\n",
              "      <td>10.4490</td>\n",
              "      <td>116.7370</td>\n",
              "      <td>g</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>75.1362</td>\n",
              "      <td>30.9205</td>\n",
              "      <td>3.1611</td>\n",
              "      <td>0.3168</td>\n",
              "      <td>0.1832</td>\n",
              "      <td>-5.5277</td>\n",
              "      <td>28.5525</td>\n",
              "      <td>21.8393</td>\n",
              "      <td>4.6480</td>\n",
              "      <td>356.4620</td>\n",
              "      <td>g</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19015</th>\n",
              "      <td>21.3846</td>\n",
              "      <td>10.9170</td>\n",
              "      <td>2.6161</td>\n",
              "      <td>0.5857</td>\n",
              "      <td>0.3934</td>\n",
              "      <td>15.2618</td>\n",
              "      <td>11.5245</td>\n",
              "      <td>2.8766</td>\n",
              "      <td>2.4229</td>\n",
              "      <td>106.8258</td>\n",
              "      <td>h</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19016</th>\n",
              "      <td>28.9452</td>\n",
              "      <td>6.7020</td>\n",
              "      <td>2.2672</td>\n",
              "      <td>0.5351</td>\n",
              "      <td>0.2784</td>\n",
              "      <td>37.0816</td>\n",
              "      <td>13.1853</td>\n",
              "      <td>-2.9632</td>\n",
              "      <td>86.7975</td>\n",
              "      <td>247.4560</td>\n",
              "      <td>h</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19017</th>\n",
              "      <td>75.4455</td>\n",
              "      <td>47.5305</td>\n",
              "      <td>3.4483</td>\n",
              "      <td>0.1417</td>\n",
              "      <td>0.0549</td>\n",
              "      <td>-9.3561</td>\n",
              "      <td>41.0562</td>\n",
              "      <td>-9.4662</td>\n",
              "      <td>30.2987</td>\n",
              "      <td>256.5166</td>\n",
              "      <td>h</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19018</th>\n",
              "      <td>120.5135</td>\n",
              "      <td>76.9018</td>\n",
              "      <td>3.9939</td>\n",
              "      <td>0.0944</td>\n",
              "      <td>0.0683</td>\n",
              "      <td>5.8043</td>\n",
              "      <td>-93.5224</td>\n",
              "      <td>-63.8389</td>\n",
              "      <td>84.6874</td>\n",
              "      <td>408.3166</td>\n",
              "      <td>h</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19019</th>\n",
              "      <td>187.1814</td>\n",
              "      <td>53.0014</td>\n",
              "      <td>3.2093</td>\n",
              "      <td>0.2876</td>\n",
              "      <td>0.1539</td>\n",
              "      <td>-167.3125</td>\n",
              "      <td>-168.4558</td>\n",
              "      <td>31.4755</td>\n",
              "      <td>52.7310</td>\n",
              "      <td>272.3174</td>\n",
              "      <td>h</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>19020 rows × 11 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-a8b049e9-85b7-47c5-ae4e-8c6e70494852')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-a8b049e9-85b7-47c5-ae4e-8c6e70494852 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-a8b049e9-85b7-47c5-ae4e-8c6e70494852');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-4adfb0f8-e57c-4ea2-8092-717db6ac428d\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-4adfb0f8-e57c-4ea2-8092-717db6ac428d')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-4adfb0f8-e57c-4ea2-8092-717db6ac428d button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df['class'].unique()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vIvlQ0BDYicS",
        "outputId": "264e22dc-4e34-4b01-c0a9-127d706d61bd"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array(['g', 'h'], dtype=object)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Since, the class labels only contains letters and we need numerical data to establish a mapping from our independent variables to our dependent variables - we need to convert the classes into numerical data."
      ],
      "metadata": {
        "id": "v1SAvIbfYmjK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df['class'] = (df['class'] == 'g').astype(int)"
      ],
      "metadata": {
        "id": "HuXtOJJeY23L"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now we're all set with our labels."
      ],
      "metadata": {
        "id": "6CCPYhxrZCzn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df['class'].unique()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2JrjxTKEY-4h",
        "outputId": "d8423b09-3361-49f4-ac1c-a1c05beb9521"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 0])"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Let's plot histograms of our labels."
      ],
      "metadata": {
        "id": "BuXmJ8nEgGnt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "for label in df.columns[:-1]:\n",
        "    plt.figure(figsize=(2, 2))\n",
        "    plt.hist(\n",
        "        df[df[\"class\"] == 1][label],\n",
        "        color=\"blue\",\n",
        "        label=\"gamma\",\n",
        "        alpha=0.7,\n",
        "        density=True,\n",
        "    )\n",
        "    plt.hist(\n",
        "        df[df[\"class\"] == 0][label],\n",
        "        color=\"red\",\n",
        "        label=\"hedron\",\n",
        "        alpha=0.5,\n",
        "        density=True,\n",
        "    )\n",
        "    plt.title(label)\n",
        "    plt.xlabel(label)\n",
        "    plt.ylabel(\"Probability\")\n",
        "    plt.legend()\n",
        "    plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "GPYejRiYeWpW",
        "outputId": "2a7df3dc-324c-47e8-d9c0-a5c2430a1916"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAANUAAADvCAYAAABloZFIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAkBElEQVR4nO3deVhU1f8H8PewzLANAyKyCLIkuYK7Bn1z5ZtLoZahqSkYX41E1B+ahqm4Bi6lqeRTloImmZJimglKoGUuCGIGuCEiJoip7LHNnN8fxM1xEGeGCzMDn9fzzCNz59x7P5fh47n33HvOETDGGAghvNHTdACEtDaUVITwjJKKEJ5RUhHCM0oqQnhGSUUIzyipCOEZJRUhPKOkIoRnlFRaJCUlBV5eXjA1NYVAIEB6erqmQyJqoKTSEjU1NfD19cWjR4+wadMm7NmzB05OTgCA9PR0vPPOO3B0dIRIJEK7du3g7e2NXbt2QSqVajhyYO3atRg7dixsbGwgEAiwYsUKTYekUQaaDoDUyc7ORm5uLnbs2IH//e9/3PKvvvoKgYGBsLGxwbRp0+Dm5obS0lIkJiYiICAA+fn5WLJkiQYjB5YuXQpbW1v06dMH8fHxGo1FG1BSaYnCwkIAgIWFBbfs3LlzCAwMhKenJ44dOwaxWMx9Nn/+fFy8eBF//PFHS4eqICcnB87Ozvjrr79gbW2t6XA0jxGN8/PzYwDkXkOGDGGjRo1iBgYGLDc3V6ntlJWVsZCQEObg4MCEQiF78cUX2YYNG5hMJpMrB4AFBQWxQ4cOsR49ejChUMi6d+/OfvrpJ4Vt3r17l7377rvMzs6OCYVC5uzszAIDA1lVVZVC2QcPHjAALCwsTK3fQ2tBNZUWeO+999CxY0d8/PHHmDt3LgYMGACxWAxfX18MHjwYnTp1eu42GGMYO3YskpKSEBAQgN69eyM+Ph4ffPAB/vzzT2zatEmu/K+//oqDBw9i9uzZEIvF2LJlCyZMmIA7d+7AysoKAHDv3j0MHDgQRUVFmDVrFrp27Yo///wTsbGxqKiogFAobJbfh87TdFaTOklJSQwAO3DgAGOMscuXLzMAbN68eUqtHxcXxwCwNWvWyC1/6623mEAgYDdv3uSWAWBCoVBuWf3+tm7dyi2bPn0609PTYykpKQr7e7r2Y4xqqnrU+qelSkpKAEDuOqoxx44dg76+PubOnSu3fMGCBWCM4aeffpJb7u3tjRdeeIF77+HhAXNzc9y6dQsAIJPJEBcXBx8fH/Tv319hfwKBQKXjaUsoqbSUubk5AKC0tFSp8rm5ubC3t1dIwm7dunGfP6mhU0pLS0s8fvwYAPDgwQOUlJSgZ8+eKsfe1lFSaanOnTvDwMAAV65caZbt6+vrN7ic0egKTUZJpaVMTEwwfPhwnD59Gnl5ec8t7+TkhHv37inUbFevXuU+V4W1tTXMzc21osle11BSabGwsDAwxjBt2jSUlZUpfJ6amoro6GgAwJgxYyCVSrFt2za5Mps2bYJAIMDo0aNV2reenh7Gjx+PI0eO4OLFiwqfU432bNSkrsW8vLwQGRmJ2bNno2vXrnJPVCQnJ+OHH37AmjVrAAA+Pj4YNmwYPvroI9y+fRu9evVCQkICDh8+jPnz58s1Sijr448/RkJCAoYMGYJZs2ahW7duyM/Px4EDB/Drr79yN6r37NmD3NxcVFRUAABOnz7NxTVt2jSVa0mdp9nGR1Lv6Sb1J6WmprIpU6Ywe3t7ZmhoyCwtLdmIESNYdHQ0k0qlXLnS0lL2f//3f1w5Nze3Rm/+Ps3JyYn5+fnJLcvNzWXTp09n1tbWTCQSMVdXVxYUFCR383fIkCEKN6/rX0lJSU37xeggAWNUjxPCJ7qmIoRnlFSE8IySihCeUVIRwjNKKkJ4pjVJFRERAYFAgPnz52s6FEKaRCtu/qakpOCLL76Ah4eHSuvJZDLcu3cPYrGYnpomzYoxhtLSUtjb20NPr/G6SONJVVZWhqlTp2LHjh3cXXhl3bt3D46Ojs0UGSGK8vLy4ODg0GgZjSdVUFAQXnvtNXh7ez83qaqqqlBVVcW9r79vnZeXx3WVIKQ5lJSUwNHRUan+bRpNqn379iEtLQ0pKSlKlQ8PD8fKlSsVlpubm1NSkRahzGWGxhoq8vLyMG/ePOzduxdGRkZKrRMaGori4mLupUyXCEJamsae/YuLi8Mbb7wh11lOKpVCIBBAT08PVVVVz+xIV6+kpAQSiQTFxcVUU5FmpcrfmsZO/0aMGKHQq3XGjBno2rUrFi9e/NyEIkRbaSypxGKxwvgHpqamsLKyahPjIjDGUFtbqxXDNpO64QUMDAx4uTWj8da/1sjHp/HPxeJqDB+eD1fXCtRXyB06NH9cpHEmJiaws7Nr8niGWpVUycnJmg6h2enryzB5cg5sbfVhZmYPgUAIQABnZ01H1nYxxlBdXY0HDx4gJycHbm5uz73B2xitSqq2QCKphlgsg1jsCH19E265kg2gpJkYGxvD0NAQubm5qK6uVrpFuiFa8+xfW6GnB9SdttOvXts0pXaS2w4vWyGEcCipCOEZJRUhPKOGCi3yvKZ4Ph050nL7amuopiKEZ5RUjfDxUe/VWpWWlmLq1KkwNTWFnZ0dNm3ahKFDh3K9tffs2YP+/ftDLBbD1tYWU6ZM4aZdBeruQwoEAsTHx6NPnz4wNjbG8OHDUVhYiJ9++gndunWDubk5pkyZwo12CwBDhw5FcHAw5s+fD0tLS9jY2GDHjh0oLy/HjBkzIBaL0blzZ7npgqRSKQICAuDi4gJjY2N06dIFn332WYv8niipiNJCQkJw5swZ/PDDDzhx4gR++eUXpKWlcZ/X1NRg9erVuHz5MuLi4nD79m34+/srbGfFihXYtm0bfvvtN+Tl5WHixInYvHkzYmJi8OOPPyIhIQFbt26VWyc6Ohrt27fHhQsXEBwcjPfffx++vr7w8vJCWloaXn31VUybNo1LRplMBgcHBxw4cACZmZlYvnw5lixZgv379zfr7wjQ4FPqfGjup9Sbo9Zp374S/v45sLFxgb7+vzcY3dy0+5qqtLQUVlZWiImJwVtvvQUAKC4uhr29PWbOnInNmzcrrHPx4kUMGDAApaWlMDMzQ3JyMoYNG4aTJ09ixIgRAOrGJgkNDUV2djZcXV0BAIGBgbh9+zaOHz8OoK6mkkql+OWXXwDU1UISiQRvvvkmdu/eDQAoKCiAnZ0dzp49i5deeqnBY5gzZw4KCgoQGxvb4OeVlZXIycmBi4uLws1fVf7WqKYiSrl16xZqamowcOBAbplEIkGXLl2496mpqfDx8UGnTp0gFosxZMgQAMCdO3fktvXkWCQ2NjYwMTHhEqp+2ZOnjU+vo6+vDysrK7i7u8utA0BuvcjISPTr1w/W1tYwMzPDl19+qRBLc6CkIrwoLy/HyJEjYW5ujr179yIlJQWHDh0CAFRXV8uVNTQ05H4WCARy7+uXyWSyZ67T0Hr1T5fXr7dv3z4sXLgQAQEBSEhIQHp6OmbMmKEQS3OgJnWiFFdXVxgaGiIlJYWb2rS4uBjXr1/H4MGDcfXqVTx8+BARERHcYDwNzWvVUs6cOQMvLy/Mnj2bW5adnd0i+6aaiihFLBbDz88PH3zwAZKSkpCRkYGAgADo6elBIBCgU6dOEAqF2Lp1K27duoUffvgBq1ev1li8bm5uuHjxIuLj43H9+nUsW7ZM6bFQmopqKi2i7TdkP/30UwQGBuL111+Hubk5Fi1ahLy8PBgZGcHa2hpRUVFYsmQJtmzZgr59+2Ljxo0YO3asRmJ97733cOnSJUyaNAkCgQCTJ0/G7Nmz5Zrdmwu1/jWipVv/dE15eTk6duyITz75BAEBAZoOp8n4av2jmooo7dKlS7h69SoGDhyI4uJirFq1CgAwbtw4DUemXSipiEo2btyIa9euQSgUol+/fvjll1/Qvn17TYelVSipiNL69OmD1NRUTYeh9aj1jxCeUVIRwjNKKkJ4RklFCM8oqQjhGSUVITyjJnVtsmKFVu9r6NCh6N27d4N9p9QPYwXi4uKQnp7O2zY1jZKKJ5OvrVCqnLDUDKY1L0NSXYgy407NGxTRCI2e/m3fvh0eHh7cTIienp4t8sAj0R0t0f+JbxqtqRwcHBAREQE3NzcwxhAdHY1x48bh0qVL6NGjh+YC++fUaPI1zYWgrWQyGRYtWoSvvvoKQqEQgYGBWPHP76uoqAgLFy7E4cOHUVVVhf79+2PTpk3o1asXt35ERAQ2bdqEiooKTJw4EdbW1nLb9/f3R1FREQYMGIDIyEiIRCLk5OTgypUrmDdvHs6ePQsTExNMmDABn376KczMzOTW+89//oNPPvkE1dXVePvtt7F582aFDo7NTaM1lY+PD8aMGQM3Nze8+OKLWLt2LczMzHDu3DlNhkUaER0dDVNTU5w/fx7r16/HqlWrcOLECQCAr68vNzJSamoq+vbtixEjRuDRo0cAgP3792PFihX4+OOPcfHiRdjZ2eHzzz9X2EdiYiKuXbuGEydO4OjRo1yvYktLS6SkpODAgQM4efIk5syZI7deUlISsrOzkZSUhOjoaERFRSEqKqrZfydP05prKqlUigMHDqC8vByenp4Nlnl6dvqSkpKWCo/8w8PDA2FhYQDqOgJu27YNiYmJMDY2xoULF1BYWAiRSASg7uHbuLg4xMbGYtasWdi8eTMCAgK4biJr1qzByZMnUVlZKbcPU1NTriYEgB07dqCyshK7d++GqakpAGDbtm3w8fHBunXruPEpLC0tsW3bNujr66Nr16547bXXkJiYiJkzZ7bI76aexpvUr1y5AjMzM4hEIgQGBuLQoUPo3r17g2XDw8MhkUi4V323bdJynhyABQDs7OxQWFiIy5cvo6ysDFZWVjAzM+NeOTk5XDf2rKwsDBo0SG79hv4DdXd3l5t4LSsrC7169eISCgBefvllyGQyXLv27zl6jx495Ka1rY+tpWm8purSpQvS09NRXFyM2NhY+Pn54dSpUw0mVmhoKEJCQrj3JSUllFgt7FmDtJSVlcHOzq7BifssLCxU2seTycNHbC1N40klFArRuXNnAEC/fv2QkpKCzz77DF988YVCWZFIxJ1atAbtKu/9++YegNJS5VYUi5slnqbo27cvCgoKYGBgAOdnTAvZrVs3nD9/HtOnT+eWKXP93K1bN0RFRaG8vJxLuDNnzkBPT09uiDRtodbpX1JSEt9xcGQymdx1E9EN3t7e8PT0xPjx45GQkIDbt2/jt99+w0cffcSNqjRv3jzs3LkTu3btwvXr1xEWFoaMjIznbnvq1KkwMjKCn58f/vjjDyQlJSE4OBjTpk3jrqe0iVo11ahRo+Dg4IAZM2bAz89P7VOw0NBQjB49Gp06dUJpaSliYmKQnJyM+Ph4tban8xYsUK6cvX3zxqEGgUCAY8eO4aOPPsKMGTPw4MED2NraYvDgwdwf/qRJk5CdnY1FixahsrISEyZMwPvvv//c79vExATx8fGYN28eBgwYINekro3UGvjlr7/+wp49exAdHY2MjAwMHz4cAQEBGD9+vEozewcEBCAxMRH5+fmQSCTw8PDA4sWL8d///lep9Ztt4Jd/7rvExPC3yXpCezM4L30ZHTt0hFDw7/9pVlYqbEQLk6o14GvglyaPppSWloZdu3bh22+/BQBMmTIFAQEBcjf8mktrSipVPDL6N6l0cRQmbaU1Y6n37dsXoaGhmDNnDsrKyrBz507069cPr7zyilLny4S0NmonVU1NDWJjYzFmzBg4OTkhPj4e27Ztw/3793Hz5k04OTnB19eXz1gJ0QlqnX8EBwfj22+/BWMM06ZNw/r169GzZ0/uc1NTU2zcuBH2dO5P2iC1kiozMxNbt27Fm2+++cz7Ru3bt2/WpnddxWQMjAFgDBBoOhryJL4Ga1br9C8sLAy+vr4KCVVbW4vTp08DAAwMDLj5ici/pMVVkNVKUc1qNB0KeUr9LIxNfapdrZpq2LBhyM/PR4cOHeSWFxcXY9iwYZBKpU0KqjWT/V2Lx0m5MBwjBCwAocAQEKhWZUml/z6A+tSzqEQNjDFUVFSgsLAQFhYWcs8PqkOtpGKMcZNsPenhw4dqP7fVljw6fAMAUDPMCXoG+qrmFMoNy7mfdXd6Ce1jYWEBW1vbJm9HpaR68803AdTdPff395c7/ZNKpfj999/h5eXV5KBaPQY8iruBx8dvwcDCCAI91bLqR5d/+xFt3853cG2ToaFhk2uoeiollUQiAVBXU4nFYhgbG3OfCYVCvPTSSy3ed0WXsUopagrKn1/wKX+J/70x+dQ9SqIFVEqqXbt2AQCcnZ2xcOFCOtUjpAFqXVPV9/wkhChSOqn69u2LxMREWFpaok+fPg02VNRLS0vjJThCdJHSSTVu3DiuYWL8+PHNFQ8hOk/ppHrylI9O/wh5No0P/EJIa6N0TWVpadnoddST6sd5I6QtUjqp+ByUnpDWTOmk8vPza844CGk1lE6qkpISrhvx80aG5bVrOyE6RqVrqvon0y0sLBq8vqp/0JaeUidtmdJJ9fPPP6Ndu3YAmnfcP0J0ndJJ9WSHQ+p8SMizqT3s8+PHj/H1118jKysLANC9e3fMmDGDq81I85GbtXHFs0o9pSWnPm3j1Lr5e/r0aTg7O2PLli14/PgxHj9+jC1btsDFxYXrTk9IW6VWTRUUFIRJkyZh+/btXMcuqVSK2bNnIygoCFeuXOE1SEJ0iVo11c2bN7FgwQK5npL6+voICQnBzZs3eQuOEF2kVlL17duXu5Z6Uv3kXIS0ZUqf/v3+++/cz3PnzsW8efNw8+ZNvPTSSwDq5hmKjIxERESE0jsPDw/HwYMHcfXqVRgbG8PLywvr1q3TyjmHCFGW0knVu3dvCAQCuQEHFy1apFBuypQpmDRpklLbPHXqFIKCgjBgwADU1tZiyZIlePXVV5GZmUld9YnOUjqpcnJyeN/58ePH5d5HRUWhQ4cOSE1NxeDBg3nfHyEtQemkcnJyas44ANQNxgngmfe6aHZ6oguaNOdvZmYm7ty5g+rqarnlY8eOVXlbMpkM8+fPx8svvyw32cGTwsPDsXLlSrViJaSlqJVUt27dwhtvvIErV67IXWfVP2SrzgO1QUFB+OOPP/Drr78+swzNTk90gVpN6vPmzYOLiwsKCwthYmKCjIwMnD59Gv3790dycrLK25szZw6OHj2KpKQkODg4PLOcSCSCubm53IsQbaNWTXX27Fn8/PPPaN++PfT09KCnp4f//Oc/CA8Px9y5c3Hp0iWltsMYQ3BwMA4dOoTk5GS4uLioEw4hWkWtmkoqlUIsFgOom4fq3r17AOoaM65du6b0doKCgvDNN98gJiYGYrEYBQUFKCgowN9//61OWIRoBbVqqp49e+Ly5ctwcXHBoEGDsH79egiFQnz55ZdwdXVVejvb/xldf+jQoXLLd+3aBX9/f3VCI0Tj1EqqpUuXory8bmD9VatW4fXXX8crr7wCKysrfPfdd0pvh6+Z69qymBjlyn2bKv/+yBH+YyF11EqqkSNHcj937twZV69exaNHj1QaxoyQ1qpJ96kAIC8vDwCoaZuQf6jVUFFbW4tly5ZBIpHA2dkZzs7OkEgkWLp0KWpqaC5b0rapVVMFBwfj4MGDWL9+PTw9PQHUNbOvWLECDx8+5BogCGmL1EqqmJgY7Nu3D6NHj+aWeXh4wNHREZMnT6akIm2aWqd/IpEIzs7OCstdXFwgFAqbGhMhOk2tpJozZw5Wr14t98R4VVUV1q5dizlz5jSyJiGtn9Knf/Uz09c7efIkHBwcuO7zly9fRnV1NUaMGMFvhIToGKWTqn5m+noTJkyQe09N6oTUUTqp6memJ4Q0rkk3fx88eMA9QNulSxdYW1vzEhQhukythory8nK8++67sLOzw+DBgzF48GDY29sjICAAFRUVfMdIiE5RK6lCQkJw6tQpHDlyBEVFRSgqKsLhw4dx6tQpLFiwgO8YCdEpap3+ff/994iNjZXrsjFmzBgYGxtj4sSJdPNXC8lNagDQxAbNSK2aqqKiAjY2NgrLO3ToQKd/pM1TK6k8PT0RFhaGyspKbtnff/+NlStXcs8CEtJWqXX6t3nzZowaNUrh5q+RkRHi4+N5DZAQXaNWUrm7u+PGjRvYu3cvrl69CgCYPHkypk6dCmNjY14DJETXqJxUNTU16Nq1K44ePYqZM2c2R0yE6DSVr6kMDQ3lrqUIIfLUaqgICgrCunXrUFtby3c8hOg8ta6pUlJSkJiYiISEBLi7uytMe3Pw4EFegmsyusfyTDQKU/NRK6ksLCwUnlInhNRRKalkMhk2bNiA69evo7q6GsOHD8eKFSuoxY+QJ6h0TbV27VosWbIEZmZm6NixI7Zs2YKgoKDmio0QnaRSUu3evRuff/454uPjERcXhyNHjmDv3r2QyWTNFR8hOkelpLpz5w7GjBnDvff29oZAIOAmKCCEqJhUtbW1MDIykltmaGio9gCap0+fho+PD+zt7SEQCBAXF6fWdgjRJio1VDDG4O/vD5FIxC2rrKxEYGCgXLO6sk3q5eXl6NWrF959912FgWUI0VUqJZWfn5/CsnfeeUftnY8ePVpuQE5CWgOVkkrTg7/Q7PREF6j1mJKmhIeHQyKRcC8aFo1oI51KqtDQUBQXF3Ov+ml8CNEmTZ6fqiWJRCK5RhJCtJFO1VSE6AKN1lRlZWW4efMm9z4nJwfp6elo164dOnXqpMHICFGfRpPq4sWLGDZsGPc+JCQEQF3TfVRUlIaiIqRpNJpUQ4cOpRnqSaujUw0VpOWpNQhnG+8cSg0VhPCMkooQnlFSEcIzSipCeEZJRQjPqPWPqESZoc2eHtYMaFtDm7WJpFJ2jDtC+ECnf4TwjJKKEJ5RUhHCM0oqQnhGSUUIzyipCOEZJRUhPGsT96lIy1LoLgK0qS4jVFMRwjNKKkJ4RklFCM8oqQjhGSUVITyjpCKEZ5RUhPCMkooQntHNX9Ii2lKPYaqpCOEZ1VREa6j1eJMWPtqkFTVVZGQknJ2dYWRkhEGDBuHChQuaDokQtWk8qb777juEhIQgLCwMaWlp6NWrF0aOHInCwkJNh0aIWgRMw9NuDBo0CAMGDMC2bdsAADKZDI6OjggODsaHH37Y6LolJSWQSCQoLi6Gubm5YoF/Tg1oNKW2Z8qURj5U45TxuX9rT9DoNVV1dTVSU1MRGhrKLdPT04O3tzfOnj2rUP7p2emLi4sBNDJL/T9lK6Q8Bk10wld7nv1Z7Lln/L0A2L+/4eX1f2PK1EEaTaq//voLUqkUNjY2csttbGxw9epVhfLh4eFYuXKlwnKapZ6o5FbEMz+SSBpftbS0FJLnFNKp1r/Q0FButkWg7lTx0aNHsLKygkAgkCtbUlICR0dH5OXlPbe61gWt7XgA3TomxhhKS0thb2//3LIaTar27dtDX18f9+/fl1t+//592NraKpRvaHZ6CwuLRvdhbm6u9V+YKlrb8QC6c0zPq6HqabT1TygUol+/fkhMTOSWyWQyJCYmwtPTU4OREaI+jZ/+hYSEwM/PD/3798fAgQOxefNmlJeXY8aMGZoOjRC1aDypJk2ahAcPHmD58uUoKChA7969cfz4cYXGC1WJRCKEhYUpnC7qqtZ2PEDrPCZAC+5TEdLaaPyJCkJaG0oqQnhGSUUIzyipCOGZTiWVql1EDhw4gK5du8LIyAju7u44duyY3OeMMSxfvhx2dnYwNjaGt7c3bty40ZyHIEeV49mxYwdeeeUVWFpawtLSEt7e3grl/f39IRAI5F6jRo1q7sPgqHI8UVFRCrEaGRnJldH096M2piP27dvHhEIh27lzJ8vIyGAzZ85kFhYW7P79+w2WP3PmDNPX12fr169nmZmZbOnSpczQ0JBduXKFKxMREcEkEgmLi4tjly9fZmPHjmUuLi7s77//1rrjmTJlCouMjGSXLl1iWVlZzN/fn0kkEnb37l2ujJ+fHxs1ahTLz8/nXo8ePWr2Y1HneHbt2sXMzc3lYi0oKJAro8nvpyl0JqkGDhzIgoKCuPdSqZTZ29uz8PDwBstPnDiRvfbaa3LLBg0axN577z3GGGMymYzZ2tqyDRs2cJ8XFRUxkUjEvv3222Y4AnmqHs/TamtrmVgsZtHR0dwyPz8/Nm7cOL5DVYqqx7Nr1y4mkUieuT1Nfz9NoROnf/VdRLy9vblljXURAYCzZ8/KlQeAkSNHcuVzcnJQUFAgV0YikWDQoEHP3CZf1Dmep1VUVKCmpgbt2rWTW56cnIwOHTqgS5cueP/99/Hw4UNeY2+IusdTVlYGJycnODo6Yty4ccjIyOA+0+T301Q6kVSNdREpKChocJ2CgoJGy9f/q8o2+aLO8Txt8eLFsLe3l/ujGzVqFHbv3o3ExESsW7cOp06dwujRoyGVNm+HMnWOp0uXLti5cycOHz6Mb775BjKZDF5eXrh79y4AzX4/TaXxx5SI6iIiIrBv3z4kJyfLXdy//fbb3M/u7u7w8PDACy+8gOTkZIwYMUIToT6Tp6en3EPTXl5e6NatG7744gusXr1ag5E1nU7UVKp2EQEAW1vbRsvX/6vKNvmizvHU27hxIyIiIpCQkAAPD49Gy7q6uqJ9+/a4efNmk2NuTFOOp56hoSH69OnDxarJ76epdCKp1Oki4unpKVceAE6cOMGVd3Fxga2trVyZkpISnD9/vtm7najb5WX9+vVYvXo1jh8/jv79+z93P3fv3sXDhw9hZ2fHS9zPwkcXHqlUiitXrnCxavL7aTJNt5Qoa9++fUwkErGoqCiWmZnJZs2axSwsLLhm2GnTprEPP/yQK3/mzBlmYGDANm7cyLKyslhYWFiDTeoWFhbs8OHD7Pfff2fjxo1r0SZ1VY4nIiKCCYVCFhsbK9cMXVpayhhjrLS0lC1cuJCdPXuW5eTksJMnT7K+ffsyNzc3VllZqXXHs3LlShYfH8+ys7NZamoqe/vtt5mRkRHLyMiQO2ZNfT9NoTNJxRhjW7duZZ06dWJCoZANHDiQnTt3jvtsyJAhzM/PT678/v372YsvvsiEQiHr0aMH+/HHH+U+l8lkbNmyZczGxoaJRCI2YsQIdu3atZY4FMaYasfj5OTEACi8wsLCGGOMVVRUsFdffZVZW1szQ0ND5uTkxGbOnKlw70dbjmf+/PlcWRsbGzZmzBiWlpYmtz1Nfz/qoq4fhPBMJ66pCNEllFSE8IySihCeUVIRwjNKKkJ4RklFCM8oqQjhGSUVITyjpNJijDHMmjUL7dq1g0AgQHp6uqZDIkqgpNJix48fR1RUFI4ePYr8/Hz07NkTBQUFCA4OhqurK0QiERwdHeHj46Pw8HBzW7t2Lby8vGBiYvLcSSLaGupPpcWys7NhZ2cHLy8vAMDt27fx8ssvw8LCAhs2bIC7uztqamoQHx+PoKCgBuf0ai7V1dXw9fWFp6cnvv766xbbr07Q8LOH5Bn8/PzkHpx1cnJio0ePZh07dmRlZWUK5R8/fsz9nJuby8aOHctMTU2ZWCxmvr6+cg/WhoWFsV69erHdu3czJycnZm5uziZNmsRKSkq4MlKplK1bt4698MILTCgUMkdHR7ZmzRqF/T5vrIm2iE7/tNRnn32GVatWwcHBAfn5+Th//jyOHz+OoKAgmJqaKpSvPwWTyWQYN24cHj16hFOnTuHEiRO4desWJk2aJFc+OzsbcXFxOHr0KI4ePYpTp04hIuLfGQZDQ0MRERGBZcuWITMzEzExMU2eNKLN0HRWk2fbtGkTc3JyYowxdv78eQaAHTx4sNF1EhISmL6+Prtz5w63LCMjgwFgFy5cYIzV1VQmJiZyNdMHH3zABg0axBhjrKSkhIlEIrZjx47nxkg1lSKqqXQEU7KHTlZWFhwdHeXmQe7evTssLCyQlZXFLXN2doZYLObe29nZobCwkNtGVVWV1o1roSsoqXSEm5sbBAIBb40RhoaGcu8FAgFkMhkAwNjYmJd9tFWUVDqiXbt2GDlyJCIjI1FeXq7weVFREQCgW7duyMvLQ15eHvdZZmYmioqK0L17d6X25ebmBmNj4xZvpm8tKKl0SGRkJKRSKQYOHIjvv/8eN27cQFZWFrZs2cINhuLt7Q13d3dMnToVaWlpuHDhAqZPn44hQ4YoNVgMABgZGWHx4sVYtGgRdu/ejezsbJw7d06u6fzOnTtIT0/HnTt3IJVKkZ6ejvT0dJSVlTXLsesSuk+lQ1xdXZGWloa1a9diwYIFyM/Ph7W1Nfr164ft27cDqDuNO3z4MIKDgzF48GDo6elh1KhR2Lp1q0r7WrZsGQwMDLB8+XLcu3cPdnZ2CAwM5D5fvnw5oqOjufd9+vQBACQlJWHo0KFNP1gdRmNUEMIzOv0jhGeUVITwjJKKEJ5RUhHCM0oqQnhGSUUIzyipCOEZJRUhPKOkIoRnlFSE8IySihCe/T9CRqIvReZdZAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPwAAADvCAYAAADW6blSAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAzFUlEQVR4nO3deVxUVf8H8M+w7/syoAhYJC4sCjJCpGb8xCSTMkUyRcNcQtTQTEzBrVBwIZckC3EJQql+PGqmEVk+P5kQEFRMUQxCHxjEBVCS/fz+4OHmdQBZBgeY7/v1mhfMueeee+5lvpw7995zjoAxxkAIUQhK8q4AIeTZoYAnRIFQwBOiQCjgCVEgFPCEKBAKeEIUCAU8IQqEAp4QBUIBT4gCoYAnHZaRkQEPDw9oa2tDIBAgJydH3lUi7UQBTzqkrq4OU6dOxb1797B9+3YcOnQI1tbW3PIVK1ZAIBDAz89PjrUkrRHQs/SkI65evYrBgwfjyy+/xNy5c3nLGGMYMGAAVFRUUFpaitLSUujq6sqppqQl1MKTDrl9+zYAwMDAQGrZr7/+ilu3bmHfvn2or6/H999//4xrR56GAp602+zZszFmzBgAwNSpUyEQCDB27FhueXx8PIYMGYKXX34ZXl5eiI+Pb7GcnTt3YujQodDS0oKhoSFcXV2RkJAAADh9+jQEAgH+93//V2q9hIQECAQCiMVirj46OjooKirCa6+9Bh0dHfTr1w+7d+8GAFy6dAnjxo2DtrY2rK2tuW0oNEZIO6WlpbFVq1YxAGzx4sXs0KFD7KeffmKMMVZdXc0MDAzYhg0bGGOMHTx4kCkrK7OSkhJeGXv37mUA2FtvvcW++OIL9tlnn7HAwEC2ePFixhhjjY2NzMrKik2ZMkVq+xMnTmTPPfcc9z4gIIBpaGiwIUOGsAULFrDdu3czDw8PBoDFxcUxS0tL9uGHH7KdO3eyoUOHMmVlZfbnn3921+HpFSjgSYecPn2aAWBJSUm89G+//ZYBYNevX2eMMVZZWck0NDTY9u3befkmT57Mhg4d2uY2QkNDmbq6OisvL+fSbt++zVRUVFh4eDiXFhAQwACwTz/9lEu7f/8+09TUZAKBgCUmJnLpV69eZQB46ysiOqUnMhEfHw9XV1c8//zzAABdXV34+PhIndYbGBjg1q1byMjIaLWsWbNmoaamBt9++y2XdvjwYdTX1+Odd96Ryv/4xUMDAwMMGjQI2tramDZtGpc+aNAgGBgY4M8//+z0PvYFFPCky8rLy3HixAmMGTMG+fn53OvFF19EZmYmrl27xuX96KOPoKOjAzc3N9jZ2SEoKAhnz57llWdvb4+RI0fy/lnEx8dj1KhR3D+UZhoaGjA1NeWl6evro3///hAIBFLp9+/fl9Vu90oU8KTLkpKSUFNTg61bt8LOzo57hYSEAAAvcAcPHoy8vDwkJibC09MT3333HTw9PREeHs4rc9asWfjtt99w69Yt3LhxA7///nuLrbuysnKLdWotnSn4XWgVeVeA9H7x8fEYNmyYVNACwBdffIGEhASsW7eOS9PW1oafnx/8/PxQW1uLN998E5988glCQ0OhoaEBAJg+fTpCQkLwzTff4NGjR1BVVaWHeWSAAp50yc2bN3HmzBmsW7cOb731ltTy2tpazJgxA+np6RCJRLh79y6MjY255WpqahgyZAh+/PFH1NXVcQFvYmKCV199FV9//TWqq6sxYcIEmJiYPLP96qso4EmXJCQkgDGG119/vcXlEydOhIqKCuLj4yESiTB+/HgIhUK8+OKLMDc3x5UrV7Br1y74+PhIPZU3a9Ys7p/Ihg0bun1fFAF9hyddEh8fjwEDBsDJyanF5QYGBvD09OSuss+fPx8PHz7Etm3bEBQUhOTkZCxevBhff/211LqTJk2CoaEh9PX1W/2HQjqGnqUnPVZ9fT0sLS0xadIkxMbGyrs6fQK18KTHSk5ORllZGWbNmiXvqvQZ1MKTHic9PR0XL17Ehg0bYGJigvPnz8u7Sn0GtfCkx9mzZw8WLlwIMzMzHDx4UN7V6VOohSdEgVALT4gCoYAnRIHQgzfdqLGxEcXFxdDV1ZXqyEGILDHG8ODBA1haWkJJqfV2nAK+GxUXF8PKykre1SAK5ObNm+jfv3+ryyngu1Hzo6I3b96Enp6enGtD+rLKykpYWVk9ddBQCvhu1Hwar6enRwFPnomnfXWki3aEKBAKeEIUCAU8IQqEvsMTKYwx1NfXo6GhQd5VIf+lrKwMFRWVLt/epYAnPLW1tSgpKcHff//dLeX/d+KaVpmZdctm+wQtLS1YWFhATU2t02VQwBNOY2MjCgoKoKysDEtLS6ipqcn8gaGnFWdjI9PN9QmMMdTW1qKsrAwFBQWws7Nr8+GatlDAE05tbS0aGxthZWUFLS2tbtlGK4PJcv47pB15gqamJlRVVfHXX3+htraWG/uvo+iiHZHS2daDdC9Z/F3oL0uIApF7wO/evRs2NjbQ0NCASCTCuXPn2syflJQEe3t7aGhowMHBASdOnOAt//777zF+/HgYGxtDIBAgJydHqozq6moEBQXB2NgYOjo6mDJlCkpLS3l5ioqK4OPjAy0tLZiZmeHDDz9EfX19l/eXEHmSa8AfPnwYISEhCA8Px/nz5+Hk5ARvb29uDvInpaWlwd/fH4GBgcjOzoavry98fX2Rm5vL5amqqoKnpyc2b97c6nY/+OADHDt2DElJSfjtt99QXFyMN998k1ve0NAAHx8f1NbWIi0tDQcOHMD+/fsRFhYmu50nRA7kOuKNSCTCyJEjsWvXLgDgLhgFBwdj5cqVUvn9/PxQVVWF48ePc2mjRo2Cs7MzYmJieHkLCwtha2uL7OxsODs7c+kVFRUwNTVFQkICN+b51atXMXjwYIjFYowaNQo//vgjXnvtNRQXF8Pc3BwAEBMTg48++ghlZWXtvi1SWVkJfX19VFRU9Ipn6aurq1FQUABbW1upi0KTJslmG1VVbS/X1m76eeyYbLbXl7T192nvZ01uLXxtbS2ysrLg5eX1T2WUlODl5QWxWNziOmKxmJcfALy9vVvN35KsrCzU1dXxyrG3t8eAAQO4csRiMRwcHLhgb95OZWUlLl++3GrZNTU1qKys5L0I6UnkFvB37txBQ0MDL6gAwNzcHBKJpMV1JBJJh/K3VoaamhoMDAxaLae17TQva01ERAT09fW5F/WFf7YePHiAGTNmQFtbGxYWFti+fTvGjh2LpUuXAgAOHToEV1dX6OrqQigU4u233+Z9ffz1118hEAhw6tQpDB8+HJqamhg3bhxu376NH3/8EYMHD4aenh7efvtt3oNJY8eORXBwMJYuXQpDQ0OYm5vjyy+/RFVVFebMmQNdXV08//zz+PHHH7l1GhoaEBgYCFtbW2hqamLQoEH47LPPuv0Yyf2iXV8SGhqKiooK7nXz5k15V0mhhISE4OzZszh69ChSUlLw73//mzfEdV1dHTZs2IALFy4gOTkZhYWFmD17tlQ5a9euxa5du5CWloabN29i2rRpiI6ORkJCAn744Qf89NNP2LlzJ2+dAwcOwMTEBOfOnUNwcDAWLlyIqVOnwsPDA+fPn8f48eMxc+ZM7h9FY2Mj+vfvj6SkJPzxxx8ICwvDqlWrcOTIkW49RnJ78MbExATKyspSV8dLS0shFApbXEcoFHYof2tl1NbWory8nNfKP16OUCiUulvQvN22tqWurg51dfV214XIzoMHD3DgwAEkJCTglVdeAQDExcXB0tKSy/Puu+9yvw8cOBA7duzAyJEj8fDhQ+jo6HDLNm7ciBdffBEAEBgYiNDQUNy4cQMDBw4EALz11ls4ffo0PvroI24dJycnrF69GkDTP/5NmzbBxMQE7733HgAgLCwMe/bswcWLFzFq1CioqqryZtS1tbWFWCzGkSNHMG3aNFkfHo7cWng1NTW4uLggNTWVS2tsbERqairc3d1bXMfd3Z2XHwBSUlJazd8SFxcXqKqq8srJy8tDUVERV467uzsuXbrEO91LSUmBnp4ehgwZ0u5tkWfnzz//RF1dHdzc3Lg0fX19DBo0iHuflZWFSZMmYcCAAdDV1cWYMWMANN2CfZyjoyP3u7m5ObS0tLhgb0578k7S4+soKyvD2NgYDg4OvHUA8NbbvXs3XFxcYGpqCh0dHezdu1eqLrIm10drQ0JCEBAQAFdXV7i5uSE6Opr73gM0zR7ar18/REREAACWLFmCMWPGYOvWrfDx8UFiYiIyMzOxd+9ersx79+6hqKgIxcXFAJqCGWhqmYVCIfT19REYGIiQkBAYGRlBT08PwcHBcHd3x6hRowAA48ePx5AhQzBz5kxERkZCIpFg9erVCAoKoha8l6qqqoK3tze8vb0RHx8PU1NTFBUVwdvbG7W1tby8qqqq3O8CgYD3vjmtsbGx1XVaWq+5T0LzeomJiVi+fDm2bt0Kd3d36OrqIioqCunp6V3f2TbINeD9/PxQVlaGsLAwSCQSODs74+TJk9x/w6KiIt7jhB4eHkhISMDq1auxatUq2NnZITk5GcOGDePyHD16lPuHAQDTp08HAISHh2Pt2rUAgO3bt0NJSQlTpkxBTU0NvL298fnnn3PrKCsr4/jx41i4cCHc3d2hra2NgIAArF+/vjsPB+mCgQMHQlVVFRkZGRgwYACApluw165dw+jRo3H16lXcvXsXmzZt4i6mZmZmyq2+Z8+ehYeHB95//30u7caNG92+Xbl3nlm0aBEWLVrU4rJff/1VKm3q1KmYOnVqq+XNnj27xQsxj9PQ0MDu3buxe/fuVvNYW1tLPcVHei5dXV0EBATgww8/hJGREczMzBAeHg4lJSUIBAIMGDAAampq2LlzJxYsWIDc3Fy5zjlvZ2eHgwcP4tSpU7C1tcWhQ4eQkZEBW1vbbt2u3AOe9A6yehDm+vW2l9vZdb7sbdu2YcGCBXjttdegp6eHFStW4ObNm9DQ0ICpqSn279+PVatWYceOHRgxYgS2bNkit3nn58+fj+zsbPj5+UEgEMDf3x/vv/8+79Zdd6C55bpRX3rSTla6M+CfVFVVhX79+mHr1q0IDAyUXcFyIosn7aiFJ31GdnY2rl69Cjc3N1RUVHDXXCZPniznmvUcFPCkT9myZQvy8vK4277//ve/YWJiIu9q9RgU8KTPGD58OLKysuRdjR6NHq0lRIFQwBOiQCjgCVEgFPCEKBAKeEIUCAU8IQqEbsuR9vlvx6OuMrr7lAzGndve2LFj4ezsjOjo6E7UqmVr165FcnJyiyMf91bUwhOiQCjgCemkJ/vR9wYU8KTPaGxsxIoVK2BkZAShUMiNfwAA5eXlmDt3LkxNTaGnp4dx48bhwoULvPU3bdoEc3Nz6OrqIjAwENXV1bzls2fPhq+vLz755BNYWlpyo+lcunQJ48aNg6amJoyNjTFv3jw8fPhQar0tW7bAwsICxsbGCAoKQl1dXfcdjFZQwJM+48CBA9DW1kZ6ejoiIyOxfv16pKSkAGgaR6F59NmsrCyMGDECr7zyCu7duwcAOHLkCNauXYtPP/0UmZmZsLCw4A2K0iw1NRV5eXlISUnB8ePHuZF0DA0NkZGRgaSkJPz8889SYzycPn0aN27cwOnTp7mJTfbv39/tx+RJdNGO9BmOjo4IDw8H0DTAxK5du5CamgpNTU2cO3cOt2/f5oYo27JlC5KTk/Htt99i3rx5iI6ORmBgINeNduPGjfj555+lWnltbW189dVX3GQkX375Jaqrq3Hw4EFo/3cWjV27dmHSpEnYvHkzN3qToaEhdu3aBWVlZdjb28PHxwepqancIJfPCrXwpM94fCBJALCwsMDt27dx4cIFPHz4kJtLsPlVUFDADSt15coViEQi3votDY7q4ODAm3noypUrcHJy4oIdAF588UU0NjZy4ykCwNChQ6H82FzZzXV71qiFJ31Ga4NNPnz4EBYWFi0OmfbkhCRP83hgy6Juzxq18KTPGzFiBCQSCVRUVPD888/zXs195QcPHiw1Yuzvv//+1LIHDx6MCxcuoOqxSfPOnj0LJSUl3hDZPUWnAv706dOyrgch3cbLywvu7u7w9fXFTz/9hMLCQqSlpeHjjz/mRq5dsmQJ9u3bh7i4OFy7dg3h4eFtziPYbMaMGdDQ0EBAQAByc3Nx+vRpBAcHY+bMmVLTlfUEnTqlnzBhAvr37485c+YgICCA5lBTBDJ60u7eU8a0M5bhmHbNBAIBTpw4gY8//hhz5sxBWVkZhEIhRo8ezQWln58fbty4gRUrVqC6uhpTpkzBwoULcerUqTbL1tLSwqlTp7BkyRKMHDkSWlpamDJlCrZt2yb7HZGBTg1ieefOHRw6dAgHDhzA5cuXMW7cOAQGBsLX17fdUykrAhrEUtqzHMSyr5HbdNEmJib44IMPkJOTg/T0dLzwwgt4//33YWlpicWLF0s90EAI6Rm6fNFuxIgRCA0NxaJFi/Dw4UPs27cPLi4ueOmll9r1HQhommPLxsYGGhoaEIlEUhM5PikpKQn29vbQ0NCAg4OD1IQRjDGEhYXBwsICmpqa8PLywvXHmpbmaYFbemVkZAAACgsLW1zengs5hPRUnQ74uro6fPvtt5g4cSKsra1x6tQp7Nq1C6WlpcjPz4e1tXWbM8Q0O3z4MEJCQhAeHo7z58/DyckJ3t7erd6jTEtLg7+/PwIDA5GdnQ1fX1/4+voiNzeXyxMZGYkdO3YgJiYG6enp0NbWhre3N/cQhYeHB0pKSnivuXPnwtbWFq6urrzt/fzzz7x8Li4unT1khMgf64RFixYxY2NjZmRkxJYsWcIuXboklaekpIQJBIKnluXm5saCgoK49w0NDczS0pJFRES0mH/atGnMx8eHlyYSidj8+fMZY4w1NjYyoVDIoqKiuOXl5eVMXV2dffPNNy2WWVtby0xNTdn69eu5tIKCAgaAZWdnP3UfmlVXV7OKigrudfPmTQaAVVRUtLsMeXr06BH7448/2KNHj7ptG9eutf0irWvr71NRUdGuz1qnWvg//vgDO3fuRHFxMaKjo3mTOTYzMTF56u272tpaZGVlwcvLi0tTUlKCl5cXxGJxi+uIxWJefgDw9vbm8hcUFEAikfDy6OvrQyQStVrm0aNHcffuXd4klM1ef/11mJmZwdPTE0ePHm1zfyIiIqCvr8+9euvdC0aTEfVIsvi7dCrgw8PDMXXqVKmpk+vr63HmzBkAgIqKCjf/dmvu3LmDhoYGqfuV5ubmkEgkLa4jkUjazN/8syNlxsbGwtvbG/379+fSdHR0sHXrViQlJeGHH36Ap6cnfH192wz60NBQVFRUcK+bN2+2mrcnan4a7O+//5ZzTUhLmv8uTz611xGdug//8ssvo6SkBGZmZrz0iooKvPzyy2hoaOh0hZ61W7du4dSpUzhy5Agv3cTEBCEhIdz7kSNHori4GFFRUa1OQKiurt6r549XVlaGgYEBd/1ES0uLm9dcVp720XiirwpBU8v+999/4/bt2zAwMOA9k99RnQp4xliLH4S7d+926FljExMTKCsro7S0lJdeWloKoVDY4jpCobDN/M0/S0tLYWFhwcvj7OwsVV5cXByMjY3bNYuoSCTiulv2Vc3Hr7s6djytWPo20ToDA4NW46K9OhTwb775JoCmJ5dmz57Na80aGhpw8eJFeHh4tLu85vm/UlNT4evrC6BpEIPU1NRW54x3d3dHamoqli5dyqWlpKRwPZtsbW0hFAqRmprKBXhlZSXS09OxcOFCXlmMMcTFxWHWrFntOk3Kycnh/RPpiwQCASwsLGBmZtYtAzRERra9fM8emW+yT1BVVe1Sy96sQwGvr68PoClQdHV1oampyS1TU1PDqFGjOty/NyQkBAEBAXB1dYWbmxuio6NRVVXFXUCbNWsW+vXrh4iICABNzzyPGTMGW7duhY+PDxITE5GZmYm9e/cCaPrALl26FBs3boSdnR1sbW2xZs0aWFpacv9Umv3yyy8oKCjA3Llzpep14MABqKmpYfjw4QCA77//Hvv27cNXX33Vof3rrZSVlWXyAXvSnTttL++mB/zIf3Uo4OPi4gAANjY2WL58eae7Cj7Oz88PZWVlCAsLg0QigbOzM06ePMlddCsqKoKS0j/XFj08PJCQkIDVq1dj1apVsLOzQ3JyMu9OwYoVK1BVVYV58+ahvLwcnp6eOHnypNTjiLGxsfDw8IC9vX2LdduwYQP++usvqKiowN7eHocPH8Zbb73V5X0mRF469Sw9aZ/e9iz9szBpUtvLjx17NvXoa9r7WWt3Cz9ixAikpqbC0NAQw4cPb/Pq7fnz5ztWW0LIM9HugJ88eTJ3ke7J78KEkN6BTum7EZ3SS6NT+u7Rrd1jCSG9U7tP6Q0NDdv91FXzWN+EkJ6l3QEvy0n6CCHy0e6ADwgI6M56kL6ohXHw/POks/HXaXk9IhvtDvjKykruYkBlZWWbeekCFSE9U4e+wzf3kDMwMGjx+3xzp5re1FuOEEXS7oD/5ZdfYGRkBIDGpSekt2p3wD8+mMXTBrYghPRMnZ5b7v79+4iNjcWVK1cAAEOGDMGcOXO4swBCSM/TqQdvzpw5AxsbG+zYsQP379/H/fv3sWPHDtja2nJDXBFCep5OtfBBQUHw8/PDnj17uD7TDQ0NeP/99xEUFIRLly7JtJKEENnoVAufn5+PZcuW8QZIUFZWRkhICPLz82VWOUKIbHUq4EeMGMF9d3/clStX4OTk1OVKEUK6R7tP6S9evMj9vnjxYixZsgT5+fkYNWoUgKa5tHfv3o1NmzbJvpaEEJlod8A7OztDIBDwBsNfsWKFVL63334bfn5+sqkdIUSm2h3wBQUF3VkPQsgz0O6At7a27s56EEKegU4/eAM0zTFXVFSE2tpaXnp7JnUghDx7nQr4P//8E2+88QYuXbrE+17f3KGGOs8Q0jN16rbckiVLYGtri9u3b0NLSwuXL1/GmTNn4Orqil9//VXGVSSEyEqnAl4sFmP9+vUwMTGBkpISlJSU4OnpiYiICCxevLjD5e3evRs2NjbQ0NCASCTCuXPn2syflJQEe3t7aGhowMHBASdOnOAtZ4whLCwMFhYW0NTUhJeXF65fv87LY2NjA4FAwHs9eUvx4sWLeOmll6ChoQErKytEPm2eJEJ6uE4FfENDA3R1dQE0TQhZXFwMoOnCXl7e04Y04Tt8+DBCQkIQHh6O8+fPw8nJCd7e3q1OZpiWlgZ/f38EBgYiOzsbvr6+8PX1RW5uLpcnMjISO3bsQExMDNLT06GtrQ1vb29UPzE16fr161FSUsK9goODuWWVlZUYP348rK2tkZWVhaioKKxdu5ab0oqQ3qhTAT9s2DBcuHABQNOMqpGRkTh79izWr1+PgQMHdqisbdu24b333sOcOXMwZMgQxMTEQEtLC/v27Wsx/2effYYJEybgww8/xODBg7FhwwaMGDECu3btAtDUukdHR2P16tWYPHkyHB0dcfDgQRQXFyM5OZlXlq6uLoRCIfd6fOqs+Ph41NbWYt++fRg6dCimT5+OxYsXY9u2bR3aP0J6kk4F/OrVq9HY2AigqZUsKCjASy+9hBMnTmDHjh3tLqe2thZZWVnw8vL6p0JKSvDy8oJYLG5xHbFYzMsPAN7e3lz+goICSCQSXh59fX2IRCKpMjdt2gRjY2MMHz4cUVFRqK+v521n9OjRUFNT420nLy8P9+/fb7FuNTU1qKys5L0I6Uk6dZXe29ub+/3555/H1atXce/evQ4NZQ0Ad+7cQUNDAzdxZDNzc3NcvXq1xXUkEkmL+SUSCbe8Oa21PEDT48EjRoyAkZER0tLSEBoaipKSEq4Fl0gksLW1lSqjeZmhoaFU3SIiIrBu3bqn7jch8tKl+/AAcPPmTQCAlZVVlyvzLIWEhHC/Ozo6Qk1NDfPnz0dERARv3vuOCA0N5ZVbWVnZ644L6ds6dUpfX1+PNWvWQF9fHzY2NrCxsYG+vj5Wr16Nurq6dpdjYmICZWVllJaW8tJLS0shFApbXEcoFLaZv/lnR8oEmq5F1NfXo7CwsM3tPL6NJ6mrq0NPT4/3IqQn6VTABwcHY+/evYiMjER2djays7MRGRmJ2NjYDt2WU1NTg4uLC1JTU7m0xsZGpKamwt3dvcV13N3defkBICUlhctva2sLoVDIy1NZWYn09PRWywSAnJwcKCkpwczMjNvOmTNneP/AUlJSMGjQoBZP5wnpDTp1Sp+QkIDExES8+uqrXJqjoyOsrKzg7++PPXv2tLuskJAQBAQEwNXVFW5uboiOjkZVVRXmzJkDAJg1axb69euHiIgIAE0P/YwZMwZbt26Fj48PEhMTkZmZyd0uEwgEWLp0KTZu3Ag7OzvY2tpizZo1sLS05Ga9FYvFSE9Px8svvwxdXV2IxWJ88MEHeOedd7hgfvvtt7Fu3ToEBgbio48+Qm5uLj777DNs3769M4eMkB6hUwGvrq4OGxsbqXRbW1veVe328PPzQ1lZGcLCwiCRSODs7IyTJ09yF8iKioqgpPTPiYiHhwcSEhKwevVqrFq1CnZ2dkhOTsawYcO4PCtWrEBVVRXmzZuH8vJyeHp64uTJk9DQ0ODqn5iYiLVr16Kmpga2trb44IMPeN+/9fX18dNPPyEoKAguLi4wMTFBWFgY5s2b16H9I6Qn6dR00evXr8fVq1cRFxfHXeCqqalBYGAg7OzsEB4eLvOK9kYKP110C1NGJSS0vcrbb7e8Hmlbez9r7W7h33zzTd77n3/+Gf379+eGtLpw4QJqa2vxyiuvdLLKhJDu1u6A19fX572fMmUK7z3dfiKk52t3wMfFxXVnPUgfMWnSP78/daZY8sx16cGbsrIyrrPMoEGDYGpqKpNKEUK6R6fuw1dVVeHdd9+FhYUFRo8ejdGjR8PS0hKBgYH4+++/ZV1HQoiMdCrgQ0JC8Ntvv+HYsWMoLy9HeXk5/vWvf+G3337DsmXLZF1HQoiMdOqU/rvvvsO3336LsWPHcmkTJ06EpqYmpk2b1qEHbwghz06nWvi///5bqjcaAJiZmdEpPSE9WKcC3t3dHeHh4bwRZB49eoR169a1+bw6IUS+OnVKHx0djQkTJkg9eKOhoYFTp07JtIKEENnpVMA7ODjg+vXriI+P5waq8Pf3x4wZM6CpqSnTChJCZKfDAV9XVwd7e3scP34c7733XnfUiRDSTTr8HV5VVVVq9FdCSO/QqYt2QUFB2Lx5M2/QR0JIz9ep7/AZGRlITU3FTz/9BAcHB97wzgDw/fffy6RyhBDZ6lTAGxgYSPWWI4T0fB0K+MbGRkRFReHatWuora3FuHHjsHbtWroyT0gv0aHv8J988glWrVoFHR0d9OvXDzt27EBQUFB31Y0QImMdCviDBw/i888/x6lTp5CcnIxjx44hPj6em4WGENKzdSjgi4qKMHHiRO69l5cXBAIBN5kkIaRn61DA19fXcyO/NlNVVe3Q5BOEEPnp0EU7xhhmz57Nm4qpuroaCxYs4N2ao9tyhPRMHQr4gIAAqbR33nlHZpUhhHSvDgV8dw1kuXv3bkRFRUEikcDJyQk7d+6Em5tbq/mTkpKwZs0aFBYWws7ODps3b+ZdW2CMITw8HF9++SXKy8vx4osvYs+ePbCzswMAFBYWYsOGDfjll18gkUhgaWmJd955Bx9//DE3kUZhYaHU7LFA06w1o0aNkvERIOTZ6NSjtbJ0+PBhhISEIDw8HOfPn4eTkxO8vb1x+/btFvOnpaXB398fgYGByM7Ohq+vL3x9fZGbm8vliYyMxI4dOxATE4P09HRoa2vD29ub6wNw9epVNDY24osvvsDly5exfft2xMTEYNWqVVLb+/nnn1FSUsK9XFxcuudAEPIMdGrmGVkSiUQYOXIkdu3aBaDp4R4rKysEBwdj5cqVUvn9/PxQVVWF48ePc2mjRo2Cs7MzYmJiwBiDpaUlli1bhuXLlwMAKioqYG5ujv3792P69Okt1iMqKgp79uzBn3/+CeCfFj47OxvOzs6d2jdFnHmGP0z12g6vTzPPdE57P2tybeFra2uRlZUFLy8vLk1JSQleXl4Qi8UtriMWi3n5AcDb25vLX1BQAIlEwsujr68PkUjUaplA0z8FIyMjqfTXX38dZmZm8PT0xNGjR9vcn5qaGlRWVvJehPQkcg34O3fuoKGhQWp8PHNzc0gkkhbXkUgkbeZv/tmRMvPz87Fz507Mnz+fS9PR0cHWrVuRlJSEH374AZ6envD19W0z6CMiIqCvr8+9aDYe0tN0aSKKvuA///kPJkyYgKlTp/IG9DAxMeHNJjty5EgUFxcjKioKr7/+eotlhYaG8taprKykoCc9ilxbeBMTEygrK6O0tJSXXlpaCqFQ2OI6QqGwzfzNP9tTZnFxMV5++WV4eHhw88u3RSQSIT8/v9Xl6urq0NPT470I6UnkGvBqampwcXFBamoql9bY2IjU1NRWR791d3fn5QeAlJQULr+trS2EQiEvT2VlJdLT03ll/uc//8HYsWPh4uKCuLg43hz0rcnJyYGFhUWH9pGQnkTup/QhISEICAiAq6sr3NzcEB0djaqqKsyZMwcAMGvWLPTr1w8REREAgCVLlmDMmDHYunUrfHx8kJiYiMzMTK6FFggEWLp0KTZu3Ag7OzvY2tpizZo1sLS0hK+vL4B/gt3a2hpbtmxBWVkZV5/ms4ADBw5ATU0Nw4cPB9D09OC+ffvw1VdfPatDQ4jMyT3g/fz8UFZWhrCwMEgkEjg7O+PkyZPcRbeioiJe6+vh4YGEhASsXr0aq1atgp2dHZKTkzFs2DAuz4oVK1BVVYV58+ahvLwcnp6eOHnyJNcPICUlBfn5+cjPz0f//v159Xn8LuWGDRvw119/QUVFBfb29jh8+DDeeuut7jwchHQrud+H78voPvzaDq9P9+E7p1fchyeEPFsU8IQoEAp4QhQIBTwhCoQCnhAFIvfbcqQX6MBVc/+87qsG6Tpq4QlRIBTwhCgQOqUnPUpCAvBNVsvLjh17tnXpi6iFJ0SBUMATokAo4AlRIBTwhCgQCnhCFAgFPCEKhAKeEAVC9+FJhyQkyLsGpCuohSdEgVALT3qcVofGaiX5n+VPy0CohSdEgVALr0ioBVR41MITokAo4AlRIBTwhCiQHhHwu3fvho2NDTQ0NCASiXDu3Lk28yclJcHe3h4aGhpwcHDAiRMneMsZYwgLC4OFhQU0NTXh5eWF69ev8/Lcu3cPM2bMgJ6eHgwMDBAYGIiHDx/y8ly8eBEvvfQSNDQ0YGVlhcjISNnsMCFyIveAP3z4MEJCQhAeHo7z58/DyckJ3t7euH37dov509LS4O/vj8DAQGRnZ8PX1xe+vr7Izc3l8kRGRmLHjh2IiYlBeno6tLW14e3tjerqai7PjBkzcPnyZaSkpOD48eM4c+YM5s2bxy2vrKzE+PHjYW1tjaysLERFRWHt2rXtmmW2t0tIaP1Feje5TzUlEokwcuRI7Nq1C0DT7LFWVlYIDg7GypUrpfL7+fmhqqoKx48f59JGjRoFZ2dnxMTEgDEGS0tLLFu2DMuXLwcAVFRUwNzcHPv378f06dNx5coVDBkyBBkZGXB1dQUAnDx5EhMnTsStW7dgaWmJPXv24OOPP4ZEIoGamhoAYOXKlUhOTsbVq1fbtW/dNtVUN19t76mB/fbbT8mgwHch2vtZk+ttudraWmRlZSE0NJRLU1JSgpeXF8RicYvriMVihISE8NK8vb2RnJwMACgoKIBEIoGXlxe3XF9fHyKRCGKxGNOnT4dYLIaBgQEX7ADg5eUFJSUlpKen44033oBYLMbo0aO5YG/ezubNm3H//n0YGhpK1a2mpgY1NTXc+4qKCgBNfwyZemwb3eHvhm4tvtO+OtT28mk1oW1n6ClCZV/P5s/Y09pvuQb8nTt30NDQwM0U28zc3LzVVlQikbSYXyKRcMub09rKY2ZmxluuoqICIyMjXh5bW1upMpqXtRTwERERWLdunVS6lZVVi/tCZOu9TfKuQTtt6r6KPnjwAPr6+q0upwdvZCg0NJR39tHY2Ii//voLzs7OuHnzpsLMINuWyspKWFlZ0fH4L1kdD8YYHjx4AEtLyzbzyTXgTUxMoKysjNLSUl56aWkphEJhi+sIhcI28zf/LC0thYWFBS+Ps7Mzl+fJi4L19fW4d+8er5yWtvP4Np6krq4OdXV1Xlrz3PZ6enr0AX8MHQ8+WRyPtlr2ZnK9Sq+mpgYXFxekpqZyaY2NjUhNTYW7u3uL67i7u/PyA0BKSgqX39bWFkKhkJensrIS6enpXB53d3eUl5cjK+uf8ZB/+eUXNDY2QiQScXnOnDmDuro63nYGDRrU4uk8Ib0Ck7PExESmrq7O9u/fz/744w82b948ZmBgwCQSCWOMsZkzZ7KVK1dy+c+ePctUVFTYli1b2JUrV1h4eDhTVVVlly5d4vJs2rSJGRgYsH/961/s4sWLbPLkyczW1pY9evSIyzNhwgQ2fPhwlp6ezv7v//6P2dnZMX9/f255eXk5Mzc3ZzNnzmS5ubksMTGRaWlpsS+++KJD+1dRUcEAsIqKis4eoj6Fjgffsz4ecg94xhjbuXMnGzBgAFNTU2Nubm7s999/55aNGTOGBQQE8PIfOXKEvfDCC0xNTY0NHTqU/fDDD7zljY2NbM2aNczc3Jypq6uzV155heXl5fHy3L17l/n7+zMdHR2mp6fH5syZwx48eMDLc+HCBebp6cnU1dVZv3792KZNmzq8b9XV1Sw8PJxVV1d3eN2+iI4H37M+HnK/D08IeXbk/qQdIeTZoYAnRIFQwBOiQCjgCVEgFPAyYmNjA4FAwHtteuIRyvZ0t31a19/erqNdoXujtWvXSn0W7O3tueXV1dUICgqCsbExdHR0MGXKFKmHvIqKiuDj4wMtLS2YmZnhww8/RH19fdcr90zuBSgAa2trtn79elZSUsK9Hj58yC2vqKhg5ubmbMaMGSw3N5d98803TFNTk3df/+zZs0xZWZlFRkayP/74g61evVrqGYPeLDExkampqbF9+/axy5cvs/fee48ZGBiw0tJSeVdNpsLDw9nQoUN5n4WysjJu+YIFC5iVlRVLTU1lmZmZbNSoUczDw4NbXl9fz4YNG8a8vLxYdnY2O3HiBDMxMWGhoaFdrhsFvIxYW1uz7du3t7r8888/Z4aGhqympoZL++ijj9igQYO499OmTWM+Pj689UQiEZs/f77M6ysPbm5uLCgoiHvf0NDALC0tWUREhBxrJXvh4eHMycmpxWXl5eVMVVWVJSUlcWlXrlxhAJhYLGaMMXbixAmmpKTEPXzGGGN79uxhenp6vM9PZ9ApvQxt2rQJxsbGGD58OKKioninYK11t83Ly8P9+/e5PI93623O01pX4d6kuSv04/v3tK7Qvdn169dhaWmJgQMHYsaMGSgqKgIAZGVloa6ujncc7O3tMWDAAO44iMViODg48Hp8ent7o7KyEpcvX+5Svai3nIwsXrwYI0aMgJGREdLS0hAaGoqSkhJs27YNQPu62z6t629v1pmu0L2VSCTC/v37MWjQIJSUlGDdunV46aWXkJubyw2oYmBgwFvnye7bLR2n5mVdQQHfhpUrV2Lz5s1t5rly5Qrs7e153WIdHR2hpqaG+fPnIyIiQqoHHenbXn31Ve53R0dHiEQiWFtb48iRI9DU1JRjzSjg27Rs2TLMnj27zTwDBw5sMV0kEqG+vh6FhYUYNGhQu7rbPq3rb2/Wma7QfYWBgQFeeOEF5Ofn43/+539QW1uL8vJyXiv/ZBfvJ+9ePK1rdnvRd/g2mJqawt7evs3X49/JH5eTkwMlJSVuZJ32dLd9Wtff3qwzXaH7iocPH+LGjRuwsLCAi4sLVFVVecchLy8PRUVFvO7bly5d4o3ZkJKSAj09PQwZMqRrlenSJT/CGGMsLS2Nbd++neXk5LAbN26wr7/+mpmamrJZs2ZxedrT3bY9XX97s6d1he4rli1bxn799VdWUFDAzp49y7y8vJiJiQm7ffs2Y6zpttyAAQPYL7/8wjIzM5m7uztzd3fn1m++LTd+/HiWk5PDTp48yUxNTem2XE+RlZXFRCIR09fXZxoaGmzw4MHs008/lery2J7utk/r+tvbtdUVuq/w8/NjFhYWTE1NjfXr14/5+fmx/Px8bvmjR4/Y+++/zwwNDZmWlhZ74403WElJCa+MwsJC9uqrrzJNTU1mYmLCli1bxurq6rpcN+oeS4gCoe/whCgQCnhCFAgFPCEKhAKeEAVCAU+IAqGAJ0SBUMATokAo4AlRIBTwpMsYY5g3bx6MjIwgEAiQk5Mj7yqRVlDAky47efIk9u/fj+PHj6OkpATDhg0D0DSQg7KyMnx8fORcQ9KMAp50WXNPMA8PDwiFQqioNPW6jo2NRXBwMM6cOYPi4mI515IAFPCki2bPno3g4GAUFRVBIBDAxsYGQFOX0MOHD2PhwoXw8fHB/v37eevdv38fM2bMgKmpKTQ1NWFnZ4e4uDgAwLhx47Bo0SJe/rKyMqipqXHdSm1sbLBx40bMmjULOjo6sLa2xtGjR1FWVobJkydDR0cHjo6OyMzM7PZj0Kt0ufsNUWjl5eVs/fr1rH///qykpITrAhobG8tcXV0ZY4wdO3aMPffcc6yxsZFbLygoiDk7O7OMjAxWUFDAUlJS2NGjRxljjMXHxzNDQ0Neb8Nt27YxGxsbrgxra2tmZGTEYmJi2LVr19jChQuZnp4emzBhAjty5AjLy8tjvr6+bPDgwbztKjoKeNJl27dvZ9bW1rw0Dw8PFh0dzRhjrK6ujpmYmLDTp09zyydNmsTmzJnTYnmPHj1ihoaG7PDhw1yao6MjW7t2Lffe2tqavfPOO9z7kpISBoCtWbOGSxOLxQyAVNdTRUan9ETm8vLycO7cOfj7+wMAVFRU4Ofnh9jYWC7PwoULkZiYCGdnZ6xYsQJpaWncMg0NDcycORP79u0DAJw/fx65ublSw405OjpyvzcP8ujg4CCV9vjIMYqOAp7IXGxsLOrr62FpaQkVFRWoqKhgz549+O6771BRUQGgaaDHv/76Cx988AGKi4vxyiuvYPny5VwZc+fORUpKCm7duoW4uDiMGzcO1tbWvO2oqqpyvwsEglbTGhsbu21fexsKeCJT9fX1OHjwILZu3YqcnBzudeHCBVhaWuKbb77h8pqamiIgIABff/01oqOjsXfvXm6Zg4MDXF1d8eWXXyIhIQHvvvuuPHanz6FRa4lMHT9+HPfv30dgYCD09fV5y6ZMmYLY2FgsWLAAYWFhcHFxwdChQ1FTU4Pjx49j8ODBvPxz587FokWLoK2tjTfeeONZ7kafRS08kanY2Fh4eXlJBTvQFPCZmZm4ePEi1NTUEBoaCkdHR4wePRrKyspITEzk5ff394eKigr8/f2hoaHxrHahT6Mx7UiPVVhYiOeeew4ZGRkYMWKEvKvTJ1DAkx6nrq4Od+/exfLly1FQUICzZ8/Ku0p9Bp3Skx7n7NmzsLCwQEZGBmJiYuRdnT6FWnhCFAi18IQoEAp4QhQIBTwhCoQCnhAFQgFPiAKhgCdEgVDAE6JAKOAJUSD/D/YeS4+3aRr8AAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPMAAADvCAYAAAAn4uLfAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAArW0lEQVR4nO3de1xN6RoH8N/eaXe/S5fRzUiFSmkk4xIyNaXRTC5TDZUmjMS4jHGJwsxxGeMWo4OjcCYNTmMwQ3I/h6ZBEiGKJkbJrSi6qPf84bSOrdtut2vX6vl+Pvujvda73vXuXY/1rsv7PgLGGAMhpN0TyrsBhBDZoGAmhCcomAnhCQpmQniCgpkQnqBgJoQnKJgJ4QkKZkJ4goKZEJ6gYO4Azp8/jwEDBkBNTQ0CgQDp6enybhJpARTMPFdZWYkxY8bgyZMnWLt2LXbt2oXY2FgIBAIIhULcvXu31jbPnj2DiooKBAIBpk2bxi1/+fIlQkJC0Lt3b2hpaUFdXR329vZYv349KisrAQC5ubkQCAQSvXJzc1vra+gQOsm7AaRl5eTk4M8//8TWrVvx+eefAwCys7MBAEpKSti9ezfmzp0rtk1iYmKddb18+RKZmZnw9PSEubk5hEIhzp07h5kzZyI1NRXx8fHQ19fHrl27xLb7/vvvce/ePaxdu1Zsub6+vqw+JgEARnjt9OnTDADbu3cvtywyMpIBYJ988gnr06dPrW1GjBjBfH19GQAWFhbW6D6mTZvGALD8/Pw613t5eTEzM7MG66iurmYvXrxodF+kftTN5rGgoCAMGTIEADBmzBgIBAK4urpy6/39/ZGeno4bN25wywoKCnDixAn4+/tLvB9zc3MAQFFRUZO2GTlyJJKSkuDk5AQVFRX8/e9/BwDExsZi2LBh6NKlC5SUlNCzZ09s3ry53jr+85//oF+/flBWVka3bt2wc+dOsXKVlZVYsmQJLC0toaysDD09PQwcOBDJyckSt7c9oGDmscmTJ2PBggUAgOnTp2PXrl1YuHAht37w4MHo2rUr4uPjuWU//fQT1NXV4eXlVW+9FRUVePToEe7evYuff/4Zq1evhpmZGbp3796k9mVlZcHPzw8jRozA+vXr0adPHwDA5s2bYWZmhgULFuD777+HiYkJpk6dik2bNtWqIzs7G6NHj8aIESPw/fffQ0dHB0FBQcjMzOTKREVFYcmSJRg6dCg2btyIhQsXwtTUFGlpaU1qb5sn764BaVknT56st5v98OFDNmfOHNa9e3du3XvvvceCg4MZY6zebvbu3bsZAO7l5OTEMjIy6m1DXd1sMzMzBoAdOXKkVvm6utvu7u6sW7duddZx5swZbllhYSFTUlJis2fP5pbZ29szLy+vetvHF3Rk7uD8/f2RnZ2N8+fPc/821sUeOnQokpOTsXfvXkyZMgWKioooLS1t8r4tLCzg7u5ea7mKigr3c3FxMR49eoQhQ4bg9u3bKC4uFivbs2dPDBo0iHuvr68PKysr3L59m1umra2NzMxM3Lp1q8ltbE/oanYH5+DgAGtra8THx0NbWxuGhoYYNmxYg9sYGBjAwMAAADB69Gj87W9/w4gRI3Dr1i0YGhpKvG8LC4s6l589exaRkZFISUnBixcvxNYVFxdDS0uLe29qalprex0dHTx9+pR7v3TpUowaNQo9evRA79694eHhgfHjx8POzk7itrYHdGQm8Pf3x08//YT4+HiMGzcOQmHT/ixGjx6NkpIS/PLLL03a7s0jcI2cnBwMHz4cjx49wpo1a/Drr78iOTkZM2fOBABUV1eLlVdQUKizbvbGbFiDBw9GTk4Otm/fjt69e2Pbtm1wdHTEtm3bmtTeto6CmcDf3x/5+fm4efNmk65i13j58iUA1OoCS+PgwYMoLy/HgQMHMHnyZHh6esLNza3OwG8KXV1dBAcHY/fu3bh79y7s7OwQFRXV7Pa2JdTNJnj33Xexbt06vHz5Ev369au33KNHj6CnpweBQCC2vOYI5+Tk1Oy21Bxp3zyyFhcXIzY2Vuo6Hz9+DD09Pe69uro6unfvXufTb+0ZBTMBAMyYMaPRMv/85z8RExMDHx8fdOvWDc+fP0dSUhKSk5Ph7e3d6Lm2JD744AOIRCJ4e3tj8uTJKCkpwdatW9GlSxfk5+dLVWfPnj3h6uqKvn37QldXFxcuXMC+ffvEHlXlAwpmIrGBAwfi3Llz2L17Nx48eIBOnTrBysoKa9asQXh4uEz2YWVlhX379iEiIgJz5syBoaEhvvjiC+jr62PixIlS1Tl9+nQcOHAAR48eRXl5OczMzPDNN9/gq6++kkmb2woBYzRvNiF8QBfACOEJCmZCeIKCmRCeoGAmhCcomAnhCQpmQniC7jNLqbq6Gvfv34eGhkatJ6IIkSXGGJ4/fw5jY+MGn5unYJbS/fv3YWJiIu9mkA7k7t276Nq1a73rKZilpKGhAeD1F6ypqSnn1hA+e/bsGUxMTLi/ufpQMEuppmutqalJwUxaRWOnc3QBjBCeoGAmhCcomAnhCTpn7oCqqqq4dDJE/hQUFNCpU6dm3+KkYO5gSkpKcO/ePbTFka+FhfWv69Kl9dohD6qqqjAyMoJIJJK6DgrmDqSqqgr37t2Dqqoq9PX129zDLg01539JM3iHMYaKigo8fPgQd+7cgaWlZZMnVKxBwdyBVFZWgjEGfX39Zk+Q1xLqmWgTAKCs3HrtaG0qKipQVFTEn3/+iYqKCihL+WEpmDsgeR6ReT4PvdSkPRqL1SGDdhBC2gAKZkJ4goKZEJ6gc2YCb+/W21dpKfC/NMxExujITAhPUDCTduH58+cICAiAmpoajIyMsHbtWri6uuLLL78EAOzatQtOTk7Q0NCAoaEh/P39UfjGUyinTp2CQCBAUlISHBwcoKKigmHDhqGwsBCHDx+GjY0NNDU14e/vL5Z50tXVFeHh4fjyyy+ho6MDAwMDbN26FaWlpQgODoaGhga6d++Ow4cPc9tUVVUhJCQEFhYWUFFRgZWVFdavX9/i3xEFM2kXZs2ahbNnz+LAgQNITk7Gv//9b6SlpXHrKysrsWzZMly+fBn79+9Hbm4ugoKCatUTFRWFjRs34ty5c7h79y7Gjh2LdevWIT4+Hr/++iuOHj2K6OhosW127NiBzp07448//kB4eDi++OILjBkzBgMGDEBaWho++OADjB8/nvtPoLq6Gl27dsXevXtx7do1LF68GAsWLMCePXta9DtqE8G8adMmmJubQ1lZGc7Ozvjjjz8aLL93715YW1tDWVkZtra2+O2337h1lZWV+Prrr2Fraws1NTUYGxtjwoQJuH//vlgdT548QUBAADQ1NaGtrY2QkBCUlJS0yOcjzVNS8hw7duzA6tWrMXz4cPTu3RuxsbGoqqriykycOBEffvghunXrhv79+2PDhg04fPhwrd/pN998g/fffx8ODg4ICQnB6dOnsXnzZjg4OGDQoEEYPXo0Tp48KbaNvb09IiIiYGlpifnz50NZWRmdO3dGaGgoLC0tsXjxYjx+/BgZGRkAAEVFRSxZsgROTk6wsLBAQEAAgoOD+R/MP/30E2bNmoXIyEikpaXB3t4e7u7uYl2kN507dw5+fn4ICQnBpUuX4OPjAx8fH1y9ehUA8OLFC6SlpWHRokVIS0tDYmIisrKy8NFHH4nVExAQgMzMTCQnJ+PQoUM4c+YMJk2a1OKflzTd3bu3UVlZKZahUktLC1ZWVtz7ixcvwtvbG6amptDQ0MCQIUMAAHl5eWJ1vZlg3cDAAKqqqujWrZvYsrf/9t7cRkFBAXp6erC1tRXbBoDYdps2bULfvn2hr68PdXV1bNmypVZbZE3uwbxmzRqEhoYiODgYPXv2RExMDFRVVbF9+/Y6y69fvx4eHh746quvYGNjg2XLlsHR0REbN24E8PqXnJycjLFjx8LKygr9+/fHxo0bcfHiRe7LvH79Oo4cOYJt27bB2dkZAwcORHR0NBISEmodwUnbV1paCnd3d2hqauLHH3/E+fPn8fPPPwMAKioqxMoqKipyPwsEArH3NcveTuheV5m36wH+nwg+ISEBc+bMQUhICI4ePYr09HQEBwfXaousyTWYKyoqcPHiRbi5uXHLhEIh3NzckJKSUuc2KSkpYuUBwN3dvd7ywOv8vgKBANra2lwd2traYvmE3dzcIBQKkZqaWmcd5eXlePbsmdiLtA4Tk25QVFTE+fPnuWXFxcW4efMmAODGjRt4/PgxVqxYgUGDBsHa2rrenl1rOHv2LAYMGICpU6fCwcEB3bt3R05OTovvV67B/OjRI1RVVXHdlBoGBgYoKCioc5uCgoImlS8rK8PXX38NPz8/bq6ugoICdHlrTF2nTp2gq6tbbz3Lly+HlpYW96KZOVuPuroGAgMD8dVXX+HkyZPIzMxESEgIhEIhBAIBTE1NIRKJEB0djdu3b+PAgQNYtmyZ3NpraWmJCxcuICkpCTdv3sSiRYvE/iNqKbx+aKSyshJjx44FYwybN29uVl3z58/HrFmzuPc1MybywcGDrbcvaQdarFmzBlOmTMHIkSOhqamJuXPn4u7du1BWVoa+vj7i4uKwYMECbNiwAY6Ojli9enWt6yStZfLkybh06RLGjRsHgUAAPz8/TJ06Vez2VYtgclReXs4UFBTYzz//LLZ8woQJ7KOPPqpzGxMTE7Z27VqxZYsXL2Z2dnZiyyoqKpiPjw+zs7Njjx49Elv3j3/8g2lra4stq6ysZAoKCiwxMVGithcXFzMArLi4WKLybcHLly/ZtWvX2MuXL+XWhps3pXu9raSkhGlpabFt27a1/odoAQ39biT9W5NrN1skEqFv3744fvw4t6y6uhrHjx+Hi4tLndu4uLiIlQeA5ORksfI1R+Rbt27h2LFj0NPTq1VHUVERLl68yC07ceIEqqur4ezsLIuPRmTs0qVL2L17N3JycpCWloaAgAAAwKhRo+TcsrZD7t3sWbNmITAwEE5OTujXrx/WrVvHPV0DABMmTMA777yD5cuXAwBmzJiBIUOG4Pvvv4eXlxcSEhJw4cIFbNmyBcDrQB49ejTS0tJw6NAhVFVVcefBurq6EIlEsLGxgYeHB0JDQxETE4PKykpMmzYNn376KYyNjeXzRZBGrV69GllZWdxB4N///jc6d+4s72a1HS3VbWiK6OhoZmpqykQiEevXrx/7/fffuXVDhgxhgYGBYuX37NnDevTowUQiEevVqxf79ddfuXV37txhAOp8nTx5kiv3+PFj5ufnx9TV1ZmmpiYLDg5mz58/l7jN1M2Wjqy62Xwji262gLE2OLNbO/Ds2TNoaWmhuLi43WS0KCsrw507d2BhYSH11DTNJe0FMEtL2bajrWnodyPp35rcHxohhMgGBTMhPEHBTAhPUDATwhMUzITwhNzvM5M2ICqq1Xal+xh4Mr1p+/vsM1e4uPTBunXrZNaOqKgo7N+/H+np6TKrU97oyEwIT1AwE1KHlh573BIomEm7UF1djblz50JXVxeGhoaIeuPUoKioCJ9//jn09fWhqamJYcOG4fLly2Lbr1ixAgYGBtDQ0EBISAjKysrE1gcFBcHHxwfffvstjI2NuVlMrly5gmHDhkFFRQV6enqYNGmS2FRENdutXr0aRkZG0NPTQ1hYmFxS5lIwk3Zhx44dUFNTQ2pqKlatWoWlS5ciOTkZADBmzBhuls2LFy/C0dERw4cPx5MnTwAAe/bsQVRUFP72t7/hwoULMDIywg8//FBrH8ePH0dWVhY3lVTNDCY6Ojo4f/489u7di2PHjmHatGli2508eRI5OTk4efIkduzYgbi4OMTFxbX4d/I2ugBG2gU7OztERkYCeD34f+PGjTh+/DhUVFTwxx9/oLCwEEpKSgBeD8jYv38/9u3bh0mTJmHdunUICQlBSEgIgNeT+h07dqzW0VlNTQ3btm3jciRv3boVZWVl2LlzJ9TU1AAAGzduhLe3N1auXMlNkqGjo4ONGzdCQUEB1tbW8PLywvHjxxEaGtoq300NOjKTduHNSfUAwMjICIWFhbh8+TJKSkqgp6cHdXV17nXnzh1uqp7r16/XGtpa1xBbW1tbsWTn169fh729PRfIAPD++++juroaWVlZ3LJevXpB4Y18tDVta210ZCbtQn0T75WUlMDIyAinTp2qtU3NnG+SejNoZdG21kbBTFqVyqvn0C1r2gyonarrv7Ls6OiIgoICdOrUCebm5nWWsbGxQWpqKiZMmMAt+/333xvdr42NDeLi4lBaWsoF+tmzZyEUCsWm+W0rpOpmvz1JOCHy4ubmBhcXF/j4+ODo0aPIzc3FuXPnsHDhQly4cAHA6wkttm/fjtjYWNy8eRORkZHIzMxstO6AgAAoKysjMDAQV69excmTJxEeHo7x48fXmlSyLZDqyOzh4YGuXbsiODgYgYGBvJnYrsNqxSfAXl6R7bzkAoEAv/32GxYuXIjg4GA8fPgQhoaGGDx4MBdw48aNQ05ODubOnYuysjL4+vriiy++QFJSUoN1q6qqIikpCTNmzMB7770HVVVV+Pr6Ys2aNTL9DLIi1eQEjx49wq5du7Bjxw5kZmZi2LBhCAkJgY+Pj9gFBD6jyQmk81jKYNaz5fd0TnKbnKBz586YOXMm0tPTkZqaih49emDq1KkwNjbG9OnTa92wJ4S0vGbfmnJ0dMT8+fMxbdo0lJSUYPv27ejbty8GDRok0XkJIUQ2pA7myspK7Nu3D56enjAzM0NSUhI2btyIBw8eIDs7G2ZmZhgzZows20oIaYBUF8DCw8Oxe/duMMYwfvx4rFq1Cr179+bWq6mpYfXq1TRtLSGtSKpgvnbtGqKjo/HJJ59wj9C9rXPnznQLq42iCVnbHln8TqTqZkdGRmLMmDG1AvnVq1c4c+YMgNeJ2Gpy5DZElonWASAxMREffPAB9PT0IBAI6hx87urqCoFAIPaaMmVKo21t72oeOWyPw/v47sWLFwBqP03WFFIdmYcOHYr8/PxamRSLi4sxdOhQsYz2DalJtB4TEwNnZ2esW7cO7u7uyMrKqlU38P9E68uXL8fIkSMRHx8PHx8fpKWlcd380tJSDBw4EGPHjm3wQffQ0FAsXbqUe6+qqipRm9uzTp06QVVVFQ8fPoSioiKEwtZ/NL+CvZJqu7cHRfAFYwwvXrxAYWEhtLW1xZ7xbiqp7jMLhUI8ePAA+vr6Ystv3rwJJycniXMXOzs747333uMSpVdXV8PExATh4eGYN29erfLjxo1DaWkpDh06xC3r378/+vTpg5iYGLGyubm5sLCwwKVLl9CnTx+xda6urujTp3nT0LTH+8zA66PynTt35PLsMACU/lUk1XZq72jLtB1tjba2NgwNDbnE7W+S9G+tSUfmTz75BMDrp26CgoLEutlVVVXIyMjAgAEDJKqrJtH6/PnzuWWSJFp/M60q8DrR+v79+5vyMQAAP/74I/75z3/C0NAQ3t7eWLRoUYNH5/LycpSXl3Pv22uydZFIBEtLS7l1tQ99sVGq7UYemdZ4oXZKUVGxWUfkGk0KZi0tLQCvuwYaGhpQUVHh1olEIvTv31/iMZwNJVq/ceNGnds0NdF6ffz9/WFmZgZjY2NkZGTg66+/RlZWFhITE+vdZvny5ViyZEmT9tNWCYVCuT0BVnG/pPFCdZBXe9uTJgVzbGwsAMDc3Bxz5syResiYvE2aNIn72dbWFkZGRhg+fDhycnLw7rvv1rkNn5OtE36Q6gJYzYwPzdG5c2coKCjgwYMHYssfPHgAQ0PDOrcxNDRsUnlJ1Qxcz87OrjeYlZSU6r0NR0hbIPHlTEdHRzx9+hQA4ODgAEdHx3pfkmipROvSqLl9ZWRk1Kx6CJEniY/Mo0aN4o5MPj4+Mtm5rBOtA8CTJ0+Ql5eH+/dfj86pmd7F0NAQhoaGyMnJQXx8PDw9PaGnp4eMjAzMnDkTgwcPrjU1DSHticTB/GbXWhbdbOD1raaHDx9i8eLFKCgoQJ8+fXDkyBHuIldeXp7YvdABAwYgPj4eERERWLBgASwtLbF//36xR0kPHDjA/WcAAJ9++inX5qioKIhEIhw7doz7j8PExAS+vr6IiIiQyWciRF4o2bqU2ut9ZnmL7xEl1Xb+N6Xbjg9kfp9ZR0enzhvadamZr5gQ0nokDmZZJu0ihMiexMEcGBjYku0ghDSTxMH87Nkzrr/e2KOMdA5JSOtr0jlzzUgpbW3tOs+fGWMQCAQSj5oihMiOxMF84sQJ6OrqAqB5swlpiyQO5jcnGpBk0gFCSOuSOj3N06dP8Y9//APXr18HAPTs2RPBwcHc0ZsQ0rqkmmrizJkzMDc3x4YNG/D06VM8ffoUGzZsgIWFBTdtECGkdUl1ZA4LC8O4ceOwefNmblB1VVUVpk6dirCwMFy5ckWmjSSENE6qI3N2djZmz54tNjuCgoICZs2ahezsbJk1jhAiOamC2dHRkTtXflNNcmpCSOuTuJudkZHB/Tx9+nTMmDED2dnZ6N+/P4DX+W43bdqEFStWyL6VhJBGSTxqSigUQiAQNDpZd0d5aIRGTUmHRk01ncxHTd25c0cmDSOEtAyJg9nMzKwl20EIaSapHxoBXuecysvLqzUH80cffdSsRhFCmk6qYL59+zY+/vhjXLlyRew8umbwRUc4ZyakrZHq1tSMGTNgYWGBwsJCqKqqIjMzE2fOnIGTkxNOnTol4yYSQiQh1ZE5JSUFJ06cQOfOnSEUCiEUCjFw4EAsX74c06dPx6VLl2TdTkJII6Q6MldVVUFDQwPA68nsa6a1NTMz46a2JYS0LqmOzL1798bly5dhYWEBZ2dnrFq1CiKRCFu2bEG3bt1k3UZCiASkOjJHRERwKUGXLl2KO3fuYNCgQfjtt9+wYcOGJtUlj2TrZWVlCAsLg56eHtTV1eHr61sr7Q0h7Y1Uwezu7s6ld+3evTtu3LiBR48eobCwEMOGDZO4nppk65GRkUhLS4O9vT3c3d1RWFhYZ/maZOshISG4dOkSfHx84OPjg6tXr3JlapKtr1y5st79zpw5EwcPHsTevXtx+vRp3L9/n/s8hLRXzZ4E/+7duwAgVUZEeSRbLy4uhr6+PuLj4zF69GgAwI0bN2BjY4OUlBTuWfPG0OOc0qHHOZtO0r81qY7Mr169wqJFi6ClpQVzc3OYm5tDS0sLERERqKyslKiOmmTrbm5u/2+MBMnW3ywPvO4l1Fe+LhcvXkRlZaVYPdbW1jA1NW2wnvLycjx79kzsRUhbItUFsPDwcCQmJmLVqlVcBsaUlBRERUXh8ePH2Lx5c6N1yCvZekFBAUQiEbS1tZtUD5+SrRN+kiqY4+PjkZCQgA8//JBbZmdnBxMTE/j5+UkUzO0NJVsnbZ1UwaykpARzc/Nayy0sLCASiSSqQ17J1g0NDVFRUYGioiKxo3Nj9VCyddLWSXXOPG3aNCxbtgzl5eXcsvLycnz77beYNm2aRHXIK9l63759oaioKFZPVlYW8vLymp20nRB5kvjI/Patm2PHjqFr167cNEGXL19GRUUFhg8fLvHO5ZFsXUtLCyEhIZg1axZ0dXWhqamJ8PBwuLi4SHwlm5C2SOJg1tLSEnvv6+sr9l6a80d5JFsHgLVr10IoFMLX1xfl5eVwd3fHDz/80OT2E9KWULJ1KdF9ZunQfeamk/m0QXV5+PAh1421srKCvr5+c6ojhDSDVBfASktLMXHiRBgZGWHw4MEYPHgwjI2NERISghcvXsi6jYQQCUgVzLNmzcLp06dx8OBBFBUVoaioCL/88gtOnz6N2bNny7qNhBAJSNXN/te//oV9+/bB1dWVW+bp6QkVFRWMHTuWlw+NENLWSXVkfvHiRa3HKgGgS5cu1M0mRE6kCmYXFxdERkairKyMW/by5UssWbKEHrwgRE6k6mavW7cOHh4etR4aUVZWRlJSkkwbSAiRjFTBbGtri1u3buHHH3/kRjj5+fkhICAAKioqMm0gIUQyTQ7myspKWFtb49ChQwgNDW2JNhFCpNDkc2ZFRUWxc2VCSNsg1QWwsLAwrFy5Eq9evZJ1ewghUpLqnPn8+fM4fvw4jh49CltbW6ipqYmtT0xMlEnjCCGSkyqYtbW1a42aIoTIV5OCubq6Gt999x1u3ryJiooKDBs2DFFRUXQFm5A2oEnnzN9++y0WLFgAdXV1vPPOO9iwYQPCwsJaqm2EkCZoUjDv3LkTP/zwA5KSkrB//34cPHgQP/74I5fdghAiP00K5ry8PHh6enLv3dzcIBAIuCl6CCHy06Rz5levXkFZWVlsmaKiosQT35OOwdu7/nV+rdeMDqdJwcwYQ1BQkNiUs2VlZZgyZYrY7Sm6NUVI62tSMAcGBtZa9tlnn8msMYQQ6TUpmGNjY1uqHYSQZpLqcU5CSNvTJoJZ1gnXGWNYvHgxjIyMoKKiAjc3N9y6dUusjLm5OQQCgdhrxYoVMv9shLQWuQdzSyRcX7VqFTZs2ICYmBikpqZCTU0N7u7utUZ7LV26FPn5+dwrPDy8RT8rIS1J7sG8Zs0ahIaGIjg4GD179kRMTAxUVVWxffv2OsuvX78eHh4e+Oqrr2BjY4Nly5bB0dGRS9jOGMO6desQERGBUaNGwc7ODjt37sT9+/exf/9+sbo0NDS4tDWGhoa1BowQ0p7INZhbIuH6nTt3UFBQIFZGS0sLzs7OtepcsWIF9PT04ODggO+++67BIZ2UbJ20dc3KaNFcLZFwvebfxpKyT58+HY6OjtDV1cW5c+cwf/585OfnY82aNXXul5Ktk7ZOrsEsT28mTrezs4NIJMLkyZOxfPnyOvMwU7L1t/wvCV9d/LJarxnk/+TazW6JhOs1/zY1KbuzszNevXqF3NzcOtcrKSlBU1NT7EVIWyLXYG6JhOsWFhYwNDQUK/Ps2TOkpqY2OKd3eno6hEIhunTp0pyPRIjcyL2bLeuE6wKBAF9++SW++eYbWFpawsLCAosWLYKxsTF8fHwAvL6IlpqaiqFDh0JDQwMpKSmYOXMmPvvsM+jo6MjleyCkueQezC2RcH3u3LkoLS3FpEmTUFRUhIEDB+LIkSPciC8lJSUkJCQgKioK5eXlsLCwwMyZM8XOiQlpbyjZupQ6fLL1Bi6AxcfLfneUbL3xvzW5PzRCCJENCmZCeIKCmRCeoGAmhCcomAnhCQpmQniCgpkQnqBgJoQnKJgJ4QkKZkJ4Qu7PZhMikQYeH22R7dohOjITwhMUzITwBAUzITxBwUwIT1AwE8ITFMyE8AQFMyE8QcFMCE/QQyOkXWhoXjF//9ZrR1tGwdzRdaAnpPiOutmE8ESbODJv2rQJ3333HQoKCmBvb4/o6Gj069ev3vJ79+7FokWLkJubC0tLS6xcuRKenp7cesYYIiMjsXXrVhQVFeH999/H5s2bYWlpyZV58uQJwsPDcfDgQQiFQvj6+mL9+vVQV1dv0c9KWllzeh7trNci9yOzvJKtBwQEIDMzE8nJyTh06BDOnDmDSZMmtfjnJaSlyH0SfGdnZ7z33ntcsvTq6mqYmJggPDwc8+bNq1V+3LhxKC0txaFDh7hl/fv3R58+fRATEwPGGIyNjTF79mzMmTMHAFBcXAwDAwPExcXh008/xfXr19GzZ0+cP38eTk5OAIAjR47A09MT9+7dg7GxcaPtbnOT4Leho0hLTILfkBa7ANZGvlNJ/9bk2s2uSbY+f/58bpkkydbfTiPj7u6O/fv3A2g82fqnn36KlJQUaGtrc4EMAG5ubhAKhUhNTcXHH39ca7/l5eUoLy/n3hcXFwOA7JOu/y+nVlu3Z4+8W/B/23bVv27s2GZU/MbfZatsV4+av7HGjrsdMtl6QUFBrWyPnTp1gq6urlhC9jfVl2y9Q+dobgdCV8hhpytaZqfPnz+HlpZWvevbxAWw9uDtZOvV1dV48uQJ9PT0IBAI6tymJiH73bt320ZXvA2i76hxjDE8f/680dM/uQZzSydbNzIyEivTp08frszbF9hevXqFJ0+e1LtfJSUlKCkpiS3T1tZu+AP+DyVnbxx9Rw1r6Ihco0MmW3dxcUFRUREuXrzIlTlx4gSqq6vh7Owss89HSKticpaQkMCUlJRYXFwcu3btGps0aRLT1tZmBQUFjDHGxo8fz+bNm8eVP3v2LOvUqRNbvXo1u379OouMjGSKiorsypUrXJkVK1YwbW1t9ssvv7CMjAw2atQoZmFhwV6+fMmV8fDwYA4ODiw1NZX95z//YZaWlszPz0+mn624uJgBYMXFxTKtl0/oO5IduQczY4xFR0czU1NTJhKJWL9+/djvv//OrRsyZAgLDAwUK79nzx7Wo0cPJhKJWK9evdivv/4qtr66upotWrSIGRgYMCUlJTZ8+HCWlZUlVubx48fMz8+PqaurM01NTRYcHMyeP38u089VVlbGIiMjWVlZmUzr5RP6jmRH7veZCSGyIfcnwAghskHBTAhPUDATwhMUzITwBAWzDOTm5iIkJAQWFhZQUVHBu+++i8jISFRUVIiVy8jIwKBBg6CsrAwTExOsWrWqVl179+6FtbU1lJWVYWtri99++621PoZcbNq0Cebm5lBWVoazszP++OMPeTep/ZL35XQ+OHz4MAsKCmJJSUksJyeH/fLLL6xLly5s9uzZXJni4mJmYGDAAgIC2NWrV9nu3buZiooK+/vf/86VOXv2LFNQUGCrVq1i165dYxEREbXuofNJQkICE4lEbPv27SwzM5OFhoYybW1t9uDBA3k3rV2iYG4hq1atYhYWFtz7H374geno6LDy8nJu2ddff82srKy492PHjmVeXl5i9Tg7O7PJkye3fIPloF+/fiwsLIx7X1VVxYyNjdny5cvl2Kr2i7rZLaS4uBi6urrc+5SUFAwePBgikYhb5u7ujqysLDx9+pQr8+bQzZoy9Q0Hbc9qhr+++XkbG/5KGkbB3AKys7MRHR2NyZMnc8vqG7pZs66hMvUNy2zPGhr+ysfP2xoomBswb948CASCBl9vj7v+66+/4OHhgTFjxiA0NFROLScdEY1nbsDs2bMRFBTUYJlu3bpxP9+/fx9Dhw7FgAEDsGXLFrFy9Q3drFnXUJn6hmW2Z9IMfyUNoyNzA/T19WFtbd3gq+Yc+K+//oKrqyv69u2L2NhYCIXiX62LiwvOnDmDyspKbllycjKsrKygo6PDlWloeCefSDP8lTRC3lfg+ODevXuse/fubPjw4ezevXssPz+fe9UoKipiBgYGbPz48ezq1assISGBqaqq1ro11djwTj5pbPgraRoKZhmIjY1lAOp8veny5cts4MCBTElJib3zzjtsxYoVtepqbHgn3zQ0/JU0DQ2BJIQn6JyZEJ6gYCaEJyiYCeEJCmZCeIKCmRCeoGAmhCcomAnhCQpmQniCgrmDYoxh0qRJ0NXVhUAgQHp6urybRJqJgrmDOnLkCOLi4nDo0CHk5+dj9erVEAgEmDJlSq2yYWFhEAgEYiPINm/eDDs7Oy7hm4uLCw4fPgwAOHXqVKNDR0+dOtVKn7TjoCGQHVROTg6MjIwwYMAAAK/zU5uYmCAhIQFr166FiooKAKCsrAzx8fEwNTUV275r165YsWIFLC0twRjDjh07MGrUKFy6dAkDBgxAfn4+V3bGjBl49uwZYmNjuWVvzsJSUVEhNgMLkQ4dmTugoKAghIeHIy8vDwKBAObm5gAAR0dHmJiYIDExkSubmJgIU1NTODg4iNXh7e0NT09PWFpaokePHvj222+hrq6O33//HSKRCIaGhtxLRUUFSkpK3PuYmBj069cP27Ztg4WFBZSVlQG87i0MHDgQ2tra0NPTw8iRI5GTk8PtMzc3FwKBAImJiRg6dChUVVVhb28vNs3Qn3/+CW9vb+jo6EBNTQ29evXi/QynNSiYO6D169dj6dKl6Nq1K/Lz83H+/Hlu3cSJE8WOoNu3b0dwcHCD9VVVVSEhIQGlpaUSj0XOzs7Gv/71LyQmJnLn66WlpZg1axYuXLiA48ePQygU4uOPP0Z1dbXYtgsXLsScOXOQnp6OHj16wM/PD69evQLw+pSgvLwcZ86cwZUrV7By5Uqoq6tL1Kb2jrrZHZCWlhY0NDSgoKBQa1aPzz77DPPnz8eff/4JADh79iwSEhLqPMe9cuUKXFxcUFZWBnV1dfz888/o2bOnRG2oqKjAzp07oa+vzy3z9fUVK7N9+3bo6+vj2rVr6N27N7d8zpw58PLyAgAsWbIEvXr1QnZ2NqytrZGXlwdfX1/Y2toCEJ8Jhu/oyEzE6Ovrw8vLC3FxcYiNjYWXlxc6d+5cZ1krKyukp6cjNTUVX3zxBQIDA3Ht2jWJ9mNmZiYWyABw69Yt+Pn5oVu3btDU1OS6/3l5eWLl7OzsuJ+NjIwAAIWFhQCA6dOn45tvvsH777+PyMhIZGRkSNQePqBgJrVMnDgRcXFx2LFjByZOnFhvOZFIhO7du6Nv375Yvnw57O3tsX79eon2oaamVmuZt7c3njx5gq1btyI1NRWpqakAUCsziKKiIvezQCAAAK4r/vnnn+P27dsYP348rly5AicnJ0RHR0vUpvaOgpnU4uHhgYqKClRWVsLd3V3i7aqrq1FeXi7VPh8/foysrCxERERg+PDhsLGx4eYTbyoTExNMmTIFiYmJmD17NrZu3SpVPe0NnTOTWhQUFHD9+nXu57rMnz8fH374IUxNTfH8+XPEx8fj1KlTSEpKkmqfOjo60NPTw5YtW2BkZIS8vDzMmzevyfV8+eWX+PDDD9GjRw88ffoUJ0+ehI2NjVRtam8omEmdNDU1G1xfWFiICRMmID8/H1paWrCzs0NSUhJGjBgh1f6EQiESEhIwffp09O7dG1ZWVtiwYQNcXV2bVE9VVRXCwsJw7949aGpqwsPDA2vXrpWqTe0NzQFGCE/QOTMhPEHBTAhPUDATwhMUzITwBAUzITxBwUwIT1AwE8ITFMyE8AQFMyE8QcFMCE9QMBPCE/8F0fuJZdawfIcAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 200x200 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Train, Validation, and Test Split"
      ],
      "metadata": {
        "id": "5aXCxoG_gYE_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train, valid, test = np.split(\n",
        "    df.sample(frac=1, random_state=42), [int(0.7 * len(df)), int(0.83 * len(df))]\n",
        ")"
      ],
      "metadata": {
        "id": "nnFq5e5vi6wJ"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Let's rebalance our training set so that the proportion of `alpha` and `hedron` particles remain same."
      ],
      "metadata": {
        "id": "_Z-xCNWSlEgX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def ScaleDataset(df, over_sample=False):\n",
        "    X = df[df.columns[:-1]].values\n",
        "    y = df[df.columns[-1]].values\n",
        "\n",
        "    scaler = StandardScaler()\n",
        "    X = scaler.fit_transform(X)\n",
        "\n",
        "    if over_sample:\n",
        "        ros = RandomOverSampler()\n",
        "        X, y = ros.fit_resample(X, y)\n",
        "\n",
        "    data = np.hstack((X, np.reshape(y, (-1, 1))))\n",
        "    return data, X, y"
      ],
      "metadata": {
        "id": "qQ7W6Yivgb5r"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train, X_train, y_train = ScaleDataset(train, over_sample=True)"
      ],
      "metadata": {
        "id": "-jGSO1gijYWb"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "valid, X_valid, y_valid = ScaleDataset(valid)"
      ],
      "metadata": {
        "id": "ToEJCWmkjeAD"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test, X_test, y_test = ScaleDataset(test)"
      ],
      "metadata": {
        "id": "xfs34vGjji6I"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sum(y_train == 1), sum(y_train == 0)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h_XYBL5Qke1P",
        "outputId": "0e14a8dd-8b46-4f7e-e270-39cf55b1693e"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(8650, 8650)"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Training our model"
      ],
      "metadata": {
        "id": "mvF8pYgVqmYr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = KNeighborsClassifier(n_neighbors=25, metric='minkowski', p=2)\n",
        "model.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 75
        },
        "id": "6_0IN1MtqoWL",
        "outputId": "dacc4d1e-c855-4225-ee84-9a0cb197c638"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "KNeighborsClassifier(n_neighbors=25)"
            ],
            "text/html": [
              "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier(n_neighbors=25)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier(n_neighbors=25)</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Making predictions from our trained model."
      ],
      "metadata": {
        "id": "rBjVJYosqwvS"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = model.predict(X_test)"
      ],
      "metadata": {
        "id": "u6CGFoHQq0fw"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(classification_report(y_test, y_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_YK6F71Bq79z",
        "outputId": "fbf57870-6270-4bde-80dd-94a4091afb8e"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.79      0.71      0.74      1126\n",
            "           1       0.85      0.90      0.87      2108\n",
            "\n",
            "    accuracy                           0.83      3234\n",
            "   macro avg       0.82      0.80      0.81      3234\n",
            "weighted avg       0.83      0.83      0.83      3234\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "joblib.dump(model, 'kNN_magic_model.sav')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l5aXQ2z_tB9i",
        "outputId": "92fea20d-7194-46c4-a6f7-9d90e3d9be1d"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['kNN_magic_model.sav']"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    }
  ]
}