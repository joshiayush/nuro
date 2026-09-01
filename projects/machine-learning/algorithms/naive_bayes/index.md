{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "view-in-github"
   },
   "source": [
    "<a href=\"https://colab.research.google.com/github/joshiayush/ai/blob/master/ai/projects/machine-learning/algorithms/naive_bayes.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "10qHpgKC_3AU"
   },
   "source": [
    "# Naive Bayes\n",
    "\n",
    "Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable. Bayes’ theorem states the following relationship, given class variable $y$ and dependent feature vector $x_{1}$ through $x_{n}$:\n",
    "\n",
    "$$P(y | x_{1}, ..., x_{n}) = \\dfrac{P(y) \\cdot P(x_{1} ..., x_{n} | y)}{P(x_{1} ..., x_{n})}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7hKMmmGYBQ9Z"
   },
   "source": [
    "Using the naive conditional independence assumption that\n",
    "\n",
    "$$P(x_{i} | y, x_{1}, ..., x_{i-1}, x_{i+1}, ..., x_{n}) = P(x_{i} | y)$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "XcoQJrf7BjZG"
   },
   "source": [
    "for all $i$, this relationship is simplified to\n",
    "\n",
    "$$P(y | x_{1}, ..., x_{n}) = \\dfrac{P(y) \\cdot \\prod_{i=1}^{n}P(x_{i} | y) }{P(x_{1} ..., x_{n})}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lY3QYYGgCPkH"
   },
   "source": [
    "Since $P(x_{1}, ..., x_{n})$ is constant given the input, we can use the following classification rule:\n",
    "\n",
    "$$P(y | x_{1}, ..., x_{n}) \\propto P(y) \\cdot \\prod_{i=1}^{n}P(x_{i} | y)$$\n",
    "\n",
    "$$⇒ \\hat y = arg \\max_{y} P(y) \\cdot \\prod_{i=1}^{n} P(x_{i} | y)$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "p4mIDeKiDtGS"
   },
   "source": [
    "and we can use __Maximum A Posteriori (MAP)__ estimation to estimate $P(y)$ and $P(x_{i} | y)$;  the former is then the relative frequency of class $y$ in the training set."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FG9mpoRfEgmr"
   },
   "source": [
    "## Gaussian Naive Bayes\n",
    "\n",
    "[__GaussianNB__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB) implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian:\n",
    "\n",
    "$$P(x_{i} | y) = \\dfrac{1}{\\sqrt{2 \\pi \\sigma_{y}^{2}}} \\exp \\left( -\\dfrac{(x_{i} - \\mu_{y})^2}{2 \\sigma_{y}^2} \\right)$$\n",
    "\n",
    "The parameters $\\sigma_{y}$ and $\\mu_{y}$ are estimated using maximum likelihood."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "xfxBelRSFhi2"
   },
   "outputs": [],
   "source": [
    "import joblib\n",
    "\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "yj3AkFZ_Fye_"
   },
   "outputs": [],
   "source": [
    "X, y = load_iris(return_X_y=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "leoUGs_UF2t5"
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 75
    },
    "id": "-bbllY_kGBZU",
    "outputId": "fe9d3b6c-5faf-422a-f726-c702ba7164a4"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" checked><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GaussianNB</label><div class=\"sk-toggleable__content\"><pre>GaussianNB()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "GaussianNB()"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = GaussianNB()\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "FYsmuWCHGFy3"
   },
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vnBJf9HeGbMz",
    "outputId": "8efa8f32-f8a0-4394-bc90-22f9cd56f060"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      1.00      1.00        10\n",
      "           1       1.00      1.00      1.00         9\n",
      "           2       1.00      1.00      1.00        11\n",
      "\n",
      "    accuracy                           1.00        30\n",
      "   macro avg       1.00      1.00      1.00        30\n",
      "weighted avg       1.00      1.00      1.00        30\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "ARhlpkHIG2Wn",
    "outputId": "5c886ac6-26bd-4962-8070-8f22c7ecf23b"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['naive_bayes_iris_model.sav']"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(model, 'naive_bayes_iris_model.sav')"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyOS179dqaQ46rd7cAODmNl+",
   "include_colab_link": true,
   "provenance": []
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
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
