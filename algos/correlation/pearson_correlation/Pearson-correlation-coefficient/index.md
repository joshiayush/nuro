# Pearson correlation coefficient

In [statistics](https://en.wikipedia.org/wiki/Statistics), the __Pearson correlation coefficient__  ― also known as __Pearson's r__, the __Pearson product-moment correlation coefficient (PPMCC)__, the __bivariate correlation__, or colloquially simply as the __correlation coefficient__ ― is the measure of linear correlation between two sets of data.

## Definition

Pearson's correlation coefficient is the covariance of the two variables divided by the product of their standard deviations. The form of the definition involves a "product moment", that is, the mean of the product of the mean-adjusted random variables; hence the modifier product-moment in the name.

### For a population

Pearson's correlation coefficient, when applied to a [population](https://en.wikipedia.org/wiki/Statistical_population), is commonly represented by the Greek letter ρ (rho) and may be referred to as the _population correlation coefficient_ or the _population Pearson correlation coefficient_. Given a pair of random variables $(X, Y)$, the formula for ρ is:

$$ρ_{X,Y}=\dfrac{cov(X,Y)}{\sigma_{X}\sigma_{Y}}$$

where:

* __cov__ is the [covariance](https://en.wikipedia.org/wiki/Covariance).
* $\sigma_{X}$ is the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of $X$.
* $\sigma_{Y}$ is the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of $Y$.

The formula for ρ can be expressed in terms of mean and expectation. Since

$$cov(X,Y)=\mathbb{E}[(X-μ_{X})(Y-μ_{Y})],$$

the formula for ρ can also be written as

$$ρ_{X,Y}=\dfrac{\mathbb{E}[(X-μ_{X})(Y-μ_{Y})]}{\sigma_{X}\sigma_{Y}}$$

where:

* $\sigma_{X}$ and $\sigma_{Y}$ are defined as above.
* $μ_{X}$ is the [mean](https://en.wikipedia.org/wiki/Mean) of $X$.
* $μ_{Y}$ is the [mean](https://en.wikipedia.org/wiki/Mean) of $Y$.
* $\mathbb{E}$ is the [expectation](https://en.wikipedia.org/wiki/Expected_Value).

The formula for ρ can be expressed in terms of uncentered moments. Since

* $μ_{X} = \mathbb{E}[X]$
* $μ_{Y} = \mathbb{E}[Y]$
* $\sigma^{2}_{X} = \mathbb{E}[(X-\mathbb{E}[X])^{2}] = \mathbb{E}[X^{2}] - (\mathbb{E}[X])^{2}$
* $\sigma^{2}_{Y} = \mathbb{E}[(Y-\mathbb{E}[Y])^{2}] = \mathbb{E}[Y^{2}] - (\mathbb{E}[Y])^{2}$
* $\mathbb{E}[(X-μ_{X})(Y-μ_{Y})] = \mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$

the formula for ρ can also be written as

$$ρ_{X,Y} = \dfrac{\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]}{\sqrt{\mathbb{E}[X^{2}] - (\mathbb{E}[X])^{2}}\sqrt{\mathbb{E}[Y^{2}] - (\mathbb{E}[Y])^{2}}}$$

Pearson's correlation coefficient does not exist when either $\sigma_{X}$ or $\sigma_{Y}$ are zero, infinite, or undefined.
https://github.com/joshiayush/ai/blob/master/ai/algos/correlation/pearson_correlation/pearson_correlation.py