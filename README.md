# py-earth

A Python implementation of Multivariate Adaptive Regression Splines (MARS) algorithm. Ref Friedman 1991. Detailed refs below.
- **It has scikit-learn style**. That is, it provides an interface that is compatible with scikit-learn's Estimator, Predictor, Transformer, and Model interfaces.
- For speed / excellent scaling, it optionally **uses "Fast MARS"**. Ref Friedman 1993. Details in appendix below.
- For more speed, it uses **Cython**.
- It can be used as a **pypi-style package. Or run directly** from this repo. Or copied as a sub-dir of another repo (with minor modifications).

This is a fork of [scikit-learn-contrib/py-earth](https://github.com/scikit-learn-contrib/py-earth), which was last changed in 2017, and archived a few years later. Changes: make it work on Python 3.12; be able to run directly (vs pypi); fix other small errors that emerged; add 2D usage example.

## Table of Contents
- [Installation](#installation)
- [Usage: 1D Model](#usage-1d-model)
- [Usage: 2D Model](#usage-2d-model)
- [Appendix: References](#appendix-references)
- [Appendix: MARS vs Fast MARS](#appendix-mars-vs-fast-mars)
- [Appendix: Other Implementations](#appendix-other-implementations)

## Installation

Requires Python 3.8+.

```bash
git clone https://github.com/trentmc/py-earth.git
cd py-earth

# Create and activate virtualenv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy cython six scikit-learn

# Build Cython extensions in-place (required for Python 3.12+)
python setup.py build_ext --inplace --cythonize
```

## Usage: 1D Model
```python
import numpy
from pyearth import Earth
from matplotlib import pyplot
    
#Create some fake data
numpy.random.seed(0)
m = 1000
n = 10
X = 80*numpy.random.uniform(size=(m,n)) - 40
y = numpy.abs(X[:,6] - 4.0) + 1*numpy.random.normal(size=m)
    
#Fit an Earth model
model = Earth(use_fast=True) # use_fast = FAST Mars y/n
model.fit(X,y)
    
#Print the model
print(model.trace())
print(model.summary())
    
#Plot the model
y_hat = model.predict(X)
pyplot.figure()
pyplot.plot(X[:,6],y,'r.')
pyplot.plot(X[:,6],y_hat,'b.')
pyplot.xlabel('x_6')
pyplot.ylabel('y')
pyplot.title('Simple Earth Example')
pyplot.show()
 ```

## Usage: 2D Model
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from pyearth import Earth

# Load the diabetes dataset (10 input features, 442 samples)
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
N, n = diabetes_X.shape
print(f"N (# samples) = {N}, n (# features) = {n}")

# Split data into train vs test
X_train, X_test = diabetes_X[:-20], diabetes_X[-20:]
y_train, y_test = diabetes_y[:-20], diabetes_y[-20:]
print(f"N_train = {len(y_train)}, N_test = {len(y_test)}")

# Create and train MARS model (with feature importance)
model = Earth(feature_importance_type='gcv', use_fast=True)
model.fit(X_train, y_train)

# Print the model
print(model.summary())

# Variable importances: find the two most important
imps = model.feature_importances_
imps_I = np.argsort(-imps)
var0, var1 = imps_I[0], imps_I[1]
print(f"Most important vars: {var0}, {var1}")

# Predict on test set
yhat_test = model.predict(X_test)

# Performance
mse = mean_squared_error(y_test, yhat_test)
R2 = r2_score(y_test, yhat_test)
print(f"MSE: {mse:.2f}, R^2: {R2:.2f}")

# 3D Plot: actual vs predicted, over the two most important variables
v0, v1 = X_test[:, var0], X_test[:, var1]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(v0, v1, y_test, marker='o', label='actual')
ax.scatter(v0, v1, yhat_test, marker='^', label='predicted')
ax.set_xlabel(f"x{var0}")
ax.set_ylabel(f"x{var1}")
ax.set_zlabel("y")
ax.set_title(f"MARS on Diabetes (MSE={mse:.1f}, R^2={R2:.2f})")
ax.legend()
plt.show()
```

## Appendix: References

1. **Friedman, J. (1991). Multivariate adaptive regression splines. The annals of statistics, 19(1), 1–67. [pdf](https://www.stat.yale.edu/~lc436/08Spring665/Mars_Friedman_91.pdf)**
2. Stephen Milborrow. Derived from mda:mars by Trevor Hastie and Rob Tibshirani. (2012). earth: Multivariate Adaptive Regression Spline Models. R package
   version 3.2-3. http://CRAN.R-project.org/package=earth
3. **Friedman, J. (1993). Fast MARS. Stanford Dept. Statistics, Tech. Report No 110. [pdf](https://www.stat.yale.edu/~lc436/08Spring665/Mars_Friedman_91.pdf)**
4. Friedman, J. (1991). Estimating functions of mixed ordinal and categorical variables using adaptive splines. Stanford University Department of Statistics, Technical Report No 108. http://media.salford-systems.com/library/MARS_V2_JHF_LCS-108.pdf
5. Stewart, G.W. Matrix Algorithms, Volume 1: Basic Decompositions. (1998). Society for Industrial and Applied Mathematics.
6. Bjorck, A. Numerical Methods for Least Squares Problems. (1996). Society for Industrial and Applied Mathematics.
7. Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning (2nd Edition). (2009).  Springer Series in Statistics
8. Golub, G., & Van Loan, C. Matrix Computations (3rd Edition). (1996). Johns Hopkins University Press.
   
Refs 7, 2, 1, 3, and 4 contain discussions likely to be useful to users of py-earth.  References 1, 2, 6, 5, 8, 3, and 4 were useful during the implementation process.

## Appendix: MARS vs Fast MARS

_Q: Does py-earth use the original MARS algorithm (ref 1)? Or, does it use Fast MARS (ref 3)?_

**A: py-earth implements both algorithms.** It defaults to the original MARS (Friedman 1991) and optionally supports Fast MARS (Friedman 1993) via the `use_fast` parameter.

**Default** (`use_fast=False`): Evaluates all basis functions as candidate parents and all candidate knots at each step.

**Fast MARS** (`use_fast=True`): Adds three optimizations from the 1993 paper:
- A priority queue (`fast_heap`) to consider only the top `fast_K` parent basis functions (ranked by MSE improvement)
- A swept knot search (`fast_update` in `_knot_search.pyx`) that evaluates knots by incrementally updating statistics (`alpha`, `beta`, `gamma`, `kappa`, etc.) instead of recomputing regressions
- An updating QR decomposition (`UpdatingQT` in `_qr.pyx`) using Householder transformations

  Both modes share the updating QR machinery for the core knot search. The `use_fast` flag controls whether the priority queue shortcut is used to skip unpromising parent basis functions. 
  
Usage: 
```python
model = Earth(use_fast=True, fast_K=20, fast_h=1)
```

The explicit reference is in `_forward.pyx:32:` .. [1] Fast MARS, Jerome H.Friedman, Technical Report No.110, May 1993.

## Appendix: Other Implementations

I am aware of the following implementations of Multivariate Adaptive Regression Splines:

1. The R package earth (coded in C by Stephen Millborrow): http://cran.r-project.org/web/packages/earth/index.html
2. The R package mda (coded in Fortran by Trevor Hastie and Robert Tibshirani): http://cran.r-project.org/web/packages/mda/index.html
3. The Orange data mining library for Python (uses the C code from 1): http://orange.biolab.si/
4. The xtal package (uses Fortran code written in 1991 by Jerome Friedman): http://www.ece.umn.edu/users/cherkass/ee4389/xtalpackage.html
5. MARSplines by StatSoft: http://www.statsoft.com/textbook/multivariate-adaptive-regression-splines/
6. MARS by Salford Systems (also uses Friedman's code): http://www.salford-systems.com/products/mars
7. ARESLab (written in Matlab by Gints Jekabsons): http://www.cs.rtu.lv/jekabsons/regression.html

The R package earth was most useful to me in understanding the algorithm, particularly because of Stephen Milborrow's 
thorough and easy to read vignette (http://www.milbo.org/doc/earth-notes.pdf).


   
