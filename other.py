import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# Load data and train a quick model
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
clf = RandomForestRegressor(n_estimators=10).fit(X, y)

# Plot the partial dependence for 'MedInc' (Median Income)
# and 'HouseAge'
features = ['MedInc', 'HouseAge']
PartialDependenceDisplay.from_estimator(clf, X, features)

plt.show()
