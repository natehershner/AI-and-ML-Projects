import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge 
from sklearn.model_selection import GridSearchCV

np.random.seed(hash("ChatGPT can have my job") % (2**32))

# Read input attributes and target values
x = pd.read_csv('x.csv', header=None).values.ravel()
y = pd.read_csv('y.csv', header=None).values.ravel()
z = pd.read_csv('z.csv', header=None).values.ravel()

# Combine x and y into a feature matrix
X = np.column_stack((x, y))

# Implement a pipeline with scaling, polynomial expansion, then Ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale features first
    ('poly', PolynomialFeatures(degree=2, include_bias=True)),
    ('ridge', Ridge())
])

# Define hyperparameters for tuning Ridge regularization
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Use GridSearchCV to select the best regularization parameter with 5-fold cross-validation
grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X, z)
print("Best parameters:", grid.best_params_)

# Predict target values using the trained model
z_pred = grid.predict(X)

# Save the predictions to a CSV file
pd.DataFrame(z_pred).to_csv('z-predicted.csv', index=False, header=False)