# -*- coding: utf-8 -*-
"""
@author: parra
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

theta_vector = np.array([-25,1,2])

np.random.seed(10)
x1_inputs = np.random.uniform(-50, 50, size=100)
np.random.seed(100)
x2_inputs = np.random.uniform(-50, 50, size=100)

row_list = []
labels = []

for x1, x2 in zip(x1_inputs, x2_inputs):
    x_i_vector = np.array([1, x1, x2])
    row_list.append(x_i_vector)

    hypothesis_prediction = np.dot(theta_vector, x_i_vector)

    if hypothesis_prediction >= 25:
        labels.append(1)
    else:
        labels.append(0)

design_matrix = np.array(row_list)
data = pd.DataFrame(row_list,columns = ['intercept','x1','x2'])
data['label'] = labels
np.random.seed(123)
test_ids = np.random.choice(np.arange(0, data.shape[0]), size=int(.20*data.shape[0]), replace=False)
test_data = data[data.index.isin(test_ids)]
training_data = data[~data.index.isin(test_ids)]

test_data['type'] = 'test'
training_data['type'] = 'train'

full_data = pd.concat([test_data,training_data])

cols_to_keeps = ['x1','x2','label']

# Create the scatter plot
plt.scatter(x1_inputs, x2_inputs, c=labels, cmap='viridis', s=100, edgecolor='k')

# Add a color bar
plt.colorbar(label='Color intensity')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Logitstic Regression with linear Decision Boundary')

# Show the plot
plt.show()
# Step 6: Train Logistic Regression

# Select input features and target labels
X_train = training_data[['x1', 'x2']]
y_train = training_data['label']

X_test = test_data[['x1', 'x2']]
y_test = test_data['label']

# Create the logistic regression model
log_model = LogisticRegression(max_iter=1000)

# Train the model
log_model.fit(X_train, y_train)

# Make predictions on the test data
log_predictions = log_model.predict(X_test)

# Calculate accuracy
log_accuracy = accuracy_score(y_test, log_predictions)

print("Logistic Regression Test Accuracy:", log_accuracy)
print("Logistic Regression Test Accuracy:", log_accuracy)

# Step 7: Train KNN Models

# KNN with k = 3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)

knn3_predictions = knn3.predict(X_test)
knn3_accuracy = accuracy_score(y_test, knn3_predictions)

print("KNN (k=3) Test Accuracy:", knn3_accuracy)


# KNN with k = 15
knn15 = KNeighborsClassifier(n_neighbors=15)
knn15.fit(X_train, y_train)

knn15_predictions = knn15.predict(X_test)
knn15_accuracy = accuracy_score(y_test, knn15_predictions)

print("KNN (k=15) Test Accuracy:", knn15_accuracy)

# Step 8: Compare Decision Boundaries

# Create a grid of points
xx, yy = np.meshgrid(
    np.linspace(-50, 50, 200),
    np.linspace(-50, 50, 200)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

# True decision boundary from the data-generating rule
true_predictions = (
    -25 + grid_points[:, 0] + 2 * grid_points[:, 1] >= 25
).astype(int)

# Model predictions
log_grid = log_model.predict(grid_points)
knn3_grid = knn3.predict(grid_points)
knn15_grid = knn15.predict(grid_points)

# Create four comparison plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

models = [
    ("True Decision Boundary", true_predictions),
    ("Logistic Regression", log_grid),
    ("KNN (k=3)", knn3_grid),
    ("KNN (k=15)", knn15_grid)
]

for ax, (title, predictions) in zip(axes.ravel(), models):

    ax.contourf(
        xx, yy,
        predictions.reshape(xx.shape),
        alpha=0.3,
        cmap="viridis",
        levels=[-0.5, 0.5, 1.5]
    )

    ax.scatter(
        data["x1"],
        data["x2"],
        c=data["label"],
        cmap="viridis",
        edgecolor="black",
        s=35,
        vmin=0,
        vmax=1
    )

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

plt.tight_layout()

# Save the comparison graph
plt.savefig("decision_boundary_comparison.png", dpi=300)

plt.show()