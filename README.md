# First Steps with scikit-learn: Training a Perceptron on the Iris Dataset

Author: Shivesh Raj Sahu

⸻

## Introduction

The Perceptron algorithm is a fundamental building block in machine learning, representing one of the earliest supervised learning methods for classification tasks.
In this project, we apply Python’s scikit-learn library to train a Perceptron classifier on the well-known Iris dataset—a standard for learning and demonstrating classification concepts.

Project Workflow Includes:
	•	Loading and understanding the dataset
	•	Preparing and filtering features and labels
	•	Splitting data into training and testing sets
	•	Standardizing features for optimal learning
	•	Training and evaluating the model
	•	Visualizing the learned decision boundary

This hands-on project not only reinforces core ML concepts but also prepares us for more sophisticated classifiers by highlighting both the strengths and weaknesses of linear models.

⸻

## Theoretical Background

What is a Perceptron?

A Perceptron is a type of linear classifier—a mathematical model that attempts to find a hyperplane (a straight line in 2D) that separates two classes. The Perceptron algorithm iteratively adjusts the weights assigned to each input feature based on the prediction error until the model converges (if possible).

Key Properties
	•	Binary classifier: Only distinguishes between two classes
	•	Linear: Can only separate data with a straight line (or hyperplane)
	•	Deterministic output: Gives hard class labels (not probabilities)
	•	Sensitive to feature scale: Standardization is necessary for good results

Limitations
	•	Only works if the data is linearly separable (can be perfectly split by a straight line)
	•	Fails on data where classes overlap in feature space
	•	Cannot output probabilities—only labels
	•	Sensitive to feature scaling and outliers

Why Use the Iris Dataset?
	•	Simple and ideal for visualization
	•	Contains three species (we focus on two for binary classification)
	•	Using petal length and width as features, Setosa and Versicolor are linearly separable

⸻

## Data Preparation and Workflow

1. Loading and Exploring the Data
	•	The dataset is loaded from a CSV file or using scikit-learn’s built-in loader.
	•	Column names are assigned for clarity and checked for data consistency.

2. Feature and Label Selection
	•	For binary classification, we select only the Setosa and Versicolor classes.
	•	Features used: petal_length and petal_width.

3. Splitting the Data
	•	Data is split into training and testing sets to evaluate generalization.
	•	Training set: Used to fit the model
	•	Testing set: Used to evaluate performance

4. Feature Standardization
	•	Standardization (zero mean, unit variance) ensures fair weight updates and model convergence.

⸻

## Model Training and Evaluation

Training the Perceptron

from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

Evaluating the Model

y_pred = ppn.predict(X_test_std)
print('Misclassified samples:', (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

	•	Misclassified samples: Number of test points incorrectly predicted
	•	Accuracy: Proportion of correct predictions

⸻

## Visualizing the Decision Boundary

	•	The plot displays the decision surface learned by the Perceptron.
	•	Different colors represent different predicted classes.
	•	The boundary is a straight line, showing linear separability.

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # plotting code (see full script for details)

(See the script for the full plotting code.)

⸻

## Results
	•	Accuracy: 1.00 (100%)
	•	Misclassified samples: 0
	•	The model perfectly separated Setosa and Versicolor using petal features.
	•	The decision boundary is a straight line—clear visual evidence of linear separability.

⸻

## Advantages and Limitations

Advantages:
	•	Fast and simple to implement
	•	Effective for linearly separable data
	•	Foundation for understanding neural networks

Limitations:
	•	Only works for binary (two-class) tasks
	•	Fails on non-linearly separable data
	•	No probabilistic output
	•	Sensitive to feature scaling and outliers

How are these limitations solved?
	•	Logistic Regression (next topic) provides probability estimates
	•	SVMs, kernel methods, and neural networks handle non-linear boundaries and multiclass problems

⸻

## Advanced Notes & Professional Tips
	•	Use cross-validation for more robust performance estimates
	•	For larger/more complex datasets, explore pipelines and model serialization
	•	Try multiclass problems (all three iris species) for further learning
	•	Use more features and try other linear classifiers for comparison

⸻

## References
	•	Python Machine Learning by Sebastian Raschka
	•	scikit-learn documentation
	•	Iris dataset description

⸻
