# Perceptron on Iris dataset using scikit-learn

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


# 1. Load and filter the data
iris = pd.read_csv('/Users/shivesh/Desktop/PythonProject/Python Machine Learning Textbook/Chapter 2/Perceptron (Iris DataSet)/iris.data.csv')

# Assign column names
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Check the data
print("\nGetting to know the dataset:")
print(iris.columns)
print(iris.shape)
print(iris.head())
print(iris.describe())
print("\nCheck for null values:")
print(iris.isnull().sum())

# Now you can use the columns as intended
X = iris[['petal_length', 'petal_width']].values
y = iris['species'].values

# Keep only Setosa and Versicolor for binary classification
mask = (y == 'Iris-setosa') | (y == 'Iris-versicolor')
X = X[mask]
y = y[mask]

# Encode the classes as integers: 0 for Setosa, 1 for Versicolor
y = np.where(y == 'Iris-setosa', 0, 1)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# 3. Standardize features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 4. Train Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# 5. Predict and evaluate
y_pred = ppn.predict(X_test_std)
print(f"Misclassified samples: {(y_test != y_pred).sum()}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 6. Plot decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x')
    colors = ('lightgreen', 'gray')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], label=f'Class {cl}')

plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Perceptron - Decision Regions (Training Set)')
plt.show()