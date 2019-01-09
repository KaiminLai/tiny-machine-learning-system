import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets
from sklearn.datasets import make_classification
try:
	from sklearn.model_selection import train_test_split
except ImportError:
	from sklearn.cross_validation import train_test_split

def generate_classification_csv():
	# Generate a random binary classification problem.
	X, y = make_classification(n_samples=100000, n_features=100,
								n_informative=80, random_state=2018,
								n_classes=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2018)
	X_train = pd.DataFrame(X_train)
	X_test = pd.DataFrame(X_test)
	y_train = pd.DataFrame(y_train)
	y_test = pd.DataFrame(y_test)
	X_train.to_csv("classification_train_features.csv", index=False, header=False)
	X_test.to_csv("classification_test_features.csv", index=False, header=False)
	y_train.to_csv("classification_train_labels.csv", index=False, header=False)
	y_test.to_csv("classification_test_labels.csv", index=False, header=False)
	np.save("classification_train_feature_values.npy", X_train.values)
	np.save("classification_test_feature_values.npy", X_test.values)
	np.save("classification_train_label_values.npy", y_train.values)
	np.save("classification_test_label_values.npy", y_test.values)

	X_train[:2000].to_csv("knn_train_features.csv", index=False, header=False)
	X_test[:2000].to_csv("knn_test_features.csv", index=False, header=False)
	y_train[:2000].to_csv("knn_train_labels.csv", index=False, header=False)
	y_test[:2000].to_csv("knn_test_labels.csv", index=False, header=False)


	X, y = make_classification(n_samples=500, n_features=5, n_informative=5,
							n_redundant=0, n_repeated=0, n_classes=3,
							random_state=1111, class_sep=1.5,)
	X_ = pd.DataFrame(X)
	y_ = pd.DataFrame(y)
	X_.to_csv("tree_sample_features.csv", index=False, header=None)
	y_.to_csv("tree_sample_labels.csv", index=False, header=None)

if __name__ == '__main__':
	generate_classification_csv()