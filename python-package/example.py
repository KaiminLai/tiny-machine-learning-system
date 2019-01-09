from logistic_regression import LogisticRegression
import numpy as np

def mean_accuracy(label, pred, size):
	num_pos, hit_pos = 0.0, 0.0
	num_neg, hit_neg = 0.0, 0.0
	for i in range(size):
		if label[i] == 1.0:
			num_pos += 1.0
			if pred[i] > 0.5:
				hit_pos += 1.0

		if label[i] == 0.0:
			num_neg += 1.0
			if pred[i] <= 0.5:
				hit_neg += 1.0
	print("pos-accuracy:{0:.5f}, neg-accuracy:{1:.5f}".format(hit_pos/num_pos, hit_neg/num_neg))
	return 0.5*hit_pos/num_pos + 0.5*hit_neg/num_neg

features = np.load('../classification_train_feature_values.npy')
labels = np.load('../classification_train_label_values.npy')

print(features.shape, labels.shape, labels.sum())

clf = LogisticRegression(max_iter=1000, alpha=0.01, l2_lambda=0.0, tolerance=1e-7, seed=2018, use_batch=False)
clf.fit(features,labels,1024,100,mean_accuracy)
print(clf.predict(features[:30]))

clf.save("/home/laikaimin/lr.model")
clf1 = LogisticRegression()
clf1.load("/home/laikaimin/lr.model")
print(clf1.predict(features[:30]))