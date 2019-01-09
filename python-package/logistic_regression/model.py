from ctypes import *
import numpy as np
import os
import shutil
from threading import Thread

liblr = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__))+'/liblr.so')

def accuracy(y, pred, size):
	hit = 0.0
	for i in range(size):
		if y[i] == 1.0 and pred[i] > 0.5:
			hit += 1.0
		if y[i] == 0.0 and pred[i] <= 0.5:
			hit += 1.0
	return hit/size


class LogisticRegression(object):
	def __init__(self, max_iter=200, alpha=0.01, l2_lambda=0.0, tolerance=0.001, seed=2018, use_batch=False):
		self.max_iter = max_iter
		self.alpha = alpha
		self.l2_lambda = l2_lambda
		self.tolerance = tolerance
		self.seed = seed
		self.use_batch = use_batch
		self.fmodel = None
		self.auto_clear = True

	# support python list, numpy array
	def fit(self, features, labels, batch_size=128, early_stopping_round=100, metric=accuracy):
		# convert to numpy array
		# if not isinstance(features, np.ndarray):
		features = np.asarray(features, dtype=np.double)
		labels = np.ascontiguousarray(np.asarray(labels, dtype=np.int32), dtype=np.int32)

		# convert to ctypes's type
		row, col = features.shape
		int_p = cast(labels.ctypes.data, POINTER(c_int))
		double_p_p = (features.ctypes.data + np.arange(features.shape[0]) * features.strides[0]).astype(np.uintp)
		char_p = c_char_p(str("0"*25).encode('utf-8'))

		# call the C function
		DOUBLEPP = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
		INTP = POINTER(c_int)
		METRIC = CFUNCTYPE(c_double, POINTER(c_double), POINTER(c_double), c_int)
		liblr.lr_fit.argtypes = [DOUBLEPP,INTP,c_int,c_int,c_int,c_double,c_double,c_double,c_int,c_bool, c_int,c_int,c_char_p, METRIC]
		liblr.lr_fit.restype = None

		# enable interrupt
		t = Thread(target=liblr.lr_fit, args=(double_p_p,int_p,c_int(row),c_int(col),c_int(self.max_iter),c_double(self.alpha),c_double(self.l2_lambda),c_double(self.tolerance),c_int(self.seed), c_bool(self.use_batch), c_int(batch_size), c_int(early_stopping_round),char_p,METRIC(metric)))
		t.daemon = True
		t.start()
		while t.is_alive():
			t.join(0.1)

		# get the result
		self.fmodel = char_p.value

	def predict_prob(self, features):
		assert self.fmodel is not None
		# convert to numpy array
		features = np.asarray(features, dtype=np.double)

		# convert to ctypes's type
		row, col = features.shape
		double_p_p = (features.ctypes.data + np.arange(features.shape[0]) * features.strides[0]).astype(np.uintp)
		ret_double_p = (c_double*row)(*([-1.0 for _ in range(row)]))
		# call C function
		DOUBLEPP = np.ctypeslib.ndpointer(dtype=np.uintp,ndim=1,flags='C')
		liblr.lr_predict_prob.argtypes = [DOUBLEPP, c_int, c_int, c_char_p, POINTER(c_double)]
		liblr.lr_predict_prob.restype = None
		# enable interrupt
		t = Thread(target=liblr.lr_predict_prob, args=(double_p_p, c_int(row), c_int(col), c_char_p(self.fmodel),ret_double_p))
		t.daemon = True
		t.start()
		while t.is_alive():
			t.join(0.1)

		return [ret_double_p[i] for i in range(row)]

	def predict(self, features):
		assert self.fmodel is not None
		prob = self.predict_prob(features)
		return [1 if p>0.5 else 0 for p in prob]

	def save(self, path):
		shutil.copy(self.fmodel, path)

	def load(self, path):
		self.fmodel = path.encode('utf-8')
		self.auto_clear = False

	def __del__(self):
		if self.auto_clear:
			os.remove(self.fmodel)