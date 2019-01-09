#include <iostream>
#include "../src/logistic_regression.h"
#include <ctime>

using namespace Eigen;
using namespace std;

void gen_random(char *s, int len) {
	srand(time(NULL));
	static const char alphanum[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	for (int i = 0; i < len; ++i){
		s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
	}
	s[len] = 0;
}

extern "C" void lr_fit(double** features, int* labels, int row, int col, int max_iter, double alpha, double lambda, 
					double tolerance, int seed, bool use_batch, int batch_size, int early_stopping_round, char* ret, 
					double (*metric)(double* y, double* pred, int size)=Utils::accuracy){
	MatrixXd X(row, col);
	VectorXd y(row);
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			X(i,j) = features[i][j];
		}
		y(i) = labels[i];
	}

	// train the logistic regression model
	LogisticRegression clf = LogisticRegression(max_iter, alpha, lambda, tolerance, seed, use_batch);
	clf.fit(X, y, batch_size, early_stopping_round, metric);

	// save the model weights
	char* fmodel = new char[21];
	gen_random(fmodel, 20);
	string model_path = "/tmp/"+string(fmodel);
	clf.saveWeights(model_path);
	strcpy(ret, model_path.c_str());
}

extern "C" void lr_predict_prob(double** features, int row, int col, char* fmodel, double* ret){
	LogisticRegression clf = LogisticRegression();
	clf.loadWeights(fmodel);
	MatrixXd X(row, col);
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			X(i,j) = features[i][j];
		}
	}
	VectorXd pred = clf.predict_prob(X);

	for(int i = 0; i < row; i++){
		ret[i] = pred(i);
	}
}

extern "C" void lr_predict(double** features, int row, int col, char* fmodel, int* ret){
	double* prob = new double[row];
	lr_predict_prob(features, row, col, fmodel, prob);
	for(int i = 0; i < row; i++){
		ret[i] = prob[i]>0.5?1:0;
	}
}


int main(){
	int row = 10, col = 2;
	double** features = new double *[row];
	for(int i = 0; i < row; i++){
		features[i] = new double[col];
	}
	int* labels = new int[row];

	double features_value[row*col] = {1.0,0.8,2.0,1.7,3.0,2.5,4.0,3.6,5.0,4.9,1.0,1.2,2.0,2.5,3.0,3.4,4.0,4.5,5.0,6.0};
	int labels_value[row] = {0,0,0,0,0,1,1,1,1,1};
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			features[i][j] = features_value[i*col+j];
		}
		labels[i] = labels_value[i];
	}
	char* ret = new char[26];
	lr_fit(features, labels, row, col, 200, 0.01, 0.0, 1e-7, 2018, false, 128, 100, ret);
	cout << ret << endl;

	int* pred = new int[row];
	lr_predict(features, row, col, ret, pred);
	for(int i = 0; i < row; i++){
		cout << pred[i] << ",";
	}
}
