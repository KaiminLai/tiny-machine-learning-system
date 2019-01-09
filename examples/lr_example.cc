#include <iostream>
#include "../src/utils.h"
#include "../src/csv.h"
#include "../src/logistic_regression.h"
#include "omp.h"
#include <ctime>

using namespace Eigen;
using namespace std;
using namespace csv;

int main(){
	CSVReader train_features_csv("./classification_train_features.csv");
	CSVReader train_labels_csv("./classification_train_labels.csv");
	CSVReader test_features_csv("./classification_test_features.csv");
	CSVReader test_labels_csv("./classification_test_labels.csv");

	MatrixXd X_train = train_features_csv.get_mat();
	VectorXd y_train = train_labels_csv.get_col(0);
	MatrixXd X_test = test_features_csv.get_mat();
	VectorXd y_test = test_labels_csv.get_col(0);

	cout << "rows: " << X_train.rows() << " cols: " << X_train.cols() << endl;
	cout << "size: " << y_train.size() << endl;
	double startTime = omp_get_wtime();	
	LogisticRegression lr(1000, 0.01, 0.0, 1e-7, 2018, false);
	lr.fit(X_train, y_train);
	VectorXi train_pred = lr.predict(X_train);
	//cout << train_pred << endl;
	cout << "X_train accuracy: " << Utils::accuracy(y_train, train_pred.cast<double>()) << endl;;
	//VectorXd train_prob = lr.predict_prob(X_train);
	//cout << "X_train pred proba: " << train_prob << endl;

	lr.saveWeights("lr_weights");
	cout << "X_test ........................" << endl;
	VectorXi test_pred = lr.predict(X_test);
	cout << "X_test accuracy: " << Utils::accuracy(y_test, test_pred.cast<double>()) << endl;;
	LogisticRegression lr_save;
	lr_save.loadWeights("lr_weights");
	cout << "X_test accuracy: " << Utils::accuracy(y_test, lr_save.predict(X_test).cast<double>()) << endl;;
	double endTime = omp_get_wtime();
	cout << "Total time : " << (double)(endTime - startTime) << "s" << endl;
	return 0;
}

