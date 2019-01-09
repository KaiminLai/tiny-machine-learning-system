#include <iostream>
#include "../src/utils.h"
#include "../src/csv.h"
#include "../src/k_nearest_neighbors.h"
#include "omp.h"
#include <ctime>
#include <cmath>

using namespace Eigen;
using namespace std;
using namespace csv;

int main(){	
	CSVReader train_features_csv("./knn_train_features.csv");
	CSVReader train_labels_csv("./knn_train_labels.csv");
	CSVReader test_features_csv("./knn_test_features.csv");
	CSVReader test_labels_csv("./knn_test_labels.csv");

	MatrixXd X_train = train_features_csv.get_mat();
	//VectorXd y_train = train_labels_csv.get_col(0);
	VectorXd y_train = train_labels_csv.get_mat();
	MatrixXd X_test = test_features_csv.get_mat();
	//VectorXd y_test = test_labels_csv.get_col(0);
	VectorXd y_test = test_labels_csv.get_mat();

	cout << "rows: " << X_train.rows() << " cols: " << X_train.cols() << endl;
	cout << "size: " << y_train.size() << endl;

	double startTime = omp_get_wtime();
	//clock_t startTime, endTime;
	//startTime = clock();	

	KNearestNeighbors knn(10);
	knn.fit(X_train, y_train);
	cout << "fit done..." << endl;

	VectorXi train_pred = knn.predict(X_train);
	cout << "X_train accuracy: " << Utils::accuracy(y_train, train_pred.cast<double>()) << endl;

	MatrixXd train_pred_prob = knn.predict_prob(X_train);
	//cout << "X_train_pred_prob: " << endl << train_pred_prob << endl;

	VectorXi test_pred = knn.predict(X_test);
	cout << "X_test accuracy: " << Utils::accuracy(y_test, test_pred.cast<double>()) << endl;

	MatrixXd test_pred_prob = knn.predict_prob(X_test);
	//cout << "X_test_pred_prob: " << endl << test_pred_prob << endl;

	double endTime = omp_get_wtime();
	cout << "Total time : " << (double)(endTime - startTime) << "s" << endl;
	//endTime = clock();
	//cout << "Total time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;	
	return 0;
}