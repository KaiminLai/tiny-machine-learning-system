#include <iostream>
#include "../src/utils.h"
#include "../src/csv.h"
#include "../src/decision_tree.h"
#include <ctime>
#include <cmath>

using namespace Eigen;
using namespace std;
using namespace csv;

int main(){
	//CSVReader train_features_csv("./knn_train_features.csv");
	//CSVReader train_labels_csv("./knn_train_labels.csv");
	//CSVReader test_features_csv("./knn_test_features.csv");
	//CSVReader test_labels_csv("./knn_test_labels.csv");

	CSVReader train_features_csv("./knn_sample_features.csv");
	CSVReader train_labels_csv("./knn_sample_labels.csv");
	CSVReader test_features_csv("./knn_sample_features.csv");
	CSVReader test_labels_csv("./knn_sample_labels.csv");

	MatrixXd X_train = train_features_csv.get_mat();
	VectorXd y_train = train_labels_csv.get_mat();
	MatrixXd X_test = test_features_csv.get_mat();
	VectorXd y_test = test_labels_csv.get_mat();

	cout << "rows: " << X_train.rows() << " cols: " << X_train.cols() << endl;
	cout << "size: " << y_train.size() << endl;

	double startTime = omp_get_wtime();	
	DecisionTree tree;
	tree.fit(X_train, y_train);
	VectorXi train_pred = tree.predict(X_train);
	cout << "X_train accuracy: " << Utils::accuracy(y_train, train_pred.cast<double>()) << endl;;

	cout << "X_test ........................" << endl;
	VectorXi test_pred = tree.predict(X_test);
	cout << "X_test accuracy: " << Utils::accuracy(y_test, test_pred.cast<double>()) << endl;;

	double endTime = omp_get_wtime();
	cout << "Total time : " << (double)(endTime - startTime) << "s" << endl;
	return 0;
}