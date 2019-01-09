#ifndef __K_NEAREST_NEIGHBORS_H__
#define __K_NEAREST_NEIGHBORS_H__

#include <eigen3/Eigen/Dense>
#include <string>
#include "utils.h"


class KNearestNeighbors{
public:
	KNearestNeighbors(int k = 5);
	~KNearestNeighbors();
	void fit(Eigen::MatrixXd X_train, Eigen::VectorXd y_train);
	Eigen::VectorXi predict(const Eigen::MatrixXd& X_test, double (*metric)(double* x1, double* x2, int size)=Utils::euclidean_distance);
	int predict(const Eigen::VectorXd& X_test, double (*metric)(double* x1, double* x2, int size)=Utils::euclidean_distance);
	Eigen::MatrixXd predict_prob(const Eigen::MatrixXd& X_test, double (*metric)(double* x1, double* x2, int size)=Utils::euclidean_distance);

private:
	int k;
	int d;  // the dimensionality of each sample
	int n;  // the classes of sample
	Eigen::MatrixXd X;
	Eigen::VectorXi y;


};





#endif