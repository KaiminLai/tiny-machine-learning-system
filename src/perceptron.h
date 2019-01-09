#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include <eigen3/Eigen/Dense>
#include <string>
#include "utils.h"

class Perceptron{
public:
	Perceptron(const int& n_iterations=2000, const double& learning_rate=0.01, const double& tolerance=0.001, const int& seed=2018, const int& early_stopping_round=100);
	~Perceptron();
	void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, Eigen::VectorXd (*activation)(Eigen::VectorXd x)=Utils::sigmoid, double (*loss)(Eigen::VectorXd y, Eigen::VectorXd y_pred)=Utils::squareLoss,
			double (*metric)(Eigen::VectorXd y, Eigen::VectorXd pred)=Utils::accuracy);
	Eigen::VectorXi predict(const Eigen::MatrixXd& X);
	Eigen::VectorXd predict_prob(const Eigen::MatrixXd& X);

private:
	int n_iterations;
	double learning_rate;
	double tolerance;
	int seed;
	int early_stopping_round;
	Eigen::VectorXd W;

};




#endif