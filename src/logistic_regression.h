#ifndef __LOGISTIC_REGRESSION_H__
#define __LOGISTIC_REGRESSION_H__

#include <eigen3/Eigen/Dense>
#include <string>
#include "utils.h"

class LogisticRegression{
public:
	LogisticRegression();
	LogisticRegression(const int& max_iter, const double& alpha, const double& lambda, const double& tolerance, const int& seed, bool use_batch);
	~LogisticRegression();
	void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const int& batch_size=128, const int& early_stopping_round=50, double (*metric)(double* y, double* y_pred, int size)=Utils::accuracy);
	//void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const int& batch_size=128, const int& early_stopping_round=50, double (*metric)(Eigen::VectorXd y, Eigen::VectorXd y_pred)=Utils::accuracy);
	Eigen::VectorXd getW();
	Eigen::VectorXd predict_prob(const Eigen::MatrixXd& X);
	Eigen::VectorXi predict(const Eigen::MatrixXd& X);
	void saveWeights(const std::string& fpath);
	void loadWeights(const std::string& fpath);

private:
	Eigen::VectorXd W;
	int max_iter;
	int seed;            // random seed
	double lambda;       // l2 regularization
	double tolerance;    // error tolerance 
	double alpha;        // learning rate
	bool use_batch;          // if use batch_size
};



#endif