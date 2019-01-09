#ifndef __NAIVE_BAYES_H__
#define __NAIVE_BAYES_H__

#include <eigen3/Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include "utils.h"

class NaiveBayes{
public:
	NaiveBayes();
	~NaiveBayes();
	void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
	Eigen::VectorXi predict(const Eigen::MatrixXd& X);
	int predict(const Eigen::VectorXd& X);

private:
	std::map<int, std::map<int, std::pair<double, double> > > parameters; // map<label, map<col, pair<mean, var>>>
	std::map<int, double>  priors; // map<label, prior>
	int d;  // the dimensionality of each sample
	double calculate_likelihood(double mean, double var, double x);

};





#endif