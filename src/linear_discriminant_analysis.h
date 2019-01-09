#ifndef __LINEAR_DISCRIMINANT_ANALYSIS_H__
#define __LINEAR_DISCRIMINANT_ANALYSIS_H__

#include <eigen3/Eigen/Dense>
#include "utils.h"

class LDA{
public:
	LDA();
	~LDA();
	void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
	Eigen::VectorXi predict(const Eigen::MatrixXd& X);
	Eigen::MatrixXd transform(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

private:
	Eigen::MatrixXd W;
};


#endif