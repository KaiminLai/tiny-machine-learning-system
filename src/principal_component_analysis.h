#ifndef __PCA_H__
#define __PCA_H__

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include "utils.h"

class PCA{
public:
	PCA(int n_components = 5);
	~PCA();
	void fit(const Eigen::MatrixXd& X);
	Eigen::MatrixXd transform(const Eigen::MatrixXd& X);

private:
	int n_components;
	Eigen::MatrixXd eigen_vectors;
}


#endif
