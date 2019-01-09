#include <iostream>
#include "linear_discriminant_analysis.h"
#include "utils.h"
#include <omp.h>
#include <ctime>
#include <eigen3/Eigen/QR>

using namespace std;
using namespace Eigen;

LDA::LDA() {}

LDA::~LDA() {}

void LDA::fit(const MatrixXd& X, const VectorXd& y){
	int count = 0;
	VectorXi y_ = y.cast<int>();
	for(int i = 0; i < y_.size(); i++){
		if(y_(i) == 1){
			count++;
		}
	}
	MatrixXd X1(count, X.cols());
	int size_x2 = X.rows() - count;
	MatrixXd X2(size_x2, X.cols());
	for(int i = 0, j = 0, k = 0; i < X.rows(); i++){
		if(y_(i) == 1){
			X1.row(j) = X.row(i);
			j++;
		}else{
			X2.row(k) = X.row(i);
			k++;
		}
	}
	MatrixXd cov1 = Utils::calculate_covariance_matrix(X1);
	MatrixXd cov2 = Utils::calculate_covariance_matrix(X2);
	MatrixXd cov_tot = cov1 + cov2;

	VectorXd mean1 = X1.colwise().mean();
	VectorXd mean2 = X2.colwise().mean();
	VectorXd mean_diff = mean1 - mean2;
	CompleteOrthogonalDecomposition<MatrixXd> cod(cov_tot);
	W = cod.pseudoInverse()*mean_diff;
}

VectorXi LDA::predict(const MatrixXd& X){
	VectorXi ret(X.rows());
	for(int i = 0; i < X.rows(); i++){
		double h = X.row(i).dot(W);
		ret(i) = h > 0 ? 1 : 0;
	}
	return ret;
}

MatrixXd LDA::transform(const MatrixXd& X, const VectorXd& y){
	fit(X, y);
	//MatrixXd X_transform = X*W;
	return X*W;
}