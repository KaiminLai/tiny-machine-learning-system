#include <iostream>
#include "pca.h"
#include "utils.h"
#include <omp.h>
#include <eigen3/Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

PCA::PCA(int n_components) n_components(n_components) {}

PCA::~PCA() {}

void PCA::fit(const MatrixXd& X){
	MatrixXd covariance_matrix = Utils::calculate_covariance_matrix(X);
	SelfAdjointEigenSolver<MatrixXd> es(covariance_matrix);
	//MatrixXd eigen_values = es.eigenvalues();
	eigen_vectors = es.eigenvectors().rightCols(n_components);
}

MatrixXd PCA::transform(const MatrixXd& X){
	return X*eigenvectors;
}