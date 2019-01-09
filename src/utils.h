#ifndef __UTILS_H__
#define __UTILS_H__

#include <eigen3/Eigen/Dense>
#include <array>

class Utils{
public:
	// sigmoid function, depend on <cmath> library
	static double sigmoid(double x);
	static Eigen::VectorXd sigmoid(Eigen::VectorXd x);
	static double squareLoss(Eigen::VectorXd y, Eigen::VectorXd y_pred);
	static double crossEntropyLoss(Eigen::VectorXd y, Eigen::VectorXd y_pred);
	static double accuracy(Eigen::VectorXd y, Eigen::VectorXd pred);
	static double accuracy(double* y, double* pred, int size);
	static double* VectorXd_to_double_array(Eigen::VectorXd pred);
	static int* VectorXi_to_int_array(Eigen::VectorXi y);
	static Eigen::MatrixXd slice(Eigen::MatrixXd X, int start_idx, int end_idx);
	static Eigen::VectorXd slice(Eigen::VectorXd X, int start_idx, int end_idx);
	static double euclidean_distance(Eigen::VectorXd x1, Eigen::VectorXd x2);
	static double euclidean_distance(double* x1, double* x2, int size);
	// divide dataset based on if sample value on feature index is larger than the givern threshold
	static std::array<Eigen::MatrixXd, 2> divide_on_feature(Eigen::MatrixXd X, int j, double threshold);
	static double calculate_entropy(Eigen::VectorXd y);
	static Eigen::VectorXd Linear(Eigen::MatrixXd X, Eigen::VectorXd y);
	static Eigen::MatrixXd calculate_covariance_matrix(Eigen::MatrixXd X);
	static Eigen::MatrixXd calculate_covariance_matrix(Eigen::MatrixXd X, Eigen::MatrixXd Y);
};



#endif