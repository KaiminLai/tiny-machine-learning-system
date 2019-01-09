#ifndef __SUPPORT_VECTOR_MACHINE_H__
#define __SUPPORT_VECTOR_MACHINE_H__

#include <eigen3/Eigen/Dense>
#include <string>
#include <utility>
#include "utils.h"

// Support vector machines implementation using simplified SMO optimization.
class SupportVectorMachine{
public:
	SupportVectorMachine(int max_iter=1000, double C=1.0, double tolerance=1e-7);
	~SupportVectorMachine();
	void fit(const Eigen::MatrixXd& X_train, const Eigen::VectorXd& y_train, Eigen::VectorXd (*kernal)(Eigen::MatrixXd x, Eigen::VectorXd y)=Utils::Linear);
	Eigen::VectorXi predict(const Eigen::MatrixXd& X_test);

private:
	double C;
	double tolerance;
	int max_iter;
	double b;
	Eigen::MatrixXd X;
	Eigen::VectorXd y;
	Eigen::VectorXd alpha;
	Eigen::VectorXi sv_idx;

	Eigen::MatrixXd X_sv;
	Eigen::VectorXd alpha_sv;
	Eigen::VectorXd y_sv;
	int size;
	int sv_count;
	int random_index(int z);
	std::pair<double, double> find_bounds(int i, int j);
	double predict_row(const Eigen::VectorXd& x_row, const Eigen::VectorXd& k_v);
	double predict_row(const Eigen::VectorXd& x_row);
	double error(int i, const Eigen::VectorXd& k_row);
	double clip(double alpha, double H, double L);



};



#endif