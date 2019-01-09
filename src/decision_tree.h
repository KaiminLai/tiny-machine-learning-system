#ifndef _DECISION_TREE_H__
#define _DECISION_TREE_H__

#include <eigen3/Eigen/Dense>
#include <string>
#include "utils.h"

class DecisionNode{
public:
	DecisionNode(int value) : value(value) {
		true_branch = nullptr;
		false_branch = nullptr;
	}
	DecisionNode(int feature, double threshold, int value, DecisionNode* true_branch, DecisionNode* false_branch) : feature(feature), 
				threshold(threshold), value(value), true_branch(true_branch), false_branch(false_branch) {}
	~DecisionNode() {}
	int feature; // index for the feature that is tested
	double threshold; // threshold value for feature
	int value;  // value if the node is a leaf in the tree
	DecisionNode* true_branch; // 'left' subtree
	DecisionNode* false_branch; // 'right' subtree
};

class DecisionTree{
public:
	DecisionTree(int min_samples_split=2, double min_impurity=1e-7, int max_depth=100);
	~DecisionTree();
	void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
	Eigen::VectorXi predict(const Eigen::MatrixXd& X);
	//void print_tree(DecisionNode* r=nullptr);

private:
	DecisionNode* root;
	int min_samples_split;
	double min_impurity;
	int max_depth;
	DecisionNode* build_tree(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int current_depth);
	double impurity_calculation(const Eigen::VectorXd& y, const Eigen::VectorXd& y1, const Eigen::VectorXd& y2);
	int leaf_value_calculation(const Eigen::VectorXd& y);	
	int predict(const Eigen::VectorXd& X, DecisionNode* r);
};



#endif