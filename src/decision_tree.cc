#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <boost/format.hpp>
#include "decision_tree.h"
#include "utils.h"

using namespace std;
using namespace Eigen;

DecisionTree::DecisionTree(int min_samples_split, double min_impurity, int max_depth) : min_samples_split(min_samples_split), min_impurity(min_impurity), max_depth(max_depth) {}

DecisionTree::~DecisionTree() {}


void DecisionTree::fit(const MatrixXd& X, const VectorXd& y){
	root = build_tree(X, y, 0);
}

DecisionNode* DecisionTree::build_tree(const MatrixXd& X, const VectorXd& y, int current_depth){
	double largest_impurity = 0.0;
	//cout << "start..." << endl;
	MatrixXd Xy = MatrixXd(X.rows(), X.cols()+1);
	Xy << X, y;
	//cout << "its ok?"<<endl;
	//cout << "..............." <<endl;
	//cout << "X.rows: " << X.rows() << "  X.cols:" << X.cols() << " y.size:" << y.size() << "  ..Xy.rows:" << Xy.rows() << "..Xy.cols:" << Xy.cols() << endl;
	int n_samples = X.rows();
	int n_features = X.cols();

	array<double, 2> best_criteria;
	array<MatrixXd, 4> best_sets;

	if(n_samples >= min_samples_split && current_depth < max_depth){
		// calculate the impurity for each feature
		for(int i = 0; i < n_features; i++){
			// all values of feature index i
			VectorXd feature_values = X.col(i);
			unordered_set<double> unique_values;
			for(int j = 0; j < feature_values.size(); j++){
				unique_values.insert(feature_values[j]);
			}

			// iterate through all unique values of feature column i and
			// calculate the impurity
			for(auto p = unique_values.begin(); p != unique_values.end(); p++){
				//cout << "start div..." << endl;
				array<MatrixXd,2> X_div = Utils::divide_on_feature(Xy, i, *p);
				MatrixXd Xy1 = X_div[0];
				MatrixXd Xy2 = X_div[1];
				//cout << "Xy1: " << Xy1.rows() << ".." << Xy1.cols() << endl;
				//cout << "Xy2: " << Xy2.rows() << ".." << Xy2.cols() << endl;

				if(Xy1.rows() > 0 && Xy2.rows() > 0){
					// select the y_values of the two sets
					VectorXd y1 = Xy1.col(Xy1.cols()-1);
					VectorXd y2 = Xy2.col(Xy2.cols()-1);
					//cout << "y1:" << y1.size() << "y2: " << y2.size() << endl;
					// calculate impurity
					double impurity = impurity_calculation(y, y1, y2);
					//cout << "impurity done " << impurity << endl;
					// if this threshold resulted in a higher information gain than previously
					// recorded save the threshod value and the feature
					if(impurity > largest_impurity){
						largest_impurity = impurity;
						best_criteria[0] = i;
						best_criteria[1] = *p;
						//cout << "best_criteria " << best_criteria[0] << " " << best_criteria[1] << endl;

						//MatrixXd Xy1_feature = Xy1.leftCols(Xy1.cols()-1);
						//MatrixXd Xy2_feature = Xy2.leftCols(Xy2.cols()-1);

						MatrixXd Xy1_feature(Xy1.rows(), Xy1.cols()-1), Xy2_feature(Xy2.rows(), Xy2.cols()-1);
						for(int k = 0; k < Xy1.cols()-1; k++){
							Xy1_feature.col(k) = Xy1.col(k);
						}
						for(int k = 0; k < Xy2.cols()-1; k++){
							Xy2_feature.col(k) = Xy2.col(k);
						}
						best_sets[0] = Xy1_feature;
						best_sets[1] = y1;
						best_sets[2] = Xy2_feature;
						best_sets[3] = y2;
					}
				}
			}
		}
	}
	//cout << "bulid tree next" << endl;
	if(largest_impurity > min_impurity){
		DecisionNode* true_branch = build_tree(best_sets[0], best_sets[1], current_depth+1);
		DecisionNode* false_branch = build_tree(best_sets[2], best_sets[3], current_depth+1);
		return new DecisionNode(static_cast<int>(best_criteria[0]), best_criteria[1], -1, true_branch, false_branch);
	}
	// we're at leaf ==> determind value
	double leaf_value = leaf_value_calculation(y);
	cout << "leaf value: " << leaf_value << endl;
	return new DecisionNode(leaf_value);
}

int DecisionTree::predict(const VectorXd& X, DecisionNode* r){
	if(r == nullptr){
		r = root;
	}
	if(r->value != -1){
		return r->value;
	}
	//cout << "r->feature:" << r->feature << endl;
	double feature_value = X(r->feature);
	//cout << "nani ..." << endl;
	DecisionNode* branch = r->false_branch;
	if(feature_value >= r->threshold){
		branch = r->true_branch;
	}
	return predict(X, branch);
}

VectorXi DecisionTree::predict(const MatrixXd& X){
	VectorXi ret(X.rows());
	for(int i = 0; i < X.rows(); i++){
		ret(i) = predict(X.row(i), root);
	}
	return ret;
}

double DecisionTree::impurity_calculation(const VectorXd& y, const VectorXd& y1, const VectorXd& y2){
	// calculate information gain
	double prob = y1.size()/double(y2.size());
	double entropy = Utils::calculate_entropy(y);
	double info_gain = entropy - prob*Utils::calculate_entropy(y1) - (1-prob)*Utils::calculate_entropy(y2);
	return info_gain;
}

int DecisionTree::leaf_value_calculation(const VectorXd& y){
	int most_common;
	int max_count = 0;
	unordered_map<int,int> m;
	for(int i = 0; i < y.size(); i++){
		m[y(i)]++;
	}
	for(auto p = m.begin(); p != m.end(); p++){
		if(p->second > max_count){
			most_common = p->first;
			max_count = p->second;
		}
	}
	return most_common;
}

