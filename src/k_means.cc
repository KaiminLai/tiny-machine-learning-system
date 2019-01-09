#include <iostream>
#include <random>
#include <vector>
#include <unordered_set>
#include <limits>
#include "k_means.h"
#include "utils.h"
#include <omp.h>

using namespace std;
using namespace Eigen;

KMeans::KMeans(int k, int max_iter) : k(k), max_iter(max_iter) {}

KMeans::~KMeans() {}

VectorXi KMeans::predict(const MatrixXd& X){
	MatrixXd centroids = init_random_centroids(X);
	vector<vector<int>> clusters;
	for(int i = 0; i < max_iter; i++){
		clusters = create_clusters(centroids, X);
		MatrixXd prev_centroids = centroids;
		centroids = calculate_centroids(clusters, X);
		MatrixXd diff = centroids - prev_centroids;
		double diff_sum = diff.array().abs().sum();
		if(diff_sum < 1e-7){
			break;
		}
	}
	return get_cluster_labels(clusters, X);

}

MatrixXd KMeans::init_random_centroids(const MatrixXd& X){
	int size = X.rows();
	uniform_int_distribution<int> u(0, size-1);
	default_random_engine e(0);
	unordered_set<int> index;
	while(index.size() != k){
		index.insert(u(e));
	}
	MatrixXd centroids(k, X.cols());
	int i = 0;
	for(auto p = index.begin(); p != index.endl(); p++){
		centroids.row(i) = X.row(*p);
		i++;
	}
	return centroids;
}

vector<vector<int>> KMeans::create_clusters(const MatrixXd& centroids, const MatrixXd& X){
	vector<vector<int>> clusters(k, vector<int>());
	int size = X.rows();
	for(int i = 0; i < size; i++){
		int centroid_i = closet_centroid(X.row(i), centroids);
		clusters[centroid_i].push_back(i);
	}
	return clusters;
}

int KMeans::closet_centroid(const VectorXd& sample, const MatrixXd& centorids){
	int closet_i = 0;
	double closet_dist = numeric_limits<double>::max();
	for(int i = 0; i < centroids.size(); i++){
		double distance = Utils::euclidean_distance(sample, centorids.row(i));
		if(distance < closet_dist){
			closet_dist = distance;
			closet_i = i;
		}
	}
	return closet_i;
}

MatrixXd KMeans::calculate_centroids(vector<vector<int>>& clusters, const MatrixXd& X){
	MatrixXd centroids(k, X.cols());
	for(int i = 0; i < clusters.size(); i++){
		VectorXd centroid = VectorXd::Zero(X.cols());
		for(int j = 0; j < clusters[i].size(); j++){
			centroid += VectorXd(X.row(clusters[i][j]));
		}
		centroid /= clusters[i].size();
		centroids.row(i) = centroid;
	}
	return centroids;
}

VectorXi KMeans::get_cluster_labels(vector<vector<int>>& clusters, const MatrixXd& X){
	VectorXi ret(X.rows());
	for(int i = 0; i < clusters.size(); i++){
		for(auto j : clusters[i]){
			ret(j) = i;
		}
	}
	return ret;
}