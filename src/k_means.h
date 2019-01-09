#ifndef __K_MEANS_H__
#define __K_MEANS_H__

#include <eigen3/Eigen/Dense>
#include <vector>
#include "utils.h"

class KMeans{
public:
	KMeans(int k = 2, int max_iter = 1000);
	~KMeans();
	Eigen::VecotrXi predict(const Eigen::MatrixXd& X);

private:
	int k;
	int max_iter;
	// initialize the centroids as k random samples of X
	Eigen::MatrixXd init_random_centroids(const Eigen::MatrixXd& X);
	// assign the samples to the closet centroids to create clusters
	std::vector<std::vector<int>> create_clusters(const Eigen::MatrixXd& centroids, const Eigen::MatrixXd& X);
	// return the index of the closet centroid to the sample
	int closest_centroid(const Eigen::VectorXd& sample, const Eigen::MatrixXd& centroids);
	// calculate new centroids as the means of the samples in each cluster
	Eigen::MatrixXd calculate_centroids(std::vector<std::vector<int>>& clusters, const Eigen::MatrixXd& X);
	Eigen::VecotrXi get_cluster_labels(std::vector<std::vector<int>>& clusters, const Eigen::MatrixXd& X);

};

#endif