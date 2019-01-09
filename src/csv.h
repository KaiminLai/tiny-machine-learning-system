/*--------------------------------------------------
# CSVReader is for data loading before applying machine learning algorithms with C++.
# so we assume that the data format is: rows*cols
# use CSVReader to convert csv file into eigen matrix
# and CSVWriter is used to convert eigen matrix into csv file
---------------------------------------------------*/

#ifndef __CSV_H__
#define __CSV_H__

#include<fstream>
#include<string>
#include<array>
#include<sstream>
#include<eigen3/Eigen/Dense>

namespace csv{
	typedef std::array<int,2> matrix_size;

	class CSVReader
	{
	private:
		std::ifstream csv_ifs;
		Eigen::MatrixXd mat;
		int mat_rows, mat_cols;
		char sep;
		bool header;
		Eigen::MatrixXd csv2mat();

	public:
		CSVReader(std::string filename, char sep=',', bool header=false);
		~CSVReader();
		int rows();
		int cols();
		matrix_size size();
		Eigen::MatrixXd get_mat();
		Eigen::VectorXd get_row(int n_row);
		Eigen::VectorXd get_col(int n_col);
		double get_element(int n_row, int n_col);
	};

	class CSVWriter
	{
	private:
		Eigen::MatrixXd mat;

	public:
		CSVWriter(Eigen::MatrixXd matrix);
		CSVWriter(Eigen::VectorXd vector);
		~CSVWriter();
		void write(std::string filename);
		
	};
}   //namespace csv

#endif