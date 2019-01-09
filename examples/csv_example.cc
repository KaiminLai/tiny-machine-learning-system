#include<iostream>
#include "../src/csv.h"
using namespace std;
using namespace Eigen;
using namespace csv;

int main(){
	CSVReader csv("a.csv");
	cout<<"rows:"<<csv.rows()<<endl;
	cout<<"columns:"<<csv.cols()<<endl;
	cout<<"size:"<<csv.size()[0]<<"*"<<csv.size()[1]<<endl;
	cout<<"element mat(0,1):"<<csv.get_element(0,1)<<endl;
	VectorXd r1 = csv.get_row(1);
	VectorXd c1 = csv.get_col(1);
	MatrixXd mat = csv.get_mat();
	cout<<"row1:\n";
	cout<<r1[0]<<" "<<r1[1]<<" "<<r1[2]<<endl;
	cout<<"column1:\n";
	cout<<c1[0]<<" "<<c1[1]<<endl;
	cout<<"mat(0,0):"<<mat(0,0)<<endl;
	CSVWriter csv1(mat);
	csv1.write("b.csv");
	CSVWriter csv2(c1);
	csv2.write("c.csv");
	CSVWriter csv3(r1);
	csv3.write("d.csv");
	return 0;
}