#obj =  logistic_regression.o lr_example.o utils.o csv.o
# 用了引用后logistic_regression.o 要在 lr_example.o前面
#lr_test: $(obj)
#	g++ -fopenmp -o lr_test $(obj)
#lr_example.o: ./examples/lr_example.cc ./src/utils.h ./src/csv.h
#	g++ -fopenmp -c ./examples/lr_example.cc
#logistic_regression.o: ./src/logistic_regression.cc ./src/logistic_regression.h ./src/utils.h
#	g++ -fopenmp -c ./src/logistic_regression.cc
#utils.o: ./src/utils.cc ./src/utils.h
#	g++ -fopenmp -c ./src/utils.cc
#csv.o: ./src/csv.cc ./src/csv.h
#	g++ -fopenmp -c ./src/csv.cc
#clean:
#	rm lr_test lr_example.o logistic_regression.o utils.o csv.o

#obj = k_nearest_neighbors.o knn_example.o utils.o csv.o

#knn_test: $(obj)
#	g++  -fopenmp -o knn_test $(obj)
#knn_example.o: ./examples/knn_example.cc ./src/utils.h ./src/csv.h
#	g++  -fopenmp -c ./examples/knn_example.cc
#k_nearest_neighbors.o: ./src/k_nearest_neighbors.cc ./src/k_nearest_neighbors.h ./src/utils.h
#	g++  -fopenmp -c ./src/k_nearest_neighbors.cc
#utils.o: ./src/utils.cc ./src/utils.h
#	g++  -fopenmp -c ./src/utils.cc
#csv.o: ./src/csv.cc ./src/csv.h
#	g++  -fopenmp -c ./src/csv.cc
#clean:
#	rm knn_test knn_example.o k_nearest_neighbors.o utils.o csv.o


#obj =  naive_bayes.o utils.o csv.o nb_example.o

#nb_test: $(obj)
#	g++ -fopenmp -o nb_test $(obj)
#nb_example.o: ./examples/nb_example.cc ./src/utils.h ./src/csv.h
#	g++ -fopenmp -c ./examples/nb_example.cc
#naive_bayes.o: ./src/naive_bayes.cc ./src/naive_bayes.h ./src/utils.h
#	g++ -fopenmp -c ./src/naive_bayes.cc
#utils.o: ./src/utils.cc ./src/utils.h
#	g++ -fopenmp -c ./src/utils.cc
#csv.o: ./src/csv.cc ./src/csv.h
#	g++ -fopenmp -c ./src/csv.cc
#clean:
#	rm nb_test nb_example.o naive_bayes.o utils.o csv.o

#obj =  perceptron.o utils.o csv.o perceptron_example.o

#pp_test: $(obj)
#	g++  -fopenmp -o pp_test $(obj)
#perceptron_example.o: ./examples/perceptron_example.cc ./src/utils.h ./src/csv.h
#	g++  -fopenmp -c ./examples/perceptron_example.cc
#perceptron.o: ./src/perceptron.cc ./src/perceptron.h ./src/utils.h
#	g++  -fopenmp -c ./src/perceptron.cc
#utils.o: ./src/utils.cc ./src/utils.h
#	g++  -fopenmp -c ./src/utils.cc
#csv.o: ./src/csv.cc ./src/csv.h
#	g++  -fopenmp -c ./src/csv.cc
#clean:
#	rm pp_test perceptron_example.o perceptron.o utils.o csv.o


#obj =  decision_tree.o decision_tree_example.o utils.o csv.o

#dt_test: $(obj)
#	g++ -fopenmp -o dt_test $(obj)
#decision_tree_example.o: ./examples/decision_tree_example.cc ./src/utils.h ./src/csv.h
#	g++ -fopenmp -c ./examples/decision_tree_example.cc
#decision_tree.o: ./src/decision_tree.cc ./src/decision_tree.h ./src/utils.h
#	g++ -fopenmp -c ./src/decision_tree.cc
#utils.o: ./src/utils.cc ./src/utils.h
#	g++ -fopenmp -c ./src/utils.cc
#csv.o: ./src/csv.cc ./src/csv.h
#	g++ -fopenmp -c ./src/csv.cc
#clean:
#	rm dt_test decision_tree_example.o decision_tree.o utils.o csv.o

#obj =  support_vector_machine.o svm_example.o utils.o csv.o

#svm_test: $(obj)
#	g++ -fopenmp -o svm_test $(obj)
#svm_example.o: ./examples/svm_example.cc ./src/utils.h ./src/csv.h
#	g++ -fopenmp -c ./examples/svm_example.cc
#support_vector_machine.o: ./src/support_vector_machine.cc ./src/support_vector_machine.h ./src/utils.h
#	g++ -fopenmp -c ./src/support_vector_machine.cc
#utils.o: ./src/utils.cc ./src/utils.h
#	g++ -fopenmp -c ./src/utils.cc
#csv.o: ./src/csv.cc ./src/csv.h
#	g++ -fopenmp -c ./src/csv.cc
#clean:
#	rm svm_test svm_example.o support_vector_machine.o utils.o csv.o


#obj =  linear_discriminant_analysis.o ld_example.o utils.o csv.o

#lda_test: $(obj)
#	g++ -fopenmp -o lda_test $(obj)
#ld_example.o: ./examples/ld_example.cc ./src/utils.h ./src/csv.h
#	g++ -fopenmp -c ./examples/ld_example.cc
#linear_discriminant_analysis.o: ./src/linear_discriminant_analysis.cc ./src/linear_discriminant_analysis.h ./src/utils.h
#	g++ -fopenmp -c ./src/linear_discriminant_analysis.cc
#utils.o: ./src/utils.cc ./src/utils.h
#	g++ -fopenmp -c ./src/utils.cc
#csv.o: ./src/csv.cc ./src/csv.h
#	g++ -fopenmp -c ./src/csv.cc
#clean:
#	rm lda_test ld_example.o linear_discriminant_analysis.o utils.o csv.o