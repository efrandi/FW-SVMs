#ifndef SYNC_PROBLEM_GENERATOR_H_
#define SYNC_PROBLEM_GENERATOR_H_

#ifndef NULL
#define NULL 0
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))	// allocating memory
#define FILENAME_LEN 1024


#include "SVM-commons.h"
#include "partitioner.h"



//implemented problems
enum {RECTANGLES, GAUSSIANS_MIX}; 


class sync_data {

public:

	 sync_data(int problem_id_ = RECTANGLES);
	 ~sync_data();
	
	svm_problem* generate(const int no_items);
	svm_problem* generate(const int *no_items);
	svm_problem* get_problem();
	Partition* get_partition();
	void plot();
	void plot_rectangles();
	void plot_gaussians();
	
private:

	int problem_id;
	bool is_done; 
	svm_problem* prob;
	svm_node *x_space; //here the data is really stored
	Partition *partition; 	
	
	void do_rectangles(int nr1,int nr2,int nr3,int nr4,int nr5,int nr6);
	void do_mix_gaussians(int nr1,int nr2,int nr3,int nr4,int nr5,int nr6,int nr7,int nr8);
};


#endif /*SYNC_PROBLEM_GENERATOR_H_*/
