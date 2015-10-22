#ifndef PARTITIONER_H_
#define PARTITIONER_H_

#ifndef NULL
#define NULL 0
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))	// allocating memory

#include "SVM-commons.h"

// Implemented Partition Methods
enum { RANDOM, RANDOM_WEIGHTED, KMEANS, PREPART}; 

// Used to group the indexes by class
struct grouped_indexes{

int total;
int nr_classes;
int *class_counts;
int *starts;
int *idxs;
};

// Class 'Partition': Checked 21-Sept-2009
// Computes and handles a partition of a dataset for LIBSVM  
// The partition is stored as a 2-dim array with the indexes of the items 
// Public access: mainly through a 'get-next-element-from-part' function

class Partition{

private:
	
	int n; //number of parts
	int** indexes; //indexes[i][j] index of the j-th element of the i-th part
	int* sizes; //size of each part
	int* currents;//array to store the last element recovered from each part
	double kmeans_tolerance;
	int range_rand01;
	int pmode;
	bool ensure_minimum_by_class;
	int minimum_per_class;
	//interface to compute the partition with different options
	bool do_partition(int mode, const svm_problem *prob, int* perm, int* ns, float* weights,bool group_classes_in_kmeans);
	
	//computes a random assignment
	//can also receive the permutation assigment and the number of elements by part    	
	bool do_random_assignment(int l, int* assignments, int* counts);
	//computes a weighted random assignment
	bool do_weighted_rand_assignment(int l, int* assignments, int*counts, float* weights);
	
	//computes an assignment based on similarity using kmeans clustering
	//requires to know the dimensionality of the items 
	bool do_kmeans_assignment(const svm_problem *prob, int* assignments, int* counts, bool separate_classes, grouped_indexes* items);
	void group_items(const svm_problem *prob, grouped_indexes *items);
	bool kmeans(svm_node** const x, int l,  int k, int* assignments, int* counts, int dim);
	
	//used at each iteration of kmeans to update the centers
	bool update_means(double **centers, struct svm_node **x, int *assignments, int *counts,int l, int dim, int k);
	
	//compute the distance of a point to a kmeans center  
	double compute_distance(const svm_node *point, const double *center, int dim);
	double compute_dot(const svm_node *point, const double *center, int dim);
	double compute_norm_item(const svm_node *point, int dim);
	
	//set default parameters
    void set_default_params();
    void set_members(int mode, int n_frags, bool do_minimal_balancing, int minimum);
    void destroy_grouped_items(grouped_indexes *git);
    
public:

    //constructors: 
    //1. RANDOM OR KMEANS PARTITION, SPECIFIED IN MODE
    //2. PRECOMPUTED PARTITION
    //3. WEIGHTED RANDOM PARTITION
	
	Partition(int n_frags, int mode, const svm_problem *prob, bool group_classes, bool do_minimal_balancing, int minimum); 
	Partition(int n_frags, const svm_problem *prob, int* perm, int* ns);
	Partition(int n_frags, const svm_problem *prob, float* weights, bool do_minimal_balancing, int minimum);
	Partition(int n_frags, int mode, const svm_problem *prob, int* perm, int* ns, float* weights, bool group_classes, bool do_minimal_balancing, int minimum);
   ~Partition();
    
	
 	//for a given part, returns the index of the item following the last item used   	
	int get_next(int part);
	//for a given part, resets the "last-used-pointer" 
	int rewind(int part);
	//reset all the "last-used-pointers"
	void rewind();
	//returns the size of a given part
	int get_size(int part);
	//returns the number of parts
	int get_nparts();
	//returns a pointer to the indexes of a given part
	int* get_indexes(int part);
	
	void set_kmeans_tolerance(double);
};


#endif /*PARTITIONER_H_*/
