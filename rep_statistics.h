#ifndef REP_STATISTICS_H_
#define REP_STATISTICS_H_

#include <stdio.h>
#include <math.h>
#include <vector>

#include "SVM-commons.h"

#define FILENAME_LEN 1024
#define TOL_VAR 10E-10

typedef unsigned long int kevals_type;

struct summary_full_comparison{

	double model1_cent; //accuracies
	double model2_cent;
	double model1_dist;
	double model2_dist;
	int type_model1;
	int type_model2;
	
};

class statistic{

private:

std::vector <double>values;

public:

	void add(double value){
		values.push_back(value);
	}
	
	void add(int value){
		double double_val = (double)value; 
		values.push_back(double_val);
	}

	void add(unsigned long int value){
		double double_val = (double)value; 
		values.push_back(double_val);
	}
	
	double get_mean(){
		double mean = 0.0;
		for (int i=0; i<values.size(); i++) {
    		mean += values[i];
		}
		if(values.size() > 0)
			mean = mean / values.size();
		return mean;
	}
	
	double get_sum(){
		double sum = 0.0;
		for (int i=0; i<values.size(); i++) {
    		sum += values[i];
		}
		return sum;
	}

	double get_std(){
		double mean = 0.0;
		double prod_values = 0.0;
		
		for (int i=0; i<values.size(); i++) {
    		mean += values[i];
    		prod_values += values[i]*values[i];
		}
		
		if(values.size() > 0)
			mean = mean / values.size();
	
		double std = 0.0;
		if(values.size() > 1){
			double var = (prod_values/values.size()) - (mean*mean);
			if(var > 0)
				std = sqrt(var);
		}		
		
		return std;
	}
	
	int get_num(){
		return values.size();
	}
	
};


class rep_statistics_cent {

public: 
			
	 rep_statistics_cent(char *prob_name_, svm_parameter *param_, int reps);
	~rep_statistics_cent();
			
	 void add(double current_accuracy, double current_time, kevals_type current_k_evals);
	 void compute_statistics();
	 void write();	
	 int size(){ return allocated; }
	 int filled() {return idx_current+1; }
	 
	double sqrt_and_check_var(double var){
		if ((var < 0.0) && (fabs(var) < TOL_VAR))
			var = fabs(var);	
		return sqrt(var);  
	}	 	 
	
	double meanAccuracy;
	double stdAccuracy;
		 
private:

	int idx_current;
	int allocated;	 
	double *accuracies;
	double *trTimes;
	kevals_type *n_kevals;
	char stats_file[FILENAME_LEN];
	char prob_name[FILENAME_LEN];
	
	svm_parameter *param; 
	
	double meanTime;
	double stdTime;
	double meanN_kevals;
	double stdN_kevals;
	 	
};


class rep_statistics_dist {

public: 
			
	 rep_statistics_dist(char *prob_name_, svm_parameter *param_, int reps);
	~rep_statistics_dist();
	
	 void add(double current_accuracy, struct distrib_statistics *one_sim_stats);
	 void compute_statistics();
	 void write();	
	 int size(){ return allocated; }
	 int filled() {return idx_current+1; }	 
	 
	 double sqrt_and_check_var(double var){
		if ((var < 0.0) && (fabs(var) < TOL_VAR))
			var = fabs(var);	
		return sqrt(var);  
	}
	
	double meanAccuracy;
	double stdAccuracy;
	
private:

	int idx_current;
	int allocated;	
	
	char stats_file[FILENAME_LEN];
	char prob_name[FILENAME_LEN];
	
	svm_parameter *param; 
	
	double *accuracies;
	double *coordTimes;
	double *nodeTimes;
	double *parTimes;
	double *totalTimes;
	double *totalCompressions;
	double *nodeCompressions;
	
	double *coordN_kevals;
	double *nodeN_kevals;
	double *parN_kevals;
	double *totalN_kevals;
	
	double intraVarN_kvals;	
	double intraVarRemoteTimes;
	double intraVarRemoteCompressions;  

	double meanTotalCompression;
	double stdTotalCompression;
	double meanNodeCompression; 
	double stdNodeCompression;
	double meanParTime;
	double meanTotalTime;
	double stdParTime; 
	double stdTotalTime;
	double meanCoordTime;
	double stdCoordTime;
	double meanNodeTime;
	double stdNodeTime;
	double meanSumParallel;
	double meanSumSequential;
	double stdSumParallel;
	double stdSumSequential;
	double meanCoordN_kevals;
	double stdCoordN_kevals;
	double meanSumParallelN_kevals;
	double stdSumParallelN_kevals;
	double meanSumSequentialN_kevals;
	double stdSumSequentialN_kevals;  
	double meanNodeN_kevals;
	double stdNodeN_kevals;
	double meanParN_kevals;
	double meanTotalN_kevals;
	double stdParN_kevals;
	double stdTotalN_kevals;

};



#endif /*REP_STATISTICS_H_*/
