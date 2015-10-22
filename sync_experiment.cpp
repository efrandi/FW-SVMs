#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <limits.h>
#include <assert.h>
#include <iomanip>
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

#include "SVM-commons.h"
#include "svm.h"
#include "MEB-based-SVMs.h"
#include "partitioner.h"
#include "sync_problem_generator.h"
#include "rep_statistics.h"

#define FILENAME_LEN 1024

void exit_with_help()
{
	printf(
	"Usage: sync_experiment [options] exp_name \n"
	"VERSION 29.12.09. Options:\n"
	"exp_name: RECTANGLES, MIX_GAUSSIANS\n"
	"-E : experiment-type (default 0)\n"
	"   0 -- save and plot simulated data\n"
	"   1 -- statistics for a centralized model\n"
	"   2 -- statistics for a distributed model\n"
	"   3 -- grid search for hyper-parameters\n"
	"   4 -- comparison of L1 and L2 models in the centralized case\n"
	"   5 -- comparison of L1 and L2 models in the distributed case\n"
	"   6 -- full comparison of L1 and L2 models on a predefined grid\n"
	"-MP : multiple params, used within experiments type 4 and 5.\n"
	"      if not specified the same parameters are used for both models except that the first is considered a L1-OVO and the second L2-OVO.\n"
	"      if specified, all the model parameters (not-algorithmic) need to be declared twice.\n"
	"      for example: -c 100 1000\n"
	"      algorithmic parameters like eps, cache_size, shrinking are common and need to be declared only once.\n"
	"-s : SVM model: (default 0)\n"
	"   0 -- one-versus-one of L2-SVMs\n"
	"   1 -- direct L2-SVM implementation\n"
	"   2 -- one-versus-one of L1-SVMs\n"
	"-NTR n : number of training examples to generate (n)\n"
	"-NTS n : number of testing examples to generate (n)\n"
	"-NS n : number of different realizations (n) of the data\n"
	"-VC n1 n2 n3 : defines the grid to search a value of C, n1=base, n2=min-power, n2=max-power\n"
	"               default: base=2, min-power=-5, max-power=+5\n" 
	"-VG n1 n2 n3 : defines the grid to search a value of G, n1=base, n2=min-power, n2=max-power\n"
	"               default: only the estimation provided by the average-distances heuristic\n"
	"-g gamma: set gamma in kernel function (default -1, which sets 1/averaged distance between patterns)\n"
	"-c value: set the regularization parameter C (default 100)\n"
	"-e epsilon: set tolerance of termination criterion for MEBs\n"
	"            (default eps=-1 which sets eps according to the bound |f(x)-f(x)^*|)\n"
	"-y : type of codes for MCVM (default 2)\n"
	"   0 -- AD-SVM codes\n"
	"   1 -- Asharaf codes\n"
	"   2 -- Normalized AD-SVM codes\n"
	"   3 -- ANN codes\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
    "	5 -- laplacian: exp(-sqrt(gamma)*|u-v|)\n"
    "	6 -- normalized poly: ((gamma*u'*v+coef0)/sqrt((gamma*u'*u+coef0)*(gamma*v'*v+coef0)))^degree\n"
	"	7 -- inverse distance: 1/(sqrt(gamma)*|u-v|+1)\n"
	"	8 -- inverse square distance: 1/(gamma*|u-v|^2+1)\n"
	"-d degree: set degree in kernel function (default 3)\n"
	"-g gamma: set gamma in kernel function (default -1, which sets 1/averaged distance between patterns)\n"
	"-r coef0: set coef0 in kernel function (default 0)\n"	
	"-m cachesize: set cache memory size in MB (default 200)\n"
	"-f max #CVs : MAX number of Core Vectors. For OVO (default 50000)\n"
	"-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-a size: sample size for probabilistic sampling (default 60)\n"
	"-iss size: initial sample size for the coreset (default 0)\n"
	);
	exit(1);
}

/////////////// GLOBAL VARIABLES /////////////////////////////
char exp_name[FILENAME_LEN];
char path[FILENAME_LEN];

int exp_id;
int no_tr_examples, no_ts_examples;
int no_repetitions;
float searchC, searchG, searchE; 
int minpowC, maxpowC, baseC;
int minpowG, maxpowG, baseG;
int minpowE, maxpowE, baseE;

struct svm_parameter* param;
struct svm_parameter* second_param_set;
int max_index;
bool multiple_params, no_file_but_plot;
bool run_centralized_when_comparing_distributed;

enum {SAVE_AND_PLOT, STATS_CENTRALIZED, STATS_DISTRIBUTED, GRID_SEARCH, COMPARATIVE_CENTRALIZED, COMPARATIVE_DISTRIBUTED, FULL_ON_GRID}; 

///////////////// FUNCTIONS INTERFACES /////////////////////////////
void parse_command_line(int argc, char **argv);
void repetitions_centralized(char* name, int reps);
void repetitions_distributed(char* name, int reps);
void comparison_centralized(char* name, int reps);
void comparison_distributed(char* name, int reps);
void full_comparison(char* name, int reps, summary_full_comparison* summary);
void full_comparison_on_grid(char* name, int reps);
void grid_search(char* name);
void check_parameters(svm_problem *prob, svm_parameter *param);
void save_and_plot(char* name);
double CalRBFWidth(svm_problem *prob);

double test_model(svm_problem *test_problem, svm_model* model);

svm_model* svm_simulate_distributed(const svm_problem *prob, const svm_parameter *param, Partition *partition, distrib_statistics *stats);

void determine_labels(svm_problem *prob, int *nr_class_ret, int **label_ret);

double sqrt_and_check_var(double var){
		if ((var < 0.0) && (fabs(var) < TOL_VAR))
			var = fabs(var);	
		return sqrt(var);  
}	

void load_default_params(svm_parameter* parameter_set);

/////////////////////////// MAIN ////////////////////////////////////////

int main(int argc, char **argv) {
	
	parse_command_line(argc, argv);	
	
	switch(param->exp_type){
		
		case SAVE_AND_PLOT:
			printf("EXP TYPE: simulate, save to file and plot\n");
			save_and_plot(exp_name);
			break;
			
		case STATS_CENTRALIZED: 
			printf("EXP TYPE: statistics for the centralized routine\n");
			repetitions_centralized(exp_name,no_repetitions);
			break;
		
		case STATS_DISTRIBUTED: 
			printf("EXP TYPE: statistics for the distributed routine\n");
			repetitions_distributed(exp_name,no_repetitions);
			break;
	
		case GRID_SEARCH:
			printf("EXP TYPE: grid search for hyper-parameters\n");
			grid_search(exp_name);
			break;
		
		case COMPARATIVE_CENTRALIZED: 
			printf("EXP TYPE: compare statistics L1 vs L2 (centralized training)\n");
			comparison_centralized(exp_name,no_repetitions);
			break;
	
		case COMPARATIVE_DISTRIBUTED:
			printf("EXP TYPE: compare statistics L1 vs L2 simulating distributed\n");
			if(run_centralized_when_comparing_distributed){
				summary_full_comparison* summary = new summary_full_comparison;
				full_comparison(exp_name, no_repetitions, summary);
				printf("\n*************** RESULTS ********************\n");
				printf("Model1 Centralized (type %d): %g",summary->type_model1, summary->model1_cent);
				printf("Model1 Distributed (type %d): %g",summary->type_model1, summary->model1_dist);
				printf("Model2 Centralized (type %d): %g",summary->type_model2, summary->model2_cent);
				printf("Model2 Distributed (type %d): %g",summary->type_model2, summary->model2_dist);
				printf("\n*********************************************\n");
				
			} else {
			
				comparison_distributed(exp_name, no_repetitions);
			}
			break;
		case FULL_ON_GRID: 
			printf("EXP TYPE: full comparison L1 vs L2 on predefined grid\n");
			full_comparison_on_grid(exp_name,no_repetitions);
			break;
		
	}
}

/////////////////////////// STATISTICS CENTRALIZED ////////////////////////////

void repetitions_centralized(char* name, int reps){

	struct svm_model *model;
	rep_statistics_cent *rep_statistics = new rep_statistics_cent(name,param,reps);
	
	sync_data *sync_dataset, *test_dataset;
	svm_problem *prob, *test_prob;
		
	for(int repIt=0; repIt < reps; repIt++){
		
		sync_dataset = new sync_data(exp_id);
		sync_dataset->generate(no_tr_examples);
		prob = sync_dataset->get_problem();
		  	
		printf("SYNC-GEN: syntehtic problem %s generated ... \n",name);
		printf("***eps is %g\n",param->eps);	
		check_parameters(prob,param);
		const char *error_msg;
		error_msg = svm_check_parameter(prob,param);	
		if(error_msg){
			fprintf(stderr,"Error: %s\n",error_msg);
			exit(1);
		}
		printf ("PARARMS_CHECK: finish checking parameters ...\n");
	
		////////////// Train a Model ////////////////////
		Kernel::reset_real_kevals();
		
			model = svm_train(prob,param);
	
		kevals_type n_kevals = Kernel::get_real_kevals();
		//////////////////////////////////////////////////
		
		test_dataset = new sync_data(exp_id);
		test_dataset->generate(no_ts_examples);
		test_prob = test_dataset->get_problem();
		
		double currentAccuracy = test_model(test_prob, model);
		rep_statistics->add(currentAccuracy,model->trainingTime,n_kevals);
		
		svm_destroy_model(model);
		delete sync_dataset;
		delete test_dataset;
		
	}//end iterations centralized case
	
	// now, compute statistics
	rep_statistics->compute_statistics();
	rep_statistics->write();
	
	delete rep_statistics;
	svm_destroy_param(param);
			
}//end repetitions centralized case routine


/////////////////////////// STATISTICS DISTRIBUTED ////////////////////////////

void repetitions_distributed(char* name, int reps){

	struct svm_model *model;
		
	rep_statistics_dist *rep_statistics = new rep_statistics_dist(name,param,reps);
		
	int counterReps = 0;
	sync_data *sync_dataset, *test_dataset;
	svm_problem *prob, *test_prob;
	Partition *partition;
		
	for(int repIt=0; repIt < reps; repIt++){
		
		double currentAccuracy;
		struct distrib_statistics statsDist;
		
		sync_dataset = new sync_data(exp_id);	
		sync_dataset->generate(no_tr_examples);
		prob = sync_dataset->get_problem();
		partition = sync_dataset->get_partition();
		partition->rewind();
		
		printf("SYNC-GEN: syntehtic problem %s generated ... \n",name);
	
		check_parameters(prob,param);
		const char *error_msg;
		error_msg = svm_check_parameter(prob,param);	
		if(error_msg){
			fprintf(stderr,"Error: %s\n",error_msg);
			exit(1);
		}
		printf ("PARARMS_CHECK: finish checking parameters ...\n");
	
		////////////// Train a Model ////////////////////	
			
			model = svm_simulate_distributed(prob,param,partition,&statsDist);
		
		//////////////////////////////////////////////////
		
		
		test_dataset = new sync_data(exp_id);
		test_dataset->generate(no_ts_examples);
		test_prob = test_dataset->get_problem();
		
		currentAccuracy = test_model(test_prob, model);
		rep_statistics->add(currentAccuracy,&statsDist);
		
		printf("STATS_DIST: end dist simulation: %d\n", repIt);
		
		//destroy dynamically allocated structures
		svm_destroy_model(model);
		
		delete sync_dataset;
		delete test_dataset;
		
		counterReps++;
		
	}//end iterations distributed case
	
	printf("STATS_DIST: end\n");

	// now, compute statistics
	rep_statistics->compute_statistics();
	rep_statistics->write();
	
	//destroy dynamically allocated structures
	delete rep_statistics;
	svm_destroy_param(param);	
	
						
}//end repetitions distributed case routine

/////////////////////////////////// SIMULATION OF THE DISTRIBUTED CASE /////////////////////////////////////////////////////////////////
svm_model* svm_simulate_distributed(const svm_problem *prob, const svm_parameter *param, Partition *partition, distrib_statistics *stats){

	int nnodes = partition->get_nparts();
	svm_model *dist_model;
	//iterate among the nodes
	
	svm_model **remote_models = Malloc(struct svm_model*,nnodes);
	unsigned long int *n_kevals = Malloc(unsigned long int,nnodes);
	printf("DS: ITERATING AMONG THE REMOTE NODES\n");
	for(int k=0;k<nnodes;k++){
	
	svm_problem subprob; 
	subprob.l = partition->get_size(k);
	printf("Requesting Remote-Node: %d, Sub-problem Size: %d examples \n",k+1,subprob.l);
	subprob.x = Malloc(struct svm_node*,subprob.l);
	subprob.y = Malloc(double,subprob.l);
		
	for(int i=0;i<subprob.l;i++)
		{
				int idx = partition->get_next(k);
				subprob.x[i] = prob->x[idx];
			    subprob.y[i] = prob->y[idx];		
		}
	 
	////////////// Train a Remote-Model ////////////////////
	
	Kernel::reset_real_kevals();
		
		remote_models[k] = svm_train(&subprob,param);
	
	n_kevals[k] = Kernel::get_real_kevals();
	
	//////////////////////////////////////////////////

	printf("Model obtained from Remote-Node: %d\n",k+1); 
	
	//destroy local problem
	free(subprob.x);
	free(subprob.y);
	}//end iteration among nodes	
	
	int size_core_coordinator = 0;
	//count the number of core-points and allocate
	for(int k=0;k<nnodes;k++){
		svm_model *submodel = remote_models[k];
		size_core_coordinator += submodel->l; 	
	}
	
	printf("Creating the training set for the Coordinator\n");
	struct svm_problem problem_coordinator;
	problem_coordinator.l = size_core_coordinator;
	problem_coordinator.x = Malloc(struct svm_node*,problem_coordinator.l);
	problem_coordinator.y = Malloc(double,problem_coordinator.l);
	int count = 0;	
	double meanRemoteTime = 0;
	double maxRemoteTime = 0;
	double sumRemoteTime = 0;
	double stdRemoteTime = 0;
	double meanNodeCompression = 0;
	double stdNodeCompression = 0;
	
	stats->mean_remote_N_kevals = 0;
	stats->std_remote_N_kevals = 0;
	stats->max_remote_N_kevals = 0;
	stats->sum_remote_N_kevals = 0; 
	
	//join the corePoints and re-train
	for(int k=0;k<nnodes;k++){
		svm_model *submodel = remote_models[k];
		meanRemoteTime = meanRemoteTime + submodel->trainingTime; 
		maxRemoteTime = max(maxRemoteTime,submodel->trainingTime);
		stdRemoteTime = stdRemoteTime + ((submodel->trainingTime)*(submodel->trainingTime));
		double actualcomp = (double) ((double) submodel->l/ (double) partition->get_size(k));
		meanNodeCompression = meanNodeCompression + actualcomp;
		stdNodeCompression = stdNodeCompression + actualcomp*actualcomp;
		printf("integrating model %d\n",k);
		for(int i=0; i<submodel->l; i++){
			problem_coordinator.y[count] = submodel->ySV[i]; 
			problem_coordinator.x[count] = submodel->SV[i];
			count++;
		}
		
		stats->mean_remote_N_kevals += (double)n_kevals[k];
		stats->std_remote_N_kevals += ((double)n_kevals[k]*(double)n_kevals[k]);
		stats->max_remote_N_kevals += max((double)n_kevals[k],stats->std_remote_N_kevals);
		stats->sum_remote_N_kevals += (double)n_kevals[k]; 
		
		//destroy local model
		svm_destroy_model(submodel);
	}
	
	sumRemoteTime = meanRemoteTime;
	meanRemoteTime = meanRemoteTime/nnodes;
	
	stdRemoteTime = sqrt_and_check_var((stdRemoteTime/nnodes) - (meanRemoteTime*meanRemoteTime));
	
	meanNodeCompression = meanNodeCompression/nnodes; 
	stdNodeCompression = sqrt_and_check_var((stdNodeCompression/nnodes) - (meanNodeCompression*meanNodeCompression)); 
	
	stats->mean_remote_N_kevals /= nnodes;
	stats->std_remote_N_kevals /= nnodes;
	stats->std_remote_N_kevals -= (stats->mean_remote_N_kevals*stats->mean_remote_N_kevals);
	stats->std_remote_N_kevals = sqrt_and_check_var(stats->std_remote_N_kevals);
	
	free(n_kevals);
		
	///////////////// TRAINING IN THE CENTRALIZED NODE ////////////////////////
	printf("Training with the union of the coresets\n");
	
	clock_t startTime = clock();
	Kernel::reset_real_kevals();
		
		dist_model = svm_train(&problem_coordinator,param);
	
	stats->N_kevals = (double)Kernel::get_real_kevals();
	clock_t endTime = clock();
	

	if (dist_model == NULL){
		printf("error: modelo NULL despues de llamada a svm_train\n");
		exit(1);
	}
	
	/////////////////////////////////////////////////////////////////////////////
	
	printf("Distributed solution finished ... \n");
	double totalTime = (double)(endTime - startTime)/CLOCKS_PER_SEC;
	info("Total CPU-TIME = %g seconds\n", totalTime);
	info("Total KERNEL-CALLS = %g calls\n", stats->sum_remote_N_kevals);
	info("Total TOTAL COMPRESSION = %g \n", (double) problem_coordinator.l/prob->l);
	info("Total SIZE TR COORDINATOR = %d \n", problem_coordinator.l);
	info("Total NODE COMPRESSION (MEAN) = %g (~ %g)\n", meanNodeCompression,2*stdNodeCompression);
	
	stats->trainingTime = totalTime;
	stats->meanRemoteTime = meanRemoteTime;
	stats->stdRemoteTime = stdRemoteTime;
	stats->meanNodeCompression = meanNodeCompression;
	stats->stdNodeCompression = stdNodeCompression;
	stats->totalCompression = (double) ((double) problem_coordinator.l)/( (double) prob->l);
	stats->maxRemoteTime = maxRemoteTime;
	stats->sumRemoteTime = sumRemoteTime;
	
	//destroy coordinator problem
	free(problem_coordinator.x);
	free(problem_coordinator.y);
		
	return dist_model;	
}//end simulate distributed training

double test_model(svm_problem* test_problem, svm_model* model){
	
	printf("TEST: generating test instances\n");
	
	if(test_problem->l <= 0){
		printf("TEST: error generating test instances\n");
		exit(1);
	} 
		
	int correct = 0;
	for(int i=0; i < test_problem->l; i++){
	
		svm_node* x_test = test_problem->x[i];
		double y_test = test_problem->y[i];
		if((i%100 == 0) && (i > 1))
			printf("TEST: calling SVM-PREDICT\n");
		double y_predicted = svm_predict(model,x_test);
		
		if(y_test == y_predicted)	
			++correct;	
		
			
	}
		
	double accuracy = (double)correct/(double)test_problem->l; 
	printf("TEST: model tested with result %g %%\n",accuracy*100.0);
		
	return accuracy*100.0;
}



/////////////////////////// GRID SEARCH FOR HYPER-PARAMETERS /////////////////////////////////////////////



void grid_search(char* name){

	printf("GRID-SEARCH: defying range\n");
		
	int nC,nG;
	double *cList;
	double *gList;
	
	
	if(searchC){
		nC = maxpowC - minpowC + 1; 			
		cList = Malloc(double,nC);
		for (int i=0; i < nC; i++){
			cList[i]=pow(baseC,minpowC + i);
		}	
	} else {
		nC = 1;
		cList = Malloc(double,nC);
		cList[0] = 100;
	}
	
	if(searchG){
		nG = maxpowG - minpowG + 1 + 1;
		gList = Malloc(double,nG);
		for (int i=0; i < nG-1; i++){
			gList[i+1]=pow(baseG,minpowG + i);
		}
	} else {
		nG = 1;
		gList = Malloc(double,nG);
		//In this case only the default value of gamme is used
		//this will be estimated after reading the problem
	}
		
}




/////////////////////////// MODELS COMPARISON CENTRALIZED ////////////////////////////

void comparison_centralized(char* name, int reps){

	struct svm_model *model1, *model2;
	svm_parameter *param_set1, *param_set2;
	
	param_set1 = param;
		
	if(multiple_params){
		param_set2 = second_param_set;
	} else {
		
		param_set1->svm_type = CVM;
		param_set2 = new svm_parameter;
	    *param_set2 = *param;
		param_set2->svm_type = C_SVC;
	}
	
	rep_statistics_cent *rep_statistics_model1 = new rep_statistics_cent(name,param_set1,reps);
	rep_statistics_cent *rep_statistics_model2 = new rep_statistics_cent(name,param_set2,reps);
	
	double current_accuracy_model1, current_accuracy_model2;
	kevals_type n_kevals_model1, n_kevals_model2;
	
	sync_data *sync_dataset, *test_dataset;
	svm_problem *prob, *test_prob;
	
	printf("COMP-CENT-MODELS: begining ... \n");
		
	for(int repIt=0; repIt < reps; repIt++){
		
		sync_dataset = new sync_data(exp_id);
		sync_dataset->generate(no_tr_examples);
		prob = sync_dataset->get_problem();
		  	
		printf("SYNC-GEN: syntehtic problem %s generated ... \n",name);
	
		check_parameters(prob,param_set1);
		check_parameters(prob,param_set2);
		
		const char *error_msg1,*error_msg2;
		error_msg1 = svm_check_parameter(prob,param_set1);	
		error_msg2 = svm_check_parameter(prob,param_set2);	
		if(error_msg1 || error_msg2){
			fprintf(stderr,"Error checking parameters, message1: %s; message2: %s\n",error_msg1,error_msg2);
			exit(1);
		}
		printf ("PARARMS_CHECK: finish checking parameters ...\n");
	
		////////////// Train Model # 1 ////////////////////
		Kernel::reset_real_kevals();
		
			model1 = svm_train(prob,param_set1);
	
		n_kevals_model1 = Kernel::get_real_kevals();
		//////////////////////////////////////////////////
		
		////////////// Train Model # 2 ////////////////////
		Kernel::reset_real_kevals();
		
			model2 = svm_train(prob,param_set2);
	
		n_kevals_model2 = Kernel::get_real_kevals();
		//////////////////////////////////////////////////
	
		test_dataset = new sync_data(exp_id);
		test_dataset->generate(no_ts_examples);
		test_prob = test_dataset->get_problem();
		
		current_accuracy_model1 = test_model(test_prob, model1);
		rep_statistics_model1->add(current_accuracy_model1,model1->trainingTime,n_kevals_model1);
		svm_destroy_model(model1);
		
		current_accuracy_model2 = test_model(test_prob, model2);
		rep_statistics_model2->add(current_accuracy_model2,model2->trainingTime,n_kevals_model2);
		svm_destroy_model(model2);
		
		delete sync_dataset;	
		delete test_dataset;	
		
	}//end iterations centralized case
	
	// now, compute statistics
	rep_statistics_model1->compute_statistics();
	rep_statistics_model1->write();
	rep_statistics_model2->compute_statistics();
	rep_statistics_model2->write();
		
	delete rep_statistics_model1;
	delete rep_statistics_model2;
	
	svm_destroy_param(param_set1);
	svm_destroy_param(param_set2);
	
			
}//end comparative centralized




/////////////////////////// MODELS COMPARISON DISTRIBUTED ////////////////////////////

void comparison_distributed(char* name, int reps){

	struct svm_model *model1, *model2;
	svm_parameter *param_set1, *param_set2;
	
	param_set1 = param;
		
	if(multiple_params){
		param_set2 = second_param_set;
	} else {
		param_set1->svm_type = CVM;
		param_set2 = new svm_parameter;
	    *param_set2 = *param;
		param_set2->svm_type = C_SVC;
	}
		
	rep_statistics_dist *rep_statistics_model1 = new rep_statistics_dist(name,param_set1,reps);
	rep_statistics_dist *rep_statistics_model2 = new rep_statistics_dist(name,param_set2,reps);
		
	int counterReps = 0;
	sync_data *sync_dataset, *test_dataset;
	svm_problem *prob, *test_prob;
	Partition *partition;
		
	for(int repIt=0; repIt < reps; repIt++){
		
		double current_accuracy_model1, current_accuracy_model2;
		struct distrib_statistics stats_one_sim_model1, stats_one_sim_model2;
		
		sync_dataset = new sync_data(exp_id);	
		sync_dataset->generate(no_tr_examples);
		prob = sync_dataset->get_problem();
		partition = sync_dataset->get_partition();
		
		
		printf("SYNC-GEN: syntehtic problem %s generated ... \n",name);
	
		check_parameters(prob,param_set1);
		check_parameters(prob,param_set2);
		const char *error_msg1,*error_msg2;
		error_msg1 = svm_check_parameter(prob,param_set1);
		error_msg2 = svm_check_parameter(prob,param_set2);
			
		if(error_msg1 || error_msg2){
			fprintf(stderr,"Error checking parameters, message1: %s, message2: %s\n",error_msg1,error_msg2);
			exit(1);
		}
		printf ("PARARMS_CHECK: finish checking parameters ...\n");
		
		partition->rewind();
		////////////// Train Model # 1 ////////////////////	
			
			model1 = svm_simulate_distributed(prob,param_set1,partition,&stats_one_sim_model1);
		
		//////////////////////////////////////////////////
		printf("COMP_DIST: end dist simulation MODEL #1: rep: %d, model-size: %d\n", repIt, model1->l);
		
		partition->rewind();
		////////////// Train Model # 2 ////////////////////	
			
			model2 = svm_simulate_distributed(prob,param_set2,partition,&stats_one_sim_model2);
		
		//////////////////////////////////////////////////
		printf("COMP_DIST: end dist simulation MODEL #2, rep: %d, model-size: %d\n", repIt, model2->l);
		
		test_dataset = new sync_data(exp_id);
		test_dataset->generate(no_ts_examples);
		test_prob = test_dataset->get_problem();
		
		current_accuracy_model1 = test_model(test_prob, model1);
		rep_statistics_model1->add(current_accuracy_model1,&stats_one_sim_model1);
		svm_destroy_model(model1);
		
		current_accuracy_model2 = test_model(test_prob, model2);
		rep_statistics_model2->add(current_accuracy_model2,&stats_one_sim_model2);
		svm_destroy_model(model2);
		
		delete test_dataset;
		delete sync_dataset;
		
		counterReps++;
		
	}//end iterations distributed case
	
	printf("COMP_DIST: end\n");

	// now, compute statistics
	rep_statistics_model1->compute_statistics();
	rep_statistics_model1->write();
	rep_statistics_model2->compute_statistics();
	rep_statistics_model2->write();
	
	//destroy dynamically allocated structures
	delete rep_statistics_model1;
	delete rep_statistics_model2;
	
	svm_destroy_param(param_set1);
	svm_destroy_param(param_set2);
						
}//end comparison of models distributed case 


/////////////////////////// FULL COMPARISON ////////////////////////////

void full_comparison(char* name, int reps, summary_full_comparison* summary){

	struct svm_model *model_dist1, *model_dist2;
	struct svm_model *model_cent1, *model_cent2;
	
	svm_parameter *param_set1, *param_set2;
	
	param_set1 = param;
		
	if(multiple_params){
		param_set2 = second_param_set;
	} else {
		param_set1->svm_type = CVM;
		param_set2 = new svm_parameter;
	    *param_set2 = *param;
		param_set2->svm_type = C_SVC;
	}
		
	rep_statistics_dist *rep_statistics_model_dist1 = new rep_statistics_dist(name,param_set1,reps);
	rep_statistics_dist *rep_statistics_model_dist2 = new rep_statistics_dist(name,param_set2,reps);
	rep_statistics_cent *rep_statistics_model_cent1 = new rep_statistics_cent(name,param_set1,reps);
	rep_statistics_cent *rep_statistics_model_cent2 = new rep_statistics_cent(name,param_set2,reps);
		
	int counterReps = 0;
	sync_data *sync_dataset, *test_dataset;
	svm_problem *prob, *test_prob;
	Partition *partition;
		
	for(int repIt=0; repIt < reps; repIt++){
		
		double current_accuracy_model_cent1, current_accuracy_model_cent2;
		kevals_type n_kevals_model_cent1, n_kevals_model_cent2;
		double current_accuracy_model_dist1, current_accuracy_model_dist2;
		struct distrib_statistics stats_one_sim_model1, stats_one_sim_model2;
		
		sync_dataset = new sync_data(exp_id);	
		sync_dataset->generate(no_tr_examples);
		prob = sync_dataset->get_problem();
		partition = sync_dataset->get_partition();
		
		
		printf("SYNC-GEN: syntehtic problem %s generated ... \n",name);
	
		check_parameters(prob,param_set1);
		check_parameters(prob,param_set2);
		const char *error_msg1,*error_msg2;
		error_msg1 = svm_check_parameter(prob,param_set1);
		error_msg2 = svm_check_parameter(prob,param_set2);
			
		if(error_msg1 || error_msg2){
			fprintf(stderr,"Error checking parameters, message1: %s, message2: %s\n",error_msg1,error_msg2);
			exit(1);
		}
		printf ("PARARMS_CHECK: finish checking parameters ...\n");
		
		////////////// Train Model # 1 CENTRALIZED ////////////////////	
		Kernel::reset_real_kevals();
		
			model_cent1 = svm_train(prob,param_set1);
	
		n_kevals_model_cent1 = Kernel::get_real_kevals();
		//////////////////////////////////////////////////
		
		printf("FULL_COMPARISON: end centralized training MODEL #1: rep: %d, model-size: %d\n", repIt, model_cent1->l);
		
		////////////// Train Model # 2 CENTRALIZED ////////////////////	
		Kernel::reset_real_kevals();
		
			model_cent2 = svm_train(prob,param_set2);
	
		n_kevals_model_cent2 = Kernel::get_real_kevals();
		//////////////////////////////////////////////////
		
		printf("FULL_COMPARISON: end centralized training MODEL #2: rep: %d, model-size: %d\n", repIt, model_cent2->l);
		
		partition->rewind();
		////////////// Train Model # 1 DISTRIBUTED ////////////////////	
			
			model_dist1 = svm_simulate_distributed(prob,param_set1,partition,&stats_one_sim_model1);
		
		//////////////////////////////////////////////////
		printf("FULL_COMPARISON: end dist simulation MODEL #1: rep: %d, model-size: %d\n", repIt, model_dist1->l);
		
		partition->rewind();
		////////////// Train Model # 2 DISTRIBUTED ////////////////////	
			
			model_dist2 = svm_simulate_distributed(prob,param_set2,partition,&stats_one_sim_model2);
		
		//////////////////////////////////////////////////
		printf("FULL_COMPARISON: end dist simulation MODEL #2, rep: %d, model-size: %d\n", repIt, model_dist2->l);
		
		test_dataset = new sync_data(exp_id);
		test_dataset->generate(no_ts_examples);
		test_prob = test_dataset->get_problem();
		
		current_accuracy_model_dist1 = test_model(test_prob, model_dist1);
		rep_statistics_model_dist1->add(current_accuracy_model_dist1,&stats_one_sim_model1);
		svm_destroy_model(model_dist1);
		
		current_accuracy_model_dist2 = test_model(test_prob, model_dist2);
		rep_statistics_model_dist2->add(current_accuracy_model_dist2,&stats_one_sim_model2);
		svm_destroy_model(model_dist2);
		
		current_accuracy_model_cent1 = test_model(test_prob, model_cent1);
		rep_statistics_model_cent1->add(current_accuracy_model_cent1,model_cent1->trainingTime,n_kevals_model_cent1);
		svm_destroy_model(model_cent1);
		
		current_accuracy_model_cent2 = test_model(test_prob, model_cent2);
		rep_statistics_model_cent2->add(current_accuracy_model_cent2,model_cent2->trainingTime,n_kevals_model_cent2);
		svm_destroy_model(model_cent2);
	
		
		delete test_dataset;
		delete sync_dataset;
		
		counterReps++;
		
	}//end iterations distributed case
	
	printf("FULL_COMPARISON: end\n");

	// now, compute statistics
	rep_statistics_model_dist1->compute_statistics();
	rep_statistics_model_dist1->write();
	rep_statistics_model_dist2->compute_statistics();
	rep_statistics_model_dist2->write();
	rep_statistics_model_cent1->compute_statistics();
	rep_statistics_model_cent1->write();
	rep_statistics_model_cent2->compute_statistics();
	rep_statistics_model_cent2->write();
	
	if(summary != NULL){	
		summary->type_model1 = param_set1->svm_type;
		summary->type_model2 = param_set2->svm_type;
		summary->model1_cent = rep_statistics_model_cent1->meanAccuracy; 
		summary->model1_dist = rep_statistics_model_dist1->meanAccuracy;
		summary->model2_cent = rep_statistics_model_cent2->meanAccuracy;
		summary->model2_dist = rep_statistics_model_dist2->meanAccuracy;
	}	
	//destroy dynamically allocated structures
	delete rep_statistics_model_dist1;
	delete rep_statistics_model_dist2;
	delete rep_statistics_model_cent1;
	delete rep_statistics_model_cent2;
	
	svm_destroy_param(param_set1);
	svm_destroy_param(param_set2);
					
}//end comparison of models in the distributed case AND centralized case 


void full_comparison_on_grid(char* name, int reps){
 
	const int no_c_values =  maxpowC -  minpowC + 1;
	const int no_g_values =  maxpowG -  minpowG + 1;
	const int no_e_values =  maxpowE -  minpowE + 1;
	
	double cList[no_c_values];
	double gList[no_g_values];
	double eList[no_e_values];
	
	for(int i=0; i < no_c_values; i++){
		cList[i]=pow(baseC,maxpowC - i);
	}
	
	for(int i=0; i < no_g_values; i++){
		gList[i]=pow(baseG,minpowG + i);
	}
	
	for(int i=0; i < no_e_values; i++){
		eList[i]=pow(baseE,maxpowE - i);
	}	
	
	summary_full_comparison* summary = new summary_full_comparison;
	char of_name[FILENAME_LEN];
	int stamp;
	srand(time(NULL));
	stamp = (int) rand()%1000000;
	sprintf(of_name,"FULL_COMPARISON_PROB_%s.%d.txt",name,stamp);
	std::ofstream ofs;
	ofs.open(of_name,std::ios_base::app);
	using std::endl;
	using std::scientific;
	using namespace std;
	const int w0=12;
	const int wi=20;
	ofs<<setw(w0)<< "EPS-val" <<setw(wi)<< "C-val" <<setw(wi)<< "G-val" <<setw(wi)<< "L1-cent";
	ofs<<setw(wi)<< "L1-dist" <<setw(wi)<< "L2-cent" <<setw(wi)<< "L2-dist" <<endl;			
	ofs.close();
	
	for(int j=0; j < no_e_values; j++){	 
		for(int i=0; i < no_c_values; i++){
			for(int k=0; k < no_g_values; k++){
				
				param->eps = eList[j];
				param->reg_param = cList[i];
				param->gamma = gList[k];
				multiple_params = false;
				full_comparison(name,reps,summary);
				
				ofs.open(of_name,std::ios_base::app);
				ofs.precision(5);
				ofs<<scientific<<setw(w0)<<param->eps<<setw(wi)<<param->reg_param<<setw(wi)<<param->gamma; 
				ofs<<fixed<<setw(wi)<<summary->model2_cent;
				ofs<<fixed<<setw(wi)<<summary->model2_dist;
				ofs<<fixed<<setw(wi)<<summary->model1_cent;
				ofs<<fixed<<setw(wi)<<summary->model1_dist<<endl;
				ofs.close();						
			
			}
		}
	}
		 
}
///////////////////////////  SAVE AND PLOT  /////////////////////////////////////////////


void save_and_plot(char* name){
		
		using std::endl;
		printf("SAVE_AND_PLOT: generating data ...\n");
		sync_data *sync_dataset = new sync_data(exp_id);
		sync_dataset->generate(no_tr_examples);	
		svm_problem *prob = sync_dataset->get_problem();
		Partition *partition = sync_dataset->get_partition();
		sync_dataset->plot();
	
		if(!no_file_but_plot){
			partition->rewind();
			printf("SAVE_AND_PLOT: data generated succesfully ...\n");
			//// SAVE TO FILE
			char of_name[FILENAME_LEN];
			int stamp;
			srand(time(NULL));
			stamp = (int) rand()%1000000;
		
			sprintf(of_name,"%sSYNC_DATA_%s.%d.txt",path,name,stamp);
			std::ofstream ofs(of_name,std::ios_base::out);
			svm_node* current_x;
			
			for(int j=0; j < prob->l; j++){
				
				current_x = prob->x[j];
				while(current_x->index != -1){
					ofs<<current_x->value<<" ";
					current_x++;
				}
				ofs<<prob->y[j]<<endl;
			}
			ofs.close();
		}	
		
		delete sync_dataset; 
}

void load_default_params(svm_parameter* parameter_set){
	
	parameter_set->exp_type = 1;
	parameter_set->svm_type = CVM; 
	
	parameter_set->kernel_type  = RBF;
	parameter_set->degree       = 3;
	parameter_set->gamma        = -1;	
	parameter_set->coef0        = 0;
	parameter_set->nu           = 0.5;
	parameter_set->mu           = 0.02;
	parameter_set->cache_size   = 500;//0.5GB	
	parameter_set->C            = INVALID_C;
	parameter_set->eps          = 1e-5;
	parameter_set->p            = 0.1;
	parameter_set->shrinking    = 1;
	parameter_set->probability  = 0;
	parameter_set->nr_weight    = 0;
	parameter_set->weight_label = NULL;
	parameter_set->weight       = NULL;
	parameter_set->sample_size  = 60;
	parameter_set->num_basis    = 50000;
	parameter_set->nrclasses = -1;
	parameter_set->mcvm_type = ADNORM;//ADSVM;
	
	parameter_set->reg_param   = 100.0;
	parameter_set->scale_param = 10000.0;
}
	
void parse_command_line(int argc, char **argv)
{
	int i;
	
	exp_id = RECTANGLES;
	strcpy(path,"");
	
	param = Malloc(svm_parameter,1);
	
	load_default_params(param);
	
	no_tr_examples = 1800;
	no_ts_examples = 900;
	no_repetitions = 20;
		
	bool epsIsSet = false;
	multiple_params = false;
	no_file_but_plot = false;
	run_centralized_when_comparing_distributed = false;
	
	searchC = true;
    searchG = true; 
    searchE = true; 
    
    baseC = 10;
	minpowC = -3;
	maxpowC = +3; 
	baseG = 10;
	minpowG = -5;
	maxpowG =  5; 
	baseE = 10;
	minpowE = -7;
	maxpowE = -5; 
				
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'f':
				param->num_basis = atoi(argv[i]);
				break;
			case 'E':
				param->exp_type = atoi(argv[i]);
				break;
			case 'M':
				if (argv[i-1][2]=='P'){
					multiple_params = true;
					second_param_set = Malloc(svm_parameter,1);
					load_default_params(second_param_set);
					--i;
				}
				break;	
			case 's':
				param->svm_type = atoi(argv[i]); 
				if(multiple_params){
					printf("first model is %d and second %d\n", atoi(argv[i]), atoi(argv[i+1]));
					second_param_set->svm_type = atoi(argv[++i]);	
				
				}						
//				if (!epsIsSet && (param->svm_type == CVDD || param->svm_type == CVM || param->svm_type == CVM_LS 
//					|| param->svm_type == CVR || param->svm_type == BVM || param->svm_type == MCVM )){
//					param->eps = -1;	
//					printf("*************setting eps -1\n");
//					if(multiple_params)
//						second_param_set->eps = -1;	
//				}	
				break;
			case 'N':
				if (argv[i-1][2]=='S'){
					no_repetitions = atoi(argv[i]);
				} else if ((argv[i-1][2]=='T') && (argv[i-1][3]=='R')){
					no_tr_examples = atoi(argv[i]);
				} else if ((argv[i-1][2]=='T') && (argv[i-1][3]=='S')){
					no_ts_examples = atoi(argv[i]);
				} else if (argv[i-1][2]=='F'){
					no_file_but_plot = true;
					i--;
				}		
				break;
			case 'R':
				if (argv[i-1][2]=='C'){
					run_centralized_when_comparing_distributed = true;
				}
				i--;
				break;	
			case 'P':
				if ((argv[i-1][2]=='A') && (argv[i-1][2]=='T') && (argv[i-1][2]=='H')){
					strcpy(path, argv[i]); 
				}	
			case 'y':
				param->mcvm_type = atoi(argv[i]);
				if(multiple_params)
					second_param_set->mcvm_type = atoi(argv[++i]);		
				break;	
			case 't':
				param->kernel_type = atoi(argv[i]);
				if(multiple_params)
					second_param_set->kernel_type = atoi(argv[++i]);		
				break;
			case 'd':
				param->degree = atoi(argv[i]);
				if(multiple_params)
					second_param_set->degree = atoi(argv[++i]);		
				break;
			case 'g':
				param->gamma = atof(argv[i]);
				if(multiple_params)
					second_param_set->gamma = atof(argv[++i]);	
				break;
			case 'r':
				param->coef0 = atof(argv[i]);
				if(multiple_params)
					second_param_set->coef0 = atof(argv[++i]);	
				break;
			case 'n':
				param->nu = atof(argv[i]);
				if(multiple_params)
					second_param_set->nu= atof(argv[++i]);	
				break;
			case 'u':
				param->mu = atof(argv[i]);
				if(multiple_params)
					second_param_set->mu = atof(argv[++i]);	
				break;			
			case 'c':
				param->reg_param = atof(argv[i]);
				if(multiple_params)
					second_param_set->reg_param = atof(argv[++i]);	
				break;
			case 'C':
				param->scale_param = atof(argv[i]);
				if(multiple_params)
					second_param_set->scale_param = atof(argv[++i]);	
				break;
			case 'm':
				param->cache_size = atof(argv[i]);
				if(multiple_params)
					second_param_set->cache_size =  atof(argv[i]);
				break;
			case 'e':
				param->eps = atof(argv[i]);
				if(multiple_params)
					second_param_set->eps =  atof(argv[i]);
				epsIsSet = true;
				break;
			case 'p':
				param->p = atof(argv[i]);
				if(multiple_params)
					second_param_set->p =  atof(argv[i]);
				break;
			case 'h':
				param->shrinking = atoi(argv[i]);
				if(multiple_params)
					second_param_set->shrinking = atoi(argv[i]);
				break;
			case 'b':
				param->probability = atoi(argv[i]);
				if(multiple_params)
					second_param_set->shrinking = atoi(argv[i]);
				break;
			case 'V': //ranges for cross-validation grid
				if (argv[i-1][2]=='C'){
					searchC = true;
					baseC = atoi(argv[i++]);
					minpowC = atoi(argv[i++]);
					maxpowC = atoi(argv[i]); 
					std::cout<<"base: "<<baseC<<" pow-Min: "<<minpowC<<" pow-Max"<<maxpowC<<"\n";
				} else if (argv[i-1][2]=='G'){
					searchG = true;
					baseG = atoi(argv[i++]);
					minpowG = atoi(argv[i++]);
					maxpowG = atoi(argv[i]); 
				} else if (argv[i-1][2]=='E'){
					searchE = true;
					baseE = atoi(argv[i++]);
					minpowE = atoi(argv[i++]);
					maxpowE = atoi(argv[i]); 
				}
				break;	
			case 'w':
				++param->nr_weight;
				param->weight_label = (int *)realloc(param->weight_label,sizeof(int)*param->nr_weight);
				param->weight = (double *)realloc(param->weight,sizeof(double)*param->nr_weight);
				param->weight_label[param->nr_weight-1] = atoi(&argv[i-1][2]);
				param->weight[param->nr_weight-1] = atof(argv[i]);
				break;
			case 'a':
				param->sample_size = atoi(argv[i]);
				if(multiple_params)
					second_param_set->sample_size = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}

	if(i>=argc)
		exit_with_help();

	strcpy(exp_name, argv[i]);
	if((strcmp(exp_name,"RECTANGLES")==0) || (strcmp(exp_name,"Rectangles")==0) || (strcmp(exp_name,"rectangles")==0)){
		exp_id = RECTANGLES;
		max_index = 2;
	}
	if((strcmp(exp_name,"MIX_GAUSSIANS")==0) || (strcmp(exp_name,"GAUSSIANS")==0) || (strcmp(exp_name,"gaussians")==0)){
		exp_id = GAUSSIANS_MIX;
		max_index = 2;
	}
	sprintf(exp_name,"SYNC-%s",argv[i]);
		
}

void check_parameters(svm_problem* prob, svm_parameter* param_set){
	
int i;

if ( param_set->svm_type == CVR )
	{
		param_set->C = (param_set->scale_param <= 0.0) ? 10000.0 : param_set->scale_param;
		if ( param_set->mu < 0.0 )
			param_set->mu = 0.02;
		
		double maxY = -INF, minY = INF;
		for (i=0; i<prob->l; i++)
		{
			maxY = max(maxY, prob->y[i]);
			minY = min(minY, prob->y[i]);
		}
		maxY     = max(maxY, -minY);
		param_set->C  = param_set->C *maxY;
		param_set->mu = param_set->mu*maxY;

		printf("MU %.16g, ", param_set->mu);
	}
	else if ( param_set->svm_type == MCVM )
	{
		param_set->C = (param_set->reg_param <= 0.0) ? 100.0 : param_set->C = param_set->reg_param;		
	
		int num_classes;
		
		if (param_set->nrclasses > 0){
			num_classes = param_set->nrclasses;
		}	
		else{
			
			int *label = NULL;
			determine_labels(prob,&num_classes,&label);
			delete [] label;
			
		}	
					
		if (param_set->mcvm_type == ADSVM){
			
			param_set->sameclassYDP = (float) num_classes - 1.0;
			param_set->diffclassYDP = (float) -1.0;
			printf("Using ADSVM codes: sameclassYDP: %.6g, diffclassYDP: %.6g \n", param_set->sameclassYDP, param_set->diffclassYDP);
		}	
		else if (param_set->mcvm_type == ASHARAF){
		
			param_set->sameclassYDP = (float) 1.0;
			param_set->diffclassYDP = (float) (3.0*num_classes - 4.0)/(num_classes*(num_classes-1.0));
			printf("Using Asharaf codes: sameclassYDP: %.6g, diffclassYDP: %.6g  \n", param_set->sameclassYDP, param_set->diffclassYDP);
		}	
		else if (param_set->mcvm_type == ADNORM){
	
			param_set->sameclassYDP = (float) 1.0;
			param_set->diffclassYDP = -1.0 / ((float)(num_classes)-1.0);
			printf("Using ADSVM-Corrected Codes: sameclassYDP: %.6g, diffclassYDP: %.6g  \n", param_set->sameclassYDP, param_set->diffclassYDP);
		}	
		else if (param_set->mcvm_type == ANN){ 	
		
			param_set->sameclassYDP = (float) num_classes;
			param_set->diffclassYDP = (float) num_classes - 4.0;
			printf("Using ANN Codes: sameclassYDP: %.6g, diffclassYDP: %.6g  \n", param_set->sameclassYDP, param_set->diffclassYDP);
			
		}
		
	}
	else if ( param_set->svm_type == CVM_LS ){
	
		param_set->C  = (param_set->scale_param <= 0.0) ? 10000.0 : param_set->scale_param;			
		param_set->mu = param_set->C/((param_set->reg_param < 0.0) ? 100.0 : param_set->reg_param)/prob->l;

		printf("MU %.16g, ", param_set->mu);
		
	}
	else // other SVM type		
	{
		param_set->C = (param_set->reg_param <= 0.0) ? 100.0 : param_set->C = param_set->reg_param;
	}

	if(param_set->gamma == 0.0)
		param_set->gamma = 1.0/max_index;
	else if (param_set->gamma < -0.5)
		param_set->gamma = 2.0/CalRBFWidth(prob);

	if(param_set->kernel_type == PRECOMPUTED)
		for(i=0;i<prob->l;i++)
		{
			if (prob->x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob->x[i][0].value <= 0 || (int)prob->x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}	

	switch(param_set->kernel_type)
	{
		case NORMAL_POLY:
		case POLY:
			printf("Degree %d, coef0 %.16g, ", param_set->degree, param_set->coef0);
			break;
		case RBF:
		case EXP:
		case INV_DIST:
		case INV_SQDIST:
			printf("Gamma %.16g, ", param_set->gamma);
			break;
		case SIGMOID:
			printf("Gamma %.16g, coef0 %.16g, ", param_set->gamma, param_set->coef0);
			break;						
	}
	printf("C = %.16g\n", param_set->C);

}

double CalRBFWidth(svm_problem *prob)
{
	double sumDiagonal    = 0.0;
	double sumWholeKernel = 0.0;

	int inc = 1;
	int count = 0;
	int numData = prob->l;

	if (numData > 5000)
	{
		inc = (int)ceil(numData/5000.0);
	}

	for(int i=0; i<numData; i+=inc)
	{
		count++;

		for (int j=i; j<numData; j+=inc)
		{
			double dot = Kernel::dot(prob->x[i], prob->x[j]);
			if (j == i)
			{
				sumDiagonal    += dot;
				sumWholeKernel += (dot/2.0);
			}
			else sumWholeKernel += dot;
		}
	}

	return (sumDiagonal - (sumWholeKernel*2)/count)*(2.0/(count-1));
}

void determine_labels(svm_problem *prob, int *nr_class_ret, int **label_ret)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);	
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	*nr_class_ret = nr_class;
	*label_ret = label;
	 free(data_label);
}
