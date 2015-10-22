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
#include <vector>
#include <iostream>

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

#include "SVM-commons.h"
#include "svm.h"
#include "MEB-based-SVMs.h"
#include "partitioner.h"
#include "rep_statistics.h"

#define FILENAME_LEN 1024
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define TOL_VAR 10E-10

void exit_with_help()
{
	printf(
	"Usage: DL2-train [options] training_set_file [model_file] [cent_model_file] \n"
	"VERSION 15.10.09. Options:\n"
	"-E : experiment-type (default 0)\n"
	"   0 -- simple run or cv evaluation (as usual in LIBSVM)\n"
	"   1 -- statistics for a centralized model\n"
	"   2 -- statistics for a distributed model\n"
	"   3 -- grid search for hyper-parameters using validation set\n"
	"   4 -- grid search for hyper-parameters using cross-validation\n"
	"-s : SVM model: (default 0)\n"
	"   0 -- one-versus-one of L2-SVMs\n"
	"   1 -- direct L2-SVM implementation\n"
	"   2 -- one-versus-one of L1-SVMs\n"
    "   **** OPTIONS FOR L2-SVMs\n"
	"-MA : MEB Algorithm: (0) BADOU&CLARKSON (default), (1)YILDRIM-ALG1,\n" 
	"                     (2) YILDRIM-ALG2 (3) SWAP goloso (4) SWAP pigrone\n"
    "                     (5) BVM-Tsang (6) PANIGRAHY\n"
	"-CO : iterate on epsilon values? (0) no (1) yes (default) \n"   
	"-RS : randomized selection of new coreSet points?: (0) no (1) yes (default)\n" 
	"-IM : initialization method: (0) MEB of random sample (default) (1) YILDRIM initialization (2) YILDRIM on sample\n"    
	"-IL : Iterations Limit\n"    
	"-D : distributed or centralized?\n"
	"   0 -- a centralized model\n"
	"   1 -- a distributed model (simulated by partitioning)\n"
	"        choices used for experiments 0 and 3. Exp 4 only supports cebtralized training\n"
	"-P : partitioning mode\n"
	"   0 -- random (default)\n"
	"   1 -- random-weighted\n"
	"   2 -- kmeans-based\n"
	"   3 -- precomputed\n"
	"-MB n  : minimal-balancing, ensures a minimum of items (n) at each part of the partition\n"
	"-SC    : separate classes to run k-means in k-means based partition\n"
	"-NN n : number of nodes in distributed learning simulation is set to n\n (default 5)\n"	  
	"-NS n : number of differents trainings (requires differentes files train/test) is set to n\n"
	"-NP n : number of different realizations (n) of the partition when simulating the distributed case\n"
	"-NF n : number of folds in cross-validation is set to n. Only used if cross-validation is active\n"
	"-VC n1 n2 n3 : defines the grid to search a value of C, n1=base, n2=min-power, n2=max-power\n"
	"               default: base=2, min-power=-5, max-power=+5\n" 
	"-VG n1 n2 n3 : defines the grid to search a value of G, n1=base, n2=min-power, n2=max-power\n"
	"               default: only the estimation provided by the average-distances heuristic\n"
	"-SM          : save all the models when computing statistics (by default it is not done)\n"
	"-g gamma: set gamma in kernel function (default -1, which sets 1/averaged distance between patterns)\n"
	"-c value: set the regularization parameter C (default 100)\n"
	"-e epsilon: set tolerance of termination criterion for MEBs\n"
	"            (default eps=-1 which sets eps according to the bound |f(x)-f(x)^*|)\n"
	"-v n: do an estimation of the cross-validation error for the given hyper-parameters \n"    
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

bool show_resource_usage;
bool write_resource_info_to_file;
bool file_resource_info_set;
bool fixed_test_in_sample_repetitions;
bool fixed_train_in_sample_repetitions;


char resource_info_name[FILENAME_LEN];
char timestamp[32];

void process_mem_usage(double& vm_usage, double& resident_set);
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void check_parameters();
double do_cross_validation(double &meansvs,double &mean_real_kevals, double &mean_requested_kevals);
void gridCV(double* cList, double* gammaList, int nC, int nG, char* name);
void gridVal(double* cList, double* gammaList, int nC, int nG, char* name);
double predictOneRep(FILE *input, FILE *output);
double predictOneRep2(FILE *input, FILE *output, double *classification_error_by_class);
double check_var(double var){
	if ((var < 0) && abs(var) < TOL_VAR)
		var = abs(var);	
	return sqrt(var);  
} 

char* give_me_the_time(){

  time_t rawtime = time(0);
  tm *now = localtime(&rawtime);

  if(rawtime != -1){
     strftime(timestamp,sizeof(timestamp),"%Y-%m-%d-%Hhrs-%Mmins-%Ssecs",now);
  }
  return(timestamp);
}


void repetitions_centralized(char* name, int reps);
void repetitions_distributed(char* name, int reps, int partitionsReps);
svm_model* svm_simulate_distributed(const svm_problem *prob, const svm_parameter *param, Partition *partition, distrib_statistics *stats, bool printDistribStats,  char* name);
bool save_models_when_computing_stats;
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node  *x_space;
int max_index;
bool estimate_cross_validation_error;
int nr_fold;
bool grid_cv;
double reg_param   = 100.0;
std::vector<double> Clist;
std::vector<double> Glist;
double scale_param = 10000.0;
int nnodes, partition_mode;
float* pweights = NULL;
bool group_classes_in_kmeans;
bool balance_partition;
int minimum_in_balancing_partition;
bool trainCentralized;
bool trainDistributed;
int nsamplerep, npartrep;
//for grid searchs
float searchC, searchG;
int minpowC, maxpowC, baseC;
int minpowG, maxpowG, baseG;
enum { SIMPLE_RUN, STATS_CENTRALIZED, STATS_DISTRIBUTED, VAL_GRID, CROSS_VAL_GRID};

void stamp_algorithm_name(char* name){

	if((param.svm_type == CVM) || (param.svm_type == MCVM)){
		switch(param.MEB_algorithm){
			case YILDRIM1:
				strcpy(name,"FW");
				break;
			case YILDRIM2:
				strcpy(name,"MFW");
				break;
			case PARTAN:
				strcpy(name,"PARTAN-FW");
				break;
			case SWAP:/*GOLOSO*/
				strcpy(name,"SWAP2o");
				break;
			case lSWAP:/*LAZY SWAP*/
				strcpy(name,"SWAP");
				break;
			case BVMtsang:/*BVM, TSANG 2005*/
				strcpy(name,"BVM");
				break;
			case PANIGRAHY:/*PANIGRAHY*/
				strcpy(name,"PANIGRAHY");
				break;
			default:
				strcpy(name,"BC");
				break;

		}
	} else {

				strcpy(name,"ALG-SMO");

	}

}
void stamp_kernel_type(char* name){

switch(param.kernel_type)
{
	case NORMAL_POLY:
		strcpy(name,"NORMAL-POLY");
		break;
	case POLY:
		strcpy(name,"POLY");
		break;
	case RBF:
		strcpy(name,"RBF");
		break;
	case EXP:
		strcpy(name,"EXP");
		break;
	case INV_DIST:
		strcpy(name,"INV_DIST");
		break;
	case INV_SQDIST:
		strcpy(name,"INV_SQDIST");
		break;
	case SIGMOID:
		strcpy(name,"SIGMOID");
		break;
	default:
		strcpy(name,"OTHER");
}
}

//for predictions
struct svm_node *x;
int predict_probability=0;
int max_nr_attr = 64;

int main(int argc, char **argv)
{
	#ifdef WIN32
		// Send all reports to STDOUT
		_CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE );
		_CrtSetReportFile( _CRT_WARN, _CRTDBG_FILE_STDOUT );
		_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE );
		_CrtSetReportFile( _CRT_ERROR, _CRTDBG_FILE_STDOUT );
		_CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE );
		_CrtSetReportFile( _CRT_ASSERT, _CRTDBG_FILE_STDOUT );

		// enable the options
		SET_CRT_DEBUG_FIELD( _CRTDBG_DELAY_FREE_MEM_DF );
		SET_CRT_DEBUG_FIELD( _CRTDBG_LEAK_CHECK_DF );
	#endif

	//printf("int %d, short int %d, char %d, double %d, float %d, node %d\n",sizeof(int),sizeof(short int), sizeof(char), sizeof(double), sizeof(float), sizeof(svm_node));

	char input_file_name[FILENAME_LEN];    
	char model_file_name[FILENAME_LEN];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);


	#ifdef WIN32
		assert(_CrtCheckMemory());
	#endif
	
	if (estimate_cross_validation_error && trainDistributed)
	{
		fprintf(stderr,"Error: cross-validation for distributed case not yet supported \n");
		exit(1);
	}
	
	if (param.exp_type == STATS_CENTRALIZED){
		//statistics for the centralized routine
		int nreps = nsamplerep;
		trainCentralized = true;
		trainDistributed = false; 
		printf("CENTRALIZED STATS\n");
	
		repetitions_centralized(input_file_name,nreps);
		
	} else if (param.exp_type == STATS_DISTRIBUTED){
		//statistics for the distributed routine	
		int nreps = nsamplerep;
		int nrepsPartitions = npartrep;
		
		trainCentralized = false;
		trainDistributed = true;
		repetitions_distributed(input_file_name,nreps,nrepsPartitions);
	
	} else if (param.exp_type == VAL_GRID){
	    
	    //Evaluation of a set of values for hype-parameters
	    //using a valiudation set provideed by the user
	     
		printf("GRID search using validation set.\n");
		
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
		
		param.C = 10;//will be overwritten
		param.gamma = -1;//force estimation of a gamma that will be also tested
			
		gridVal(cList, gList, nC, nG, input_file_name);
	
	} else if(param.exp_type == CROSS_VAL_GRID){
	    
	    //Evaluation of a set of values for hype-parameters
	    //using CROSS-VALIDATION 
	     
		printf("Cross-Validation GRID search. Bins= %d\n",nr_fold);
		
		int nC,nG;
		double *cList;
		double *gList;
		int nreps = nsamplerep;

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
				//std::cout<<"new value of G "<<gList[i+1]<<"\n";
			}
		} else {
			nG = 1;
			gList = Malloc(double,nG);
			//In this case only the default value of gamme is used
			//this will be estimated after reading the problem
		}
		
		param.C = 10;//will be overwritten
		param.gamma = -1;//force estimation of a default gamma that will be also tested
		printf("calling GCV");
		gridCV(cList, gList, nC, nG, input_file_name);
			
	} else { //SIMPLE EXPERIMENT: TRAIN A MODEL AND SAVE IT
		
		read_problem(input_file_name);
		
		if(file_resource_info_set == false){
			srand(time(NULL));
			int stamp = (int) rand()%1000000;
			sprintf(resource_info_name,"%s.resource-usage.%d.txt",input_file_name,stamp);
		}
		
		printf ("Finish reading input files!\n");
		check_parameters();
		error_msg = svm_check_parameter(&prob,&param);	

		if(error_msg)
		{
			fprintf(stderr,"Error: %s\n",error_msg);
			exit(1);
		}
		
		 double average_num_svs;
		 double real_kevals;
		 double requested_kevals;
		 		
		 if (estimate_cross_validation_error) {//ESTIMATE CROSS-VALIDATION ERROR OF THE MODEL
		
			do_cross_validation(average_num_svs,real_kevals,requested_kevals);
	
		} else {
		
			double duration;
			double start = getRunTime();
		
			printf("kernel: %d\n",param.kernel_type);
		
			if (trainCentralized) {
				printf("centralized training of a single model\n");
				model = svm_train(&prob,&param);
			}
			else { //simulate distributed
				
			struct distrib_statistics statsDist;
			bool flag = true;
			
			Partition *data;
			
			if(partition_mode == RANDOM_WEIGHTED){
				printf("partition mode is RANDOM_WEIGHTED, number %d\n",partition_mode);
				data = new Partition(nnodes,&prob,pweights,balance_partition,minimum_in_balancing_partition);
			}else if(partition_mode == PREPART){ 
				printf("partition mode is PREDEFINED, number %d\n",partition_mode);
				data = new Partition(nnodes,&prob,NULL,NULL);
			}else{
				printf("partition mode is RANDOM OR KMEANS, number %d\n",partition_mode);
				data = new Partition(nnodes,partition_mode,&prob,group_classes_in_kmeans,balance_partition,minimum_in_balancing_partition);
			}
			
			printf("distributed training of a single model\n");
			model = svm_simulate_distributed(&prob,&param,data,&statsDist,flag,input_file_name);
	
			}	
	
    	#ifdef WIN32
			assert(_CrtCheckMemory());
		#endif
	
		double finish = getRunTime();	    
    	duration = (double)(finish - start);
	
		svm_save_model(model_file_name,model);
		printf("model saved as: %s\n",model_file_name);
		svm_destroy_model(model);
		printf("CPU Time = %f second\n", duration);
    	FILE* fModel = fopen(model_file_name, "a+t");					// append mode
		fprintf(fModel, "CPU Time = %f second\n", duration);
		fclose(fModel);
		}
	
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	
	}
	  
	#ifdef WIN32
		assert(_CrtCheckMemory());
	#endif

    return 0;
}

void iterated_distributed_routine(char *tr_file_name, char *model_name){


}

void process_mem_usage(double& vm_usage, double& resident_set)
{
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   //
   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1048576.0; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize / 1048576.0;
   resident_set = rss * page_size_kb;
}

void show_memory_usage(char *header){

	FILE* resource_file;
	if(write_resource_info_to_file){
		resource_file = fopen(resource_info_name, "a+t");
	}
	
	if(!write_resource_info_to_file || resource_file == NULL){
		resource_file = stdout;
	}
		
	fprintf(resource_file,"\n");
	fprintf(resource_file,"resources-usage in: %s\n",header);
	rusage ru;
	getrusage(RUSAGE_SELF,&ru);
	
    double vm, rss;
	process_mem_usage(vm, rss);
	fprintf(resource_file,"cpu-time: %ld.%ld\n", ru.ru_stime.tv_sec, ru.ru_stime.tv_usec);
	fprintf(resource_file,"memory usage: %f Mb\n", vm);
	fprintf(resource_file,"resident set: %f\n", rss);
	fprintf(resource_file,"\n");	
  	
  	if(write_resource_info_to_file && resource_file != NULL){
		fclose(resource_file);
	}
}


void repetitions_centralized(char* name_orig, int reps){

	char name[2*FILENAME_LEN];
	sprintf(name,"%s-eps-%.1E",name_orig,param.eps);
	
	double *accuracies = Malloc(double,reps);
	double *trTimes = Malloc(double,reps);
	int *sizes = Malloc(int,reps);
	int nsubmodels;
	
	unsigned long int *n_real_kevals = Malloc(unsigned long int,reps);
	unsigned long int *n_requested_kevals = Malloc(unsigned long int,reps);
	
	double *current_error_by_class = NULL;
	double *error_by_class = NULL;
	
	statistic* obj_values = NULL;
	statistic* size_submodels = NULL;
	statistic* smo_it = NULL;
	statistic* greedy_it = NULL;
		
	const char *error_msg;
	char repetitionsCentralizedResults[FILENAME_LEN];

	char stamp[64];
	sprintf(stamp,"%s",give_me_the_time());

	int nr_class;
	int *labels = NULL;
	
	char training_alg[FILENAME_LEN];
	stamp_algorithm_name(training_alg);
	
	char kernel_type_string[25];
	stamp_kernel_type(kernel_type_string);
	
	if (param.svm_type == CVM){
	sprintf(repetitionsCentralizedResults,"%s.cent.OVO-CVM.%s-K-%s.repStats.%s.txt",name,training_alg,kernel_type_string,stamp);
		if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.cent.OVO-CVM.%s-K-%s.repStats.%s.txt",name,training_alg,kernel_type_string,stamp);
		}
	} else if (param.svm_type == C_SVC){ 
	sprintf(repetitionsCentralizedResults,"%s.cent.OVO-CSVC.K-%s.repStats.%s.txt",name,kernel_type_string,stamp);
		if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.cent.OVO-CSVC.repStats.resource-usage.%s.txt",name,stamp);
		}
	
	} else {
		switch(param.mcvm_type){
		case ADSVM:
			sprintf(repetitionsCentralizedResults,"%s.cent.MCVM-ADCODES.repStats.%s.txt",name,stamp);
			break;
		case ASHARAF:
			sprintf(repetitionsCentralizedResults,"%s.cent.MCVM-ASHARAF.repStats.%s.txt",name,stamp);
			break;
		case ADNORM:
			sprintf(repetitionsCentralizedResults,"%s.cent.MCVM-ADNORM.repStats.%s.txt",name,stamp);
			break;
		case ANN:
			sprintf(repetitionsCentralizedResults,"%s.cent.MCVM-ANN.repStats.%s.txt",name,stamp);
			break;
		}	
			if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.cent.MCVM.repStats.resource-usage.%s.txt",name,stamp);
		}
	}
		

	for(int repIt=0; repIt < reps; repIt++){

	char problemRepName[FILENAME_LEN];
	char testRepName[FILENAME_LEN];
	char predictionsRepName[FILENAME_LEN];
	char modelName[2048];
	
	double currentAccuracy;
	
	if(fixed_train_in_sample_repetitions){

		sprintf(problemRepName,"%s.train.01.txt",name_orig);

	} else{

		sprintf(problemRepName,"%s.train.0%d.txt",name_orig,repIt+1);

	}

	if(fixed_test_in_sample_repetitions){

		sprintf(testRepName,"%s.test.01.txt",name_orig);
		sprintf(predictionsRepName,"%s.predictions.01.txt",name_orig);

	} else{

		sprintf(testRepName,"%s.test.0%d.txt",name_orig,repIt+1);
		sprintf(predictionsRepName,"%s.predictions.0%d.txt",name_orig,repIt+1);

	}

	if(Clist.size() != 0){
		reg_param = Clist[repIt%Clist.size()];
		param.C = reg_param;
	}

	if(Glist.size() != 0){
		param.gamma = Glist[repIt%Glist.size()];
	}

	read_problem(problemRepName);
	
	if(show_resource_usage){
		char *message = new char[50];
		sprintf(message, "rep.centralized - iteration %d",repIt);
		show_memory_usage(message);
	}

	printf ("Finish reading input files!\n");
	check_parameters();
	printf ("Finish checking parameters!\n");
	
	error_msg = svm_check_parameter(&prob,&param);	
	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}
	

	////////////// Train a Model ////////////////////
	printf("using params C = %g , gamma = %g", param.C, param.gamma);

	char pathFileName[2*FILENAME_LEN];
	sprintf(pathFileName,"%s.CVM.%s-K-%s.ALG-PATH.%s.Rep.%d.txt",name,training_alg,kernel_type_string,stamp,repIt+1);
	
	param.filename = pathFileName;

	Kernel::reset_real_kevals();
	Kernel::reset_requested_kevals();
	
		model = svm_train(&prob,&param);
	    
	n_real_kevals[repIt] = Kernel::get_real_kevals();
	n_requested_kevals[repIt] = Kernel::get_requested_kevals();
	
	//////////////////////////////////////////////////
	
	FILE* testF = fopen(testRepName,"r");
	FILE* predF = NULL; 
	
	if (save_models_when_computing_stats){
		
		predF = fopen(predictionsRepName,"w");
	
	}

	nr_class = model->nr_class;
	
	if (current_error_by_class == NULL){
			current_error_by_class = new double[nr_class];
	}
	
	if (obj_values == NULL){
		obj_values = new statistic[model->nsubmodels];
	}
	
	if (size_submodels == NULL){
		size_submodels = new statistic[model->nsubmodels];
	}
	
	if (smo_it == NULL){
		smo_it = new statistic[model->nsubmodels];
	}
	
	if (greedy_it == NULL){
		greedy_it = new statistic[model->nsubmodels];
	}
	
	for(int k=0; k < model->nsubmodels; k++){
		obj_values[k].add(model->obj[k]);
		size_submodels[k].add(model->subsizes[k]);
		smo_it[k].add(model->smo_it[k]);
		greedy_it[k].add(model->greedy_it[k]);
		
		
	}
		
	currentAccuracy = predictOneRep2(testF, predF,current_error_by_class);
	
	if (save_models_when_computing_stats){
	
		if (param.svm_type == CVM)
			sprintf(modelName,"%s.train.0%d.cent.OVOCVM.model",name_orig,repIt+1);
		else
			sprintf(modelName,"%s.train.0%d.cent.MCVM.model",name_orig,repIt+1);
	
		svm_save_model(modelName,model);
		printf("model saved as: %s\n",modelName);
	}
	
	accuracies[repIt] = currentAccuracy; 	
	
	if (labels == NULL){
		labels = new int[nr_class];
		for(int i=0; i< nr_class; i++)
			labels[i] = model->label[i];
	}
	
	if (error_by_class == NULL){
		error_by_class = new double[nr_class];
		for(int i=0; i < nr_class; i++){
			error_by_class[i]=0;
		}
	}
	
	for(int i=0; i < nr_class; i++){
			error_by_class[i]+= (double)current_error_by_class[i]/((double)reps);
	}
	 
	trTimes[repIt] = model->trainingTime;
	sizes[repIt] = model->l;
	nsubmodels=model->nsubmodels;
	svm_destroy_model(model);
	free(prob.y);
	free(prob.x);
	free(x_space);
	
	if (save_models_when_computing_stats){
		
		fclose(predF);
	}
	
	fclose(testF);
	
	}//end iterations centralized case
	
	// now, compute statistics
	double meanAccuracy=0,stdAccuracy=0;
	double meanTime=0, stdTime=0;
	double meanN_real_kevals = 0, stdN_real_kevals = 0;
	double meanN_requested_kevals = 0, stdN_requested_kevals = 0;
	double mean_sizes = 0.0, std_sizes = 0.0;
	for(int repIt=0; repIt < reps; repIt++){
	
		meanAccuracy += accuracies[repIt];
		stdAccuracy += accuracies[repIt]*accuracies[repIt];
		meanTime += trTimes[repIt];
		meanN_real_kevals += (double)n_real_kevals[repIt]; 
		meanN_requested_kevals += (double)n_requested_kevals[repIt]; 
		stdTime += trTimes[repIt]*trTimes[repIt];
		stdN_real_kevals +=  (double)n_real_kevals[repIt]*n_real_kevals[repIt];
		stdN_requested_kevals +=  (double)n_requested_kevals[repIt]*n_requested_kevals[repIt];
		
		mean_sizes += (double)sizes[repIt];
		std_sizes +=  (double)sizes[repIt]*sizes[repIt];
	}
	
	meanAccuracy = meanAccuracy/reps;
	stdAccuracy = sqrt((stdAccuracy/reps) - (meanAccuracy*meanAccuracy));
	meanTime = meanTime/reps;
	meanN_real_kevals = (double)meanN_real_kevals/reps;
	meanN_requested_kevals = (double)meanN_requested_kevals/reps;
	
	stdTime = check_var((stdTime/reps) - (meanTime*meanTime));
	stdN_real_kevals = check_var((stdN_real_kevals/reps) - (meanN_real_kevals*meanN_real_kevals));
	stdN_requested_kevals = check_var((stdN_requested_kevals/reps) - (meanN_requested_kevals*meanN_requested_kevals));
	
	mean_sizes = mean_sizes/reps;
	std_sizes = check_var((std_sizes/reps) - (mean_sizes*mean_sizes));
	
	printf("Results saved in: %s\n",repetitionsCentralizedResults);	
	FILE* repStats = fopen(repetitionsCentralizedResults, "w");		
	
	fprintf(repStats, "Statistics for Centralized Model, %d Repetitions\n",reps);
	fprintf(repStats, "Problem: %s\n",name);
	
	if (param.svm_type == CVM){
		fprintf(repStats, "Model-Type: OVO-CVM\n");
	} else if (param.svm_type == C_SVC){
		fprintf(repStats, "Model-Type: OVO-CSVC\n");
	} else {
		fprintf(repStats, "Model-Type: MCVM (Modified Asharaf, Codes ADSVM)\n");
	}
	
	char the_MEB_algorithm[FILENAME_LEN];
	stamp_algorithm_name(the_MEB_algorithm);
	fprintf(repStats, "MEB Algorithm: %s\n",the_MEB_algorithm);

	if((param.svm_type == CVM) || (param.svm_type == MCVM)){

		if(param.randomized){
			fprintf(repStats, "Randomization: YES (%d points)\n", param.sample_size);
			if(param.safe_stopping)
				fprintf(repStats, "Safe Stopping: YES (-SS %d)\n", param.nsamplings_safe_stopping);
			else
				fprintf(repStats, "Safe Stopping: NO\n");
		} else {
			fprintf(repStats, "Randomization: NO\n");
		}
		if(param.cooling)
			fprintf(repStats, "Cooling: YES\n");
		else
			fprintf(repStats, "Cooling: NO\n");
		
		switch(param.MEB_init_method){
			case YILDRIM_INIT:
				fprintf(repStats, "Initialization: YILDIRIM\n");
				break;
			case YILDRIM_SAMPLE_INIT:
				fprintf(repStats, "Initialization: YILDIRIM ON SAMPLE (%d points)\n", param.sample_size);
				break;
			default:
				fprintf(repStats, "Initialization: RANDOM MEB (%d points)\n",INITIAL_CS);
				break;
		}

	}

	fprintf(repStats, "C=%E, Gamma=%E, EPS-Balls=%E\n",param.C,param.gamma,param.eps);
	fprintf(repStats, "Mean- Correct-Classification-Rate = %g\n",meanAccuracy);
	printf("Mean- Correct-Classification-Rate = %g\n",meanAccuracy);
	
	fprintf(repStats, "Stdv- Correct-Classification-Rate = %g\n",stdAccuracy);
	
	// fprintf(repStats, "Means Correct-Classification Rates by Class\n");
	// for(int i=0; i < nr_class; i++){
	// 	fprintf(repStats, "class %d: %g,  ",labels[i],error_by_class[i]);
	// }
	// fprintf(repStats, "\n");

	fprintf(repStats, "Mean- TrainingTime = %E\n",meanTime);
	fprintf(repStats, "Stdv- TrainingTime = %E\n",stdTime);
	printf("Mean- TrainingTime = %E\n",meanTime);
	
	fprintf(repStats, "Mean- N°REAL-Kernel-Evals = %E\n",meanN_real_kevals);
	printf("Mean- N°REAL-Kernel-Evals = %E\n",meanN_real_kevals);
	
	fprintf(repStats, "Stdv- N°REAL-Kernel-Evals = %E\n",stdN_real_kevals);

	fprintf(repStats, "Mean- N°REQUESTED-Kernel-Evals = %E\n",meanN_requested_kevals);
	fprintf(repStats, "Stdv- N°REQUESTED-Kernel-Evals = %E\n",stdN_requested_kevals);
	printf("Mean- N°REQUESTED-Kernel-Evals = %E\n",meanN_requested_kevals);

	fprintf(repStats, "Mean- Size Model = %E\n",mean_sizes);
	fprintf(repStats, "Stdv- Size Model = %E\n",std_sizes);
	printf("Mean- Size Model = %E\n",mean_sizes);

	statistic means_greedy_iterations;
	statistic stds_greedy_iterations;
	statistic means_smo_iterations;
	statistic stds_smo_iterations;

	for(int k=0; k < nsubmodels; k++){

		means_greedy_iterations.add(greedy_it[k].get_mean());
		stds_greedy_iterations.add(greedy_it[k].get_std());

		if(param.MEB_algorithm == BADOUCLARKSON){

			means_smo_iterations.add(smo_it[k].get_mean());
			stds_smo_iterations.add(smo_it[k].get_std());
		}
	}

	fprintf(repStats, "Mean- GREEDY Iterations = %E\n",means_greedy_iterations.get_sum());
	printf("Mean- GREEDY Iterations = %E\n",means_greedy_iterations.get_sum());
	fprintf(repStats, "Stdv- GREEDY Iterations = %E\n",stds_greedy_iterations.get_sum());
	
	if(param.MEB_algorithm == BADOUCLARKSON){
		fprintf(repStats, "Mean- SMO Iterations = %E\n",means_smo_iterations.get_sum());
		fprintf(repStats, "Stdv- SMO Iterations = %E\n",stds_smo_iterations.get_sum());
	}

	fprintf(repStats, "Means OBJ-function (and size) by submodel=\n");
	
	for(int k=0; k < nsubmodels; k++){
		double current_obj = obj_values[k].get_mean();
		double current_size = size_submodels[k].get_mean();

		fprintf(repStats, "Sub Model N%d: Means- Objective=%.6g, Active Points=%g, Greedy_it=%g\n",k,current_obj,current_size,greedy_it[k].get_mean());
		fprintf(repStats, "Sub Model N%d: Stds- Objective=%.6g, Active Points=%g, Greedy_it=%g\n",k,obj_values[k].get_std(),size_submodels[k].get_std(),greedy_it[k].get_std());
		
		if(param.MEB_algorithm == BADOUCLARKSON)
			fprintf(repStats, " ,smo_it=%g",smo_it[k].get_mean());
		fprintf(repStats, "\n");
		
	}
	
	fprintf(repStats, "\n");
	
	fclose(repStats);
	
	svm_destroy_param(&param);
	
	free(accuracies);
	free(trTimes); 
	free(n_real_kevals);
	free(n_requested_kevals);
	
	free(current_error_by_class);
	free(error_by_class);
	free(sizes);
	
		
}//end repetitions centralized case routine

void repetitions_distributed(char* name_orig, int reps, int partitionsReps){
	

	char name[2*FILENAME_LEN];
	sprintf(name,"%s-nodes-%d-eps-%.1E",name_orig,nnodes,param.eps);

	
	double *accuracies = Malloc(double,reps*partitionsReps);
	double *coordTimes = Malloc(double,reps*partitionsReps);
	double *nodeTimes = Malloc(double,reps*partitionsReps);
	double *parTimes = Malloc(double,reps*partitionsReps);
	double *totalTimes = Malloc(double,reps*partitionsReps);
	double *totalCompressions = Malloc(double,reps*partitionsReps);
	double *nodeCompressions = Malloc(double,reps*partitionsReps);
	int *coordSizes = Malloc(int,reps*partitionsReps);
	
	double *coordN_kevals = Malloc(double,reps*partitionsReps);
	double *nodeN_kevals = Malloc(double,reps*partitionsReps);
	double *parN_kevals = Malloc(double,reps*partitionsReps);
	double *totalN_kevals = Malloc(double,reps*partitionsReps);
	
	double intraVarN_kvals = 0;	
	double intraVarRemoteTimes = 0;
	double intraVarRemoteCompressions = 0;
	
	double *current_error_by_class = NULL;
	double *error_by_class = NULL;
	int nr_class;
	int *labels = NULL;
		 
	const char *error_msg;
	char repetitionsDistributedResults[FILENAME_LEN];

	char stamp[64];
	sprintf(stamp,"%s",give_me_the_time());
	char training_alg[FILENAME_LEN];
	stamp_algorithm_name(training_alg);

	//naming files according to the type of model
	if (param.svm_type == CVM){
		printf("MODEL TYPE IS: CVM\n");
		sprintf(repetitionsDistributedResults,"%s.dist.OVO-CVM-ALG-%s.repStats.%d.txt",name,training_alg,stamp);
		if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.dist.OVO-CVM.repStats.resource-usage.%d.txt",name,stamp);
		}
	} else if(param.svm_type == C_SVC) { 
		printf("MODEL TYPE IS: C-SVM\n");
		sprintf(repetitionsDistributedResults,"%s.dist.C-SVC-ALG-%s.repStats.%d.txt",name,training_alg,stamp);
		if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.dist.C-SVC.repStats.resource-usage.%d.txt",name,stamp);
		}
	} else {
		printf("MODEL TYPE IS: MCVM\n");
		
		switch(param.mcvm_type){
		
			case ADSVM:
				sprintf(repetitionsDistributedResults,"%s.dist.MCVM-ADCODES.repStats.%d.txt",name,stamp);
				break;
		
			case ASHARAF:
				sprintf(repetitionsDistributedResults,"%s.dist.MCVM-ASHARAF.repStats.%d.txt",name,stamp);
				break;
			
			case ADNORM:
				sprintf(repetitionsDistributedResults,"%s.dist.MCVM-ADNORM.repStats.%d.txt",name,stamp);
				break;
			
			case ANN:
				sprintf(repetitionsDistributedResults,"%s.dist.MCVM-ANN.repStats.%d.txt",name,stamp);
				break;
		}	
		
		if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.dist.MCVM.repStats.resource-usage.%d.txt",name,stamp);
		}
	}
	
	/////////// Repetitions of the centralized routine /////////////
	int counterReps = 0;	
	for(int repIt=0; repIt < reps; repIt++){
	
	char problemRepName[FILENAME_LEN];
	char testRepName[FILENAME_LEN];
	char predictionsRepName[FILENAME_LEN];
	double currentAccuracy;
	
	
	sprintf(problemRepName,"%s.train.0%d.txt",name_orig,repIt+1);
	sprintf(testRepName,"%s.test.0%d.txt",name_orig,repIt+1);
	sprintf(predictionsRepName,"%s.predictions.0%d.txt",name_orig,repIt+1);
	
	// read new version of the problem	
	read_problem(problemRepName);
	printf ("Finish reading input files!\n");
	check_parameters();
	printf ("Finish checking parameters!\n");
	
	error_msg = svm_check_parameter(&prob,&param);	
	
	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}
			
	for (int nParReps=0; nParReps<partitionsReps ; nParReps++){
		struct distrib_statistics statsDist;
		bool flag = false;
		
		Partition *data; //destructor is called automatically after each repetition by end of scope
		
	   if(show_resource_usage){
			char *message = new char[50];
			sprintf(message, "rep.distributed - iteration %d x %d",repIt,nParReps);
			show_memory_usage(message);
		}	
	
		printf("NNODES IS: %d\n",nnodes);		
		if(partition_mode == RANDOM_WEIGHTED)
			data = new Partition(nnodes,&prob,pweights,balance_partition,minimum_in_balancing_partition);
		else if(partition_mode == PREPART) 
			data = new Partition(nnodes,&prob,NULL,NULL);
		else
			data = new Partition(nnodes,partition_mode,&prob,group_classes_in_kmeans,balance_partition,minimum_in_balancing_partition);
	
		///////////////// OBTAIN A MODEL SIMULATING THE DISTRIBUTED CASE ///////////////////////////
		
		data->rewind();
		
		model = svm_simulate_distributed(&prob,&param,data,&statsDist,flag,name);
		
		nr_class = model->nr_class;
		
		/////////////////////////////////////////////////////////////////////////////////////////////
		
				
		if (save_models_when_computing_stats){
		
			char modelName[2048];	
			if (param.svm_type == CVM){
				sprintf(modelName,"%s.train.0%d.dist.%d.OVO-CVM.model",name_orig,repIt+1,nParReps+1);
			} else if (param.svm_type == C_SVC) {
				sprintf(modelName,"%s.train.0%d.dist.%d.OVO-CSVC.model",name_orig,repIt+1,nParReps+1);
			} else {
				sprintf(modelName,"%s.train.0%d.dist.%d.MCVM.model",name_orig,repIt+1,nParReps+1);
			}
			svm_save_model(modelName,model);
			printf("model saved as: %s\n",modelName);
		
		}
			
		FILE* testF = fopen(testRepName,"r");
		FILE* predF = NULL;
		
		if (save_models_when_computing_stats){
			predF = fopen(predictionsRepName,"w");
		}
		
		if (current_error_by_class == NULL)
			current_error_by_class = new double[nr_class];
		
		for(int i=0; i < nr_class; i++)
			current_error_by_class[i]=0;
		
		currentAccuracy = predictOneRep2(testF, predF,current_error_by_class);
		
		accuracies[counterReps] = currentAccuracy; //classification accuracy
		coordTimes[counterReps] = statsDist.trainingTime;//solver time for coordinator problem 
		coordSizes[counterReps] = statsDist.size_cent_model;//size for coordinator model 
		coordN_kevals[counterReps] = statsDist.N_kevals;//calls to kernel function for coordinator problem
		totalCompressions[counterReps] = statsDist.totalCompression;//fraction of total data imported 
		nodeCompressions[counterReps] = statsDist.meanNodeCompression;//mean of the fraction of data imported from each node
		nodeTimes[counterReps] = statsDist.meanRemoteTime;//mean of remote solver times
		nodeN_kevals[counterReps] = statsDist.mean_remote_N_kevals;//mean of calls to kernel function in remote nodes
		parTimes[counterReps] = statsDist.maxRemoteTime;//max of remote solver times
		parN_kevals[counterReps] = statsDist.max_remote_N_kevals;//max of calls to kernel function in remote nodes
		totalTimes[counterReps] = statsDist.sumRemoteTime;//sum of remote solver times
		totalN_kevals[counterReps] = statsDist.sum_remote_N_kevals;//sum of calls to kernel function in remote nodes
		  
		intraVarRemoteTimes += (statsDist.stdRemoteTime*statsDist.stdRemoteTime);
		intraVarRemoteCompressions += (statsDist.stdNodeCompression*statsDist.stdNodeCompression);
		intraVarN_kvals += (statsDist.std_remote_N_kevals*statsDist.std_remote_N_kevals);
		
		if (labels == NULL){
			labels = new int[nr_class];
		
			for(int i=0; i< nr_class; i++)
				labels[i] = model->label[i];
		}
	
		if (error_by_class == NULL){
	
			error_by_class = new double[nr_class];
			for(int i=0; i < nr_class; i++){
				error_by_class[i]=0;
			}
		}
				
		for(int i=0; i < nr_class; i++){
				error_by_class[i]+= current_error_by_class[i]/((double)reps*partitionsReps);
		}
		
	
		svm_destroy_model(model);
	
		if (save_models_when_computing_stats){
		
		fclose(predF);
		
		}
		
		fclose(testF);
		counterReps++;	
	
	}//end while partitions
	
	// destroy current version of the problem		
	free(prob.y);
	free(prob.x);
	free(x_space);
		
	}//end while sample repetitions
	
	double meanAccuracy = 0,stdAccuracy = 0;
	double meanTotalCompression = 0, stdTotalCompression = 0;
	double meanNodeCompression = 0, stdNodeCompression = 0;
	double meanParTime = 0, meanTotalTime=0, stdParTime = 0, stdTotalTime=0;
	double meanCoordTime = 0, stdCoordTime = 0;
	double meanNodeTime = 0, stdNodeTime = 0;
	double meanSumParallel = 0, meanSumSequential = 0;
	double stdSumParallel = 0, stdSumSequential = 0;
	double meanCoordN_kevals = 0, stdCoordN_kevals = 0;
	double meanSumParallelN_kevals = 0, stdSumParallelN_kevals = 0;
	double meanSumSequentialN_kevals = 0, stdSumSequentialN_kevals = 0;  
	double meanNodeN_kevals = 0, stdNodeN_kevals = 0;
	double meanParN_kevals = 0, meanTotalN_kevals = 0, stdParN_kevals = 0, stdTotalN_kevals = 0;
	double mean_sizes = 0, std_sizes = 0;
	
	for(int repIt=0; repIt < counterReps; repIt++){
	
		meanAccuracy += accuracies[repIt];
		stdAccuracy += accuracies[repIt]*accuracies[repIt];
		
		meanTotalCompression += totalCompressions[repIt];
		meanNodeCompression += nodeCompressions[repIt];
		meanCoordTime += coordTimes[repIt];
		meanNodeTime += nodeTimes[repIt];
		meanParTime += parTimes[repIt];
		meanTotalTime += totalTimes[repIt];
		meanSumParallel += parTimes[repIt] + coordTimes[repIt]; 
		meanSumSequential += totalTimes[repIt] + coordTimes[repIt];
		
		meanCoordN_kevals += coordN_kevals[repIt];
		meanNodeN_kevals += nodeN_kevals[repIt];
		meanParN_kevals += parN_kevals[repIt];
		meanTotalN_kevals += totalN_kevals[repIt];
		meanSumParallelN_kevals +=	parN_kevals[repIt] + coordN_kevals[repIt];
		meanSumSequentialN_kevals += totalN_kevals[repIt] + coordN_kevals[repIt];
		
		mean_sizes += (double)coordSizes[repIt];
		std_sizes += (double)coordSizes[repIt]*coordSizes[repIt];
				
		stdNodeCompression += (nodeCompressions[repIt]*nodeCompressions[repIt]); 
		stdTotalCompression += (totalCompressions[repIt]*totalCompressions[repIt]);
		stdParTime += (parTimes[repIt]*parTimes[repIt]); 
		stdTotalTime += (totalTimes[repIt]*totalTimes[repIt]); 
		stdNodeTime += (nodeTimes[repIt]*nodeTimes[repIt]);
		stdCoordTime += (coordTimes[repIt]*coordTimes[repIt]);
	    stdSumParallel += (parTimes[repIt] + coordTimes[repIt])*(parTimes[repIt] + coordTimes[repIt]); 
		stdSumSequential += (totalTimes[repIt] + coordTimes[repIt])*(totalTimes[repIt] + coordTimes[repIt]);
	
		stdCoordN_kevals += coordN_kevals[repIt]*coordN_kevals[repIt];
		stdNodeN_kevals += nodeN_kevals[repIt]*nodeN_kevals[repIt];
		stdParN_kevals += parN_kevals[repIt]*parN_kevals[repIt];
		stdTotalN_kevals += totalN_kevals[repIt]*totalN_kevals[repIt];
		stdSumParallelN_kevals += (parN_kevals[repIt] + coordN_kevals[repIt])*(parN_kevals[repIt] + coordN_kevals[repIt]);
		stdSumSequentialN_kevals += (totalN_kevals[repIt] + coordN_kevals[repIt])*(totalN_kevals[repIt] + coordN_kevals[repIt]);
	
	}	
	 
	meanAccuracy = meanAccuracy/counterReps;
	stdAccuracy = check_var((stdAccuracy/counterReps) - (meanAccuracy*meanAccuracy));
	meanTotalCompression = meanTotalCompression/counterReps;
	meanNodeCompression = meanNodeCompression/counterReps;
	meanCoordTime = meanCoordTime/counterReps;
	meanNodeTime = meanNodeTime/counterReps;
	meanParTime = meanParTime/counterReps;
	meanTotalTime = meanTotalTime/counterReps;
	meanSumParallel = meanSumParallel/counterReps; 
	meanSumSequential = meanSumSequential/counterReps;
	
	meanCoordN_kevals = meanCoordN_kevals/counterReps;
	meanNodeN_kevals = meanNodeN_kevals/counterReps;
	meanParN_kevals = meanParN_kevals/counterReps;
	meanTotalN_kevals = meanTotalN_kevals/counterReps;
	meanSumParallelN_kevals =	meanSumParallelN_kevals/counterReps;
	meanSumSequentialN_kevals = meanSumSequentialN_kevals/counterReps;
		 
	mean_sizes = mean_sizes/counterReps;
	std_sizes = check_var((std_sizes/counterReps) - (mean_sizes*mean_sizes));;
				
	stdNodeCompression = check_var((intraVarRemoteCompressions/counterReps) + (stdNodeCompression/counterReps) - (meanNodeCompression*meanNodeCompression));  	
	stdTotalCompression = check_var((stdTotalCompression/counterReps) - (meanTotalCompression*meanTotalCompression));
	stdCoordTime = check_var((stdCoordTime/counterReps) - (meanCoordTime*meanCoordTime));
	stdNodeTime = check_var((intraVarRemoteTimes/counterReps) + (stdNodeTime/counterReps) - (meanNodeTime*meanNodeTime)); 
	stdParTime = check_var((stdParTime/counterReps) - (meanParTime*meanParTime)); 
	stdTotalTime = check_var((stdTotalTime/counterReps) - (meanTotalTime*meanTotalTime));
    stdSumParallel = check_var((stdSumParallel/counterReps) - (meanSumParallel*meanSumParallel));
    stdSumSequential = check_var((stdSumSequential/counterReps) - (meanSumSequential*meanSumSequential));
    
	stdCoordN_kevals = check_var((stdCoordN_kevals/counterReps) - (meanCoordN_kevals*meanCoordN_kevals));
	stdNodeN_kevals = check_var((intraVarN_kvals/counterReps) + (stdNodeN_kevals/counterReps) - (meanNodeN_kevals*meanNodeN_kevals));
	stdParN_kevals = check_var((stdParN_kevals/counterReps) - (meanParN_kevals*meanParN_kevals));
	stdTotalN_kevals = check_var((stdTotalN_kevals/counterReps) - (meanTotalN_kevals*meanTotalN_kevals));
	stdSumParallelN_kevals = check_var((stdSumParallelN_kevals/counterReps) - (meanSumParallelN_kevals*meanSumParallelN_kevals));
	stdSumSequentialN_kevals = check_var((stdSumSequentialN_kevals/counterReps) - (meanSumSequentialN_kevals*meanSumSequentialN_kevals));
	
	FILE* repStats = fopen(repetitionsDistributedResults, "w");		
	fprintf(repStats, "Performance Statistics for the Distributed Model\n",reps,partitionsReps,counterReps);
	fprintf(repStats, "N° Trials: %d = %d  training sample realizations X %d partitions\n",counterReps,reps,partitionsReps);
	
	fprintf(repStats, "Problem: %s\n",name);
	if (param.svm_type == CVM){
		fprintf(repStats, "Model-Type: OVO-CVM\n");
	} else if (param.svm_type == C_SVC){
		fprintf(repStats, "Model-Type: OVO-CSVC\n");
	} else {
		fprintf(repStats, "Model-Type: MCVM (Modified Asharaf, Codes ADSVM)\n");
	}
	fprintf(repStats, "C=%E, Gamma=%E, EPS-Balls=%E\n",param.C,param.gamma,param.eps);

	char partitionModeString[20];
	if (partition_mode == RANDOM){
			sprintf(partitionModeString,"Random-Simple");	
	} else if (partition_mode == RANDOM_WEIGHTED){
			sprintf(partitionModeString,"Random-Weighted");	
	} else if (partition_mode == KMEANS){
			sprintf(partitionModeString,"K-means");	
	} else if (partition_mode == PREPART){
			sprintf(partitionModeString,"Pre-computed");	
	}
	
	fprintf(repStats, "N°Nodes=%d, Partition-Mode = %s\n",nnodes,partitionModeString);
	if (partition_mode == KMEANS && group_classes_in_kmeans){
	fprintf(repStats, "K-means done grouping the classes\n");
	} else if (partition_mode == KMEANS && !group_classes_in_kmeans){
	fprintf(repStats, "K-means done WITHOUT grouping the classes\n");
	}
	
	if (balance_partition){
	fprintf(repStats, "Minimal-Balancing applied with Minimum = %d\n",minimum_in_balancing_partition);
	} else {
	fprintf(repStats, "Minimal-Balancing was NOT applied.\n\n");
	}

	fprintf(repStats, "\nSUMMARIZED-STATISTICS (*see below for human-friendly format):\n");
	fprintf(repStats, ">ACCURACY:\n");
	fprintf(repStats, "%g\n",meanAccuracy);
	fprintf(repStats, "%g\n",stdAccuracy);
	fprintf(repStats, ">SIZES:\n");
	fprintf(repStats, "%g\n",mean_sizes);
	fprintf(repStats, "%g\n",std_sizes);
	
	fprintf(repStats, ">COMPRESSIONS:\n");	
	fprintf(repStats, "%E  %E\n",meanTotalCompression,meanNodeCompression);
	fprintf(repStats, "%E  %E\n",stdTotalCompression,stdNodeCompression);
	fprintf(repStats, ">KERNEL-CALLS:\n");	
	fprintf(repStats, "%E  %E  %E  %E  %E  %E\n",meanTotalN_kevals,meanParN_kevals,meanCoordN_kevals,meanNodeN_kevals,meanSumSequentialN_kevals,meanSumParallelN_kevals);
	fprintf(repStats, "%E  %E  %E  %E  %E  %E\n",stdTotalN_kevals,stdParN_kevals,stdCoordN_kevals,stdNodeN_kevals,stdSumSequentialN_kevals,stdSumParallelN_kevals);
	fprintf(repStats, ">TIMES:\n");	
	fprintf(repStats, "%E  %E  %E  %E  %E  %E\n",meanTotalTime,meanParTime,meanCoordTime,meanNodeTime,meanSumSequential,meanSumParallel);
	fprintf(repStats, "%E  %E  %E  %E  %E  %E\n",stdTotalTime,stdParTime,stdCoordTime,stdNodeTime,stdSumSequential,stdSumParallel);
	
	fprintf(repStats, "\n");
	
	fprintf(repStats, "Mean- Correct-Classification-Rates = %g\n",meanAccuracy);
	fprintf(repStats, "Stdv- Correct-Classification-Rates = %g\n",stdAccuracy);
	
	fprintf(repStats, "Means Correct-Classification Rates by Class\n");
	for(int i=0; i < nr_class; i++){
		fprintf(repStats, "class %d: %g, ",labels[i], error_by_class[i]);
	}
	
	fprintf(repStats, "Mean- Coord.Model Size= %g\n",mean_sizes);
	fprintf(repStats, "Stdv- Coord.Model Size = %g\n",std_sizes);
		
	fprintf(repStats, "Mean- Total-Compressions = %E\n",meanTotalCompression);
	fprintf(repStats, "Stdv- Total-Compressions = %E\n",stdTotalCompression);
	
	fprintf(repStats, "Mean- Node-Compressions = %E\n",meanNodeCompression);
	fprintf(repStats, "Stdv- Node-Compressions = %E\n",stdNodeCompression);
	
	fprintf(repStats, "Mean- SumNodes-Times = %E\n",meanTotalTime);
	fprintf(repStats, "Stdv- SumNodes-Times = %E\n",stdTotalTime);
	
	fprintf(repStats, "Mean- MaxNodes-Times = %E\n",meanParTime);
	fprintf(repStats, "Stdv- MaxNodes-Times = %E\n",stdParTime);
	
	fprintf(repStats, "Mean- Coord-Times = %E\n",meanCoordTime);
	fprintf(repStats, "Stdv- Coord-Times = %E\n",stdCoordTime);
	
	fprintf(repStats, "Mean- Node-Times = %E\n",meanNodeTime);
	fprintf(repStats, "Stdv- Node-Times = %E\n",stdNodeTime);
	
	fprintf(repStats, "Mean- Sum SumNodes+Coord-Times = %E\n", meanSumSequential);
	fprintf(repStats, "Stdv- Sum SumNodes+Coord-Times = %E\n", stdSumSequential);
	
	fprintf(repStats, "Mean- Sum MaxNodes+Coord-Times = %E\n", meanSumParallel);
	fprintf(repStats, "Stdv- Sum MaxNodes+Coord-Times = %E\n", stdSumParallel);
	
	fprintf(repStats, "Intra-var Time = %g\n",(intraVarRemoteTimes/reps));
	fprintf(repStats, "Intra-var Compression: %g\n",(intraVarRemoteCompressions/reps));
	
	fprintf(repStats, "Mean- SumNodes-Kernel-Calls = %E\n",meanTotalN_kevals);
	fprintf(repStats, "Stdv- SumNodes-Kernel-Calls = %E\n",stdTotalN_kevals);
	
	fprintf(repStats, "Mean- MaxNodes-Kernel-Calls = %E\n",meanParN_kevals);
	fprintf(repStats, "Stdv- MaxNodes-Kernel-Calls = %E\n",stdParN_kevals);
	
	fprintf(repStats, "Mean- Coord-Kernel-Calls = %E\n",meanCoordN_kevals);
	fprintf(repStats, "Stdv- Coord-Kernel-Calls = %E\n",stdCoordN_kevals);
	
	fprintf(repStats, "Mean- Node-Kernel-Calls = %E\n",meanNodeN_kevals);
	fprintf(repStats, "Stdv- Node-Kernel-Calls = %E\n",stdNodeN_kevals);
	
	fprintf(repStats, "Mean- Sum SumNodes+Coord-Kernel-Calls = %E\n", meanSumSequentialN_kevals);
	fprintf(repStats, "Stdv- Sum SumNodes+Coord-Kernel-Calls = %E\n", stdSumSequentialN_kevals);
	
	fprintf(repStats, "Mean- Sum MaxNodes+Coord-Kernel-Calls = %E\n", meanSumParallelN_kevals);
	fprintf(repStats, "Stdv- Sum MaxNodes+Coord-Kernel-Calls = %E\n", stdSumParallelN_kevals);
	
	fprintf(repStats, "Intra-var Kernel-Calls = %g\n",(intraVarN_kvals/counterReps));
	
	fclose(repStats);
		
	svm_destroy_param(&param);
	free(accuracies);
	free(coordTimes);
	free(nodeTimes);
	free(parTimes);
	free(totalTimes);
	free(totalCompressions);
	free(nodeCompressions);
	free(coordN_kevals);
	free(nodeN_kevals);
	free(parN_kevals);
	free(totalN_kevals);
	free(coordSizes);
	
	
		
}//end repetitions distributed case

void gridCV(double* cList, double* gammaList, int nC, int nG, char* name){

double currentPerf;
double bestPerf =0; 
double bestC;
double bestG;
double bestNSVs;

char CVGrid_file_name[2*FILENAME_LEN];    
const char *error_msg;
int reps = nsamplerep;

char stamp[64];
sprintf(stamp,"%s",give_me_the_time());
printf("bien\n");
char training_alg[FILENAME_LEN];
char kernel_type_string[25];
stamp_algorithm_name(training_alg);
stamp_kernel_type(kernel_type_string);

if (param.svm_type == CVM){
		
	sprintf(CVGrid_file_name,"%s.CV10.CVM-OVO-model-ALG-%s-%s-%s",name,kernel_type_string,training_alg,stamp);
	if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.CV10.CVM-OVO-model.resource-usage.txt",name);
	}

} else if (param.svm_type == C_SVC) {
	sprintf(CVGrid_file_name,"%s.CV10.CSVC-OVO-model-ALG-%s-%s-%s",name,kernel_type_string,training_alg,stamp);
	if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.CV10.CSVC-OVO-model.resource-usage.txt",name);
	}
} else {

	sprintf(CVGrid_file_name,"%s.CV10.MCVM-model-ALG-%s-%s-%s",name,kernel_type_string,training_alg,stamp);
	if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.CV10.MCVM-model.resource-usage.txt",name);
	}
}

FILE* CVResults_File = fopen(CVGrid_file_name, "a+t");			
fprintf(CVResults_File, "***THIS IS CROSS-VALIDATION: columns\n C, gamma, accuracy, average number of svs, average kernel evaluations, average REQUESTED kernel evaluations\n");
fclose(CVResults_File);
				
//read_problem(name);
//check_parameters();
//error_msg = svm_check_parameter(&prob,&param);
//gammaList[0] = param.gamma;
gammaList[0] = -1;

	
	//Search for Hyperparameters
	for(int cit=0; cit < nC; cit++){
		for(int git=0; git < nG; git++){
				
				double average_num_svs;
				double av_real_kevals;
				double av_requested_kevals;
				
				param.C = cList[cit];
				param.gamma = gammaList[git];
				currentPerf = 0.0;

				for(int repIt=0; repIt < reps; repIt++){

					char problemRepName[FILENAME_LEN];
					char testRepName[FILENAME_LEN];
					char predictionsRepName[FILENAME_LEN];
					char modelName[2048];

					double currentAccuracy;

					sprintf(problemRepName,"%s.train.0%d.txt",name,repIt+1);
					sprintf(testRepName,"%s.test.0%d.txt",name,repIt+1);
					sprintf(predictionsRepName,"%s.predictions.0%d.txt",name,repIt+1);
					printf("params %g %g\n", param.C, param.gamma);
					read_problem(problemRepName);

					if(show_resource_usage){
						char *message = new char[50];
						sprintf(message, "rep.centralized - iteration %d",repIt);
						show_memory_usage(message);
					}
					printf ("Finish reading input files!\n");
					check_parameters();
					printf ("Finish checking parameters!\n");

					error_msg = svm_check_parameter(&prob,&param);
					if(error_msg)
					{
						fprintf(stderr,"Error checking parameters%s\n",error_msg);
						exit(1);
					}

					//  COMPUTE CROSS-VALIDATION ERROR
					currentAccuracy = do_cross_validation(average_num_svs,av_real_kevals,av_requested_kevals);
					currentPerf += (currentAccuracy/reps);
	//			currentPerf = do_cross_validation(average_num_svs,av_real_kevals,av_requested_kevals);
				}

				param.C = cList[cit];
				param.gamma = gammaList[git];

				FILE* CVResults_File = fopen(CVGrid_file_name, "a+t");			
				fprintf(CVResults_File, "%g, %g, %g, %g, %g, %g\n",param.C,param.gamma,currentPerf,average_num_svs,av_real_kevals,av_requested_kevals);
				fclose(CVResults_File);
				if (currentPerf >= bestPerf){
				bestPerf = currentPerf; 
				bestC = param.C;
				bestG = param.gamma;
				bestNSVs = average_num_svs;
				//printf("Cross-Validation Grid: New Best-Values are C=%g, G=%g, Accuracy=%g\n",bestC,bestG,currentPerf);
				}
				
				if(show_resource_usage){
					char *message = new char[50];
					sprintf(message, "iteration of crossval-grid %d x %d (of %d x %d)",cit,git,nC,nG);
					show_memory_usage(message);
				}	
		}
	} 
	
	CVResults_File = fopen(CVGrid_file_name, "a+t");		
	fprintf(CVResults_File, "BEST C=%g,Gamma=%g,Accuracy=%g,SV=%g\n",bestC,bestG,bestPerf,bestNSVs);
	fclose(CVResults_File);
	
}

void gridVal(double* cList, double* gammaList, int nC, int nG, char* name){

double currentPerf=0;
double bestPerf = 0; 
double bestC;
double bestG;
int bestNSVs;
char CVGrid_file_name[FILENAME_LEN];    
char trainName[FILENAME_LEN];
char valName[FILENAME_LEN];
char predictionsName[FILENAME_LEN];
const char *error_msg;

sprintf(trainName,"%s.train.vgrid.txt",name);
sprintf(valName,"%s.val.vgrid.txt",name);
sprintf(predictionsName,"%s.predictions.vgrid.txt",name);

char stamp[64];
sprintf(stamp,"%s",give_me_the_time());

char training_alg[FILENAME_LEN];
stamp_algorithm_name(training_alg);

char kernel_type_string[25];
stamp_kernel_type(kernel_type_string);

if (param.svm_type == CVM){
		
	sprintf(CVGrid_file_name,"%s.VGRID.CVM-OVO-model-ALG-%s-%s-%s",name,kernel_type_string,training_alg,stamp);
	if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.VGRID.CVM-OVO-model.resource-usage.txt",name);
	}

} else if (param.svm_type == C_SVC) {
	sprintf(CVGrid_file_name,"%s.VGRID.CSVC-OVO-model-ALG-%s-%s-%s",name,kernel_type_string,training_alg,stamp);
	if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.VGRID.CSVC-OVO-model.resource-usage.txt",name);
	}
} else {

	sprintf(CVGrid_file_name,"%s.VGRID.MCVM-model-ALG-%s-%s-%s",name,kernel_type_string,training_alg,stamp);
	if(file_resource_info_set == false){
			sprintf(resource_info_name,"%s.VGRID.MCVM-model.resource-usage.txt",name);
	}
	
}
		
read_problem(trainName);
check_parameters();
error_msg = svm_check_parameter(&prob,&param);	
gammaList[0] = param.gamma;

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}
	
printf ("Validation Grid: Starting search for hyper-parameters!\n");
	
	//Search for Hyperparameters
	for(int cit=0; cit < nC; cit++){
		for(int git=0; git < nG; git++){
			
				int num_svs=0;
				param.C = cList[cit];
				param.gamma = gammaList[git];
				
				Kernel::reset_real_kevals();
				Kernel::reset_requested_kevals();
	
				if (trainCentralized){				
					printf("validation with a centralized model\n");
					model = svm_train(&prob,&param);
				} else {
				
					printf("validation with a distributed model\n");
					struct distrib_statistics statsDist;
					bool flag = false;
					Partition data(nnodes,partition_mode,&prob,NULL,NULL,pweights,group_classes_in_kmeans,balance_partition,minimum_in_balancing_partition);
					model = svm_simulate_distributed(&prob,&param,&data,&statsDist,flag,name);
				
					
				}
				
				double n_real_kevals = ((double)Kernel::get_real_kevals())/(1.0);
				double n_requested_kevals = ((double)Kernel::get_requested_kevals())/(1.0);
		
				if(show_resource_usage){
					char *message = new char[50];
					sprintf(message, "iteration of validation-grid %d x %d (of %d x %d)",cit,git,nC,nG);
					show_memory_usage(message);
				}	
	
				num_svs = model->l; 
				printf("beginning to predict\n");
				FILE* testF = fopen(valName,"r");
				FILE* predF = fopen(predictionsName,"w");
	
				currentPerf = predictOneRep(testF, predF);
				
				//printf("current Perf: %g\n",currentPerf);
				FILE* CVResults_File = fopen(CVGrid_file_name, "a+t");			
				fprintf(CVResults_File, "%g, %g, %g, %d, %g, %g\n",param.C,param.gamma,currentPerf,num_svs,n_real_kevals,n_requested_kevals);
				fclose(CVResults_File);
				
				if (currentPerf >= bestPerf){
				bestPerf = currentPerf; 
				bestC = param.C;
				bestG = param.gamma;
				bestNSVs = num_svs;
				//printf("Validation Grid: New Best-Values are C=%g, G=%g, Accuracy=%g\n",bestC,bestG,currentPerf);
				}
				
				svm_destroy_model(model);
				
		}
	} 
	
	FILE* CVResults_File = fopen(CVGrid_file_name, "a+t");		
	fprintf(CVResults_File, "BEST C=%g,Gamma=%g,Accuracy=%g,SV=%d\n",bestC,bestG,bestPerf,bestNSVs);
	fclose(CVResults_File);
	
}


svm_model* svm_simulate_distributed(const svm_problem *prob, const svm_parameter *param, Partition *partition, distrib_statistics *stats, bool printDistribStats,  char* name){

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
	
	Kernel::reset_requested_kevals();
		
		remote_models[k] = svm_train(&subprob,param);
	
	n_kevals[k] = Kernel::get_requested_kevals();
	
	//////////////////////////////////////////////////
	 
	if(show_resource_usage){
		char *message = new char[50];
		sprintf(message, "simulated distributed - submodel %d",k);
		show_memory_usage(message);
	}	
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
		stats->max_remote_N_kevals = max((double)n_kevals[k],stats->max_remote_N_kevals);
		stats->sum_remote_N_kevals += (double)n_kevals[k]; 
		
		//destroy local model
		svm_destroy_model(submodel);
	}
	
	sumRemoteTime = meanRemoteTime;
	meanRemoteTime = meanRemoteTime/nnodes;
		
	stdRemoteTime = check_var((stdRemoteTime/nnodes) - (meanRemoteTime*meanRemoteTime));
	
	meanNodeCompression = meanNodeCompression/nnodes; 
	stdNodeCompression = check_var((stdNodeCompression/nnodes) - (meanNodeCompression*meanNodeCompression)); 
	
	stats->mean_remote_N_kevals /= nnodes;
	stats->std_remote_N_kevals /= nnodes;
	stats->std_remote_N_kevals -= (stats->mean_remote_N_kevals*stats->mean_remote_N_kevals);
	stats->std_remote_N_kevals = check_var(stats->std_remote_N_kevals);
	
	free(n_kevals);
		
	///////////////// TRAINING IN THE CENTRALIZED NODE ////////////////////////
	printf("Training with the union of the coresets\n");
	
	clock_t startTime = clock();
	Kernel::reset_requested_kevals();
		
		dist_model = svm_train(&problem_coordinator,param);
	
	stats->N_kevals = (double)Kernel::get_requested_kevals();
	clock_t endTime = clock();
	

	if (dist_model == NULL){
		printf("error: modelo NULL despues de llamada a svm_train\n");
		exit(1);
	}
	
	/////////////////////////////////////////////////////////////////////////////
	
	printf("Distributed solution finished ... \n");
	double totalTime = (double)(endTime - startTime)/CLOCKS_PER_SEC;
	info("Total COORD CPU-TIME = %g seconds\n", totalTime);
	info("Total REMOTE KERNEL-CALLS = %g calls\n", stats->sum_remote_N_kevals);
	info("MAX REMOTE KERNEL-CALLS = %g calls\n", stats->max_remote_N_kevals);
	info("Total REMOTE TIME = %g calls\n", sumRemoteTime);
	info("MAX REMOTE TIME = %g calls\n",maxRemoteTime);
	
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
	stats->size_cent_model = dist_model->l;
	//destroy coordinator problem
	free(problem_coordinator.x);
	free(problem_coordinator.y);
	
	if (printDistribStats){
	
	char Distrib_Statistics_file_name[FILENAME_LEN];    
	sprintf(Distrib_Statistics_file_name,"%s.DistribStats.txt",name);
	FILE* DistribStats = fopen(Distrib_Statistics_file_name, "w");			
	fprintf(DistribStats, "Coordinator Time= %E \n",stats->trainingTime);
	fprintf(DistribStats, "Total Compression = %E\n",stats->totalCompression);
	fprintf(DistribStats, "Mean Node Compression = %E\n",stats->meanNodeCompression);
	fprintf(DistribStats, "Std Node Compression = %E\n",stats->stdNodeCompression);
	fprintf(DistribStats, "Mean Remote Node Time = %E\n",stats->meanRemoteTime);
	fprintf(DistribStats, "Sts Remote Node Time =%E\n",stats->stdRemoteTime);
	fclose(DistribStats);
	}
		
	return dist_model;	
}//end simulate distributed training

double predictOneRep(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	
	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	
	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;
	
	int type, dim;

	type = 0; // sparse format
	dim = 0;
	j = 0;
	double cumTime=0;
	
	for(int c = fgetc(input); (c != EOF) && (dim == 0); c = fgetc(input))
	{
		
		switch(c)
		{
			case '\n':
				dim = j;
				break;

			case ':':
				++j;
				break;

			case ',':
				++j;
				type = 1;
				break;

			default:
				;
		}
	}
	rewind(input);

	while(1)
	{
		double start = getRunTime();
		double duration;
	
		int i = 0;
		int c;
		double target,v;

		if (type == 0) // sparse format
		{
			if (fscanf(input,"%lf",&target) == EOF)
				break;
		}
		else if (type == 1) // dense format
		{
			c = getc(input);

			if (c == EOF)
			{
				break;
			}
			else
			{
				ungetc(c,input);
			}
		}
		
		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
				if(show_resource_usage){
					char *message = new char[50];
					sprintf(message, "predicting-reallocating memory");
					show_memory_usage(message);
				}	
				
			}

			do {
				c = getc(input);
				if((c=='\n') || (c==EOF)) break;
			} while(isspace(c));
			if((c=='\n') || (c==EOF)) break;

			ungetc(c,input);
			
			if (type == 0) // sparse format
			{
#ifdef INT_FEAT
				int tmpindex;
				int tmpvalue;
				fscanf(input,"%d:%d",&tmpindex,&tmpvalue);
                x[i].index = tmpindex;
				x[i].value = tmpvalue;
#else
				
				fscanf(input,"%d:%lf",&x[i].index,&x[i].value);
#endif
				++i;
			}
			else if ((type == 1) && (i < dim)) // dense format, read a feature
			{
				x[i].index = i+1;
#ifdef INT_FEAT
				int tmpvalue;
                fscanf(input, "%d,", &tmpvalue);
				x[i].value = tmpvalue;
#else
				fscanf(input, "%lf,", &(x[i].value));
#endif
				++i;
			}
			else if ((type == 1) && (i >= dim)) // dense format, read the label
			{
				fscanf(input,"%lf",&target);
			}
		}	

		//printf("one testing example read\n");
		x[i++].index = -1;
		//printf("predicting a model: %d\n",model->param.svm_type);	
		v = svm_predict(model,x);
		//printf("end prediction\n");
		double finish = getRunTime();	
        duration = (double)(finish - start);
		cumTime += duration;
		if (total%5000==0)
		printf("%d predictions, %d correct, CUMTIME:%g\n",total,correct,cumTime);
		//fprintf(output,"prediction: %g real: %g\n",v,target);

		if(v == target)
			++correct;
			error += (v-target)*(v-target);
			sumv += v;
			sumy += target;
			sumvv += v*v;
			sumyy += target*target;
			sumvy += v*target;
			++total;
		}
		
	printf("Accuracy = %g%% (%d/%d) (classification)\n",
		(double)correct/total*100,correct,total);
	printf("Mean squared error = %g (regression)\n",error/total);
	free(x);
	double perfObtained =  (double)correct/total;
	return perfObtained*100;
}

double predictOneRep2(FILE *input, FILE *output, double *classification_error_by_class)
{
	int correct = 0;
	int *correct_by_class, *total_test_by_class;
	int total = 0;
	double error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	
	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	
	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;


	correct_by_class = new int[nr_class];
	total_test_by_class = new int[nr_class];
	

	for(int i=0; i < nr_class; i++){

		classification_error_by_class[i]=0.0;
		correct_by_class[i] = 0;
		total_test_by_class[i] = 0;
	}	
	int type, dim;

	type = 0; // sparse format
	dim = 0;
	j = 0;
	double cumTime=0;
	
	for(int c = fgetc(input); (c != EOF) && (dim == 0); c = fgetc(input))
	{
		
		switch(c)
		{
			case '\n':
				dim = j;
				break;

			case ':':
				++j;
				break;

			case ',':
				++j;
				type = 1;
				break;

			default:
				;
		}
	}
	rewind(input);

	while(1)
	{
		double start = getRunTime();
		double duration;
	
		int i = 0;
		int c;
		double target,v;

		if (type == 0) // sparse format
		{
			if (fscanf(input,"%lf",&target) == EOF)
				break;
		}
		else if (type == 1) // dense format
		{
			c = getc(input);

			if (c == EOF)
			{
				break;
			}
			else
			{
				ungetc(c,input);
			}
		}
		
		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			do {
				c = getc(input);
				if((c=='\n') || (c==EOF)) break;
			} while(isspace(c));
			if((c=='\n') || (c==EOF)) break;

			ungetc(c,input);
			
			if (type == 0) // sparse format
			{
#ifdef INT_FEAT
				int tmpindex;
				int tmpvalue;
				fscanf(input,"%d:%d",&tmpindex,&tmpvalue);
                x[i].index = tmpindex;
				x[i].value = tmpvalue;
#else
				
				fscanf(input,"%d:%lf",&x[i].index,&x[i].value);
#endif
				++i;
			}
			else if ((type == 1) && (i < dim)) // dense format, read a feature
			{
				x[i].index = i+1;
#ifdef INT_FEAT
				int tmpvalue;
                fscanf(input, "%d,", &tmpvalue);
				x[i].value = tmpvalue;
#else
				fscanf(input, "%lf,", &(x[i].value));
#endif
				++i;
			}
			else if ((type == 1) && (i >= dim)) // dense format, read the label
			{
				fscanf(input,"%lf",&target);
			}
		}	

		//printf("one testing example read\n");
		x[i++].index = -1;
		//printf("predicting a model: %d\n",model->param.svm_type);	
		v = svm_predict(model,x);
		//printf("end prediction\n");
		double finish = getRunTime();	
        duration = (double)(finish - start);
		cumTime += duration;
		if (total%5000==0)
		printf("%d predictions, %d correct, CUMTIME:%g\n",total,correct,cumTime);
		//fprintf(output,"prediction: %g real: %g\n",v,target);

		if(v == target)	
			++correct;
			
		error += (v-target)*(v-target);
		sumv += v;
		sumy += target;
		sumvv += v*v;
		sumyy += target*target;
		sumvy += v*target;
		++total;
		
		
		//compute the error_by_class
		for(int i=0; i < nr_class; i++){
				if(target == model->label[i]){		
					total_test_by_class[i]+=1;
					if(v == target)
						correct_by_class[i]+=1;	
				}
		}
		
	}
		
	printf("Accuracy = %g%% (%d/%d) (classification)\n", (double)correct/total*100,correct,total);
	printf("Mean squared error = %g (regression)\n",error/total);
	
	printf("Accuracy by class:\n");	
	for(int i=0; i < nr_class; i++){
		classification_error_by_class[i] = ((double)correct_by_class[i]/total_test_by_class[i])*100; 
		printf("class %d: %g%%  ",model->label[i],classification_error_by_class[i]);
	}
	printf("\n");	
	free(x);
	double perfObtained =  (double)correct/ ((double)total);
	return perfObtained*100;
}

double do_cross_validation(double &meansvs,double &mean_real_kevals, double &mean_requested_kevals)
{
	int i;
	int total_correct  = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	
	svm_cross_validation_2(&prob,&param,nr_fold,target,meansvs,mean_real_kevals,mean_requested_kevals);
	
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	
	free(target);
	return total_error/prob.l;
			
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	

	free(target);

	return  100.0*total_correct/prob.l;
	}
	
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;

	nnodes = 5;
	nr_fold = 10;
	param.exp_type     = 0;
	param.svm_type = CVM; 
	trainDistributed = false;
	trainCentralized = true;
	param.kernel_type  = RBF;
	param.degree       = 3;
	param.gamma        = -1;	
	param.coef0        = 0;
	param.nu           = 0.5;
	param.mu           = 0.02;
	param.cache_size   = 500;//1GB	
	param.C            = INVALID_C;
	param.eps          = 1e-3;
	param.p            = 0.1;
	param.shrinking    = 1;
	param.probability  = 0;
	param.nr_weight    = 0;
	param.weight_label = NULL;
	param.weight       = NULL;
	param.sample_size  = 60;
	param.num_basis    = 50000;
	estimate_cross_validation_error   = false;
	partition_mode     = RANDOM;
	param.max_iterations = INFINITY;
	bool epsIsSet      = false;
	save_models_when_computing_stats = false;
	nsamplerep = 1;
	npartrep = 20;
	param.nrclasses = -1;
	param.MEB_algorithm = BADOUCLARKSON;
	param.cooling = true;
	param.randomized = true;
	param.stop_with_real_meb = true;
	param.safe_stopping = false;
	param.nsamplings_safe_stopping = 1;
	param.nsamplings_iterations = 1;
	param.MEB_init_method = RANDOM_MEB_INIT;
	param.mcvm_type = ADNORM;//ADSVM;
	param.frecuency_messages = 500;
	grid_cv = false;
	group_classes_in_kmeans=false;
	balance_partition=false;//set at true only for kmeans mode at least the user say other thing
	minimum_in_balancing_partition=0;
    searchC = true;
    searchG = false; 
    minpowC = -4, maxpowC = +4; baseC = 2;
    fixed_test_in_sample_repetitions = false;
    fixed_train_in_sample_repetitions = false;

    
	show_resource_usage=false;
	write_resource_info_to_file=false;
	file_resource_info_set = false;
	
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'f':
				param.num_basis = atoi(argv[i]);
				break;
			case 'l':
				nnodes = atoi(argv[i]);
				break;
			case 'E':
				param.exp_type = atoi(argv[i]);
				break;
			case 'S':
				if(argv[i-1][2]=='M'){
					//all the models will be saved when computing statistics;						
					save_models_when_computing_stats = true;
				} else if(argv[i-1][2]=='C'){
					//separate classes for doing k-means
					group_classes_in_kmeans = true;
				} else if(argv[i-1][2]=='S'){
					//safe stopping
					param.safe_stopping = true;
					param.nsamplings_safe_stopping = atoi(argv[i]);
				} 
				break;
			case 's':
				param.svm_type = atoi(argv[i]); 				
				if (!epsIsSet && (param.svm_type == CVDD || param.svm_type == CVM || param.svm_type == CVM_LS 
					|| param.svm_type == CVR || param.svm_type == BVM || param.svm_type == MCVM ))
					param.eps = -1;
				break;
			case 'D':
				if (atoi(argv[i]) == 0){
					trainDistributed = false;
					trainCentralized = true;
				}
				else{
					trainDistributed = true;
					trainCentralized = false;
				}  
				break;	
			
			case 'P':
				partition_mode = atoi(argv[i]);
				if(partition_mode == RANDOM_WEIGHTED){
					pweights=new float[nnodes];
					for(int s=0; s < nnodes; s++)
						pweights[s] = atof(argv[++i]);
				}
				if(partition_mode == KMEANS){
					balance_partition = true;	
					minimum_in_balancing_partition = 1;	 
				}
				
				break;
			case 'M':
				if(argv[i-1][2]=='B'){
				balance_partition = true;	
				minimum_in_balancing_partition = atoi(argv[i]);	 
				printf("minimum in minimal-balancing set to %d\n",minimum_in_balancing_partition);
				}
				else if(argv[i-1][2]=='A'){
				param.MEB_algorithm = atoi(argv[i]);
				printf("MEB-Algorithm ...\n");
				switch(param.MEB_algorithm){
					case BADOUCLARKSON:
						printf("MEB-Algorithm selected: BADOU-CLARKSON\n");
						break;
					case YILDRIM1:
						printf("MEB-Algorithm selected: STANDARD FRANK-WOLFE\n");
						break;
					case YILDRIM2:
						printf("MEB-Algorithm selected: MODIFIED FRANK-WOLFE\n");
						break;
					case PARTAN:/*GOLOSO*/
						printf("MEB Algorithm selected: PARTANIZED FRANK-WOLFE\n");
						break;
					case SWAP:/*GOLOSO*/
						printf("MEB Algorithm selected: SWAP (PAIRWISE FRANK-WOLFE) SECOND ORDER\n");
						break;
					case lSWAP:/*LAZY SWAP*/
						printf("MEB Algorithm selected: SWAP (PAIRWISE FRANK-WOLFE) FIRST ORDER\n");
						break;
					case BVMtsang:/*BVM*/
						printf("MEB Algorithm selected: BVM Tsang, 2005.\n");
						break;
					case PANIGRAHY:/*PANIGRAHY*/
						printf("MEB Algorithm selected: Panigrahy.\n");
						break;
					default:
						param.MEB_algorithm = BADOUCLARKSON;
						printf("MEB-Algorithm selected: OTHER ALGORITHM\n");	
				}	 
				}
				break;
			case 'I':
				if(argv[i-1][2]=='M'){
					param.MEB_init_method =  atoi(argv[i]);	
				switch(param.MEB_init_method){
					case RANDOM_MEB_INIT:
						printf("MEB-initialization: RANDOM-SET + SMO\n");
						break;
					case YILDRIM_INIT:
						printf("MEB-initialization: RANDOM-POINT + FURTHEST POINT FROM THE FIRST\n");
						break;
					case YILDRIM_SAMPLE_INIT:
						printf("MEB-initialization: YILDRIM ON SAMPLE\n");
						break;
					default:
						param.MEB_init_method = RANDOM_MEB_INIT;
						printf("Invalid MEB-initialization method, using RANDOM-SET\n");
						break;
				}	
				}
				else if(argv[i-1][2]=='L'){	
					param.max_iterations = atoi(argv[i]);
					printf("Iteration limit:%d\n",param.max_iterations);	
				}
				break;
			case 'N':
				if(argv[i-1][2]=='F'){
					nr_fold = atoi(argv[i]);
					if(nr_fold < 2)
					{
						fprintf(stderr,"n-fold cross validation: n must >= 2\n");
						exit_with_help();
					}
				} else if (argv[i-1][2]=='S'){
					nsamplerep = atoi(argv[i]);
				} else if (argv[i-1][2]=='P'){
					npartrep = atoi(argv[i]);
				} else if (argv[i-1][2]=='N'){
					nnodes = atoi(argv[i]);
				}	
				break;	
			case 'F':
				if(argv[i-1][2]=='T'){
					fixed_test_in_sample_repetitions = true;
				}
				else if(argv[i-1][2]=='S'){
					fixed_train_in_sample_repetitions = true;
				} 
				else if(argv[i-1][2]=='M'){
					param.frecuency_messages = atoi(argv[i]);
					i=i+1;
				} 
				i=i-1;
				break;
			case 'y':
				param.mcvm_type = atoi(argv[i]);	
				break;	
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				Glist.push_back(param.gamma);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'u':
				param.mu = atof(argv[i]);
				break;			
			case 'c':
				reg_param = atof(argv[i]);
				Clist.push_back(reg_param);
				break;
			case 'C':
				if(argv[i-1][2]=='\0'){
					scale_param = atof(argv[i]);
				} else if (argv[i-1][2]=='O'){
					if(atoi(argv[i]) == 0)
						param.cooling = false;
					else 	
						param.cooling = true;
				}
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				epsIsSet = true;
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'v':
				estimate_cross_validation_error = true;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
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
				}
				break;	
			case 'R':
				if(argv[i-1][2]=='\0'){
					show_resource_usage=true;
					write_resource_info_to_file=true;
					sprintf(resource_info_name,"%s",argv[i]);
					file_resource_info_set = true;
				} else if (argv[i-1][2]=='S'){
					if(atoi(argv[i]) == 0)
						param.randomized = false;
					else 	
						param.randomized = true;
				} else if (argv[i-1][2]=='M'){
					if(atoi(argv[i]) == 0)
						param.stop_with_real_meb = false;
					else
						param.stop_with_real_meb = true;
				}
				 break;				 
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			case 'a':
				param.sample_size = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1) //Name file to save model 
	{
		strcpy(model_file_name,argv[i+1]);
	}
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		if (trainCentralized && param.svm_type == CVM)
			sprintf(model_file_name,"%s.centralized.OVO-CVM.model",p);
		else if (trainCentralized && param.svm_type == MCVM)
			sprintf(model_file_name,"%s.centralized.MCVM.model",p);
		else if (trainDistributed && param.svm_type == MCVM)
			sprintf(model_file_name,"%s.distributed.MCVM.model",p);
		else if (trainDistributed && param.svm_type == CVM)
			sprintf(model_file_name,"%s.distributed.OVO-CVM.model",p);
		else if (trainCentralized && param.svm_type == C_SVC)
			sprintf(model_file_name,"%s.centralized.OVO-CVC.model",p);
		else if (trainDistributed && param.svm_type == C_SVC)
			sprintf(model_file_name,"%s.distributed.OVO-CVC.model",p);
			
			
	}	
	
}

double CalRBFWidth()
{
	double sumDiagonal    = 0.0;
	double sumWholeKernel = 0.0;

	int inc = 1;
	int count = 0;
	int numData = prob.l;

	if (numData > 5000)
	{
		inc = (int)ceil(numData/5000.0);
	}

	for(int i=0; i<numData; i+=inc)
	{
		count++;

		for (int j=i; j<numData; j+=inc)
		{
			double dot = Kernel::dot(prob.x[i], prob.x[j]);
			if (j == i)
			{
				sumDiagonal    += dot;
				sumWholeKernel += dot;
			}
			else sumWholeKernel += 2.0*dot;
		}
	}

	double sum_all_pairs = 2.0*(((double)count*sumDiagonal) - (sumWholeKernel));
	double estimate = (double)sum_all_pairs/((double)count*(count-1)); 
	return estimate;
}

void count_pattern(FILE *fp, svm_problem &prob, int &elements, int &type, int &dim)
{
    int c;
    do
    {
	    c = fgetc(fp);
	    switch(c)
	    {
		    case '\n':
		    	c = fgetc(fp);
		    	if (c != EOF && c!= '\n'){
			    	++prob.l;
			    	ungetc(c,fp);
			    // fall through,
			    // count the '-1' element
			    if ((type == 1) && (dim == 0)) // dense format
			        dim = elements;				    
		    	}
			    break;

		    case ':':
			    ++elements;
			    break;

		    case ',':
			    ++elements;
			    type = 1;
			    break;

		    default:
			    ;
	    }
    } while  (c != EOF);
    rewind(fp);
}

double labelMapping(double val, double *orgLab, double *newLab, int numMap)
{
    for (int i=0; i<numMap; i++)
    {
        if (val == orgLab[i])        
            return newLab[i];        
    }
    return val;
}

void load_pattern(FILE *fp, svm_problem &prob, int type, int dim, int bidx, int eidx, int &max_index, int &j, double *orgLab=NULL, double *newLab=NULL, int numMap=0)
{

    for(int i=bidx; i<eidx; i++)
	{	
		if(i%5000==0){
			printf("%i datos cargados...\n",i);
		}
		double label;
		prob.x[i] = &x_space[j];
		if (type == 0) // sparse format
		{
			fscanf(fp,"%lf",&label);
			prob.y[i] = labelMapping(label,orgLab,newLab,numMap);
		}

		int elementsInRow = 0;
		while(1)
		{	
			int c;
			
			do {
				c = getc(fp);	
				if(c=='\n') break;
			} while(isspace(c));
			if(c=='\n') break;
			
			ungetc(c,fp);

			if (type == 0) // sparse format
			{
#ifdef INT_FEAT
				int tmpindex;
				int tmpvalue;
				fscanf(fp,"%d:%d",&tmpindex,&tmpvalue);
                x_space[j].index = tmpindex;
			    x_space[j].value = tmpvalue;
#else
				fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
#endif			
				++j;
			}
			else if ((type == 1) && (elementsInRow < dim)) // dense format, read a feature
			{
				x_space[j].index = elementsInRow+1;
				elementsInRow++;
#ifdef INT_FEAT
				int tmpvalue;
                fscanf(fp, "%d,", &tmpvalue);				
			    x_space[j].value = tmpvalue;
#else
				fscanf(fp, "%lf,", &(x_space[j].value));
#endif
				++j;
			}
			else if ((type == 1) && (elementsInRow >= dim)) // dense format, read the label
			{
                fscanf(fp,"%lf",&label);
				prob.y[i] = labelMapping(label,orgLab,newLab,numMap);
			}
		}	

		if(j>=1 && x_space[j-1].index > max_index)
			max_index = x_space[j-1].index;
		x_space[j++].index = -1;
		
//		if(true){
//			printf("showing a 100-th pattern\n");
//			svm_node * pattern = prob.x[i];
//			while(pattern->index != -1){
//			printf("dimension %d: value %lf\n",pattern->index,pattern->value);
//			pattern++;
//			}
//		}
	}
	printf("problem is of dimension: %d\n",max_index);
	prob.input_dim = max_index;
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

void check_parameters(){
	
int i;
int gamma_estimated = false;

if(param.gamma == 0.0){
		param.gamma = 1.0/max_index;
		
} else if (param.gamma < -0.5){
		param.gamma = 2.0/CalRBFWidth();
		gamma_estimated = true;
		printf("gamma : %g",param.gamma);
}

if ( param.svm_type == CVR )
	{
		param.C = (scale_param <= 0.0) ? 10000.0 : scale_param;
		if ( param.mu < 0.0 )
			param.mu = 0.02;
		
		double maxY = -INF, minY = INF;
		for (i=0; i<prob.l; i++)
		{
			maxY = max(maxY, prob.y[i]);
			minY = min(minY, prob.y[i]);
		}
		maxY     = max(maxY, -minY);
		param.C  = param.C *maxY;
		param.mu = param.mu*maxY;

		printf("MU %.16g, ", param.mu);
	}
	else if ( param.svm_type == MCVM )
	{
		param.C = (reg_param <= 0.0) ? 100.0 : param.C = reg_param;		
	
		int num_classes;
		
		if (param.nrclasses > 0){
			num_classes = param.nrclasses;
		}	
		else{
			
			int *label = NULL;
			determine_labels(&prob,&num_classes,&label);
			delete [] label;
			
		}	
					
		if (param.mcvm_type == ADSVM){
			
			param.sameclassYDP = (float) num_classes - 1.0;
			param.diffclassYDP = (float) -1.0;
			printf("Using ADSVM codes: sameclassYDP: %.6g, diffclassYDP: %.6g \n", param.sameclassYDP, param.diffclassYDP);
		}	
		else if (param.mcvm_type == ASHARAF){
		
			param.sameclassYDP = (float) 1.0;
			param.diffclassYDP = (float) (3.0*num_classes - 4.0)/(num_classes*(num_classes-1.0));
			printf("Using Asharaf codes: sameclassYDP: %.6g, diffclassYDP: %.6g  \n", param.sameclassYDP, param.diffclassYDP);
		}	
		else if (param.mcvm_type == ADNORM){
	
			param.sameclassYDP = (float) 1.0;
			param.diffclassYDP = (float) -1.0 / (num_classes-1);
			printf("Using ADSVM-Corrected Codes: sameclassYDP: %.6g, diffclassYDP: %.6g  \n", param.sameclassYDP, param.diffclassYDP);
		}	
		else if (param.mcvm_type == ANN){ 	
		
			param.sameclassYDP = (float) num_classes;
			param.diffclassYDP = (float) num_classes - 4.0;
			printf("Using ANN Codes: sameclassYDP: %.6g, diffclassYDP: %.6g  \n", param.sameclassYDP, param.diffclassYDP);
			
		}
		
		if(gamma_estimated){
		
			//param.gamma = (num_classes)*param.gamma;
			
		}
	}
	else if ( param.svm_type == CVM_LS ){
	
		param.C  = (scale_param <= 0.0) ? 10000.0 : scale_param;			
		param.mu = param.C/((reg_param < 0.0) ? 100.0 : reg_param)/prob.l;

		printf("MU %.16g, ", param.mu);
		
	}
	else // other SVM type		
	{
		param.C = (reg_param <= 0.0) ? 100.0 : param.C = reg_param;
	}

	
	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}	

	switch(param.kernel_type)
	{
		case NORMAL_POLY:
		case POLY:
			printf("Degree %d, coef0 %.16g, ", param.degree, param.coef0);
			break;
		case RBF:
		case EXP:
		case INV_DIST:
		case INV_SQDIST:
			printf("Gamma %.16g, ", param.gamma);
			break;
		case SIGMOID:
			printf("Gamma %.16g, coef0 %.16g, ", param.gamma, param.coef0);
			break;						
	}
	printf("C = %.16g\n", param.C);

}
/*
  corrected solution with dense format extension
*/
void read_problem(const char *filename)
{
	int elements, i, j;
	int type, dim;

	FILE *fp = fopen(filename,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

    // Support multiple data files and label renaming map
    bool mfilemode = false;
	int ch1 = getc(fp);
	if (ch1 == '#')
	{
		int ch2 = getc(fp);
		if (ch2 == 'm')
            mfilemode = true;
		do { ch2 = getc(fp); } while (ch2 != '\n');
	}
    else 
        ungetc(ch1,fp);

    if (mfilemode == true)
    {
        int numFile;     
        int numMap;
        fscanf(fp,"%d %d\n", &numFile, &numMap); 
        printf("#files : %d",numFile);
        char** filenames = new char*[numFile];
        int*   numPats   = new int[numFile];
        int fidx;
        for (fidx = 0; fidx<numFile; fidx++)
        {
            filenames[fidx] = new char[FILENAME_LEN];
            fscanf(fp,"%s\n", filenames[fidx]);
            printf(", %s", filenames[fidx]);
        }
        printf("\n");
        double *orgLab = new double[numMap];
        double *newLab = new double[numMap];
        for (int midx = 0; midx<numMap; midx++)
        {            
            fscanf(fp,"%lf>%lf\n", orgLab+midx, newLab+midx);
            printf("%g>%g\n", orgLab[midx],newLab[midx]);
        }

   	    prob.l   = 0;
	    elements = 0;
	    type     = 0; // sparse format
	    dim      = 0;
        for (fidx = 0; fidx<numFile; fidx++)
        {
            FILE *fp2 = fopen(filenames[fidx],"r+t");
            if(fp2 == NULL)
		        fprintf(stderr,"can't open input file %s\n",filenames[fidx]);	
	        else
    	        count_pattern(fp2, prob, elements, type, dim);
            numPats[fidx] = prob.l;
            fclose(fp2);
        }

        prob.y  = Malloc(double,prob.l);
	    prob.x  = Malloc(struct svm_node *,prob.l);
	    x_space = Malloc(struct svm_node,elements + prob.l);

	    if (!prob.y || !prob.x || !x_space)
	    {
		    fprintf(stdout, "ERROR: not enough memory!\n");

		    prob.l = 0;
		    return;
	    }

	    max_index = 0;
	    j         = 0;
        for (fidx = 0; fidx<numFile; fidx++)
        {
            FILE *fp2 = fopen(filenames[fidx],"r+t");
            if(fp2 == NULL)
		        fprintf(stderr,"can't open input file %s\n",filenames[fidx]);	
	        else
                load_pattern(fp2, prob, type, dim, (fidx>0)?numPats[fidx-1]:0, numPats[fidx],  max_index, j, orgLab, newLab, numMap);            
            fclose(fp2);
        }
        for (fidx = 0; fidx<numFile; fidx++)
            delete [] filenames[fidx];
        delete [] filenames;
        delete [] numPats;
        delete [] orgLab;
        delete [] newLab;
    }
    else
    {
	    prob.l   = 0;
	    elements = 0;
	    type     = 0; // sparse format
	    dim      = 0;
        count_pattern(fp, prob, elements, type, dim);
    	
	    prob.y  = Malloc(double,prob.l);
	    prob.x  = Malloc(struct svm_node *,prob.l);
	    x_space = Malloc(struct svm_node,elements+prob.l);

	    if (!prob.y || !prob.x || !x_space)
	    {
		    fprintf(stdout, "ERROR: not enough memory!\n");

		    prob.l = 0;
		    return;
	    }

	    max_index = 0;
	    j         = 0;
	    printf("loading patterns\n");
        load_pattern(fp, prob, type, dim, 0, prob.l, max_index, j);     
    }
    fclose(fp);
	printf("closing input files\n");
	
	}

