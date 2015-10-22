#include "rep_statistics.h"
#include <time.h>

rep_statistics_cent::rep_statistics_cent(char* prob_name_, svm_parameter *param_, int reps){

	accuracies = Malloc(double,reps);
	trTimes = Malloc(double,reps);
	n_kevals = Malloc(unsigned long int,reps);
	
	param = param_;	
	strcpy(prob_name,prob_name_);
		
	int stamp;
	
	srand(time(NULL));
	stamp = (int) rand()%1000000;
	
	
	switch(param->svm_type){
		case CVM:
			sprintf(stats_file,"%s.cent.OVO-CVM.repStats.%d.txt",prob_name,stamp);
			break;
		case C_SVC: 
			sprintf(stats_file,"%s.cent.OVO-CSVC.repStats.%d.txt",prob_name,stamp);
			break;
		case MCVM:
			sprintf(stats_file,"%s.cent.MCVM.repStats.%d.txt",prob_name,stamp);	
			break;
		default:
			sprintf(stats_file,"%s.cent.Other.repStats.%d.txt",prob_name,stamp);
	}
	
	
	idx_current = -1;
	allocated = reps;
	
}

rep_statistics_cent::~rep_statistics_cent(){

	free(accuracies);
	free(trTimes); 
	free(n_kevals);
	
}	

void rep_statistics_cent::add(double current_accuracy, double current_time, kevals_type current_k_evals){

		idx_current++;
		if(idx_current >= allocated){
			printf("attempt to write out of the statistics array limits\n");
			exit(1);
				
		}
	    accuracies[idx_current] = current_accuracy; 	
		trTimes[idx_current] = current_time;
		n_kevals[idx_current] = current_k_evals;
		
}


void rep_statistics_cent::compute_statistics(){

	meanAccuracy=0;
	stdAccuracy=0;
	meanTime=0;
	stdTime=0;
	meanN_kevals = 0;
	stdN_kevals = 0;
	
	for(int rep_it=0; rep_it < idx_current; rep_it++){
	
		meanAccuracy += accuracies[rep_it];
		stdAccuracy += accuracies[rep_it]*accuracies[rep_it];
		meanTime += trTimes[rep_it];
		meanN_kevals += (double)n_kevals[rep_it]; 
		stdTime += trTimes[rep_it]*trTimes[rep_it];
		stdN_kevals +=  (double)n_kevals[rep_it]*n_kevals[rep_it];
	}
	
	meanAccuracy = meanAccuracy/idx_current;
	stdAccuracy = sqrt_and_check_var((stdAccuracy/idx_current) - (meanAccuracy*meanAccuracy));
	meanTime = meanTime/idx_current;
	meanN_kevals = (double)meanN_kevals/idx_current;
	stdTime = sqrt_and_check_var((stdTime/idx_current) - (meanTime*meanTime));
	stdN_kevals = sqrt_and_check_var((stdN_kevals/idx_current) - (meanN_kevals*meanN_kevals));
	
}

void rep_statistics_cent::write(){

	FILE* repStats = fopen(stats_file, "w");		
	fprintf(repStats, "Statistics for Centralized Model, %d Repetitions\n",idx_current+1);
	fprintf(repStats, "PROBLEM: %s\n",prob_name);

	switch(param->svm_type){
		case CVM:
			fprintf(repStats, "Model-Type: OVO-CVM\n");
			break;
		case C_SVC: 
			fprintf(repStats, "Model-Type: OVO-CSVC\n");
			break;
		case MCVM:
			fprintf(repStats, "Model-Type: MCVM\n");
			break;
		default:
			fprintf(repStats, "Model-Type: Other\n");
	}
		
	fprintf(repStats, "C=%E, Gamma=%E, EPS-Balls=%E\n",param->C,param->gamma,param->eps);
	fprintf(repStats, "Mean- Correct-Classification-Rate = %g\n",meanAccuracy);
	fprintf(repStats, "Stdv- Correct-Classification-Rate = %g\n",stdAccuracy);
	
	fprintf(repStats, "Mean- TrainingTime = %E\n",meanTime);
	fprintf(repStats, "Stdv- TrainingTime = %E\n",stdTime);
	
	fprintf(repStats, "Mean- N°Kernel-Calls = %E\n",meanN_kevals);
	fprintf(repStats, "Stdv- N°Kernel-Calls = %E\n",stdN_kevals);
	
	fclose(repStats);

}

rep_statistics_dist::rep_statistics_dist(char *prob_name_, svm_parameter *param_, int reps){
 
	accuracies = new double [reps];
	coordTimes = new double [reps];
	nodeTimes = new double [reps];
	parTimes = new double [reps];
	totalTimes = new double [reps];
	totalCompressions = new double [reps];
	nodeCompressions = new double [reps];
	
	coordN_kevals = new double [reps];
	nodeN_kevals = new double [reps];
	parN_kevals = new double [reps];
	totalN_kevals = new double [reps];
	
	intraVarN_kvals = 0;	
	intraVarRemoteTimes = 0;
	intraVarRemoteCompressions = 0; 

	strcpy(prob_name,prob_name_);
	param = param_;	
	
	int stamp;
	
	srand(time(NULL));
	stamp = (int) rand()%1000000;
	
	
	switch(param->svm_type){
		case CVM:
			sprintf(stats_file,"%s.DIST.OVO-CVM.repStats.%d.txt",prob_name,stamp);
			break;
		case C_SVC: 
			sprintf(stats_file,"%s.DIST.OVO-CSVC.repStats.%d.txt",prob_name,stamp);
			break;
		case MCVM:
			sprintf(stats_file,"%s.DIST.MCVM.repStats.%d.txt",prob_name,stamp);	
			break;
		default:
				sprintf(stats_file,"%s.DIST.Other.repStats.%d.txt",prob_name,stamp);
	}
	
	idx_current = -1;
	allocated = reps;
	 
 }

rep_statistics_dist::~rep_statistics_dist(){

	delete [] accuracies;
	delete [] coordTimes;
	delete [] nodeTimes;
	delete [] parTimes;
	delete [] totalTimes;
	delete [] totalCompressions;
	delete [] nodeCompressions;
	delete [] coordN_kevals;
	delete [] nodeN_kevals;
	delete [] parN_kevals;
	delete [] totalN_kevals;
	
}

void rep_statistics_dist::add(double current_accuracy, struct distrib_statistics *one_sim_stats){
 
	idx_current++;
	if(idx_current >= allocated){
		printf("attempt to write out of the statistics array limits\n");
		exit(1);
			
	}
	
	accuracies[idx_current] = current_accuracy; //classification accuracy
	coordTimes[idx_current] =one_sim_stats->trainingTime;//solver time for coordinator problem 
	coordN_kevals[idx_current] = one_sim_stats->N_kevals;//calls to kernel function for coordinator problem
	totalCompressions[idx_current] = one_sim_stats->totalCompression;//fraction of total data imported 
	nodeCompressions[idx_current] = one_sim_stats->meanNodeCompression;//mean of the fraction of data imported from each node
	nodeTimes[idx_current] = one_sim_stats->meanRemoteTime;//mean of remote solver times
	nodeN_kevals[idx_current] = one_sim_stats->mean_remote_N_kevals;//mean of calls to kernel function in remote nodes
	parTimes[idx_current] = one_sim_stats->maxRemoteTime;//max of remote solver times
	parN_kevals[idx_current] = one_sim_stats->max_remote_N_kevals;//max of calls to kernel function in remote nodes
	totalTimes[idx_current] = one_sim_stats->sumRemoteTime;//sum of remote solver times
	totalN_kevals[idx_current] = one_sim_stats->sum_remote_N_kevals;//sum of calls to kernel function in remote nodes
		  
	intraVarRemoteTimes += (one_sim_stats->stdRemoteTime*one_sim_stats->stdRemoteTime);
	intraVarRemoteCompressions += (one_sim_stats->stdNodeCompression*one_sim_stats->stdNodeCompression);
	intraVarN_kvals += (one_sim_stats->std_remote_N_kevals*one_sim_stats->std_remote_N_kevals);
		
 
}
 
 
void rep_statistics_dist::compute_statistics(){
  
  	
	meanAccuracy = 0; stdAccuracy = 0;
	meanTotalCompression = 0; stdTotalCompression = 0;
	meanNodeCompression = 0; stdNodeCompression = 0;
	meanParTime = 0; meanTotalTime=0; stdParTime = 0; stdTotalTime=0;
	meanCoordTime = 0; stdCoordTime = 0;
	meanNodeTime = 0; stdNodeTime = 0;
	meanSumParallel = 0; meanSumSequential = 0;
	stdSumParallel = 0; stdSumSequential = 0;
	meanCoordN_kevals = 0; stdCoordN_kevals = 0;
	meanSumParallelN_kevals = 0; stdSumParallelN_kevals = 0;
	meanSumSequentialN_kevals = 0; stdSumSequentialN_kevals = 0;  
	meanNodeN_kevals = 0; stdNodeN_kevals = 0;
	meanParN_kevals = 0; meanTotalN_kevals = 0; stdParN_kevals = 0; stdTotalN_kevals = 0;
	
	for(int rep_it=0; rep_it < idx_current; rep_it++){
	
		meanAccuracy += accuracies[rep_it];
		stdAccuracy += accuracies[rep_it]*accuracies[rep_it];
		
		meanTotalCompression += totalCompressions[rep_it];
		meanNodeCompression += nodeCompressions[rep_it];
		meanCoordTime += coordTimes[rep_it];
		meanNodeTime += nodeTimes[rep_it];
		meanParTime += parTimes[rep_it];
		meanTotalTime += totalTimes[rep_it];
		meanSumParallel += parTimes[rep_it] + coordTimes[rep_it]; 
		meanSumSequential += totalTimes[rep_it] + coordTimes[rep_it];
		
		meanCoordN_kevals += coordN_kevals[rep_it];
		meanNodeN_kevals += nodeN_kevals[rep_it];
		meanParN_kevals += parN_kevals[rep_it];
		meanTotalN_kevals += totalN_kevals[rep_it];
		meanSumParallelN_kevals +=	parN_kevals[rep_it] + coordN_kevals[rep_it];
		meanSumSequentialN_kevals += totalN_kevals[rep_it] + coordN_kevals[rep_it];
			
		stdNodeCompression += (nodeCompressions[rep_it]*nodeCompressions[rep_it]); 
		stdTotalCompression += (totalCompressions[rep_it]*totalCompressions[rep_it]);
		stdParTime += (parTimes[rep_it]*parTimes[rep_it]); 
		stdTotalTime += (totalTimes[rep_it]*totalTimes[rep_it]); 
		stdNodeTime += (nodeTimes[rep_it]*nodeTimes[rep_it]);
		stdCoordTime += (coordTimes[rep_it]*coordTimes[rep_it]);
	    stdSumParallel += (parTimes[rep_it] + coordTimes[rep_it])*(parTimes[rep_it] + coordTimes[rep_it]); 
		stdSumSequential += (totalTimes[rep_it] + coordTimes[rep_it])*(totalTimes[rep_it] + coordTimes[rep_it]);
	
		stdCoordN_kevals += coordN_kevals[rep_it]*coordN_kevals[rep_it];
		stdNodeN_kevals += nodeN_kevals[rep_it]*nodeN_kevals[rep_it];
		stdParN_kevals += parN_kevals[rep_it]*parN_kevals[rep_it];
		stdTotalN_kevals += totalN_kevals[rep_it]*totalN_kevals[rep_it];
		stdSumParallelN_kevals += (parN_kevals[rep_it] + coordN_kevals[rep_it])*(parN_kevals[rep_it] + coordN_kevals[rep_it]);
		stdSumSequentialN_kevals += (totalN_kevals[rep_it] + coordN_kevals[rep_it])*(totalN_kevals[rep_it] + coordN_kevals[rep_it]);
	
	}	
	 
	meanAccuracy = meanAccuracy/idx_current;
	stdAccuracy = sqrt_and_check_var((stdAccuracy/idx_current) - (meanAccuracy*meanAccuracy));
	meanTotalCompression = meanTotalCompression/idx_current;
	meanNodeCompression = meanNodeCompression/idx_current;
	meanCoordTime = meanCoordTime/idx_current;
	meanNodeTime = meanNodeTime/idx_current;
	meanParTime = meanParTime/idx_current;
	meanTotalTime = meanTotalTime/idx_current;
	meanSumParallel = meanSumParallel/idx_current; 
	meanSumSequential = meanSumSequential/idx_current;
	
	meanCoordN_kevals = meanCoordN_kevals/idx_current;
	meanNodeN_kevals = meanNodeN_kevals/idx_current;
	meanParN_kevals = meanParN_kevals/idx_current;
	meanTotalN_kevals = meanTotalN_kevals/idx_current;
	meanSumParallelN_kevals =	meanSumParallelN_kevals/idx_current;
	meanSumSequentialN_kevals = meanSumSequentialN_kevals/idx_current;
		 
		
	stdNodeCompression = sqrt_and_check_var((intraVarRemoteCompressions/idx_current) + (stdNodeCompression/idx_current) - (meanNodeCompression*meanNodeCompression));  	
	stdTotalCompression = sqrt_and_check_var((stdTotalCompression/idx_current) - (meanTotalCompression*meanTotalCompression));
	stdCoordTime = sqrt_and_check_var((stdCoordTime/idx_current) - (meanCoordTime*meanCoordTime));
	stdNodeTime = sqrt_and_check_var((intraVarRemoteTimes/idx_current) + (stdNodeTime/idx_current) - (meanNodeTime*meanNodeTime)); 
	stdParTime = sqrt_and_check_var((stdParTime/idx_current) - (meanParTime*meanParTime)); 
	stdTotalTime = sqrt_and_check_var((stdTotalTime/idx_current) - (meanTotalTime*meanTotalTime));
    stdSumParallel = sqrt_and_check_var((stdSumParallel/idx_current) - (meanSumParallel*meanSumParallel));
    stdSumSequential = sqrt_and_check_var((stdSumSequential/idx_current) - (meanSumSequential*meanSumSequential));
    
	stdCoordN_kevals = sqrt_and_check_var((stdCoordN_kevals/idx_current) - (meanCoordN_kevals*meanCoordN_kevals));
	stdNodeN_kevals = sqrt_and_check_var((intraVarN_kvals/idx_current) + (stdNodeN_kevals/idx_current) - (meanNodeN_kevals*meanNodeN_kevals));
	stdParN_kevals = sqrt_and_check_var((stdParN_kevals/idx_current) - (meanParN_kevals*meanParN_kevals));
	stdTotalN_kevals = sqrt_and_check_var((stdTotalN_kevals/idx_current) - (meanTotalN_kevals*meanTotalN_kevals));
	stdSumParallelN_kevals = sqrt_and_check_var((stdSumParallelN_kevals/idx_current) - (meanSumParallelN_kevals*meanSumParallelN_kevals));
	stdSumSequentialN_kevals = sqrt_and_check_var((stdSumSequentialN_kevals/idx_current) - (meanSumSequentialN_kevals*meanSumSequentialN_kevals));
		
  
 }
 
 void rep_statistics_dist::write(){
  
  	FILE* repStats = fopen(stats_file, "w");		
	fprintf(repStats, "Performance Statistics for the Distributed Model\n");
	fprintf(repStats, "N° Trials: %d realizations of the synthetic dataset\n",idx_current+1);
	
	fprintf(repStats, "PROBLEM: %s\n",prob_name);
	
	switch(param->svm_type){
		case CVM:
			fprintf(repStats, "Model-Type: OVO-CVM\n");
			break;
		case C_SVC: 
			fprintf(repStats, "Model-Type: OVO-CSVC\n");
			break;
		case MCVM:
			fprintf(repStats, "Model-Type: MCVM\n");
			break;
		default:
			fprintf(repStats, "Model-Type: Other\n");
	}
	
	fprintf(repStats, "C=%E, Gamma=%E, EPS-Balls=%E\n",param->C,param->gamma,param->eps);
	
	fprintf(repStats, "\nSUMMARIZED-STATISTICS (*see below for human-friendly format):\n");
	fprintf(repStats, ">ACCURACY:\n");
	fprintf(repStats, "%g\n",meanAccuracy);
	fprintf(repStats, "%g\n",stdAccuracy);
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
	
	fprintf(repStats, "Intra-var Time = %g\n",(intraVarRemoteTimes/idx_current));
	fprintf(repStats, "Intra-var Compression: %g\n",(intraVarRemoteCompressions/idx_current));
	
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
	
	fprintf(repStats, "Intra-var Kernel-Calls = %g\n",(intraVarN_kvals/idx_current));
	
	fclose(repStats);
	
 }