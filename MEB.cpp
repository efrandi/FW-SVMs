
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ios>
#include <fstream>


#include "MEB.h"

void Solver_Core::Init(const svm_problem *_prob, const svm_parameter* _param)
{	
	prob  = _prob;
	param = _param;
	showCoreSolverInfo = false;
	tempD = Malloc(double,2*INITIAL_CS);
	actual_size_tmp_D= 2*INITIAL_CS;
	tmpAlpha = Malloc(double,2*INITIAL_CS);
	actual_size_tmp_alpha= 2*INITIAL_CS;
	allocated_size_for_weights = 2*INITIAL_CS;
	id_fw1_it  = 0;
	id_fw2_it  = 1;
	id_swap_it = 2;
	_Init();
}

bool Solver_Core::Create(double CVM_eps)
{
	printf("CONTROL ** begin create child\n");
	bool flag = _Create(CVM_eps);
	//printf("end create child\n");
	if (flag) 
	{
		// use a small subset for initial MEB
		 solver->Solve(coreIdx,coreNum,tempD);
		 outAlpha = solver->getAlpha();
		 allocated_size_for_weights = solver->get_allocated_size();
		 ComputeRadius2();
	}
	return flag;
}


void Solver_Core::process_mem_usage(double& vm_usage, double& resident_set)
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

void Solver_Core::show_memory_usage(char *header){

	FILE* resource_file;
	char resource_info_name[1024];

	sprintf(resource_info_name,"resource-usage-solver-MEB.txt");
	
	
	resource_file = fopen(resource_info_name, "a+t");
	
	if(resource_file == NULL){
		resource_file = stdout;
	}
		
	fprintf(resource_file,"resources-usage in: %s\n",header);
	
    double vm, rss;
	process_mem_usage(vm, rss);
	fprintf(resource_file,"memory usage: %f Mb\n", vm);
	fprintf(resource_file,"resident set: %f\n", rss);
	fprintf(resource_file,"\n");	
  	
  	if(resource_file != NULL && resource_file != stdout){
		fclose(resource_file);
	}
}

//Solve a MEB problem selecting an specific algorithm
//(0) BADOU-CLARKSON, (1) YILDRIM1, (2) YILDRIM2
int Solver_Core::Solve(int num_basis, double cvm_eps, int method, bool cooling, bool randomized){
	
	 int status;
	 bool glotonous = true;
	 id_fw1_it  = 0;
	 id_fw2_it  = 1;
	 id_swap_it = 2;
	 previous_violator_idx = -1;
	 previousQidx = NULL;
	 next_dot_violator_c = 0.0;
	 previous_lambda = 0.0;

	 switch(method)
	 {
	 	case BADOUCLARKSON:
	 		status = Badou_Clarkson_Algorithm(num_basis,cvm_eps,cooling,randomized);
	 		break;
	 	case YILDRIM1: 
	 		//status = Yildrim_Algorithm(num_basis,cvm_eps,false,cooling,randomized);
	 		status = Yildrim_Algorithm1(num_basis,cvm_eps,cooling,randomized);
	 		break;
	 	case YILDRIM2: 
	 		//status = Yildrim_Algorithm(num_basis,cvm_eps,true,cooling,randomized);
	 		status = Yildrim_Algorithm2(num_basis,cvm_eps,cooling,randomized);
	 		break;
	 	case SWAP: /*GOLOSO*/
	 		glotonous = true;
	 		status = SWAP_Algorithm(num_basis,cvm_eps,cooling,randomized,glotonous);
	 		break;	 		
	 	case lSWAP: /*LAZY SWAP*/
	 		glotonous = false;
	 		status = SWAP_Algorithm(num_basis,cvm_eps,cooling,randomized,glotonous);
	 		break;

	 }
	 
	 return status;
}


int Solver_Core::Badou_Clarkson_Algorithm(int num_basis, double cvm_eps, bool cooling, bool randomized)
{	
	
	printf("MEB Solver: Badou & Clarkson, ");
	cooling ? printf("cooling YES, ") : printf("cooling NOT, ");
	randomized ? printf("randomized YES\n") : printf("randomized NOT\n");
	
	greedy_it = 0;
	smo_it = 0;
	this->maxNumBasis = num_basis;

    double epsilonFactor = EPS_SCALING;
	double currentEpsilon;
	
	if(!cooling) //avoid cooling
		currentEpsilon = cvm_eps/epsilonFactor;
	else //do cooling 	
		currentEpsilon = INITIAL_EPS;
		
	int fremessages = 250;
	 	
	while(currentEpsilon > cvm_eps){	
		
		currentEpsilon *= epsilonFactor;
		if (currentEpsilon < cvm_eps)
			currentEpsilon = cvm_eps;
				
		printf("MEB Solver: Iterating to achieve EPS=%g currentEPS=%g coreNum=%d\n",cvm_eps,currentEpsilon,coreNum);
		
		// solve problem with current epsilon (warm start from the previous solution)
		double maxDistance2 = 0.0;
		int maxDistance2Idx = 0;
		double factor       = 1.0 + currentEpsilon;
		factor             *= factor;

		while (maxDistance2Idx != -1)
		{
			
			maxDistance2    = r2 * factor;
			maxDistance2Idx = -1;			
			
			if(randomized){
				for(int sampleIter = 0; (sampleIter < NUM_SAMPLINGS) && (maxDistance2Idx == -1); sampleIter++)			
					maxDistance2 = _maxDistFromSampling(maxDistance2, maxDistance2Idx);
				if((maxDistance2Idx == -1) && param->stop_with_real_meb){
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
				}
			} else{
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
			}
			
			if (maxDistance2Idx != -1)
			{	
				fremessages = (coreNum > 50000) ? 5 : 50;
				bool show_information = (greedy_it%fremessages == 0);
				
				if(show_information)
					printf("## coreset size: %d eps: %g |c|: %.10f  R: %.10f |c-x|: %.10f r: %.10f\n",coreNum, currentEpsilon, coreNorm2, r2, maxDistance2, sqrt(maxDistance2/r2)-1.0);
			
				_UpdateCoreSet(maxDistance2Idx);
								
				if (show_information)			
					printf("calling numerical solver: coreset size %d\n",coreNum);
				
				
				clock_t init_time_2 = clock ();
				smo_it +=  solver->Solve(coreIdx,coreNum,tempD);
				clock_t final_time_2 = clock ();
				
				greedy_it++;
				
				if(show_information)
					printf("end calling numerical solver. TIME %g (seconds)\n", (double)(final_time_2 - init_time_2)/CLOCKS_PER_SEC);
			

				outAlpha = solver->getAlpha();				
				ComputeRadius2();
				
				if (show_information){
					char *message = new char[50];
					sprintf(message, "iteration of MEB solver");
					show_memory_usage(message);
				}
				
			}
			
			if (IsExitOnMaxIter())
			{
				currentEpsilon = cvm_eps;
				break;
			}
			
		}
	}	
	info("###### end computing MEB. Size coreset: %d, iterations: %d\n",coreNum, greedy_it);

	return coreNum;
}

int Solver_Core::Yildrim_Algorithm1(int num_basis, double cvm_eps, bool cooling, bool randomized)
{
	printf("MEB Solver: Yildrim Algorihtm I, ");
	cooling ? printf("cooling YES, ") : printf("cooling NOT, ");
	randomized ? printf("randomized YES\n") : printf("randomized NOT\n");
	
	greedy_it = 0;
	this->maxNumBasis = num_basis;
	
	//Reallocation of WEIGHTS
		allocated_size_for_weights = 1000;
		double* temp_array_weights =  Malloc(double,allocated_size_for_weights);
		for(int m=0; m<coreNum; m++)
			temp_array_weights[m] = outAlpha[m];
		outAlpha = Malloc(double,allocated_size_for_weights);
		for(int m=0; m<coreNum; m++)
			outAlpha[m] = temp_array_weights[m];

		Q_center_dot_in = Malloc(Qfloat,allocated_size_for_weights);
		r2 = update_radius_and_cache(-1,r2,0,0,0);

		furthest_coreidx = -1;
		closest_coreidx = -1;

		for(int m=coreNum; m<allocated_size_for_weights; m++){
			outAlpha[m] = 0.0;
			Q_center_dot_in[m] = 0.0;
		}

		free(temp_array_weights);
	//End reallocation of weights
	
	//Allocation of inverted-index	
	inverted_coreIdx = new int[prob->l];
	for(int m=0; m<prob->l; m++)
		inverted_coreIdx[m] = -1;
	for(int m=0; m<coreNum; m++)
		inverted_coreIdx[coreIdx[m]] = m;
	//End allocation of inverted-index	
	
	
    double epsilonFactor = EPS_SCALING;
	double currentEpsilon = cooling ? INITIAL_EPS : cvm_eps/epsilonFactor;
		
	int fremessages = 250;
	
	//iterate until desired precision 
	while(currentEpsilon > cvm_eps){	
		
		currentEpsilon *= epsilonFactor;		
		currentEpsilon = (currentEpsilon < cvm_eps) ? cvm_eps: currentEpsilon;
		printf("EPS-Iteration: Iterating to achieve EPS=%g currentEPS=%g coreNum=%d\n",cvm_eps,currentEpsilon,coreNum);
		
		// solve problem with current epsilon (warm start from the previous solution)
		
		double maxDistance2 = 0.0;
		int maxDistance2Idx = 0;
		double delta = 2.0;
		double lambda;
		double yfactor;
		double old_radius2,new_radius2;
				
		double tol_factor = (1.0 + currentEpsilon);
		tol_factor *= tol_factor;

		while (maxDistance2Idx != -1)
		//while (delta - (tol_factor -1) > TAU)
		{
			//distance2 covered by the current ball
			maxDistance2    = r2 * tol_factor;			
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			//find the furthest point from the center : -1 if no one exists (and hence delta = 0)
			
			maxDistance2Idx = -1;
							
			if(randomized){ //search on a sample
				for(int sampleIter = 0; (sampleIter < NUM_SAMPLINGS) && (maxDistance2Idx == -1); sampleIter++){
					maxDistance2 = _maxDistFromSampling(maxDistance2, maxDistance2Idx);
				}
				if((maxDistance2Idx == -1) && param->stop_with_real_meb){
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
				}
				//if(maxDistance2Idx == -1){
				//	maxDistance2 = _maxDistFromCoreSet(maxDistance2, maxDistance2Idx);
				//}
			} else { //search among the entire dataset
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			
			delta = (maxDistance2/r2) - 1.0;
			
			//if there is a point out of the ball
			if(maxDistance2Idx != -1)
			//if (delta - (tol_factor -1) > TAU)
			{	
				fremessages = (coreNum > 50000) ? 100 : 50000;
				bool show_information = (greedy_it%fremessages == 0);
				
				if(show_information)
					printf(" ## MEB-Solver: radius = %f, out-of-MEB %d, dist-from-center %f, dist_tol %f, cNorm %f\n",r2,maxDistance2Idx,maxDistance2,r2*tol_factor,coreNorm2);
			
				//////////////////////////////////////////////////////////////////////////////////////////////////////
				// Update current ball using yildirim method (Alg.1)
				
				greedy_it++; 
				clock_t time_update_start = clock ();
				
				// STR  check if the point already is in the coreset
	   					double value_inv_index = (double) inverted_coreIdx[maxDistance2Idx];

	   					if(value_inv_index < -0.5)
	   					{
							inverted_coreIdx[maxDistance2Idx] = coreNum;	
	   					} 
	   						   			
	   			// STR  update the coreset and counters if required
	  					
	  					_UpdateCoreSet(maxDistance2Idx);
				
				// STR  check space to store weights 
					    if (coreNum >= allocated_size_for_weights )
					    {			
							allocated_size_for_weights = (int)(1.5*allocated_size_for_weights);
							outAlpha = (double*)realloc(outAlpha,allocated_size_for_weights*sizeof(double));
							Q_center_dot_in = (Qfloat*)realloc(Q_center_dot_in,allocated_size_for_weights*sizeof(Qfloat));

							for(int k=coreNum; k <  allocated_size_for_weights; k++){
								outAlpha[k] = 0.0;
								Q_center_dot_in[k] = 0.0;
							}
						}
	  			
	  			// ***************  UPDATE RULE  ****************** //

				lambda = delta/(2*(1+delta));
				next_dot_violator_c = (1-lambda)*center_dot_out + lambda*Eta;

				//compute the gain of a standard Frank-Wolfe step
				double fw_gain =  ((r2/4.0)*(maxDistance2/r2 + r2/maxDistance2)) - (r2/2.0);

				// determine the point of core-set that will be changed
	   			int coreset_idx = inverted_coreIdx[maxDistance2Idx];
				
				// update the weights of old core-set points
				for(int i=0; i < coreNum; i++)  
					outAlpha[i] = outAlpha[i]*(1-lambda);
	   			
				// define the weight of the new point
	   			outAlpha[coreset_idx] += lambda;
	   			
				// recompute radius
				old_radius2 = r2;
				new_radius2 = r2 + fw_gain;
				r2 = update_radius_and_cache(id_fw1_it,new_radius2,lambda,coreset_idx,0);
							
				// ***************  END UPDATE RULE  ****************** //
				
				clock_t time_update_end = clock ();
				
				//end update current ball
				///////////////////////////////////////////////////////////////////////////////////////////////////////
				
			
				if(show_information)
					printf(" ## time last update = %g (seconds)\n", (double)(time_update_end - time_update_start)/CLOCKS_PER_SEC);
				
				//Check theorerical facts
				 
				if (r2 < old_radius2)
					printf("WARNING! The radius is decreasing! \n");
				
				double difference = r2 - old_radius2*yfactor;
				if (difference < -TAU)
				{
					printf("WARNING! Lemma 3.2 does not hold! Difference = %g \n",difference);

				}

				if (show_information){
					char *message = new char[50];
					sprintf(message, "iteration of MEB solver");
					show_memory_usage(message);
				}
				
				if(show_information)
					info(" ## %d eps: %g |c|: %.10f  R: %.10f |c-x|: %.10f r: %.10f\n",coreNum, currentEpsilon, coreNorm2, r2, maxDistance2, sqrt(maxDistance2/r2)-1.0);				
			
			}//end change for a violating point
			
			if (IsExitOnMaxIter())
			{
				currentEpsilon = cvm_eps;
				break;
			}
		}//end iteration ... (delta > (factor-1))
			
	}//end cooling	
	
	info("###### end computing MEB: size coreset %d, ITERATIONS: %d\n",coreNum, greedy_it);
	
	return coreNum;
}//End YILDRIM ALGORITHM 1


int Solver_Core::Yildrim_Algorithm2(int num_basis, double cvm_eps, bool cooling, bool randomized)
{
	printf("MEB Solver: Yildrim Algorihtm II, ");
	cooling ? printf("cooling YES, ") : printf("cooling NOT, ");
	randomized ? printf("randomized YES\n") : printf("randomized NOT\n");
	
	greedy_it = 0;
	this->maxNumBasis = num_basis;
	
	//Reallocation of WEIGHTS
		allocated_size_for_weights = 1000;
		double* temp_array_weights =  Malloc(double,allocated_size_for_weights);
		for(int m=0; m<coreNum; m++)
			temp_array_weights[m] = outAlpha[m];
		outAlpha = Malloc(double,allocated_size_for_weights);
		for(int m=0; m<coreNum; m++)
			outAlpha[m] = temp_array_weights[m];
		
		Q_center_dot_in = Malloc(Qfloat,allocated_size_for_weights);
		r2 = update_radius_and_cache(-1,r2,0,0,0);

		furthest_coreidx = -1;
		closest_coreidx = -1;

		for(int m=coreNum; m<allocated_size_for_weights; m++){
			outAlpha[m] = 0.0;
			Q_center_dot_in[m] = 0.0;
		}

		free(temp_array_weights);
	//End reallocation of weights
	
	//Allocation of inverted-index	
	inverted_coreIdx = new int[prob->l];
	for(int m=0; m<prob->l; m++)
		inverted_coreIdx[m] = -1;
	for(int m=0; m<coreNum; m++)
		inverted_coreIdx[coreIdx[m]] = m;
	//End allocation of inverted-index	
	
	
    double epsilonFactor = EPS_SCALING;
	double currentEpsilon = cooling ? INITIAL_EPS : cvm_eps/epsilonFactor;
		
	int fremessages = 1;
	int maxDistance2Idx = 0;
		
	//iterate until desired precision 
	while(currentEpsilon > cvm_eps){	
		
		currentEpsilon *= epsilonFactor;		
		currentEpsilon = (currentEpsilon < cvm_eps) ? cvm_eps: currentEpsilon;
		printf("EPS-Iteration: Iterating to achieve EPS=%g currentEPS=%g coreNum=%d\n",cvm_eps,currentEpsilon,coreNum);
		
		// solve problem with current epsilon (warm start from the previous solution)
		
		double maxDistance2 = 0.0;
		double minDistance2 = INF;
		maxDistance2Idx = 0;
		int minDistance2_coreIdx = -1;
		double delta_pos = 1.0;
		double delta_neg = 1.0;
		double delta = 2.0;
		double lambda;
		double yfactor;
		double old_radius2,new_radius2;
				
		double tol_factor = (1.0 + currentEpsilon);
		tol_factor *= tol_factor;

        //while(delta - (tol_factor -1) > TAU)
		while (maxDistance2Idx != -1)
		{
			//distance2 covered by the current ball
			maxDistance2    = r2 * tol_factor;			
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			//find the furthest point from the center : -1 if no one exists (and hence delta = 0)
			
			maxDistance2Idx = -1;
							
			if(randomized){ //search on a sample
				for(int sampleIter = 0; (sampleIter < NUM_SAMPLINGS) && (maxDistance2Idx == -1); sampleIter++){
					maxDistance2 = _maxDistFromSampling(maxDistance2, maxDistance2Idx);
				}
				if((maxDistance2Idx == -1) && param->stop_with_real_meb){
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
				}
			} else { //search among the entire dataset
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			

			//////////////////////////////////////////////////////////////////////////////////////////////////////
			//find the nearest point from the center among coreset points
			
			minDistance2_coreIdx = -1;
			minDistance2 =  INF;
			minDistance2 = _minDistFromCoreSet(minDistance2, minDistance2_coreIdx);
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			
			//compute deltas
			delta_pos = (maxDistance2/r2) - 1.0;
			delta_neg = 1.0 - (minDistance2/r2); 
			delta = (delta_pos > delta_neg) ? delta_pos : delta_neg;
			int iteration_type;// 0:plus-iteration, 1:minus-iteration, 2:drop-iteration
		    	
			//if stopping criteria has not been reached
			//if(delta - (tol_factor -1) > TAU)
			if (maxDistance2Idx != -1)
			{	
				greedy_it++; 
				fremessages = (coreNum > 50000) ? 100 : 50000;
				bool show_information = (greedy_it%fremessages == 0);
				
				iteration_type = (delta > delta_pos) ? 3 : 0; 
				 
				//if(show_information)
					//printf(" ## Yildrim2-Solver: radius = %f, out-of-MEB %d, dist-from-center %f, dist_tol %f, cNorm %f\n",r2,maxDistance2Idx,maxDistance2,r2*tol_factor,coreNorm2);
			    
			    	//////////////////////////////////////////////////////////////////////////////////////////////////////
				// Update MEB according to Yildrim ALgorithm II
				
				clock_t time_update_start = clock ();
				
				if(iteration_type == 0){
				//////////////////////////////////////////////////////////////////////////////////////////////////////
				//PLUS-ITERATION -STANDARD FRANK-WOLFE
					if(show_information)
						printf("PLUS-IT: delta is %f, delta_pos is %f, delta_neg is %f, tol: %f\n", delta, delta_pos,delta_neg, tol_factor-1);
					
									
					// STR  check if the point already is in the coreset
		   					double value_inv_index = (double) inverted_coreIdx[maxDistance2Idx];
		   					if(value_inv_index < -0.5)
		   					{
								inverted_coreIdx[maxDistance2Idx] = coreNum;	
		   					} 
							    				   						   			
		   			// STR  update the coreset and counters if required
		  					
		  					_UpdateCoreSet(maxDistance2Idx);
				
					// STR  check space to store weights 
							if (coreNum >= allocated_size_for_weights )
							{			
								allocated_size_for_weights = (int)(1.5*allocated_size_for_weights);
								outAlpha = (double*)realloc(outAlpha,allocated_size_for_weights*sizeof(double));
								Q_center_dot_in = (Qfloat*)realloc(Q_center_dot_in,allocated_size_for_weights*sizeof(Qfloat));

								for(int k=coreNum; k <  allocated_size_for_weights; k++){
										outAlpha[k] = 0.0;
										Q_center_dot_in[k] = 0.0;
								}

							}
					
		  			// ***************  UPDATE RULE  ****************** //
		  			
					delta = (maxDistance2/r2) - 1.0;
					lambda = delta/(2*(1+delta));

					next_dot_violator_c = (1-lambda)*center_dot_out + lambda*Eta;

					//compute the gain of a standard Frank-Wolfe step
					double fw_gain =  ((r2/4.0)*(maxDistance2/r2 + r2/maxDistance2)) - (r2/2.0);
					
					if(show_information)
						printf("maxDist %f, dist-tol %f, Idx %d, CoreIdx %d, weight %f, lambda %f\n",maxDistance2,r2*tol_factor,maxDistance2Idx,inverted_coreIdx[maxDistance2Idx],outAlpha[inverted_coreIdx[maxDistance2Idx]],lambda);

					// determine the point of core-set that will be changed
		   			int coreset_idx = inverted_coreIdx[maxDistance2Idx];
				
					// update the weights of old core-set points
					for(int i=0; i < coreNum; i++)  
						outAlpha[i] = outAlpha[i]*(1-lambda);
		   			
		   			
					// define the weight of the new point
		   			outAlpha[coreset_idx] += lambda;
		   			
		   			// recompute radius
					old_radius2 = r2;
					new_radius2 = r2 + fw_gain;
					r2 = update_radius_and_cache(id_fw1_it,new_radius2,lambda,coreset_idx,0);
				
					// ***************  END UPDATE RULE  ****************** //
					
				}//END PLUS-ITERATION
				
				else {
				//////////////////////////////////////////////////////////////////////////////////////////////////////
				//MINUS OR DROP-ITERATION
		
					double lambda_drop =  (outAlpha[minDistance2_coreIdx]/(1.0-outAlpha[minDistance2_coreIdx]));
					double lambda_minus = (delta_neg/(2.0*(1.0-delta_neg)));
					
					//determine iteration type
					lambda = (lambda_minus < lambda_drop) ? lambda_minus : lambda_drop;
					Qfloat dot_out_inner = Q_center_dot_out[minDistance2_coreIdx];
					next_dot_violator_c = (1+lambda)*center_dot_out - lambda*dot_out_inner;

					double mfw_gain =  lambda*(r2-minDistance2) - (lambda*lambda)*(minDistance2);

					if(show_information){
						printf("MFW-IT: delta is %f, delta_pos is %f, delta_neg is %f, tol: %f\n",delta, delta_pos,delta_neg, tol_factor-1);
						printf("dist-center %f, radius %f, core-idx %d, idx %d,  weight %f, lambda %f\n",minDistance2, r2*(tol_factor), minDistance2_coreIdx,coreIdx[minDistance2_coreIdx], outAlpha[minDistance2_coreIdx],lambda);
					}
					//scale the current center i.e. all the weights		
					for(int i=0; i < coreNum; i++){
						outAlpha[i] = outAlpha[i]*(1+lambda);
	   				}	
	   							
					//update weight of the nearest point
	    			double prev_weight = outAlpha[minDistance2_coreIdx];
	    			outAlpha[minDistance2_coreIdx] -= lambda;	

	    			if(show_information)
	    				printf("dist-center %f, radius %f, idx %d,  weight-before %f, weight-after %f, lambda %f, yfactor %f\n",minDistance2, r2*(tol_factor), minDistance2_coreIdx, prev_weight, outAlpha[minDistance2_coreIdx],lambda, yfactor);
		    
	    			// recompute radius
					old_radius2 = r2;
					new_radius2 = r2 + mfw_gain;
					r2 = update_radius_and_cache(id_fw2_it,new_radius2,lambda,0,minDistance2_coreIdx);
				
				}//END MINUS-ITERATION 		
								
				clock_t time_update_end = clock ();
				
				//end update current ball
				///////////////////////////////////////////////////////////////////////////////////////////////////////
				
			
				if(show_information)
					printf(" ## time last update = %g (seconds)\n", (double)(time_update_end - time_update_start)/CLOCKS_PER_SEC);
				
				//Check theorerical facts
				 
				if (r2 < old_radius2)
					printf("WARNING! The radius is decreasing! \n");
				
				double difference = r2 - old_radius2*yfactor;
				if (difference < -TAU)
				{
					printf("WARNING! Lemma 3.2 does not hold! Difference = %g \n",difference);
					printf("Iteration type: %d \n",iteration_type);
				}

				if (show_information){
					char *message = new char[50];
					sprintf(message, "iteration of MEB solver");
					show_memory_usage(message);
				}
				
				if (show_information) 
					info(" ## %d eps: %g |c|: %.10f  R: %.10f |c-x|: %.10f r: %.10f\n",coreNum, currentEpsilon, coreNorm2, r2, maxDistance2, sqrt(maxDistance2/r2)-1.0);				
			
			}//end change for a violating point
			
			if (IsExitOnMaxIter())
			{
				currentEpsilon = cvm_eps;
				break;
			}
		}//end iteration ... (delta > (factor-1))
			
	}//end cooling	
	
	info("###### end computing MEB: size coreset %d, ITERATIONS: %d\n",coreNum, greedy_it);
	
	return coreNum;
}//End YILDRIM ALGORITHM 2



int Solver_Core::SWAP_Algorithm(int num_basis, double cvm_eps, bool cooling, bool randomized, bool glotonous)
{
	printf("MEB Solver: FW with SWAPs Algorithm, ");
	cooling ? printf("cooling YES, ") : printf("cooling NOT, ");
	randomized ? printf("randomized YES\n") : printf("randomized NOT\n");
	
	greedy_it = 0;
	this->maxNumBasis = num_basis;

		//Reallocation of WEIGHTS
		allocated_size_for_weights = 1000;
		double* temp_array_weights =  Malloc(double,allocated_size_for_weights);
		
		for(int m=0; m<coreNum; m++){
			temp_array_weights[m] = outAlpha[m];
		
		}
		outAlpha = Malloc(double,allocated_size_for_weights);
		
		for(int m=0; m<coreNum; m++)
			outAlpha[m] = temp_array_weights[m];
		
		Q_center_dot_in = Malloc(Qfloat,allocated_size_for_weights);
		r2 = update_radius_and_cache(-1,r2,0,0,0);
	
		furthest_coreidx = -1;
		closest_coreidx = -1;

		for(int m=coreNum; m<allocated_size_for_weights; m++){
			outAlpha[m] = 0.0;
			Q_center_dot_in[m] = 0.0;
		}
		
		free(temp_array_weights);
	//End reallocation of weights
	 
	//Allocation of inverted-index	
	inverted_coreIdx = new int[prob->l];
	for(int m=0; m<prob->l; m++)
		inverted_coreIdx[m] = -1;
	for(int m=0; m<coreNum; m++)
		inverted_coreIdx[coreIdx[m]] = m;
	//End allocation of inverted-index	
	
	
    double epsilonFactor = EPS_SCALING;
	double currentEpsilon = cooling ? INITIAL_EPS : cvm_eps/epsilonFactor;
		
	int fremessages = 1;
	int maxDistance2Idx = 0;
		
	//iterate until desired precision 
	while(currentEpsilon > cvm_eps){	
		
		currentEpsilon *= epsilonFactor;		
		currentEpsilon = (currentEpsilon < cvm_eps) ? cvm_eps: currentEpsilon;

		printf("EPS-Iteration: Iterating to achieve EPS=%g currentEPS=%g coreNum=%d\n",cvm_eps,currentEpsilon,coreNum);
		
		// solve problem with current epsilon (warm start from the previous solution)
		
		double maxDistance2 = 0.0;
		maxDistance2Idx = 0;
		double old_radius2, new_radius2;
				
		double tol_factor = (1.0 + currentEpsilon);
		tol_factor *= tol_factor;

       		while (maxDistance2Idx != -1)
		{
			//distance2 covered by the current ball
			maxDistance2    = r2 * tol_factor;			
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			//find the furthest point from the center : -1 if no one exists (and hence delta = 0)
			
			maxDistance2Idx = -1;
						
			if(randomized){ //search on a sample
				for(int sampleIter = 0; (sampleIter < NUM_SAMPLINGS) && (maxDistance2Idx == -1); sampleIter++){
					maxDistance2 = _maxDistFromSampling(maxDistance2, maxDistance2Idx);
				}
				if((maxDistance2Idx == -1) && param->stop_with_real_meb){
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
				}
			} else { //search among the entire dataset
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			
			
			if (maxDistance2Idx != -1)
			{	
				greedy_it++; 
				
				//////////////////////////////////////////////////////////////////////////////////////////////////////
				// Update MEB according to the SWAP ALgorithm 
				
				clock_t time_update_start = clock ();
				
	
				//compute the gain of a standard Frank-Wolfe step
				double fw_gain =  ((r2/4.0)*(maxDistance2/r2 + r2/maxDistance2)) - (r2/2.0);
				
				//////////////////////////////////////////////////////////////////////////////////////////////////////
				//find the inner point maximizing the gain of a swap 
				int coreset_idx_in = -1;
				double alpha_swap = 0;
				double dist_in, dist_pts;
				double swap_gain;
				//printf("looking for violator\n");
				if(glotonous) {
					swap_gain = _best_point_for_SWAP(r2 * (1.0 - currentEpsilon)*(1.0 - currentEpsilon),maxDistance2,maxDistance2Idx,coreset_idx_in,alpha_swap,dist_in,dist_pts);
				} else {
					swap_gain = _most_inner_point_for_SWAP(r2 * (1.0 - currentEpsilon)*(1.0 - currentEpsilon),maxDistance2,maxDistance2Idx,coreset_idx_in,alpha_swap,dist_in,dist_pts);
				}

			
				//////////////////////////////////////////////////////////////////////////////////////////////////////
			
				//choose iteration according to the gain			
				int iteration_type;// 0:FW-iteration, 1:SWAP-iteration
		    		iteration_type = (fw_gain >= swap_gain) ? 0 : 1; 
				
				fremessages = (coreNum > 50000) ? 100 : 50000;
				bool show_information = (greedy_it%fremessages == 0);
				
			
				// STR  check if the point already is in the coreset
		   					double value_inv_index = (double) inverted_coreIdx[maxDistance2Idx];
		   					if(value_inv_index < -0.5)
		   					{
								inverted_coreIdx[maxDistance2Idx] = coreNum;	
		   					} 
							    				   						   			
		   		// STR  update the coreset and counters if required
		  					
		  					_UpdateCoreSet(maxDistance2Idx);
				
				// STR  check space to store weights 
							if (coreNum >= allocated_size_for_weights )
							{			
								allocated_size_for_weights = (int)(1.5*allocated_size_for_weights);
								outAlpha = (double*)realloc(outAlpha,allocated_size_for_weights*sizeof(double));
								Q_center_dot_in = (Qfloat*)realloc(Q_center_dot_in,allocated_size_for_weights*sizeof(Qfloat));
	
								for(int k=coreNum; k <  allocated_size_for_weights; k++){
									outAlpha[k] = 0.0;
									Q_center_dot_in[k] = 0.0;
	
								}
							}
				// determine the point of core-set that will be changed
		   		int coreset_idx_out = inverted_coreIdx[maxDistance2Idx];
				
				
				if(iteration_type == 0){
				
					//FRANK-WOLFE-ITERATION
					//printf("FW iteration: coreNum=%d, gain_fw=%g, gain_swap=%g\n",coreNum,fw_gain,swap_gain);							
					double alpha_fw = 0.5 - (0.5*r2/maxDistance2);

					next_dot_violator_c = (1-alpha_fw)*center_dot_out + alpha_fw*Eta;

					// update the weights of old core-set points
					for(int i=0; i < coreNum; i++)  
						outAlpha[i] = outAlpha[i]*(1-alpha_fw);
		   			
		   			
					// define the weight of the new point
		   			outAlpha[coreset_idx_out] += alpha_fw;
		   			
		   			// recompute radius
					old_radius2 = r2;
					new_radius2 = r2 + fw_gain;
					r2 = update_radius_and_cache(id_fw1_it,new_radius2,alpha_fw,coreset_idx_out,coreset_idx_in);
				
				}				
				else {
				
					//SWAP-ITERATION
					//printf("SWAP iteration: coreNum=%d,r2=%g\n",coreNum,r2); 
					//printf("............... gain_fw=%g, gain_swap=%g\n",fw_gain,swap_gain);
					//printf("............... idx-in=%d, idx-out=%d, alpha=%g, weight-to-change=%g\n",coreset_idx_in,coreset_idx_out,alpha_swap,outAlpha[coreset_idx_in]);
					
					//double alpha_swap_lim =  (outAlpha[minDistance2_coreIdx]/(1.0-outAlpha[minDistance2_coreIdx]));
					
					//do the swap		
					outAlpha[coreset_idx_out] += alpha_swap;
					outAlpha[coreset_idx_in] -= alpha_swap;
					Qfloat violator_dot_in = Eta - (dist_pts/2.0);
					next_dot_violator_c = center_dot_out + alpha_swap*Eta -alpha_swap*violator_dot_in;
	    				// recompute radius
					old_radius2 = r2;
					new_radius2 = r2 + swap_gain;
					r2 = update_radius_and_cache(id_swap_it,new_radius2,alpha_swap,coreset_idx_out,coreset_idx_in);
					
				}//END SWAP-ITERATION 		
								
				clock_t time_update_end = clock ();
				
				//end update current ball
				///////////////////////////////////////////////////////////////////////////////////////////////////////
				
			
				if(show_information)
					printf(" ## time last update = %g (seconds)\n", (double)(time_update_end - time_update_start)/CLOCKS_PER_SEC);
				
				//Check theorerical facts
				 
				if (r2 < old_radius2)
					printf("WARNING! The radius is decreasing! \n");
				
				if (show_information){
					char *message = new char[50];
					sprintf(message, "iteration of MEB solver");
					show_memory_usage(message);
				}
				
				if (show_information) 
					info(" ## %d eps: %g |c|: %.10f  R: %.10f |c-x|: %.10f r: %.10f\n",coreNum, currentEpsilon, coreNorm2, r2, maxDistance2, sqrt(maxDistance2/r2)-1.0);				
			
			}//end change for a violating point
			
			if (IsExitOnMaxIter())
			{
				currentEpsilon = cvm_eps;
				break;
			}
		}//end iteration ... (delta > (factor-1))
			
	}//end cooling	
	
	info("###### end computing MEB: size coreset %d, ITERATIONS: %d\n",coreNum, greedy_it);
	
	return coreNum;
}//End PAM ALGORITHM


int Solver_Core::Yildrim_Algorithm(int num_basis, double cvm_eps, bool dropping, bool cooling, bool randomized)
{
	greedy_it = 0;
	this->maxNumBasis = num_basis;
	allocated_size_for_weights = 2*coreNum;
	double* temp_array_weights =  Malloc(double,allocated_size_for_weights);
	double sum_initial = 0.0;
	printf("### ");
	
	for(int m=0; m<coreNum; m++){
		temp_array_weights[m] = outAlpha[m];
		printf("weight[%d]=%f, ",m,outAlpha[m]);
		sum_initial += outAlpha[m];
	}	 
	printf("### initial_meb SUM=%f, ",sum_initial);
	
	outAlpha = Malloc(double,allocated_size_for_weights);
	for(int m=0; m<coreNum; m++)
		outAlpha[m] = temp_array_weights[m];
	for(int m=coreNum; m<allocated_size_for_weights; m++)
		outAlpha[m] = 0.0;
		
	free(temp_array_weights);
		
	inverted_coreIdx = new int[prob->l];
	for(int m=0; m<prob->l; m++)
		inverted_coreIdx[m] = -1;
	
	for(int m=0; m<coreNum; m++)
		inverted_coreIdx[coreIdx[m]] = m;
	
    double epsilonFactor = EPS_SCALING;
	double currentEpsilon;
	
	if(!cooling)
		currentEpsilon = cvm_eps/epsilonFactor;
	else //do cooling 	
		currentEpsilon = INITIAL_EPS;
		
	int fremessages = 250;
		
	while(currentEpsilon > cvm_eps){	
		currentEpsilon *= epsilonFactor;
		
		if (currentEpsilon < cvm_eps)
			currentEpsilon = cvm_eps;
				
		printf("MEB Solver: Iterating to achieve EPS=%g currentEPS=%g coreNum=%d\n",cvm_eps,currentEpsilon,coreNum);
		
		// solve problem with current epsilon (warm start from the previous solution)
		double maxDistance2 = 0.0;
		double minDistance2 = 0.0;
		int maxDistance2Idx = 0;
		int maxDist2_invIdx = 0;
		int minDist2_invIdx = 0;
		
		
		double factor       = 1.0 + currentEpsilon;
		factor             *= factor;

       
		while (maxDistance2Idx != -1)
		{
			
			maxDistance2    = r2 * factor;
			//if (counter > 100 && counter%100 == 1)
				//printf("## %d eps: %g |c|: %.10f  R: %.10f |c-x|: %.10f r: %.10f\n",coreNum, currentEpsilon, coreNorm2, r2, maxDistance2, sqrt(maxDistance2/r2)-1.0);
			
			maxDistance2Idx = -1;			
			
			if(randomized)  
				for(int sampleIter = 0; (sampleIter < NUM_SAMPLINGS) && (maxDistance2Idx == -1); sampleIter++)			
					maxDistance2 = _maxDistFromSampling(maxDistance2, maxDistance2Idx);
			else 
					maxDistance2 = _maxDist(maxDistance2, maxDistance2Idx);
			
			printf("1. radius = %f, out-of-MEB %d, dist-from-center %f, dist_tol %f, cNorm %f\n",r2,maxDistance2Idx,maxDistance2,r2 * factor,coreNorm2);
			
			if(dropping){
				minDistance2 =  1E20;
				minDistance2 = _minDistFromCoreSet(minDistance2, minDist2_invIdx);// check maximal distance
				printf("2. min-distance^2 from core-set: %f\n",minDistance2);
			}
		
			
			if (maxDistance2Idx != -1)
			{	
				if (coreNum > 50000)
					fremessages = 1;
				if (coreNum > 5000)
					fremessages = 25;
								
				
				if (coreNum%fremessages < 1)			
					printf("using yildrim's updating rule: coreset size %d\n",coreNum);
				
				clock_t init_time_2 = clock ();
				int iteration_type;
					double yfactor = _Yildrim_Update(maxDistance2, maxDistance2Idx, dropping, minDistance2, minDist2_invIdx, iteration_type);
					greedy_it++;
				clock_t final_time_2 = clock ();
				
				if (coreNum%fremessages < 1)
				printf("end calling numerical solver. TIME %g (seconds)\n", (double)(final_time_2 - init_time_2)/CLOCKS_PER_SEC);
			
				//update radius considering the change in the coreset
				//ComputeRadius2();
				printf("4. radius = %f, out-of-MEB %d, dist-from-center %f, dist_tol %f, cNorm %f\n",r2,maxDistance2Idx,maxDistance2,r2 * factor,coreNorm2);
			    double old_radius2 = r2;
			    yildrim_computeRadius2(iteration_type, yfactor,maxDistance2);
				printf("Old radius2 %f, New_radius2 %f, yfactor %f \n",old_radius2,r2,yfactor);
				if (r2 < old_radius2)
					printf("WARNING! The radius is decreasing! \n");
				double difference = r2 - old_radius2*yfactor;
				if (difference < -TAU)
					{
					printf("WARNING! Lemma 3.2 does not hold! Difference = %g \n",difference);
					printf("Iteration type: %d \n",iteration_type);
					}

				if (coreNum%fremessages < 1){
					char *message = new char[50];
					sprintf(message, "iteration of MEB solver");
					show_memory_usage(message);
				}
				printf("6. radius = %f, out-of-MEB %d, dist-from-center %f, dist_tol %f, cNorm %f\n",r2,maxDistance2Idx,maxDistance2,r2 * factor,coreNorm2);
			
				if (coreNum%fremessages < 1) 
					info("## %d eps: %g |c|: %.10f  R: %.10f |c-x|: %.10f r: %.10f\n",coreNum, currentEpsilon, coreNorm2, r2, maxDistance2, sqrt(maxDistance2/r2)-1.0);				
			}
			
			if (IsExitOnMaxIter())
			{
				currentEpsilon = cvm_eps;
				break;
			}
		}
	}	
	
	info("###### end computing MEB: size coreset %d, ITERATIONS: %d\n",coreNum, greedy_it);
	
	return coreNum;
}

//maxdist2_idx and mindist2_idx are indexes ranging from 0 to coreNum
//and refer to elements of the current coreSet NOT elements of the dataset 
double Solver_Core::_Yildrim_Update(double maxdist2, int maxdist2_tr_idx, bool dropping, double mindist2, int mindist2_idx, int &iteration_type ){
	//update the coreset weights after the inclusion of a new point  
    // iteration_type = 0 (plus iteration), 1 (minus iteration), 2 (drop iteration)
	double delta_pos, delta_neg;
	double lambda;
	
	delta_pos = (maxdist2/r2) - 1.0;
	delta_neg = 1.0 - (mindist2/r2); 
	double yfactor;
	
	if(dropping && (delta_neg >= delta_pos)){
		
		printf("3. yildrim rule-2\n");
		printf("   before weight[%d]=%f, ",mindist2_idx,outAlpha[mindist2_idx]);
			
		if((delta_neg/(2*(1-delta_neg))) < (outAlpha[mindist2_idx]/(1-outAlpha[mindist2_idx]))){ 
			lambda = delta_neg/(2*(1-delta_neg));
			iteration_type = 1;
			yfactor = 1 + ((delta_neg*delta_neg)/(4*(1 - delta_neg)));

		} else{
			lambda = outAlpha[mindist2_idx]/(1-outAlpha[mindist2_idx]);
			iteration_type = 2;
			yfactor = lambda;
		}
		for(int i=0; i < coreNum; i++){
			//update the weights of old core-set points
			outAlpha[i] = outAlpha[i]*(1+lambda);
	   	}	
		//define the weight of the new point
	    outAlpha[mindist2_idx] -= lambda;	
	    printf("   after weight[%d]=%f coreset-size %d,\n",mindist2_idx,outAlpha[mindist2_idx],coreNum);
					
	} else {
	  //yildrim's standard procedure 	
		iteration_type = 0;
	  printf("3. yildrim rule-1\n");
	   double value_inv_index = (double) inverted_coreIdx[maxdist2_tr_idx];
	   if(value_inv_index < -0.5){
			inverted_coreIdx[maxdist2_tr_idx] = coreNum;	
	   }
	   
	   int maxdist2_idx = inverted_coreIdx[maxdist2_tr_idx]; 
			    		
	  _UpdateCoreSet(maxdist2_tr_idx);
				
	   if (coreNum >= allocated_size_for_weights )
	   {			
			allocated_size_for_weights = (int)(1.5*allocated_size_for_weights);
			outAlpha = (double*)realloc(outAlpha,allocated_size_for_weights*sizeof(double));
			for(int k=coreNum; k <  allocated_size_for_weights; k++)
				outAlpha[k] = 0.0;
		}
	  
	  printf("   before weight[%d]=%f, ",maxdist2_idx,outAlpha[maxdist2_idx]);
	  				
	  lambda = delta_pos/(2*(1+delta_pos));
	  for(int i=0; i < coreNum; i++){
		//update the weights of old core-set points
		outAlpha[i] = outAlpha[i]*(1-lambda);
	   }	
		//define the weight of the new point
	   outAlpha[maxdist2_idx] += lambda;
	   printf("   after weight[%d]=%f coreset-size %d,\n",maxdist2_idx,outAlpha[maxdist2_idx],coreNum);

	   yfactor = 1 + ((delta_pos*delta_pos)/(4*(1 + delta_pos)));
	}

return yfactor;
}

		
