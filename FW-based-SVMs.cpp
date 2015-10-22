#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "FW-based-SVMs.h"
#include "MEB-utilities.h" //solver parameters
#include <stdexcept>

enum { INIT_STEP, FW_STEP, MFW_STEP, SWAP_STEP, PARTAN_STEP};	/* step_type */

inline bool PosNeg(bool PosTurn, int pNum, int nNum)//controls turns for balanced sampling
{
	if (pNum<=0)
		PosTurn = false;
	else if (nNum<=0)
		PosTurn = true;
	else
		PosTurn = !PosTurn;
	return PosTurn;
}

int FW_L2SVM_Solver::ChooseRandomIndex(bool balanced){//choose a random vertex from the m-dimensional simplex
	int idx = 0;
	int rand32bit = random();
	if( balanced ){	//balanced sampling: try to choose positive and negative examples in equal proportions
			bool posTurn = PosNeg(posTurn,pNum,nNum);
			idx = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
	} else {
			idx	= rand32bit%prob->l;
	}
	return idx;
}

double FW_L2SVM_Solver::ComputeGradientCoordinate(int idx, Qfloat** Qcolumn){

	Qfloat gradientValue = 0.0;//requested gradient value
	Qfloat* Qidx = NULL;//kernel product between idx and active points

	if(param->randomized) {

		if(inverted_coreIdx[idx] >= 0){//the point is active
			gradientValue = gradientActivePoints[inverted_coreIdx[idx]];//the new toward vertex was also the previous one
		} else {
			Qidx = kernelQ->get_Q(idx, coreNum, coreIdx);
			if(Qidx != NULL)
				for (int j=0; j<coreNum; j++)
					gradientValue += Qidx[j]*outAlpha[j];
		}

	} else {

		gradientValue = gradientALLPoints[idx];
	}

	if(Qcolumn != NULL)
		*Qcolumn = Qidx;

	return gradientValue;

}

bool FW_L2SVM_Solver::AllocateMemoryForFW(int initial_size){ 

		allocated_size_for_alpha= initial_size;
		double* temp_array_weights =  Malloc(double,initial_size);
		for(int m=0; m<coreNum; m++)
			temp_array_weights[m] = outAlpha[m];
		
		outAlpha = Malloc(double,initial_size);
		
		for(int m=0; m<coreNum; m++){
			outAlpha[m] = temp_array_weights[m];
		}

		if(param->randomized)
			gradientActivePoints = Malloc(Qfloat,initial_size);
		
		if(!param->randomized)
			gradientALLPoints = Malloc(Qfloat,prob->l);
		
		if(param->randomized){//initialize array for caching gradient of ACTIVE points
			for(int m=0; m<coreNum; m++){
				Qfloat* Qm = kernelQ->get_Q(coreIdx[m],coreNum, coreIdx);
				gradientActivePoints[m] = 0.0;
				for (int j=0; j<coreNum; j++)
					gradientActivePoints[m] += (Qfloat)Qm[j]*outAlpha[j]; 
			}
		} else {

			for(int m=0; m<prob->l; m++){//initialize array for caching gradient of ALL points
				Qfloat* Qm = kernelQ->get_Q(m,coreNum,coreIdx);
				gradientALLPoints[m] = 0.0;
				for (int j=0; j<coreNum; j++)
					gradientALLPoints[m] += (Qfloat)Qm[j]*outAlpha[j]; 
			}
		}

		for(int m=coreNum; m<initial_size; m++){
			outAlpha[m] = 0.0;
		}

		free(temp_array_weights);
		return true;
}


bool FW_L2SVM_Solver::CheckMemoryForFW(){

	if (this->coreNum >= allocated_size_for_alpha) {	
		allocated_size_for_alpha = (int)(1.5*allocated_size_for_alpha);
		outAlpha = (double*)realloc(outAlpha,allocated_size_for_alpha*sizeof(double));
		if(param->randomized)
			gradientActivePoints = (Qfloat*)realloc(gradientActivePoints,allocated_size_for_alpha*sizeof(Qfloat));
		for(int k=this->coreNum;k<allocated_size_for_alpha; k++){
			outAlpha[k] = 0.0;
			if(param->randomized)
				gradientActivePoints[k] = 0.0;
		}
	}
}

bool FW_L2SVM_Solver::FreeMemoryForFW(){

	if(param->randomized){
			free(gradientActivePoints); 	
	} else{
			free(gradientALLPoints);
	}
}

//Problem is: min f(alpha) = 0.5(alpha^T Q alpha) s.t. 1^Talpha = 1, alpha >= 0
int FW_L2SVM_Solver::Solve(int num_basis, double FW_eps, int method, bool cooling, bool randomized){
	
	 int status = -1;
	 bool glotonous = true;

	 this->Initialize();

	 switch(method)
	 {
	 	case YILDRIM1:
			status = this->StandardFW(num_basis, FW_eps, cooling, randomized);
	 		break;
	 	case YILDRIM2:			
	 		status = this->MFW(num_basis, FW_eps, cooling, randomized);
	 		break;
	 	case PARTAN:			
	 		status = this->PartanFW(num_basis, FW_eps, cooling, randomized);
	 		break;
	 	case SWAP:			
	 		status = this->SWAPFW(num_basis, FW_eps, cooling, randomized);
	 		break;
	 	case lSWAP:			
	 		status = this->SWAPFW(num_basis, FW_eps, cooling, randomized);
	 		break;	
	 	default:
	 		throw std::invalid_argument( "FW-based-SVMs ... unvailable algorithm" );
	 		break;
	 }

	 this->FreeMemoryForFW();
	 return status;
}

//Problem is: min f(alpha) = 0.5(alpha^T Q alpha) s.t. 1^Talpha = 1, alpha >= 0
//Search i* = arg min_i grad f(alpha)
double FW_L2SVM_Solver::TowardVertex(int &towardIdx){


    int randomCoordinate = -1;
    double randomCoordinateGrad = INFINITY;
    double minGradient = INFINITY;

	if(param->randomized){//randomized search of toward vertex
		Qfloat *Qcolumn = NULL;
		previousQcolumn = Q_actives_dot_toward;
		int count_sampling = 0;
		for(int count_repetitions=0; count_repetitions< this->nsamplings_randomized_iterations; count_repetitions++){
			while(count_sampling < param->sample_size){
			 	randomCoordinate = ChooseRandomIndex(false);
			 	randomCoordinateGrad = ComputeGradientCoordinate(randomCoordinate, &Qcolumn);
			 	if(randomCoordinateGrad < minGradient){
			 		towardIdx = randomCoordinate;
			 		minGradient = randomCoordinateGrad;	
			 		Q_actives_dot_toward = Qcolumn; 
			 	}
			 	count_sampling++;
			}
		}
		if(Q_actives_dot_toward == NULL)
			Q_actives_dot_toward = kernelQ->get_Q(towardIdx,coreNum,coreIdx);
					
	} else {//not randomized search of toward vertex
			for (int m=0; m<prob->l; m++){
					if(gradientALLPoints[m] < minGradient){
				 		minGradient = gradientALLPoints[m];
				 		towardIdx = m;
				 	}	
				}
			previousQcolumn = NULL;
			Q_actives_dot_toward = NULL;
	}

	return minGradient;
}

double FW_L2SVM_Solver::safe_stopping_check(double Objective, double &dualGap, int &towardIdx, double &towardGrad){
	printf("** SAFE STOPPING CHECK FOR RANDOMIZED ITERATIONS ...\n");
	printf("** .. CURRENT DUAL GAP: %g ...\n",dualGap);
	nsamplings_randomized_iterations = param->nsamplings_safe_stopping;
	int toward_vertex_check = toward_vertex;
	double toward_gradient_check = TowardVertex(toward_vertex_check);
	if(toward_gradient_check < toward_gradient){
		towardGrad = toward_gradient_check;
		towardIdx = toward_vertex_check;
	}
	dualGap = 4.0*Objective - 2.0*towardGrad;
	nsamplings_randomized_iterations = param->nsamplings_iterations;
	printf("** .. FINAL DUAL GAP: %g ...\n",dualGap);
	return dualGap;	
}

int FW_L2SVM_Solver::StandardFW(int num_basis, double convergence_eps, bool cooling, bool randomized){

	printf("FW Solver: Standard FW (with line search), ");
	cooling ? printf("cooling : YES, ") : printf("cooling : NO, ");
	randomized ? printf("randomized : YES\n") : printf("randomized : NO\n");


    FILE* pathFile;
    pathFile = fopen(param->filename,"w");   
 	
	greedy_it = 0;

	AllocateMemoryForFW(1000);

    double epsilonFactor = EPS_SCALING;
	double currentEpsilon = cooling ? INITIAL_EPS : convergence_eps/epsilonFactor;
		
	int fremessages = param->frecuency_messages;
	if(fremessages <= 0)
		fremessages = 500;
	
	
	//iterate until desired precision 
	while(currentEpsilon > convergence_eps){// solve problem with current epsilon (warm start from the previous solution)
		currentEpsilon *= epsilonFactor;		
		currentEpsilon = (currentEpsilon < convergence_eps) ? convergence_eps: currentEpsilon;
		printf("EPS-Iteration: Iterating to achieve EPS = %g Current.EPS = %g N.Active.Points = %d\n",convergence_eps,currentEpsilon,coreNum);
		
	
		double objective = this->objective;
		
		toward_vertex = -1;
		toward_gradient = TowardVertex(toward_vertex);
			
		double dual_gap = 4.0*objective - 2.0*toward_gradient;

		while(dual_gap < -TAU){
			toward_gradient = TowardVertex(toward_vertex);
			dual_gap = 4.0*objective - 2.0*toward_gradient;
		}

		if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping))
			safe_stopping_check(objective, dual_gap, toward_vertex, toward_gradient);
	
		printf("** INITIAL OBJECTIVE = %g\n",objective);
		printf("** INITIAL COMPUTED DUAL GAP = %g\n",dual_gap);
	
		fprintf(pathFile,"%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);

		while((dual_gap > currentEpsilon) && (greedy_it <= param->max_iterations)){

				greedy_it++; 
				
				if(greedy_it%fremessages==0){
					fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);
					printf("Iterations=%d, Objective = %g Computed-GAP=%g\n",greedy_it,objective,dual_gap);
				}

				/* Toward Step */

				if(inverted_coreIdx[toward_vertex] < 0){
				//there is a new active point ...
					this->CheckMemoryForFW();
					coreIdx[coreNum] = toward_vertex;
					inverted_coreIdx[toward_vertex] = coreNum;
					coreNum++;
				}

				//Qfloat *Q_actives_dot_toward = kernelQ->get_Q(toward_vertex,coreNum,coreIdx);//we only explicitly need Qii but having the updated column in cache is likely useful and probably we only will need to calculate the last entry, i.e., Qii
				//double Qii = Q_actives_dot_toward[inverted_coreIdx[toward_vertex]];//recall: the toward vertex is now an active point
				
				double Qii = kernelQ->kernel_eval(toward_vertex,toward_vertex); //equivalent computation of Qii
			
				double step_size = dual_gap/(2.0*Qii - 4.0*toward_gradient + 4.0*objective);
				double delta_objective_paper = 0.5*(dual_gap*step_size)/2.0;
				
				double new_objective = (1.0-step_size)*(1.0-step_size)*objective + step_size*(1.0-step_size)*toward_gradient + 0.5*step_size*step_size*Qii; 
				double improvement = objective - new_objective;//should be the same as delta_objective_paper
				
				objective = new_objective;//update the objective function value

				for(int i=0; i < coreNum; i++)  
					outAlpha[i] = outAlpha[i]*(1.0-step_size);

				outAlpha[inverted_coreIdx[toward_vertex]] += step_size;
	   			
	   			clock_t time_update_start = clock ();
					
				if(param->randomized){//case RANDOMIZED: update gradient of ACTIVE points
					for (int m=0; m<coreNum; m++){
							if(m==inverted_coreIdx[toward_vertex])
								gradientActivePoints[m] = (1.0-step_size)*toward_gradient + step_size*Qii;
							else
								gradientActivePoints[m] = (1.0-step_size)*gradientActivePoints[m] + step_size*Q_actives_dot_toward[m];
						    //nabla g_{k+1,m} = (1-lambda)*nabla g_{k,m} + lambda* Q_{m,i} ; where m is the current active point and i is the toward vertex
					}

				} else{ //case NOT RANDOMIZED: update gradient of ALL points
					for (int m=0; m<prob->l; m++){
						if(m==toward_vertex)
							gradientALLPoints[m] = (1.0-step_size)*toward_gradient + step_size*Qii;
						else
							gradientALLPoints[m] = (1.0-step_size)*gradientALLPoints[m] + step_size*kernelQ->kernel_eval(toward_vertex,m);
					    //nabla g_{k+1,m} = (1-lambda)*nabla g_{k,m} + lambda* Q_{m,i} ; where m is the current active point and i is the toward vertex 		
					}
				}
					
	   			toward_gradient = TowardVertex(toward_vertex);
				clock_t time_update_end = clock ();
				
				if(greedy_it%fremessages==0){
					//printf("EPS-Iteration: Iterating to achieve EPS = %g Current.EPS = %g N.Active.Points = %d\n",convergence_eps,currentEpsilon,coreNum);
					printf("TIME LAST TOWARD = %g\n",(double)(time_update_end - time_update_start)/CLOCKS_PER_SEC);
				}

				dual_gap = 4.0*objective - 2.0*toward_gradient;

					
				while(dual_gap < -TAU){
					toward_gradient = TowardVertex(toward_vertex);
					dual_gap = 4.0*objective - 2.0*toward_gradient;
				}

				if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping))
					safe_stopping_check(objective, dual_gap, toward_vertex, toward_gradient);
			}

		printf("** FINAL OBJECTIVE = %g\n",objective);
		printf("** FINAL DUAL GAP = %g\n",dual_gap);
		printf("** ITERATIONS = %d\n",greedy_it);
	    fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);

	}	

	fclose(pathFile);

	return 1;
}


//Problem is: min f(alpha) = 0.5(alpha^T Q alpha) s.t. 1^Talpha = 1, alpha >= 0
//Choose j* = arg max_i grad f(alpha) s.t. i is in the set of active variables
double FW_L2SVM_Solver::AwayVertex(int &awayIdx){

	int tempCoordinate = -1;
    double tempCoordinateGrad = -INFINITY;
    double maxGradient = -INFINITY;
    Qfloat *Qcolumn = NULL;

   
	for(int j=0; j<coreNum; j++){
		tempCoordinate = coreIdx[j];
		tempCoordinateGrad = ComputeGradientCoordinate(tempCoordinate, &Qcolumn);
		if((tempCoordinateGrad > maxGradient) && (outAlpha[j] > TAU)){
			awayIdx = tempCoordinate;
			maxGradient = tempCoordinateGrad;	
			Q_actives_dot_away = Qcolumn; 
		}
	}
	
	if((param->randomized) && (Q_actives_dot_away == NULL))
			Q_actives_dot_away = kernelQ->get_Q(awayIdx,coreNum,coreIdx);
	
	return maxGradient;
}

int FW_L2SVM_Solver::MFW(int num_basis, double convergence_eps, bool cooling, bool randomized){

	printf("FW Solver: Modified FW (with line search), ");
	cooling ? printf("cooling : YES, ") : printf("cooling : NO, ");
	randomized ? printf("randomized : YES\n") : printf("randomized : NO\n");

    FILE* pathFile;
    pathFile = fopen(param->filename,"w");   
 	
	greedy_it = 0;

	AllocateMemoryForFW(1000);

    double epsilonFactor = EPS_SCALING;
	double currentEpsilon = cooling ? INITIAL_EPS : convergence_eps/epsilonFactor;
		
	int fremessages = param->frecuency_messages;
	if(fremessages <= 0)
		fremessages = 500;
	 

	int away_steps = 0;
	int toward_steps = 0;
	int drop_steps = 0;

	//iterate until desired precision 
	while(currentEpsilon > convergence_eps){// solve problem with current epsilon (warm start from the previous solution)
		currentEpsilon *= epsilonFactor;		
		currentEpsilon = (currentEpsilon < convergence_eps) ? convergence_eps: currentEpsilon;
		printf("EPS-Iteration: Iterating to achieve EPS = %g Current.EPS = %g N.Active.Points = %d\n",convergence_eps,currentEpsilon,coreNum);
		
	
		double objective = this->objective;
		
		toward_vertex = -1;
		away_vertex = -1;
		toward_gradient = TowardVertex(toward_vertex);
		away_gradient = AwayVertex(away_vertex);
	 	
				
		double dual_gap = 4.0*objective - 2.0*toward_gradient;
		double the_other_gap = 2.0*away_gradient - 4.0*objective;//nabla g_j* - g

		while(dual_gap < -TAU){
			toward_gradient = TowardVertex(toward_vertex);
			dual_gap = 4.0*objective - 2.0*toward_gradient;
		}

		if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping))
			safe_stopping_check(objective, dual_gap, toward_vertex, toward_gradient);
	

		printf("** INITIAL OBJECTIVE = %g\n",objective);
		printf("** INITIAL DUAL GAP = %g\n",dual_gap);

		fprintf(pathFile,"%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,the_other_gap);
	
		while((dual_gap > currentEpsilon) && (greedy_it <= param->max_iterations)){

				greedy_it++; 

				/* Decide Type of Step */
				double Qii = kernelQ->kernel_eval(toward_vertex,toward_vertex); 
				double Qjj = kernelQ->kernel_eval(away_vertex,away_vertex); 
			
				double step_toward = dual_gap/(2.0*Qii - 4.0*toward_gradient + 4.0*objective);
				double step_away = (2.0*away_gradient-4.0*objective)/(2.0*Qjj - 4.0*away_gradient + 4.0*objective);		

				double delta_pos = 0.5*dual_gap;
				double delta_neg = 0.5*(2.0*away_gradient-4.0*objective);
				//double delta_objective_toward = 0.5*(dual_gap*step_toward)/2.0;
				//double delta_objective_toward = objective - ((1.0-step_toward)*(1.0-step_toward)*objective + step_toward*(1.0-step_toward)*toward_gradient + 0.5*step_toward*step_toward*Qii); 
				//double delta_objective_away = 0.5*((2.0*away_gradient-4.0*objective)*step_away)/2.0; //this is the un-constrained delta objective
				//double delta_objective_away = objective - ((1.0+step_away)*(1.0+step_away)*objective - step_away*(1.0+step_away)*away_gradient + 0.5*step_away*step_away*Qjj);
				
				/* Perform Step */

				if(delta_pos >= delta_neg){
						/* Toward Step */
						toward_steps++;
						if(greedy_it%fremessages==0){
							fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,the_other_gap);
							printf("Iterations=%d, Step=TOWARD, Objective = %g Dual-GAP = %g, N.Active.Points = %d\n",greedy_it,objective,dual_gap,coreNum);
								
						}

						if(inverted_coreIdx[toward_vertex] < 0){
						//there is a new active point ...
							this->CheckMemoryForFW();
							coreIdx[coreNum] = toward_vertex;
							inverted_coreIdx[toward_vertex] = coreNum;
							coreNum++;
						}

						double step_size = step_toward;
					
						if(param->randomized){//case RANDOMIZED: update gradient of ACTIVE points
							for (int m=0; m<coreNum; m++){
								if(m==inverted_coreIdx[toward_vertex])
									gradientActivePoints[m] = (1.0-step_size)*toward_gradient + step_size*Qii;
								else
									gradientActivePoints[m] = (1.0-step_size)*gradientActivePoints[m] + step_size*Q_actives_dot_toward[m];
							    //nabla g_{k+1,m} = (1-lambda)*nabla g_{k,m} + lambda* Q_{m,i} ; where m is the current active point and i is the toward vertex
							}
						} else{
							for (int m=0; m<prob->l; m++){
								if(m==toward_vertex)
									gradientALLPoints[m] = (1.0-step_size)*toward_gradient + step_size*Qii;
								else
									gradientALLPoints[m] = (1.0-step_size)*gradientALLPoints[m] + step_size*kernelQ->kernel_eval(toward_vertex,m);
							    //nabla g_{k+1,m} = (1-lambda)*nabla g_{k,m} + lambda* Q_{m,i} ; where m is the current active point and i is the toward vertex
							}
						}
							
						double new_objective = (1.0-step_size)*(1.0-step_size)*objective + step_size*(1.0-step_size)*toward_gradient + 0.5*step_size*step_size*Qii; 
						double improvement = objective - new_objective;//should be the same as delta_objective_paper
						objective = new_objective;

						for(int i=0; i < coreNum; i++)  
							outAlpha[i] = outAlpha[i]*(1.0-step_size);

						outAlpha[inverted_coreIdx[toward_vertex]] += step_size;
	   			
				}//end away step

				if(delta_pos < delta_neg){
						/* Away Step */
						away_steps++;

						double alpha_j = outAlpha[inverted_coreIdx[away_vertex]];
						double limit_step_away = alpha_j/(1.0-alpha_j);
						if(step_away > limit_step_away)//feasible step-size
							step_away = limit_step_away;

	
						if(greedy_it%fremessages==0){
								fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,the_other_gap);
								printf("Iterations=%d, Step=AWAY, Objective = %g Dual-GAP = %g, N.Active.Points = %d\n",greedy_it,objective,dual_gap,coreNum);
						}

						double step_size = step_away;
						if(step_away == limit_step_away)
							drop_steps++;
			
						if(param->randomized){//case RANDOMIZED: update gradient of ACTIVE points
							for (int m=0; m<coreNum; m++){
								if(m==inverted_coreIdx[away_vertex])
									gradientActivePoints[m] = (1.0+step_size)*away_gradient - step_size*Qjj;
								else
									gradientActivePoints[m] = (1.0+step_size)*gradientActivePoints[m] - step_size*Q_actives_dot_away[m];
							 //nabla g_{k+1,m} = (1+lambda)*nabla g_{k,m} - lambda* Q_{m,j} ; where m is the current active point and j is the away vertex
							}				
						} else {//case NOT RANDOMIZED: update gradient of ALL points
							for (int m=0; m<prob->l; m++){
								if(m==away_vertex)
									gradientALLPoints[m] = (1.0+step_size)*away_gradient - step_size*Qjj;
								else
									gradientALLPoints[m] = (1.0+step_size)*gradientALLPoints[m] -  step_size*kernelQ->kernel_eval(away_vertex,m);
							    //nabla g_{k+1,m} = (1+lambda)*nabla g_{k,m} - lambda* Q_{m,j} ; where m is the current active point and j is the away vertex
							}
						}

						double new_objective = (1.0+step_size)*(1.0+step_size)*objective - step_size*(1.0+step_size)*away_gradient + 0.5*step_size*step_size*Qjj; 
						
						objective = new_objective;

						for(int i=0; i < coreNum; i++)  
							outAlpha[i] = outAlpha[i]*(1.0+step_size);

						outAlpha[inverted_coreIdx[away_vertex]] -= step_size;

				}

	   			clock_t time_update_start = clock ();
			
	   			toward_gradient = TowardVertex(toward_vertex);
				away_gradient = AwayVertex(away_vertex);

				clock_t time_update_end = clock ();
				
				dual_gap = 4.0*objective - 2.0*toward_gradient;
				the_other_gap = 2.0*away_gradient - 4.0*objective;//nabla g_j* - g

				while(dual_gap < -TAU){
					toward_gradient = TowardVertex(toward_vertex);
					dual_gap = 4.0*objective - 2.0*toward_gradient;
				}

				if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping))
					safe_stopping_check(objective, dual_gap, toward_vertex, toward_gradient);
				
		}

		printf("** FINAL OBJECTIVE = %g\n",objective);
		printf("** FINAL DUAL GAP = %g\n",dual_gap);
		printf("** TOWARD-STEPS=%d, AWAY-STEPS=%d, DROP-STEPS=%d\n",toward_steps,away_steps,drop_steps);

		fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,the_other_gap);
		
	}	

	fclose(pathFile);
	return 1;
}

int FW_L2SVM_Solver::SWAPFW(int num_basis, double convergence_eps, bool cooling, bool randomized){


	printf("FW Solver: SWAP-FW (1st order), ");
	cooling ? printf("cooling : YES, ") : printf("cooling : NO, ");
	randomized ? printf("randomized : YES\n") : printf("randomized : NO\n");


    FILE* pathFile;
    pathFile = fopen(param->filename,"w");   
 	
	greedy_it = 0;

	AllocateMemoryForFW(1000);

    double epsilonFactor = EPS_SCALING;
	double currentEpsilon = cooling ? INITIAL_EPS : convergence_eps/epsilonFactor;
		
	int fremessages = 500;  

	int swap_steps = 0;
	int toward_steps = 0;
	int drop_steps = 0;

	//iterate until desired precision 
	while(currentEpsilon > convergence_eps){// solve problem with current epsilon (warm start from the previous solution)
		currentEpsilon *= epsilonFactor;		
		currentEpsilon = (currentEpsilon < convergence_eps) ? convergence_eps: currentEpsilon;
		printf("EPS-Iteration: Iterating to achieve EPS = %g Current.EPS = %g N.Active.Points = %d\n",convergence_eps,currentEpsilon,coreNum);
		
	
		double objective = this->objective;
		
		toward_vertex = -1;
		away_vertex = -1;
		toward_gradient = TowardVertex(toward_vertex);
		away_gradient = AwayVertex(away_vertex);


		double dual_gap = 4.0*objective - 2.0*toward_gradient;
		double the_other_gap = 2.0*away_gradient - 4.0*objective;//nabla g_j* - g
	
		while(dual_gap < -TAU){
			toward_gradient = TowardVertex(toward_vertex);
			dual_gap = 4.0*objective - 2.0*toward_gradient;
		}

		if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping))
			safe_stopping_check(objective, dual_gap, toward_vertex, toward_gradient);
	

		printf("** INITIAL OBJECTIVE = %g\n",objective);
		printf("** INITIAL DUAL GAP = %g\n",dual_gap);

		fprintf(pathFile,"%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,the_other_gap);
	
		while((dual_gap > currentEpsilon) && (greedy_it <= param->max_iterations)){

				greedy_it++; 
				double Qii=0.0, Qjj=0.0, Qij=0.0;
				double step_toward = 0.0, delta_objective_toward = 0.0;
				double step_swap = 0.0;
				double alpha_j = 0.0;
				double delta_objective_swap = -INFINITY;
				double limit_step_away = 0.0;

				/* Decide Type of Step */
				Qii = kernelQ->kernel_eval(toward_vertex,toward_vertex); 
				Qjj = kernelQ->kernel_eval(away_vertex,away_vertex); 
				Qij = kernelQ->kernel_eval(away_vertex,toward_vertex); 

				step_toward = dual_gap/(2.0*Qii - 4.0*toward_gradient + 4.0*objective);
				delta_objective_toward = 0.5*(dual_gap*step_toward)/2.0;
					
				if((2.0*Qii - 4.0*Qij + 2.0*Qjj)>TAU){

					step_swap = (2.0*away_gradient-2.0*toward_gradient)/(2.0*Qii - 4.0*Qij + 2.0*Qjj);		
					alpha_j = outAlpha[inverted_coreIdx[away_vertex]];
					limit_step_away = alpha_j;
					if(step_swap > limit_step_away)//feasible step-size
						step_swap = limit_step_away;
					delta_objective_swap = step_swap*(away_gradient-toward_gradient) - 0.5*step_swap*step_swap*(Qii - 2.0*Qij + Qjj);

				}
			 
				/* Perform Step */

				if(delta_objective_toward > delta_objective_swap){

						/* Toward Step */
						toward_steps++;
						if(greedy_it%fremessages==0){
							fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,the_other_gap);
							printf("Iterations=%d, Step=TOWARD, Objective = %g Dual-GAP = %g, N.Active.Points = %d\n",greedy_it,objective,dual_gap,coreNum);
								
						}

						if(inverted_coreIdx[toward_vertex] < 0){
						//there is a new active point ...
							this->CheckMemoryForFW();
							coreIdx[coreNum] = toward_vertex;
							inverted_coreIdx[toward_vertex] = coreNum;
							coreNum++;
						}

						double step_size = step_toward;
					
						if(param->randomized){//case RANDOMIZED: update gradient of ACTIVE points
							for (int m=0; m<coreNum; m++){
								if(m==inverted_coreIdx[toward_vertex])
									gradientActivePoints[m] = (1.0-step_size)*toward_gradient + step_size*Qii;
								else
									gradientActivePoints[m] = (1.0-step_size)*gradientActivePoints[m] + step_size*Q_actives_dot_toward[m];
							    //nabla g_{k+1,m} = (1-lambda)*nabla g_{k,m} + lambda* Q_{m,i} ; where m is the current active point and i is the toward vertex
							}
						} else{
							for (int m=0; m<prob->l; m++){
								if(m==toward_vertex)
									gradientALLPoints[m] = (1.0-step_size)*toward_gradient + step_size*Qii;
								else
									gradientALLPoints[m] = (1.0-step_size)*gradientALLPoints[m] + step_size*kernelQ->kernel_eval(toward_vertex,m);
							    //nabla g_{k+1,m} = (1-lambda)*nabla g_{k,m} + lambda* Q_{m,i} ; where m is the current active point and i is the toward vertex
							}
						}
							
						double new_objective = (1.0-step_size)*(1.0-step_size)*objective + step_size*(1.0-step_size)*toward_gradient + 0.5*step_size*step_size*Qii; 
						double improvement = objective - new_objective;//should be the same as delta_objective_paper
						objective = new_objective;

						for(int i=0; i < coreNum; i++)  
							outAlpha[i] = outAlpha[i]*(1.0-step_size);

						outAlpha[inverted_coreIdx[toward_vertex]] += step_size;
	  
				}//end away step

				if(delta_objective_swap >= delta_objective_toward){
						/* Away Step */
						swap_steps++;	
						if(greedy_it%fremessages==0){
								fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,the_other_gap);
								printf("Iterations=%d, Step=SWAP, Objective = %g Dual-GAP = %g, N.Active.Points = %d\n",greedy_it,objective,dual_gap,coreNum);
						}

						if(inverted_coreIdx[toward_vertex] < 0){
						//there is a new active point ...
							this->CheckMemoryForFW();
							coreIdx[coreNum] = toward_vertex;
							inverted_coreIdx[toward_vertex] = coreNum;
							coreNum++;
						}

						double step_size = step_swap;
						if(step_swap == limit_step_away)
							drop_steps++;

			
						if(param->randomized){//case RANDOMIZED: update gradient of ACTIVE points
							for (int m=0; m<coreNum; m++){
								if(m==inverted_coreIdx[away_vertex])
									gradientActivePoints[m] = away_gradient + step_size*Qij - step_size*Qjj;
								else if(m==inverted_coreIdx[toward_vertex])
									gradientActivePoints[m] = toward_gradient + step_size*Qii - step_size*Qij;		
								else
									gradientActivePoints[m] = gradientActivePoints[m] + step_size*Q_actives_dot_toward[m] - step_size*Q_actives_dot_away[m];
							}				
						} else {//case NOT RANDOMIZED: update gradient of ALL points
							for (int m=0; m<prob->l; m++){
								if(m==away_vertex)
									gradientALLPoints[m] = away_gradient + step_size*Qij - step_size*Qjj;
								else if(m==toward_vertex)
									gradientALLPoints[m] = toward_gradient + step_size*Qii - step_size*Qij;		
								else
									gradientALLPoints[m] = gradientALLPoints[m] + step_size*kernelQ->kernel_eval(toward_vertex,m) - step_size*kernelQ->kernel_eval(away_vertex,m);
							}
						}

						double new_objective = objective-delta_objective_swap;

						objective = new_objective;
						outAlpha[inverted_coreIdx[toward_vertex]] += step_size;
						outAlpha[inverted_coreIdx[away_vertex]] -= step_size;
				}

	   			clock_t time_update_start = clock ();
			
	   			toward_gradient = TowardVertex(toward_vertex);
				away_gradient = AwayVertex(away_vertex);

				clock_t time_update_end = clock ();
	
				dual_gap = 4.0*objective - 2.0*toward_gradient;
				the_other_gap = 2.0*away_gradient - 4.0*objective;//nabla g_j* - g

				while(dual_gap < -TAU){
					toward_gradient = TowardVertex(toward_vertex);
					dual_gap = 4.0*objective - 2.0*toward_gradient;
				}

				if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping))
					safe_stopping_check(objective, dual_gap, toward_vertex, toward_gradient);
				

		}

		printf("** FINAL OBJECTIVE = %g\n",objective);
		printf("** FINAL DUAL GAP = %g\n",dual_gap);
		printf("** TOWARD-STEPS=%d, SWAP-STEPS=%d, DROP-STEPS=%d\n",toward_steps,swap_steps,drop_steps);
		printf("** ITERATIONS = %d\n",greedy_it);

		fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,the_other_gap);
		
	}	

	fclose(pathFile);

	return 1;
}


int FW_L2SVM_Solver::PartanFW(int num_basis, double convergence_eps, bool cooling, bool randomized){

	printf("FW Solver: PARTAN FW (with line search), ");
	cooling ? printf("cooling : YES, ") : printf("cooling : NO, ");
	randomized ? printf("randomized : YES\n") : printf("randomized : NO\n");


    FILE* pathFile;
    pathFile = fopen(param->filename,"w");   
 	
	greedy_it = 0;

	AllocateMemoryForFW(1000);

	double* previous_alpha = Malloc(double,allocated_size_for_alpha);
	for(int i=0; i < allocated_size_for_alpha; i++)
		previous_alpha[i]=0.0;
	
	Qfloat* prevGradientALLPoints=NULL;
	Qfloat* prevGradientActivePoints=NULL;
	
	if(param->randomized)
		prevGradientActivePoints = Malloc(Qfloat,allocated_size_for_alpha);
	else 
		prevGradientALLPoints = Malloc(Qfloat,prob->l); 

	int   	previous_toward_vertex;
	double  new_objective = 0.0;
	double  previous_objective = 0.0;
	double  previous_step_PT = 0.0;
	double  previous_step_FW = 0.0;
	double  previous_toward_gradient = 0.0;
	double  step_PT = 0.0;
	double  Wk = 0.0;
	
	double epsilonFactor = EPS_SCALING;
	double currentEpsilon = cooling ? INITIAL_EPS : convergence_eps/epsilonFactor;
	
	double previous_gradient_at_previous_toward = 0.0;
	double previous_gradient_at_current_toward = 0.0;
	
	int fremessages = param->frecuency_messages;
	
	//iterate until desired precision 
	while(currentEpsilon > convergence_eps){// solve problem with current epsilon (warm start from the previous solution)
		currentEpsilon *= epsilonFactor;		
		currentEpsilon = (currentEpsilon < convergence_eps) ? convergence_eps: currentEpsilon;
		printf("EPS-Iteration: Iterating to achieve EPS = %g Current.EPS = %g N.Active.Points = %d\n",convergence_eps,currentEpsilon,coreNum);
		
	
		double objective = this->objective;
		
		toward_vertex = -1;
		toward_gradient = TowardVertex(toward_vertex);
		
		double dual_gap = 4.0*objective - 2.0*toward_gradient;
		
		while(dual_gap < -TAU){
			toward_gradient = TowardVertex(toward_vertex);
			dual_gap = 4.0*objective - 2.0*toward_gradient;
		}

		if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping))
			safe_stopping_check(objective, dual_gap, toward_vertex, toward_gradient);
	

		printf("** INITIAL OBJECTIVE = %g\n",objective);
		printf("** INITIAL COMPUTED DUAL GAP = %g\n",dual_gap);
	
		fprintf(pathFile,"%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);

		bool perform_partan = false;
		previous_objective = objective;

		while((dual_gap > currentEpsilon) && (greedy_it <= param->max_iterations)){

				greedy_it++; 
				
				if(greedy_it%fremessages==0){
					fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);
					printf("Iterations=%d, Objective = %g Computed-GAP=%g, N.Active.Points = %d\n",greedy_it,objective,dual_gap,coreNum);
				}

				/* Toward Step */
				bool there_is_new_active = false;
				if(inverted_coreIdx[toward_vertex] < 0){
				//there is a new active point ...
				   if (this->coreNum >= allocated_size_for_alpha) {	
						allocated_size_for_alpha = (int)(1.5*allocated_size_for_alpha);
						
						outAlpha = (double*)realloc(outAlpha,allocated_size_for_alpha*sizeof(double));
						if(param->randomized)
							gradientActivePoints = (Qfloat*)realloc(gradientActivePoints,allocated_size_for_alpha*sizeof(Qfloat));
						
						previous_alpha = (double*)realloc(previous_alpha,allocated_size_for_alpha*sizeof(double));
						if(param->randomized)
							prevGradientActivePoints = (Qfloat*)realloc(prevGradientActivePoints,allocated_size_for_alpha*sizeof(Qfloat));
						
						for(int k=this->coreNum;k<allocated_size_for_alpha; k++){
							outAlpha[k] = 0.0;
							previous_alpha[k] = 0.0;
							if(param->randomized){
								prevGradientActivePoints[k] = 0.0;
								gradientActivePoints[k]=0.0;
							}
						}
					}

					there_is_new_active = true;
					coreIdx[coreNum] = toward_vertex;
					inverted_coreIdx[toward_vertex] = coreNum;
					outAlpha[coreNum]= 0.0;
					previous_alpha[coreNum] = 0.0;
					
					if(param->randomized)
						gradientActivePoints[coreNum] = toward_gradient;

					coreNum++;
				}
				
				double Qii = kernelQ->kernel_eval(toward_vertex,toward_vertex); //equivalent computation of Qii
			
				double step_FW = dual_gap/(2.0*Qii - 4.0*toward_gradient + 4.0*objective);
				double new_objective_FW = (1.0-step_FW)*(1.0-step_FW)*objective + step_FW*(1.0-step_FW)*toward_gradient + 0.5*step_FW*step_FW*Qii; 
				

				if(perform_partan){ //Perform a PARTAN Step using the Intermediate FW Step
					
					previous_gradient_at_previous_toward = 0.0;
					previous_gradient_at_current_toward = 0.0;

					if(!param->randomized){
						previous_gradient_at_previous_toward = previous_toward_gradient;
						previous_gradient_at_current_toward = prevGradientALLPoints[toward_vertex];
					} else {//randomized
						previous_gradient_at_previous_toward = previous_toward_gradient;

						if(!there_is_new_active){//current toward was active
							previous_gradient_at_current_toward = prevGradientActivePoints[inverted_coreIdx[toward_vertex]];
						} else {//we need to calculate the gradient using the previous weights 
							previous_gradient_at_current_toward = 0.0;
							Qfloat* Qidx = kernelQ->get_Q(toward_vertex, coreNum, coreIdx);
							if(Qidx != NULL){
								for (int j=0; j<coreNum; j++){
									if(j!=inverted_coreIdx[toward_vertex])
										previous_gradient_at_current_toward += Qidx[j]*previous_alpha[j];
								}
							}	
						}
					}

					if(greedy_it==2){//first PARTAN iteration, k=1, we are going to compute alpha_{k+1} = alpha_2

						Wk  = 2.0*(1.0-previous_step_FW)*previous_objective + previous_step_FW*previous_gradient_at_previous_toward;

					} else { // k > 1

						double previousWk = Wk;
						Wk  = 2.0*(1.0+previous_step_PT)*(1.0-previous_step_FW)*previous_objective;
						Wk += (1.0+previous_step_PT)*previous_step_FW*previous_gradient_at_previous_toward;
						Wk -= previous_step_PT*previousWk;
					}
					
					double numerator_step_PT  = 2.0*new_objective_FW - step_FW*previous_gradient_at_current_toward - (1.0-step_FW)*Wk;
					double denominator_step_PT = 2.0*(step_FW*previous_gradient_at_current_toward + (1.0-step_FW)*Wk - new_objective_FW - previous_objective);
					step_PT = numerator_step_PT/denominator_step_PT;

					//CLIPPING STEP_PT
					double upper_limit = INFINITY;
					double lower_limit = -INFINITY;
					double limit_i = INFINITY;

					for(int i=0; i < coreNum; i++){ 
						double difference = previous_alpha[i] - (1.0-step_FW)*outAlpha[i];
						limit_i = (1.0-step_FW)*outAlpha[i];
						if(inverted_coreIdx[toward_vertex]==i){
								limit_i += step_FW;
								difference -= step_FW; 
						}
						
						limit_i /= difference;

						if((difference>0.0) && (limit_i < upper_limit)){
							upper_limit = limit_i;
						} 

						if((difference<0.0) && (limit_i > lower_limit)){
							lower_limit = limit_i;
						} 
					}

					if(step_PT > upper_limit)
						step_PT = upper_limit;

					if(step_PT < lower_limit)
						step_PT = lower_limit;
					
					//END CLIPPING STEP_PT

					double new_objective_PT  = (1.0+step_PT)*(1.0+step_PT)*new_objective_FW - step_PT*(1.0+step_PT)*(1.0-step_FW)*Wk;
					new_objective_PT -= step_PT*(1.0+step_PT)*step_FW*previous_gradient_at_current_toward;
					new_objective_PT += step_PT*step_PT*previous_objective;  
				
					double new_weight = 0.0;
				

					for(int i=0; i < coreNum; i++){ 

						new_weight = (1.0 + step_PT - step_FW - step_PT*step_FW)*outAlpha[i] - step_PT*previous_alpha[i];
						if(inverted_coreIdx[toward_vertex]==i)
							new_weight += step_FW + step_FW*step_PT;

						if(new_weight < TAU){
							//printf("NEGATIVE WEIGHT %g - %g\n",new_weight,outAlpha[i]);
							new_weight = 0.0;
						}

						if(new_weight > 1)
	   						printf("WARNING: WEIGHT IS > 1\n");


						previous_alpha[i] = outAlpha[i];
						outAlpha[i] = new_weight;

					}

	   				new_objective = new_objective_PT;
	   				previous_step_PT = step_PT;

				} else {//Perform a Classic FW Step: only first iteration

					for(int i=0; i < coreNum; i++){  
						previous_alpha[i] = outAlpha[i];
						outAlpha[i] = outAlpha[i]*(1.0-step_FW);
					}

					outAlpha[inverted_coreIdx[toward_vertex]] += step_FW;
	   				
	   				new_objective = new_objective_FW;
	   				
				}

				previous_objective = objective;
				objective = new_objective;

				clock_t time_update_start = clock ();
				

				//Update Gradients 
				if(!perform_partan){ //Standard FW Step
						if(param->randomized){//case RANDOMIZED: update gradient of ACTIVE points
							for (int m=0; m<coreNum; m++){
								prevGradientActivePoints[m] = gradientActivePoints[m];
								if(m==inverted_coreIdx[toward_vertex])
									gradientActivePoints[m] = (1.0-step_FW)*toward_gradient + step_FW*Qii;
								else
									gradientActivePoints[m] = (1.0-step_FW)*gradientActivePoints[m] + step_FW*Q_actives_dot_toward[m];
							}
						} else {
							for (int m=0; m<prob->l; m++){
								prevGradientALLPoints[m] = gradientALLPoints[m];
								if(m==toward_vertex)
									gradientALLPoints[m] = (1.0-step_FW)*toward_gradient + step_FW*Qii;
								else
									gradientALLPoints[m] = (1.0-step_FW)*gradientALLPoints[m] + step_FW*kernelQ->kernel_eval(toward_vertex,m);	
							}
						}
				}

				//Update Gradients 
				if(perform_partan){ //Partan Step
						double new_gradient = 0.0;
						if(param->randomized){//case RANDOMIZED: update gradient of ACTIVE points
							for (int m=0; m<coreNum; m++){
									if(m==inverted_coreIdx[toward_vertex])
										new_gradient = (1.0 + step_PT - step_FW - step_PT*step_FW)*toward_gradient - step_PT*previous_gradient_at_current_toward + (step_FW + step_FW*step_PT)*Qii;
									else
										new_gradient = (1.0 + step_PT - step_FW - step_PT*step_FW)*gradientActivePoints[m] - step_PT*prevGradientActivePoints[m] + (step_FW + step_FW*step_PT)*Q_actives_dot_toward[m];
								
								prevGradientActivePoints[m] = gradientActivePoints[m];
								gradientActivePoints[m] = new_gradient;
							}
						} else {
							for (int m=0; m<prob->l; m++){
								if(m==toward_vertex)
									new_gradient = (1.0 + step_PT - step_FW - step_PT*step_FW)*toward_gradient - step_PT*prevGradientALLPoints[m] + (step_FW + step_FW*step_PT)*Qii;
								else
									new_gradient = (1.0 + step_PT - step_FW - step_PT*step_FW)*gradientALLPoints[m] - step_PT*prevGradientALLPoints[m] + (step_FW + step_FW*step_PT)*kernelQ->kernel_eval(toward_vertex,m);
							 	prevGradientALLPoints[m] = gradientALLPoints[m];
							 	gradientALLPoints[m] = new_gradient;
							}
						}
				}

				previous_toward_vertex = toward_vertex;
				previous_toward_gradient = toward_gradient;
				previous_step_FW = step_FW;

	   			toward_gradient = TowardVertex(toward_vertex);
				clock_t time_update_end = clock ();
				
				if(greedy_it%fremessages==0){
					//printf("EPS-Iteration: Iterating to achieve EPS = %g Current.EPS = %g N.Active.Points = %d\n",convergence_eps,currentEpsilon,coreNum);
					printf("TIME LAST TOWARD = %g\n",(double)(time_update_end - time_update_start)/CLOCKS_PER_SEC);
				}

				dual_gap = 4.0*objective - 2.0*toward_gradient;
				int counter = 0;
				while(dual_gap < -TAU){
					toward_gradient = TowardVertex(toward_vertex);
					dual_gap = 4.0*objective - 2.0*toward_gradient;
					counter++;
					if(counter%500==0)
						printf("WARNING\n");
				}

				if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping))
					safe_stopping_check(objective, dual_gap, toward_vertex, toward_gradient);
				

				if(!perform_partan){//activate Partan from the second iteration
					perform_partan = true;
				}

		}

		printf("** FINAL OBJECTIVE = %g\n",objective);
		printf("** FINAL DUAL GAP = %g\n",dual_gap);
		printf("** ITERATIONS = %d\n",greedy_it);
		
	    fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);

	}	

	fclose(pathFile);

	if(param->randomized)
		free(prevGradientActivePoints);
	else
		free(prevGradientALLPoints);

	free(previous_alpha);
	return 1;
}

bool FW_L2SVM_Solver::Initialize(){
	
	coreNum  = 0; 

    if ((init_method == YILDRIM_INIT) || (init_method == YILDRIM_SAMPLE_INIT)){
    //choose initial active points as suggested by Yildirim in "Two Algorithms for the Minimum Enclosing Ball Problem"
    	coreNum  = Yildirim_Initialization();
	}	
    
    else { //choose a random set of active points 
		coreNum  = RandomSet_Initialization();
    }
   

    //Solve Initial Problem Using SMO
	double eps_nested_SMO = param->eps;
	SMO_Solver = new MEB_SMO(coreIdx,coreNum,*kernelQ,tempD,outAlpha,eps_nested_SMO,param->eps,coreNum);
	SMO_Solver->Solve(coreIdx,coreNum,tempD);
	outAlpha = SMO_Solver->getAlpha();//get initial solution
	allocated_size_for_alpha = SMO_Solver->get_allocated_size();
	this->objective = SMO_Solver->computeObj();
	//printf("OBJECTIVE: %f",this->objective);

return true;

}

double FW_L2SVM_Solver::ComputeSVMSolution(double *alpha, double Threshold)
{
	double bias      = 0.0;
	double sumAlpha  = 0.0;
	int i;
	
	for(i = 0; i < coreNum; i++)
	{  
		if (outAlpha[i] > Threshold)
		{
			int ii    = coreIdx[i];
			alpha[ii] = outAlpha[i]*y[ii];
			bias     += alpha[ii];
			sumAlpha += outAlpha[i];
		}
	}


	bias /= sumAlpha;

	for(i = 0; i < coreNum; i++)
		alpha[coreIdx[i]] /= sumAlpha;

	return bias;
}


int FW_L2SVM_Solver::RandomSet_Initialization(){

		posTurn  = true;		
		int initialSampleSize = INITIAL_CS;
		
		if (initialSampleSize > prob->l){
			initialSampleSize = prob->l;
		}

		for(int sampleNum = 0; sampleNum < initialSampleSize; sampleNum++)
		{					
			posTurn = PosNeg(posTurn,pNum,nNum);
			int idx;
			do
			{	
				//balanced random sampling
				int rand32bit = random();
				idx			  = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
				
			} while (inverted_coreIdx[idx] > 0);
			

			if(y[idx] > 0)
				pNum--;
			else
				nNum--;
			

			coreIdx[coreNum]  = idx;
			inverted_coreIdx[idx] = coreNum;
			outAlpha[sampleNum] = 1.0/initialSampleSize;
			tempD[sampleNum]    = 0.0;
			coreNum++;
		}	

		return coreNum;

}

int FW_L2SVM_Solver::FullSet_Initialization(){

        coreNum = 0;
        printf("SOLVING THE FULL PROBLEM USING SMO\n");
		for(int sampleNum = 0; sampleNum < prob->l; sampleNum++)
		{					
			coreIdx[coreNum]  = sampleNum;
			inverted_coreIdx[sampleNum] = coreNum;
			outAlpha[sampleNum] = 1.0/prob->l;
			tempD[sampleNum]    = 0.0;
			coreNum++;
		}	

		return coreNum;

}

int FW_L2SVM_Solver::Yildirim_Initialization(){

    	double min_kernel_product = INFINITY;
		int more_distant_idx = 0;
	
		int idx_limit = (init_method == YILDRIM_INIT) ? prob->l : param->sample_size;
 		int sel_idx;
		
    	for(int idx = 0; idx < idx_limit; idx++)
		{	
			if(init_method == YILDRIM_INIT){
				sel_idx = idx;
							
			} else {
				posTurn = PosNeg(posTurn,posNum,negNum);
				int rand32bit = random();
				sel_idx = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
				
			}
	
			Qfloat Q_i = kernelQ->kernel_eval(0,sel_idx);
			if(Q_i < min_kernel_product){
				min_kernel_product = Q_i;
				more_distant_idx = sel_idx;
			}				
    	
		}
		
		coreNum = 0;
		coreIdx[coreNum]  = more_distant_idx;
		inverted_coreIdx[more_distant_idx] = coreNum;
		outAlpha[coreNum] = 0.5;
		tempD[coreNum]    = 0.0;
		coreNum++;
		
					
		min_kernel_product = INFINITY;
		more_distant_idx = 0;
		
    	for(int idx = 0; idx < idx_limit; idx++)
		{	
			if(init_method == YILDRIM_INIT){
				sel_idx = idx;
							
			} else {
				posTurn = PosNeg(posTurn,posNum,negNum);
				int rand32bit = random();
				sel_idx = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
				
			}

			Qfloat Q_i = kernelQ->kernel_eval(coreIdx[0],sel_idx);
			if((inverted_coreIdx[sel_idx] < 0) && (Q_i < min_kernel_product)){
				//new selected point needs to be inactive and better than the previous one
				min_kernel_product = Q_i;
				more_distant_idx = sel_idx;
			}				
    	
		}
		
		coreIdx[coreNum]  = more_distant_idx;
		inverted_coreIdx[more_distant_idx] = coreNum;
		outAlpha[coreNum] = 0.5;
		tempD[coreNum]    = 0.0;
		coreNum++;
		
		return coreNum;
}