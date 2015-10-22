#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>

	
#include "MEB-solvers.h"
	
inline bool PosNeg(bool PosTurn, int pNum, int nNum)
{
	if (pNum<=0)
		PosTurn = false;
	else if (nNum<=0)
		PosTurn = true;
	else
		PosTurn = !PosTurn;
	return PosTurn;
}

void Solver_CVM::_Init()
{
	// count the distribution of data
	printf("_Init() NO WARM\n");
	
	posIdx  = new int[prob->l];
	posNum	= 0;
	negIdx  = new int[prob->l];
	negNum	= 0;
	y       = new schar[prob->l];
	chklist = new char[prob->l];
	coreIdx = new int[prob->l];
	for(int i = 0; i < prob->l; i++)
	{	
		if (prob->y[i] > 0)
		{
			y[i]             = 1;
			posIdx[posNum++] = i;
		}
		else
		{
			y[i]             = -1;
			negIdx[negNum++] = i;
		}
		
		chklist[i] = 0;
		coreIdx[i] = -1;
	}

	// initialized the kernel
	outAlpha = &tmpAlpha[0];
	
	kernelQ  = new CVC_Q(prob, param, y);		
	
	Eta		 = kernelQ->get_Eta();

	pNum     = posNum;
	nNum     = negNum;	
	
	
	coreNum  = 0; 
	
    if ((init_method == YILDRIM_INIT) || (init_method == YILDRIM_SAMPLE_INIT)){
    	
    	//find the more distant or equivalently the point minimizing the kernel product 
    	//with the current center, which is actually the first point
    	double min_kernel_product = 100*Eta;
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
		chklist[more_distant_idx] = 1;
		outAlpha[coreNum] = 0.5;
		tempD[coreNum]    = 0.0;
		coreNum++;
		
					
		min_kernel_product = 100*Eta;
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
			if((chklist[sel_idx] != 1) && Q_i < min_kernel_product){
				min_kernel_product = Q_i;
				more_distant_idx = sel_idx;
			}				
    	
		}
		
		coreIdx[coreNum]  = more_distant_idx;
		chklist[more_distant_idx] = 1;
		outAlpha[coreNum] = 0.5;
		tempD[coreNum]    = 0.0;
		coreNum++;
		
				
		r2 = (kernelQ->get_Eta() - min_kernel_product)/2;
		coreNorm2 = (kernelQ->get_Eta() + min_kernel_product)/2;


	} //end YILDRIM_INIT	
    
    else { //choose a small subset an build the MEB as initialization
		
		posTurn  = true;		
		int initialSampleSize = INITIAL_CS;
		
		if (initialSampleSize > prob->l){
			printf("changing sampling size l= %d",prob->l);
			initialSampleSize = prob->l-1;
		}
		for(int sampleNum = 0; sampleNum < initialSampleSize; sampleNum++)
		{					
			posTurn = PosNeg(posTurn,pNum,nNum);
			int idx;
			do
			{
				// balanced and random sample
				int rand32bit = random();
				idx			  = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
				
			} while (chklist[idx] > 0);
			chklist[idx] = 1;

			if(y[idx] > 0)
				pNum--;
			else
				nNum--;
			

			coreIdx[coreNum++]  = idx;
			outAlpha[sampleNum] = 1.0/initialSampleSize;
			tempD[sampleNum]    = 0.0;
		}	
	
    }//end random MEB initialization
}

void Solver_CVM::Init(const svm_problem *_prob, const svm_parameter* _param)
{	
	prob  = _prob;
	param = _param;
	showCoreSolverInfo = false;
	tempD = Malloc(double,2*INITIAL_CS);
	tmpAlpha = Malloc(double,2*INITIAL_CS);
	
	_Init();
}

void Solver_CVM::Init(const svm_problem *_prob, const svm_parameter* _param, int* subset_,double *alpha_) 
{
	
	int sample_num = 0;

	double suma_alphas_init = 0;
	for(int prob_it = 0; prob_it < _prob->l; prob_it++)
	{	
		if(subset_[prob_it]>0){
		sample_num++;
		}
		suma_alphas_init += alpha_[prob_it];
	}
	
	//printf("suma alphas iniciales: %f\n",suma_alphas_init);
	prob  = _prob;
	param = _param;
	showCoreSolverInfo = false;
	
	tempD = Malloc(double,2*sample_num);
	tmpAlpha = Malloc(double,2*sample_num);
		
	posIdx  = new int[prob->l];
	posNum	= 0;
	negIdx  = new int[prob->l];
	negNum	= 0;
	y       = new schar[prob->l];
	chklist = new char[prob->l];
	coreIdx = new int[prob->l];
	
	for(int i = 0; i < prob->l; i++)
	{	
		if (prob->y[i] > 0)
		{
			y[i]             = 1;
			posIdx[posNum++] = i;
		}
		else
		{
			y[i]             = -1;
			negIdx[negNum++] = i;
		}
		chklist[i] = 0;
		coreIdx[i] = -1;
	}

	// initialized the kernel
	outAlpha = &tmpAlpha[0];		
	kernelQ  = new CVC_Q(prob, param, y);		

	Eta		 = kernelQ->get_Eta();	
	pNum     = posNum;
	nNum     = negNum;


	// set a subset as initialization	
	
	coreNum  = 0; 	
	sample_num = 0;
	
	for(int prob_it = 0; prob_it < prob->l; prob_it++)
	{	
		if(subset_[prob_it]>0){
			
			chklist[prob_it] = 1;			
			
						 
		tempD[sample_num]    = 0.0;
		outAlpha[sample_num++]    = alpha_[prob_it];
		coreIdx[coreNum++]     = prob_it;
		
		}
	}	
	
	posTurn = PosNeg(posTurn,posNum,negNum);


}

bool Solver_CVM::_Create(double cvm_eps)
{	
	//printf("before SMO constructor\n");
	double eps_SMO_used;
	eps_SMO_used = 0.1*param->eps;
	printf("SMO-eps = %g \n",eps_SMO_used);
	solver = new MEB_SMO(coreIdx,coreNum,*kernelQ,tempD,outAlpha,eps_SMO_used,param->eps,1000);
	return true;
}

double Solver_CVM::_maxDistFromSampling(double maxDistance2, int &maxDistance2Idx)
{
	double tmpdist = coreNorm2 + Eta;
	double dist_tol = maxDistance2;
	
	posTurn        = true;
	
	int sampleNum = 0;
	int samplingSize = param->sample_size;
		
	if (coreNum >= prob->l){
		return maxDistance2; 
	}
	
	if(param->MEB_algorithm != BADOUCLARKSON){

		Qfloat* Qidx = NULL;
		int idx;
		double dist2 = 0.0;
		double dot_c_idx = 0.0;

		//test first the furthest point in the coreset
/*
		if(furthest_coreidx >= 0){
			dist2 = tmpdist - 2.0*Q_center_dot_in[furthest_coreidx];
			if(dist2 > maxDistance2){
				maxDistance2 = dist2;
				maxDistance2Idx = coreIdx[furthest_coreidx];
				center_dot_out = Q_center_dot_in[furthest_coreidx];
				Q_center_dot_out = Q_furthest;
				//printf("violating inner: coreidx %d dataidx %d\n",furthest_coreidx,maxDistance2Idx);
			}
		}
*/
		while(sampleNum < samplingSize)
		{
			//posTurn = PosNeg(posTurn,posNum,negNum);
			//int rand32bit = random();
			//idx	= posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
			int rand32bit = random();
			idx			  = rand32bit%prob->l;
			dist2 = 0.0;
			dot_c_idx = 0.0;

			if((previous_violator_idx == idx) && (previous_violator_idx > 0)){

				dot_c_idx = next_dot_violator_c;
				Qidx = previousQidx;
				//center_dot_out = dot_c_idx;
				dist2 = tmpdist - 2.0*dot_c_idx;
				//printf("accesing\n");
				//Q_center_dot_out = previousQidx;

			} else {

				if(chklist[idx] > 0){//if the point is in the coreset use cache

					Qidx = NULL;
					int core_idx = inverted_coreIdx[idx];
					dist2 = tmpdist - 2.0*Q_center_dot_in[core_idx];
					dot_c_idx = Q_center_dot_in[core_idx];
				} else {
					Qidx = kernelQ->get_Q(idx, coreNum, coreIdx);
					if(Qidx != NULL)
					{
							for (int j=0; j<coreNum; j++)
								dot_c_idx += Qidx[j]*outAlpha[j];
							dist2  = tmpdist - 2.0 * dot_c_idx;
					}
				}

			}

			if (dist2 > maxDistance2)
			{
				//printf("changing violator\n");
				Q_center_dot_out = Qidx;
				center_dot_out = dot_c_idx;
				maxDistance2 = dist2;
				maxDistance2Idx = idx;
				//printf("******* radius: %f, distance tolerated %f, distance found %f, index %d\n",r2,dist_tol,maxDistance2,idx);
			}
			sampleNum++;

		}

		if((maxDistance2Idx >= 0) && (Q_center_dot_out == NULL)){
			//printf("computing Qidx\n");
			Qidx = kernelQ->get_Q(maxDistance2Idx, coreNum, coreIdx);
			Q_center_dot_out = Qidx;
		}

		//printf("accesing\n");
		previous_violator_idx = maxDistance2Idx;
		previousQidx = Q_center_dot_out;

		//printf("END computing Qidx. Violator: %d\n",maxDistance2Idx);
	} else {

		while(sampleNum < samplingSize)
		{
					posTurn = PosNeg(posTurn,pNum,nNum);
					int idx;
					do{
						//balanced and random sample
						int rand32bit = random();
						idx			  = posTurn ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum];
						//printf("sampling %d %d\n",pNum,nNum);
						//int rand32bit = random();
						//idx	= rand32bit%prob->l;
					} while ((chklist[idx] > 0) || ((pNum==0) && (nNum==0)));
					//compute the distance to the center
					double dist2 = 0.0;
					double dot_c_idx = 0.0;

					Qfloat* Qidx = kernelQ->get_Q(idx, coreNum, coreIdx);
					if (Qidx != NULL)
					{
						for (int j=0; j<coreNum; j++)
							dot_c_idx += Qidx[j]*outAlpha[j];
						dist2  = tmpdist - 2.0 * dot_c_idx;
					}
					if (dist2 > maxDistance2)
					{
						maxDistance2 = dist2;
						maxDistance2Idx = idx;
					}
					sampleNum++;
		}


	}

	return maxDistance2;
}


double Solver_CVM::_maxDist(double maxDistance2, int &maxDistance2Idx)
{
	double tmpdist = coreNorm2 + Eta;
	

	for(int idx = 0; idx < prob->l; idx++)
	{
		//if(chklist[idx] == 0){

		double dist2 = 0.0;
		double dot_c_idx = 0.0;

		Qfloat* Qidx = kernelQ->get_Q(idx, coreNum, coreIdx);
		if (Qidx != NULL)
		{
			for (int j=0; j<coreNum; j++)
				dot_c_idx += Qidx[j]*outAlpha[j];
			dist2  = tmpdist - 2.0 * dot_c_idx;
		}

		if (dist2 > maxDistance2)
		{
			Q_center_dot_out = Qidx;
			center_dot_out = dot_c_idx;
			maxDistance2 = dist2;
			maxDistance2Idx = idx;
		}
		//}
	}

	return maxDistance2;
}

//returns the index of the core-set element nearest to the center
//this index ranges from 0 to coreNum-1
double Solver_CVM::_minDistFromCoreSet(double minDistance2, int &minDistance2Idx)
{
	double tmpdist = coreNorm2 + Eta;
	minDistance2Idx=closest_coreidx;

	if(closest_coreidx != -1)
		minDistance2 = tmpdist - 2*Q_center_dot_in[closest_coreidx];

	return minDistance2;
}

double Solver_CVM::_maxDistFromCoreSet(double maxDistance2, int &maxDistance2Idx){

	double tmpdist = coreNorm2 + Eta;
	double dot_c_idx = 0.0;
	double dist2 = 0.0;
	double maxDistance2_found = maxDistance2;
	int maxIdx_found = -1;
	Qfloat* Qidx = NULL;

	if(furthest_coreidx >= 0){
		dist2 = tmpdist - 2.0*Q_center_dot_in[furthest_coreidx];
		if(dist2 > maxDistance2){
			maxDistance2 = dist2;
			maxDistance2Idx = coreIdx[furthest_coreidx];
			center_dot_out = Q_center_dot_in[furthest_coreidx];
			Q_center_dot_out = Q_furthest;
			//printf("violating inner: coreidx %d dataidx %d\n",furthest_coreidx,maxDistance2Idx);
		}
	} else{
		for(int j=0; j<coreNum; j++){

			dot_c_idx = Q_center_dot_in[j];
			dist2 = tmpdist - 2.0*Q_center_dot_in[j];
			if (dist2 > maxDistance2)
			{
					center_dot_out = dot_c_idx;
					maxDistance2_found = dist2;
					maxIdx_found = j;
			}

		}
		//printf("ok\n");

		if(maxIdx_found != -1){
			maxDistance2Idx=coreIdx[maxIdx_found];
			maxDistance2 = 	maxDistance2_found;
			dot_c_idx = Q_center_dot_in[maxIdx_found];

			if((previous_violator_idx == maxDistance2Idx) && (previous_violator_idx > 0)){

				Qidx = previousQidx;

			}

			if (Qidx == NULL){
				Qidx = kernelQ->get_Q(maxDistance2Idx, coreNum, coreIdx);

			}

			Q_center_dot_out = Qidx;
			center_dot_out = dot_c_idx;
		}
	}

	previous_violator_idx = maxDistance2Idx;

	//printf("ok2\n");
	return maxDistance2;
}

double Solver_CVM::_best_point_for_SWAP(double app_r2, double out_dist2, int out_idx, int &coreset_idx_in, double &alpha_swap, double &dist2_in, double &dist2_pts){
//  LO SWAP GOLOSO
//  Inputs:
//  		app_r2 		: (1-eps)^2 * r_k^2 where c_k current center, r_k current radius
//			out_dist2 	: || c_k - z_i* ||^2 where z_i* is the point maximizing || c_k - z_i ||^2
//			out_idx 	: index of the point z_i* in the range of the data, that is it belongs to 1:m
//  Outputs:
//  		coreset_idx_in 		: index of the point selected for SWAP with respect to the corset, i.e. it belongs to 1:coreNum
//			alpha_swap 			: value of the weight to assign to the new point after entering the coreset
//			dist2_in			: || c_k - z_j* ||^2 distance between the point selected for swap and the center
//			dist2_pts			: || z_i* - z_j* ||^2 distance between the point selected for swap and the center
//  Returns:
//			maximum_gain 		: improvement of the objective function value
//								  M_k = ( || c_k - z_i* ||^2 - || c_k - z_j* ||^2) / ( 2 || z_i* - z_j* ||^2)

	double tmpdist = coreNorm2 + Eta;
	double dist2_c = 0;
	double dist2_out_inner = 0;
	double gain = 0;
	double maximum_gain = 0;
			
	Qfloat *Q_i = Q_center_dot_out;			
	
	int idx, selected_idx;
	double alpha_theo, alpha_selected, alpha_limit, alpha, diff;
	double dist_in_selected, dist_pts_selected;
		
	for(idx = 0; idx < coreNum; idx++) {	
		
		if (outAlpha[idx] > TAU)
		{	
			dist2_c = tmpdist - 2*Q_center_dot_in[idx];
			dist2_out_inner = 2*(Eta - Q_i[idx]);
			
			//if(dist2_c < app_r2){ 
			//	printf("candidate: dist2c=%.10g, r2=%.10g\n",dist2_c,app_r2);
			//}	
			if((dist2_c < app_r2) && (dist2_out_inner > 0)){
				diff = 	out_dist2 - dist2_c;
				alpha_limit = outAlpha[idx];
				alpha_theo = diff/(2*dist2_out_inner);
				alpha  = (alpha_theo > alpha_limit) ? alpha_limit : alpha_theo; 
				gain = 	(alpha*diff) - (alpha*alpha*dist2_out_inner); 			
		
				if(gain > maximum_gain){
					maximum_gain = gain;
					selected_idx = idx;
					alpha_selected = alpha;
					dist_in_selected = dist2_c;
					dist_pts_selected = dist2_out_inner;
								
				}
			}	
		}	 
	}
	coreset_idx_in = selected_idx;
	alpha_swap =  alpha_selected;
	dist2_in = dist_in_selected;
	dist2_pts = dist_pts_selected;
	//printf("selecting inner point ... out-dist2=%g, in-dist2=%g, in-out-dist2=%g, r2=%g, Eta=%g\n",out_dist2, dist2_c, dist2_out_inner, r2, Eta);
	
	return maximum_gain;
		
}
	
double Solver_CVM::_most_inner_point_for_SWAP(double app_r2, double out_dist2, int out_idx, int &coreset_idx_in, double &alpha_swap, double &dist2_in, double &dist2_pts){
//  LO SWAP PIGRO
//  Inputs:
//  		app_r2 		: (1-eps)^2 * r_k^2 where c_k current center, r_k current radius
//			out_dist2 	: || c_k - z_i* ||^2 where z_i* is the point maximizing || c_k - z_i ||^2
//			out_idx 	: index of the point z_i* in the range of the data, that is it belongs to 1:m
//  Outputs:
//  		coreset_idx_in 		: index of the point selected for SWAP with respect to the corset, i.e. it belongs to 1:coreNum
//			alpha_swap 			: value of the weight to assign to the new point after entering the coreset
//			dist2_in			: || c_k - z_j* ||^2 distance between the point selected for swap and the center
//			dist2_pts			: || z_i* - z_j* ||^2 distance between the point selected for swap and the center
//  Returns:
//			gain 				: improvement of the objective function value
//								  M_k = ( || c_k - z_i* ||^2 - || c_k - z_j* ||^2) / ( 2 || z_i* - z_j* ||^2)

	double tmpdist = coreNorm2 + Eta;
	double dist2_c = 0;
	double dist2_out_inner = 0;
	double gain = 0;
	double min_dist2_c = app_r2;
	double dist_pts_selected = 0.0;
	int selected_idx = coreset_idx_in;
	double alpha_theo=0, alpha_limit=0, alpha=0, diff=0;

	if(closest_coreidx != -1){
			dist2_c = tmpdist - 2*Q_center_dot_in[closest_coreidx];
			dist2_out_inner = 2*(Eta - Q_center_dot_out[closest_coreidx]);
			if((dist2_c < app_r2) && (dist2_out_inner > 0) && (dist2_c < min_dist2_c)){
				selected_idx=closest_coreidx;
				diff = 	out_dist2 - dist2_c;
				alpha_limit = outAlpha[selected_idx];
				alpha_theo = diff/(2*dist2_out_inner);
				alpha  = (alpha_theo > alpha_limit) ? alpha_limit : alpha_theo;
				gain = 	(alpha*diff) - (alpha*alpha*dist2_out_inner);
				min_dist2_c = dist2_c;
				dist_pts_selected = dist2_out_inner;
			}
	}

	coreset_idx_in = selected_idx;
	alpha_swap =  alpha;
	dist2_in = min_dist2_c;
	dist2_pts = dist_pts_selected;
	return gain;

}
				
inline void Solver_CVM::_UpdateCoreSet(int maxDistance2Idx)
{
	if((param->MEB_algorithm == BADOUCLARKSON)){
		if(y[maxDistance2Idx] > 0)
			pNum--;
		else
			nNum--;
	}

	if(chklist[maxDistance2Idx] < 1){

		coreIdx[coreNum++]       = maxDistance2Idx;
		chklist[maxDistance2Idx] = 1;

	}
	
}	

double Solver_CVM::ComputeSolution(double *alpha, double Threshold)
{
	double bias      = 0.0;
	double sumAlpha  = 0.0;
	int i;
	
	FILE* file_svs = fopen("support-vectors.out","w");
	fprintf(file_svs,"support vectors:\n");

	for(i = 0; i < coreNum; i++)
	{  
		if (outAlpha[i] > Threshold)
		{
			int ii    = coreIdx[i];
			alpha[ii] = outAlpha[i]*y[ii];
			bias     += alpha[ii];
			sumAlpha += outAlpha[i];
			fprintf(file_svs,"idx: %d   weight: %g\n",ii,alpha[ii]);

		}
	}


	bias /= sumAlpha;
	fprintf(file_svs,"bias: %g\n",bias);
	fclose(file_svs);


	for(i = 0; i < coreNum; i++)
		alpha[coreIdx[i]] /= sumAlpha;
		
	
	return bias;
}


//---------------------------------------------------------------------------------------------------------------------
// METHODS FOR MCVM 


void Solver_MCVM::_Init()
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);	
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int) prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		
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
	
	nclasses = nr_class;
	classesNums = new int[nclasses];
	classesIdxs = Malloc(int *,nclasses);
	cNums = new int[nclasses];
	// count the distribution of data
	
	for (int j = 0; j < nclasses; j++) {

		classesNums[j] = count[j];
		cNums[j] = 0; 
  		classesIdxs[j] = Malloc(int,count[j]);
	}
	
	for(int i = 0; i < prob->l; i++){
			for (int j = 0; j < nclasses; j++){	
				int this_label = (int) prob->y[i];
				if (this_label == label[j]){
					classesIdxs[j][cNums[j]] = i;
					cNums[j] = cNums[j] + 1;
					break;
				}
			}
	}								
	
//	printf("#### INFO INIT:\n");
//	printf("nclasses: %d\n",nr_class);
//	printf("labels: ");
//	for (i=0;i < nr_class; i++){
//		printf(" %d ",label[i]);
//	}
//	printf("\n");
//	printf("indexes for each class:\n");
//	for (i=0;i < nr_class; i++){
//		int *currentIdxs = classesIdxs[i];	
//		printf("class %d. size: %d free: %d indexes:\n",i,count[i],cNums[i]);
//		for(int s=1; s < count[i]; s++){
//			printf("%d ",currentIdxs[s]);
//		}
//		printf("\n");
//	}
	
	y       = new int[prob->l];
	chklist = new char[prob->l];
	coreIdx = new int[prob->l];
	coreNum  = 0;
		
	for(int i = 0; i < prob->l; i++)
	{	
		y[i]       = (int) prob->y[i];
		chklist[i] = 0;
		coreIdx[i] = -1;
	}
	
	// initialized the kernel
	outAlpha = &tmpAlpha[0];
	kernelQ  = new MCVC_Q(prob, param, y);		
	Eta		 = kernelQ->get_Eta();	

	// choose a small subset as initialization
	classTurn  = 0;		
	int sampleNum = 0;
	int initialSampleSize = INITIAL_CS;
	
	if (initialSampleSize > prob->l)
		initialSampleSize = prob->l-1;
		
	while(sampleNum < initialSampleSize)
	{	
		classTurn = (classTurn+1)%nclasses;				
		if (cNums[classTurn] <= 0)
			break;
		
		int *currentIdxs;	
		int currentNum;
		int idx;
		do
		{
			// balanced and random sample
			int rand32bit = random();
			currentIdxs = classesIdxs[classTurn];
			currentNum  = classesNums[classTurn];
			idx			= currentIdxs[rand32bit%currentNum];
		
		} while (chklist[idx] > 0);
		
		chklist[idx] = 1;
		cNums[classTurn] = cNums[classTurn] - 1; 
		coreIdx[coreNum++]  = idx;
		outAlpha[sampleNum] = 1.0/initialSampleSize;
		tempD[sampleNum]    = 0;
		//printf("eta: %g\n",Eta);
		sampleNum++;	
	}	
}

bool Solver_MCVM::_Create(double cvm_eps)
{
	solver = new MEB_SMO(coreIdx,coreNum,*kernelQ,tempD,outAlpha,SMO_EPS,param->eps);
	return (solver != NULL);
}

double Solver_MCVM::_maxDistFromSampling(double maxDistance2, int &maxDistance2Idx)
{
	double tmpdist = coreNorm2 + Eta;
	int sampleNum = 0;
	int samplingSize = param->sample_size;
//	int remainingPatt = 0;
//	
//	for(int i = 0; i < prob->l; i++)
//	{	
//		if(chklist[i] == 0)
//			remainingPatt++;
//	}
//		
//	if (samplingSize > remainingPatt)
//		samplingSize = remainingPatt;
//	
//	if (samplingSize == 0){
//		double dist2 = kernelQ->dist2c_wc(0, coreNum, coreIdx, outAlpha, tmpdist);
//		maxDistance2Idx = 0; 
//		maxDistance2 = dist2;
//		return maxDistance2; 
//		}
//	
	while(sampleNum < samplingSize)
	{	
		classTurn = (classTurn+1)%nclasses;	
		int *currentIdxs;	
		int currentNum;
		int idx;
		
		if (cNums[classTurn] <= 0)
			break;
		
		//do
		//{
			// balanced and random sample
			int rand32bit = random();
			currentIdxs = classesIdxs[classTurn];
			currentNum  = classesNums[classTurn];
			idx			= currentIdxs[rand32bit%currentNum];
		
		//} while (chklist[idx] > 0);
		//printf("violating: %d of class %d\n",idx,y[idx]);
		double dist2 = kernelQ->dist2c_wc(idx, coreNum, coreIdx, outAlpha, tmpdist);
		if (dist2 > maxDistance2)
		{						
			maxDistance2 = dist2;
			maxDistance2Idx = idx;
		}					
		sampleNum++;	
	}
	
	return maxDistance2;
}

double Solver_MCVM::_maxDist(double maxDistance2, int &maxDistance2Idx)
{
	double tmpdist = coreNorm2 + Eta;
	
	for(int idx = 0; idx < prob->l; idx++)
	{	
		//if(chklist[idx] == 0){
		
			double dist2 = kernelQ->dist2c_wc(idx, coreNum, coreIdx, outAlpha, tmpdist);
			if (dist2 > maxDistance2)
			{						
			maxDistance2 = dist2;
			maxDistance2Idx = idx;
			}
		
		//}
			
	}
		
		return maxDistance2;
}


//returns the index of the core-set element nearest to the center
//this index ranges from 0 to coreNum-1
double Solver_MCVM::_minDistFromCoreSet(double minDistance2, int &minDistance2Idx)
{
	double tmpdist = coreNorm2 + Eta;
	
	for(int idx = 0; idx < coreNum; idx++) {	
		
		double dist2 = kernelQ->dist2c_wc(coreIdx[idx],coreNum, coreIdx, outAlpha, tmpdist);
		
		if (dist2 < minDistance2)
		{						
			minDistance2 = dist2;
			minDistance2Idx = idx;
		}	 
	}

	return minDistance2;
}

			
inline void Solver_MCVM::_UpdateCoreSet(int maxDistance2Idx)
{
	for (int j = 0; j < nclasses; j++){
				int this_label = (int) prob->y[maxDistance2Idx];	
				if (this_label == label[j]){
					cNums[j] = cNums[j] - 1;
					break;
				}
	}			
	
	if(chklist[maxDistance2Idx] == 0){
		
		coreIdx[coreNum++]       = maxDistance2Idx;
		chklist[maxDistance2Idx] = 1;
	
	}	
}	

double Solver_MCVM::_best_point_for_SWAP(double app_r2, double out_dist2, int out_idx, int &coreset_idx_in, double &alpha_swap, double &dist2_in, double &dist2_pts){
//  LO SWAP GOLOSO
//  Inputs:
//  		app_r2 		: (1-eps)^2 * r_k^2 where c_k current center, r_k current radius
//			out_dist2 	: || c_k - z_i* ||^2 where z_i* is the point maximizing || c_k - z_i ||^2
//			out_idx 	: index of the point z_i* in the range of the data, that is it belongs to 1:m
//  Outputs:
//  		coreset_idx_in 		: index of the point selected for SWAP with respect to the corset, i.e. it belongs to 1:coreNum
//			alpha_swap 			: value of the weight to assign to the new point after entering the coreset
//			dist2_in			: || c_k - z_j* ||^2 distance between the point selected for swap and the center
//			dist2_pts			: || z_i* - z_j* ||^2 distance between the point selected for swap and the center
//  Returns:
//			maximum_gain 		: improvement of the objective function value
//								  M_k = ( || c_k - z_i* ||^2 - || c_k - z_j* ||^2) / ( 2 || z_i* - z_j* ||^2)

	double tmpdist = coreNorm2 + Eta;
	double dist2_c = 0;
	double dist2_out_inner = 0;
	double gain = 0;
	double maximum_gain = 0;

	Qfloat *Q_i = Q_center_dot_out;

	int idx, selected_idx;
	double alpha_theo, alpha_selected, alpha_limit, alpha, diff;
	double dist_in_selected, dist_pts_selected;
		
	for(idx = 0; idx < coreNum; idx++) {	
		
		if (outAlpha[idx] > TAU)
		{	
			dist2_c = tmpdist - 2*Q_center_dot_in[idx];
			dist2_out_inner = 2*(Eta - Q_i[idx]);


			//if(dist2_c < app_r2){
			//	printf("candidate: dist2c=%.10g, r2=%.10g\n",dist2_c,app_r2);
			//}
			if((dist2_c < app_r2) && (dist2_out_inner > 0)){
				diff = 	out_dist2 - dist2_c;
				alpha_limit = outAlpha[idx];
				alpha_theo = diff/(2*dist2_out_inner);
				alpha  = (alpha_theo > alpha_limit) ? alpha_limit : alpha_theo;
				gain = 	(alpha*diff) - (alpha*alpha*dist2_out_inner);

				if(gain > maximum_gain){
					maximum_gain = gain;
					selected_idx = idx;
					alpha_selected = alpha;
					dist_in_selected = dist2_c;
					dist_pts_selected = dist2_out_inner;

				}
			}
		}
	}
	coreset_idx_in = selected_idx;
	alpha_swap =  alpha_selected;
	dist2_in = dist_in_selected;
	dist2_pts = dist_pts_selected;
	//printf("selecting inner point ... out-dist2=%g, in-dist2=%g, in-out-dist2=%g, r2=%g, Eta=%g\n",out_dist2, dist2_c, dist2_out_inner, r2, Eta);

	return maximum_gain;

}
	
double Solver_MCVM::_most_inner_point_for_SWAP(double app_r2, double out_dist2, int out_idx, int &coreset_idx_in, double &alpha_swap, double &dist2_in, double &dist2_pts){
//  LO SWAP PIGRO
//  Inputs:
//  		app_r2 		: (1-eps)^2 * r_k^2 where c_k current center, r_k current radius
//			out_dist2 	: || c_k - z_i* ||^2 where z_i* is the point maximizing || c_k - z_i ||^2
//			out_idx 	: index of the point z_i* in the range of the data, that is it belongs to 1:m
//  Outputs:
//  		coreset_idx_in 		: index of the point selected for SWAP with respect to the corset, i.e. it belongs to 1:coreNum
//			alpha_swap 			: value of the weight to assign to the new point after entering the coreset
//			dist2_in			: || c_k - z_j* ||^2 distance between the point selected for swap and the center
//			dist2_pts			: || z_i* - z_j* ||^2 distance between the point selected for swap and the center
//  Returns:
//			gain 				: improvement of the objective function value
//								  M_k = ( || c_k - z_i* ||^2 - || c_k - z_j* ||^2) / ( 2 || z_i* - z_j* ||^2)

	double tmpdist = coreNorm2 + Eta;
	double dist2_c = 0;
	double dist2_out_inner = 0;
	double gain = 0;
	double min_dist2_c = app_r2;

	Qfloat *Q_i = Q_center_dot_out;//inner products between the point i* and the points in the center

	int idx, selected_idx;
	double alpha_theo=0, alpha_limit=0, alpha=0, diff=0;
	printf("ENTERING MAIN FOR\n");

	for(idx = 0; idx < coreNum; idx++) {

		if (outAlpha[idx] > TAU)
		{
			printf("BEFORE USING Q_CENTER_ %d\n",idx);
			
			dist2_c = tmpdist - 2*Q_center_dot_in[idx];//distance of the candidate to the center
			dist2_out_inner = 2*(Eta - Q_i[idx]);//distance between the current candidate and i*

			if((dist2_c < app_r2) && (dist2_out_inner > 0) && (dist2_c < min_dist2_c)){
				diff = 	out_dist2 - dist2_c;
				alpha_limit = outAlpha[idx];
				alpha_theo = diff/(2*dist2_out_inner);
				alpha  = (alpha_theo > alpha_limit) ? alpha_limit : alpha_theo;
				gain = 	(alpha*diff) - (alpha*alpha*dist2_out_inner);
				selected_idx = idx;
				min_dist2_c = dist2_c;
			}
		}
	}

	coreset_idx_in = selected_idx;
	alpha_swap =  alpha;
	dist2_in = min_dist2_c;
	dist2_pts = dist2_out_inner;

	//printf("selecting inner point ... out-dist2=%g, in-dist2=%g, in-out-dist2=%g, r2=%g, Eta=%g\n",out_dist2, dist2_c, dist2_out_inner, r2, Eta);

	return gain;

}

double Solver_MCVM::ComputeSolution(double *alpha, double Threshold)
{
	double bias      = 0.0;
	double sumAlpha  = 0.0;
	int i;
	for(i = 0; i < coreNum; i++)
	{
		if (outAlpha[i] > 0)
		{
			int ii    = coreIdx[i];
			alpha[ii] = outAlpha[i];
			sumAlpha += outAlpha[i];
		}
	}

	for(i = 0; i < coreNum; i++){
		alpha[coreIdx[i]] /= sumAlpha;
		//printf("corevector %d index %d alpha %g\n",i,coreIdx[i],alpha[coreIdx[i]]); 
	}
	return bias;
}

double Solver_BVM::_update(double maxDistance2, int maxDistance2Idx)
{
    double beta = sqrt(r2/(maxDistance2 + c*c));
	double rate = 1.0 - beta;

    // update center
    int i;
	for(i = 0; i < coreNum; i++)
		alpha[coreIdx[i]] *= beta;

	// update constant and center norm
	c        *= beta;
	coreNorm2 = coreNorm2*(beta) + kappa*(rate) - maxDistance2*(beta*rate);

	// update gradient
	Qfloat* kernelColumn = kernelQ->get_Q(maxDistance2Idx, coreNum, coreIdx);
	for (i = 0; i < coreNum; i++)
	{
		int coreIdx_i = coreIdx[i];
		if (coreGrad[coreIdx_i] != 0.0)
			coreGrad[coreIdx_i] = (Qfloat)(coreGrad[coreIdx_i]*beta + rate*kernelColumn[i]);
	}

    return rate;
}

double Solver_BVM::ComputeSolution(double *_alpha, double Threshold)
{
	double bias      = 0.0;
	double sumAlpha  = 0.0;
	int i;
	for(i = 0; i < coreNum; i++)
	{
        int ii = coreIdx[i];
		if (alpha[ii] > Threshold)
		{
            sumAlpha  += alpha[ii];
            _alpha[ii] = alpha[ii]*y[ii];
			bias      += _alpha[ii];
		}
	}
	bias /= sumAlpha;

	for(i = 0; i < coreNum; i++)
		_alpha[coreIdx[i]] /= sumAlpha;
	return bias;
}

void Solver_BVM::Init(const svm_problem *prob, const svm_parameter *_param, double *_alpha)
{
    // init
    param   = _param;
    alpha   = _alpha;
    numData = prob->l;
    posIdx  = new int[numData];
	negIdx  = new int[numData];
	y       = new schar[numData];
    posNum  = 0;
	negNum  = 0;

    int i;
	for(i = 0; i < numData; i++)
	{
		if (prob->y[i] > 0)
		{
			y[i]             = 1;
			posIdx[posNum++] = i;
		}
		else
		{
			y[i]             = -1;
			negIdx[negNum++] = i;
		}
	}

    // initialize the kernel
 	kernelQ   = new BVM_Q(prob, param, y);
	kappa     = kernelQ->getKappa();    // square radius of kernel feature space
	r2        = kappa;					// square radius of EB
	c         = sqrt(r2);				// augmented center coeff.
	coreNorm2 = kappa;					// square normal of the center

    // initialize the coreset
	coreIdx	 = new int[numData];
	coreNum	 = 0;
	coreGrad = new Qfloat[numData];
	chklist  = new char[numData];
	for (i = 0; i < numData; i++)
	{
		coreGrad[i] = 0.0;
		chklist[i]  = 0;
	}

	coreIdx[coreNum++]   = 0;
	alpha   [coreIdx[0]] = 1.0;
	coreGrad[coreIdx[0]] = (Qfloat)kappa;
	chklist [coreIdx[0]] = 1;
}

//---------------------------------------------------------------------------------------------------------------------

int Solver_BVM::Solve(int _num_basis, double _bvm_eps, bool _cooling, bool _randomized)
{
    if(param->MEB_algorithm == BVMtsang){

    	coreNum = _Tsang_Algorithm(_num_basis,_bvm_eps,_cooling,_randomized);

    } else if(param->MEB_algorithm == PANIGRAHY){


    	coreNum = _Panigraphy_Algorithm(_num_basis,_bvm_eps,_cooling,_randomized);
    }

	return coreNum;
}


int Solver_BVM::_Tsang_Algorithm(int num_basis, double bvm_eps, bool cooling, bool randomized)
{
	printf("MEB Solver: BVM, TSANG(2005), ");
	cooling ? printf("cooling YES, ") : printf("cooling NOT, ");
	randomized ? printf("randomized YES\n") : printf("randomized NOT\n");

	greedy_it = 0;
	smo_it = 0;

	// iterate on epsilons
    maxNumBasis          = num_basis;
	int updateNum        = 0;
	double epsilonFactor = EPS_SCALING;
	double currentEpsilon;

	if(!cooling) //avoid cooling
		currentEpsilon = bvm_eps/epsilonFactor;
	else //do cooling
		currentEpsilon = INITIAL_EPS;


	while(currentEpsilon > bvm_eps){

		currentEpsilon *= epsilonFactor;

		// check epsilon
		if (currentEpsilon < bvm_eps)
			currentEpsilon = bvm_eps;

		double sepThres = kappa * (1.0 - (currentEpsilon + currentEpsilon * currentEpsilon * 0.5));

		// solve problem with current epsilon (warm start from the previous solution)

		//do{//refine the radius

		double dist2_violator = 0.0;
		int idx_violator = 0;
		//set the radius of the enclosing ball to search at this iteration
		//double radius_eps = r2*(1.0 + currentEpsilon)*(1.0 + currentEpsilon) - c*c;

		while (idx_violator != -1)
		{
            // increase the usage of internal cache by constraining the search points when #BV is more than COMPRESS_THRES
			int refineIter = 0;
            // search any point
			{
				// compute (1+eps)^2*radius^2 - c^2
				double radius_eps = r2*(1.0 + currentEpsilon)*(1.0 + currentEpsilon) - c*c;

				// get a probabilistic sample
				double tolerated_distance  = radius_eps;
				idx_violator = -1;
				double tmpdist  = coreNorm2 + kappa;

				clock_t init_time_2 = clock ();

				if(randomized){

					for(int sampleIter = 0; (sampleIter < NUM_SAMPLINGS) && (idx_violator == -1); sampleIter++)
					{
						for(int sampleNum = 0; sampleNum < param->sample_size; sampleNum++)
						{
							// balanced and random sample
							int rand32bit = random();
							int idx       = (sampleNum+sampleNum < param->sample_size ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum]);

							// check violation
							if (coreGrad[idx] != 0.0)
							{
								double dist2_idx = tmpdist - 2.0*coreGrad[idx];
								if (dist2_idx > tolerated_distance)
								{
									idx_violator = idx;
									dist2_violator = dist2_idx;
									break;
								}

							}
							else
							{
								bool depend = true;
								double dot_c = kernelQ->dot_c_wc(idx, coreNum, coreIdx,alpha,depend,sepThres);
								if (depend == true)
									continue;
								if (chklist[idx] == 1)
									coreGrad[idx] = (Qfloat) dot_c;

						        double dist2_idx = tmpdist - 2.0*dot_c;
								if (dist2_idx > tolerated_distance)
								{
									idx_violator = idx;
									dist2_violator = dist2_idx;
									break;
								}

							}
						}
					}
				} //end if randomized
				else {//not randomized

					for(int idx = 0; (idx < numData) && (idx_violator == -1); idx++){

							// compute distance
							if (coreGrad[idx] != 0.0)
							{
								double dist2_idx = tmpdist - 2.0*coreGrad[idx];
								if (dist2_idx > tolerated_distance)
								{
									idx_violator = idx;
									dist2_violator = dist2_idx;
									break;
								}
							}
							else
							{
								bool depend = true;
								double dot_c = kernelQ->dot_c_wc(idx, coreNum, coreIdx,alpha,depend,sepThres);
								if (depend == true)
									continue;
								if (chklist[idx] == 1)
									coreGrad[idx] = (Qfloat) dot_c;

						        double dist2_idx = tmpdist - 2.0*dot_c;
								if (dist2_idx > tolerated_distance)
								{
									idx_violator = idx;
									dist2_violator = dist2_idx;
									break;
								}

							}
					}

				}
				// check maximal distance
				if (idx_violator != -1)
				{
					double rate = _update (dist2_violator, idx_violator);
                    if (alpha[idx_violator] == 0.0)
					{
						coreIdx[coreNum++]       = idx_violator;
						chklist[idx_violator] = 1;
					}
					alpha[idx_violator] += rate;
					greedy_it++;


					// info
					updateNum++;
					//if (greedy_it%20 < 1) info(".");
#ifndef RELEASE_VER
					//printf("#%d, #cv: %d, R: %.8f, |c-x|: %.8f, r: %g\n",updateNum, coreNum, r2-c*c, dist2_violator, rate);
#endif
				}//end update

				printf("#cv: %d, tolerated_dist: %g, c: %g, r2: %.8f, |c-x|^2: %.8f, coreNorm2: %g\n", coreNum, tolerated_distance, c, r2, dist2_violator,coreNorm2);

				clock_t final_time_2 = clock ();

			}

			if (IsExitOnMaxIter())
			{
				currentEpsilon = bvm_eps;
				break;
			}
		}//end while there is a violator
		//} while((c*c) > r2*(2*currentEpsilon + currentEpsilon*currentEpsilon));
	}

	double end_radius_eps = r2*(1.0 + currentEpsilon)*(1.0 + currentEpsilon) - c*c;
	printf("TSANG-05 END: #cv: %d, tolerated_dist: %g, c: %g, r2: %.8f, coreNorm2: %g\n", coreNum, end_radius_eps, c, r2,coreNorm2);
	info("###### end computing MEB. Size coreset: %d, iterations: %d\n",coreNum, greedy_it);

    return coreNum;
}


int Solver_BVM::_Panigraphy_Algorithm(int num_basis, double bvm_eps, bool cooling, bool randomized)
{
	printf("MEB Solver: PANIGRAHY, ");
	cooling ? printf("cooling YES, ") : printf("cooling NOT, ");
	randomized ? printf("randomized YES\n") : printf("randomized NOT\n");

	greedy_it = 0;
	smo_it = 0;

	// iterate on epsilons
    maxNumBasis          = num_basis;
	int updateNum        = 0;
	double epsilonFactor = EPS_SCALING;
	double currentEpsilon;

	if(!cooling) //avoid cooling
		currentEpsilon = bvm_eps/epsilonFactor;
	else //do cooling
		currentEpsilon = INITIAL_EPS;


	while(currentEpsilon > bvm_eps){

		currentEpsilon *= epsilonFactor;

		// check epsilon
		if (currentEpsilon < bvm_eps)
			currentEpsilon = bvm_eps;

		double sepThres = kappa * (1.0 - (currentEpsilon + currentEpsilon * currentEpsilon * 0.5));

		// solve problem with current epsilon (warm start from the previous solution)
		//do{//refine the radius

		double maxDistance2 = 0.0;
		int maxDistance2Idx = 0;
		//set the radius of the enclosing ball to search at this iteration
		//double radius_eps = r2*(1.0 + currentEpsilon)*(1.0 + currentEpsilon) - c*c;

		while (maxDistance2Idx != -1)
		{
            // increase the usage of internal cache by constraining the search points when #BV is more than COMPRESS_THRES
			int refineIter = 0;
            // search any point
			{
				// compute (1+eps)^2*radius^2 - c^2
				double radius_eps = r2*(1.0 + currentEpsilon)*(1.0 + currentEpsilon) - c*c;

				// get a probabilistic sample
				maxDistance2    = radius_eps;
				maxDistance2Idx = -1;
				double tmpdist  = coreNorm2 + kappa;

				clock_t init_time_2 = clock ();

				if(randomized){

					for(int sampleIter = 0; (sampleIter < NUM_SAMPLINGS) && (maxDistance2Idx == -1); sampleIter++)
					{
						for(int sampleNum = 0; sampleNum < param->sample_size; sampleNum++)
						{
							// balanced and random sample
							int rand32bit = random();
							int idx       = (sampleNum+sampleNum < param->sample_size ? posIdx[rand32bit%posNum] : negIdx[rand32bit%negNum]);

							// compute distance
							if (coreGrad[idx] != 0.0)
							{
								_maxDistInCache(idx, tmpdist, maxDistance2, maxDistance2Idx);
							}
							else
							{
								bool depend = true;
								double dot_c = kernelQ->dot_c_wc(idx, coreNum, coreIdx,alpha,depend,sepThres);
								if (depend == true)
									continue;
								if (chklist[idx] == 1)
									coreGrad[idx] = (Qfloat) dot_c;
								_maxDistCompute(idx, dot_c, tmpdist, maxDistance2, maxDistance2Idx);
							}
						}
					}
				} //end if randomized
				else {//not randomized

					for(int idx = 0; (idx < numData) && (maxDistance2Idx == -1); idx++){

							// compute distance
							if (coreGrad[idx] != 0.0)
							{
								_maxDistInCache(idx, tmpdist, maxDistance2, maxDistance2Idx);
							}
							else
							{
								bool depend = true;
								double dot_c = kernelQ->dot_c_wc(idx, coreNum, coreIdx,alpha,depend,sepThres);
								if (depend == true)
									continue;
								if (chklist[idx] == 1)
									coreGrad[idx] = (Qfloat) dot_c;
								_maxDistCompute(idx, dot_c, tmpdist, maxDistance2, maxDistance2Idx);
							}


					}

				}
				// check maximal distance
				if (maxDistance2Idx != -1)
				{
					double rate = _update (maxDistance2, maxDistance2Idx);
                    if (alpha[maxDistance2Idx] == 0.0)
					{
						coreIdx[coreNum++]       = maxDistance2Idx;
						chklist[maxDistance2Idx] = 1;
					}
					alpha[maxDistance2Idx] += rate;
					greedy_it++;


					// info
					updateNum++;
					if (updateNum%20 < 1) info(".");
#ifndef RELEASE_VER
					//printf("#%d, #cv: %d, R: %.8f, |c-x|: %.8f, r: %g\n",updateNum, coreNum, r2-c*c, maxDistance2, rate);
#endif
				}//end update

				printf("#cv: %d, tolerated_dist: %g, c: %g, r2: %.8f, |c-x|^2: %.8f, coreNorm: %g\n",coreNum,radius_eps,c, r2, maxDistance2,coreNorm2);

				clock_t final_time_2 = clock ();

			}

			if (IsExitOnMaxIter())
			{
				currentEpsilon = bvm_eps;
				break;
			}
		}//end while there is a violator
		//} while((c*c) > r2*(2*currentEpsilon + currentEpsilon*currentEpsilon));
	}
	double end_radius_eps = r2*(1.0 + currentEpsilon)*(1.0 + currentEpsilon) - c*c;
	printf("PANIGRAPHY END: #cv: %d, tolerated_dist: %g, c: %g, r2: %.8f, coreNorm2: %g\n", coreNum, end_radius_eps, c, r2,coreNorm2);
	info("###### end computing MEB. Size coreset: %d, iterations: %d\n",coreNum, greedy_it);

    return coreNum;

}


