#ifndef MEBSOLVERS_H_
#define MEBSOLVERS_H_

#include "SVM-commons.h"
#include "MEB.h"
#include "MEB-utilities.h"
#include "MEB-kernels.h"


// Problem-specific solvers for SVMs that can be reduced to MEB problems
class Solver_CVM : public Solver_Core
{
public:
	Solver_CVM() {}
	~Solver_CVM()
	{
		// free memory
		delete [] posIdx;
		delete [] negIdx;
		delete [] y;
		delete [] chklist;
		delete [] coreIdx;
		delete solver;
		delete kernelQ;
	}
	
	bool   IsExitOnMaxIter() const { return (coreNum >= svm_commons::min(maxNumBasis,prob->l)); }
	double ComputeSolution(double *alpha, double Threshold);
	double ComputeRadius2() 
	{   
		coreNorm2 = solver->computeCNorm();
		r2        = Eta - coreNorm2;
		return r2;
	}
	
	double update_radius_and_cache(int step, double new_radius2, double alpha, int coreset_idx_out, int coreset_idx_in){
	
		r2        = new_radius2;
		coreNorm2 = Eta - new_radius2;
		Qfloat minQCi =  10*Eta;
		Qfloat maxQCi = -10*Eta;
		//int previous_furthest_idx = furthest_coreidx;
		furthest_coreidx = -1;
		closest_coreidx = -1;

		if(step == 0){//fw-step

			for (int m=0; m<coreNum; m++){
				if(m == coreset_idx_out){ 
					Q_center_dot_in[m] = (1-alpha)*center_dot_out + alpha*Eta; 
				}
				else { Q_center_dot_in[m] = (1-alpha)*Q_center_dot_in[m] + alpha*Q_center_dot_out[m]; }

				if (Q_center_dot_in[m] < minQCi){
					minQCi = Q_center_dot_in[m];
					furthest_coreidx = m;
				}

				if ((Q_center_dot_in[m] > maxQCi) && (outAlpha[m] > TAU)){
					maxQCi = Q_center_dot_in[m];
					closest_coreidx = m;
				}
			}
		}

		if(step == 1){//mfw-step
			Qfloat* Q_in = kernelQ->get_Q(coreIdx[coreset_idx_in],coreNum, coreIdx);
			for (int m=0; m<coreNum; m++){
				if(m == coreset_idx_in){ Q_center_dot_in[m] = (1+alpha)*Q_center_dot_in[m] - alpha*Eta; }
				else { Q_center_dot_in[m] = (1+alpha)*Q_center_dot_in[m] - alpha*Q_in[m]; }

				if (Q_center_dot_in[m] < minQCi){
					minQCi = Q_center_dot_in[m];
					furthest_coreidx = m;
				}

				if ((Q_center_dot_in[m] > maxQCi) && (outAlpha[m] > TAU)){
					maxQCi = Q_center_dot_in[m];
					closest_coreidx = m;
				}
			}

		}
		
		if(step == 2){//swap-step
			Qfloat* Q_in = kernelQ->get_Q(coreIdx[coreset_idx_in],coreNum, coreIdx);
			for (int m=0; m<coreNum; m++){
				if(m == coreset_idx_out){ Q_center_dot_in[m] = center_dot_out + alpha*Eta - alpha*Q_center_dot_out[coreset_idx_in]; } 
				else { Q_center_dot_in[m] = Q_center_dot_in[m] + alpha*Q_center_dot_out[m] - alpha*Q_in[m];}

				if (Q_center_dot_in[m] < minQCi){
					minQCi = Q_center_dot_in[m];
					furthest_coreidx = m;
				}

				if ((Q_center_dot_in[m] > maxQCi) && (outAlpha[m] > TAU)){
					maxQCi = Q_center_dot_in[m];
					closest_coreidx = m;
				}
			}
		}
		
		if(step == -1){//initialization
		for(int m=0; m<coreNum; m++){
			Q_center_dot_in[m] = 0.0;
			Qfloat* Qm = kernelQ->get_Q(coreIdx[m],coreNum, coreIdx);
			for (int j=0; j<coreNum; j++)
				Q_center_dot_in[m] += (Qfloat)Qm[j]*outAlpha[j]; 

			if (Q_center_dot_in[m] < minQCi){
				minQCi = Q_center_dot_in[m];
				furthest_coreidx = m;
			}

			if ((Q_center_dot_in[m] > maxQCi) && (outAlpha[m] > TAU)){
				maxQCi = Q_center_dot_in[m];
				closest_coreidx = m;
			}

			}

		}
		
		return r2;
	
	}
	
	double yildrim_computeRadius2(int iteration_type, double yfactor, double dist2)
	{
		if (iteration_type == 2)
		{   		
			coreNorm2 = ((yfactor*(1+yfactor))*dist2) + ((1+yfactor)*coreNorm2) - (yfactor*Eta);
			//explicit_compute_radius();
			//printf("coreNorm2 theo: %f coreNorm2 explicit: %f\n", coreNorm_theo, coreNorm2);			
			
			//if(fabs(coreNorm2 - coreNorm_theo) > 1E-10)
				//printf("different theo and compt coreNorm2\n");			

			r2 = Eta - coreNorm2;
			//explicit_compute_radius();
		}
		else
		{
			r2 = r2*yfactor;
	 		coreNorm2 = Eta - r2;
			//explicit_compute_radius();
		}
		
		return r2;
	}
	
	double explicit_compute_radius(){
		
		double coreNorm2 = 0.0;
		double sum_weights = 0.0;
		bool there_are_negatives = false;
		
		for(int k=0; k<coreNum; k++){
			sum_weights += outAlpha[k]; 
			if(outAlpha[k] < 0.0){
				there_are_negatives = true;
				printf("******** THERE ARE NEGATIVE WEIGHTS: %f\n",outAlpha[k]);
			}
		}
		
			
		for(int i=0;i<coreNum; i++){
			Qfloat *Q_i = kernelQ->get_Q(coreIdx[i], coreNum, coreIdx);			
			if (Q_i != NULL)
			{
				for (int j=0; j<coreNum; j++)
					coreNorm2 += (Q_i[j]*outAlpha[j]*outAlpha[i]);		
				
			}				
		}		
				
	   	
		r2        = Eta - coreNorm2;
		return r2;	
	}
	
	double cosa_computeRadius2(double maxDistance2, double maxCorePointDistance2, double beta){
			printf("cosaNorm\n");
			double exp_r2 = explicit_compute_radius();	
			double exp_coreNorm = Eta - exp_r2;
		   	r2 = maxDistance2 - (beta*beta*maxCorePointDistance2) - (beta*maxCorePointDistance2);				
			coreNorm2 = coreNorm2 - (beta*beta*maxCorePointDistance2);
			if(abs(coreNorm2-exp_coreNorm)> TAU)
				   printf("radius and coreNorm is not correctly computed\n");
			
			return r2;
	}
	
	double GetEta() const { return kernelQ->get_Eta(); }
	double GetKappa() const { return kernelQ->get_Kappa(); }
	void   Init(const svm_problem *_prob, const svm_parameter* _param, int* subset, double* alpha_);		
	void   Init(const svm_problem* prob, const svm_parameter* param);
	void   set_init_method(int im){
		   init_method = im;
	}
	
protected:
	void   _Init();
	bool   _Create(double cvm_eps);
	double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx);
	double _maxDist(double maxDistance2, int &maxDistance2Idx);
	double _minDistFromCoreSet(double minDistance2, int &minDistance2Idx);
	double _maxDistFromCoreSet(double maxDistance, int &maxDistanceIdx);
	double _most_inner_point_for_SWAP(double app_r2, double out_dist2,int out_idx, int &coreset_idx_in,double &alpha_swap, double &dist2_in, double &dist2_pts);
	double _best_point_for_SWAP(double app_r2, double out_dist2,int out_idx, int &coreset_idx_in,double &alpha_swap, double &dist2_in, double &dist2_pts);
	void   _UpdateCoreSet(int maxDistance2Idx);

private:
	int posNum;
	int negNum;	
	int *posIdx;
	int *negIdx;
	int pNum;
	int nNum;
	bool posTurn;	
	schar *y;
	CVC_Q *kernelQ;
	char *chklist;
	int init_method;

	
	//double Eta;
};

//------------------------------------------------------------------------------------------------------------------

//
// Solver for BVM
//
class Solver_BVM
{
public:
	Solver_BVM() {}
	~Solver_BVM()
	{
		// free memory
		delete [] y;
		delete [] chklist;
		delete [] coreIdx;
        delete [] coreGrad;
        delete [] posIdx;
		delete [] negIdx;

		delete kernelQ;
	}

   	void   Init(const svm_problem* prob, const svm_parameter* param, double *_alpha);
	int    Solve(int _num_basis, double _bvm_eps,  bool cooling, bool randomized);
	int    _Tsang_Algorithm(int num_basis, double bvm_eps,  bool cooling, bool randomized);
	int    _Panigraphy_Algorithm(int num_basis, double bvm_eps,  bool cooling, bool randomized);

	double ComputeSolution(double *alpha, double Threshold);

    bool   IsExitOnMaxIter() const { return (coreNum >= svm_commons::min(maxNumBasis,numData)); }
    double GetCoreNorm2 () const { return coreNorm2; }
	double ComputeRadius2() const { return r2; }
	double GetKappa() const { return kappa; }
	int greedy_it; //for greedy algorithms
	unsigned long int smo_it; //for algorithms using SMO

protected:
	inline void _maxDistInCache(int idx, double tmpdist, double &maxDistance2, int &maxDistance2Idx)
    {
        double dist2 = tmpdist - 2.0*coreGrad[idx];
		if (dist2 > maxDistance2)
		{
			maxDistance2    = dist2;
			maxDistance2Idx = idx;
		}
    }
	inline void _maxDistCompute(int idx, double dot_c, double tmpdist, double &maxDistance2, int &maxDistance2Idx)
    {
        double dist2 = tmpdist - 2.0*dot_c;
		if (dist2 > maxDistance2)
		{
			maxDistance2    = dist2;
			maxDistance2Idx = idx;
		}
    }
    double _update (double maxDistance2, int maxDistance2Idx);

private:
	int posNum;
	int negNum;
	int *posIdx;
	int *negIdx;
	int pNum;
	int nNum;
    int numData;

    double *alpha;
	schar  *y;
	BVM_Q  *kernelQ;
   	double kappa;		// square radius of kernel feature space
	double r2;			// square radius of EB
	double c;			// augmented center coeff.
	double coreNorm2;	// square normal of the center

    int     maxNumBasis;
    int    *coreIdx;
	int     coreNum;
	Qfloat *coreGrad;
	char   *chklist;

    const svm_parameter *param;
};


//------------------------------------------------------------------------------------------------------------------
//
// Solver for MCVM
//
class Solver_MCVM : public Solver_Core
{
public:
	Solver_MCVM() {}
	~Solver_MCVM()
	{
		// free memory	
		delete [] classesNums;
		delete [] cNums;
		
		for (int i = 0; i < nclasses; i++) {
  		delete [] classesIdxs[i];
  		classesIdxs[i] = 0;
		}
		
		delete [] classesIdxs;
		delete [] y;
		delete [] chklist;
		delete [] coreIdx;
		delete solver;
		delete kernelQ;
	}
	bool   IsExitOnMaxIter() const { return (coreNum >= svm_commons::min(maxNumBasis,prob->l)); }
	double ComputeSolution(double *alpha, double Threshold);
	double ComputeRadius2() 
	{ 
		coreNorm2 = solver->computeCNorm();
		r2        = Eta - coreNorm2;
		return r2;
	}
	double explicit_computeRadius2() 
	{   
		r2 = 0.0;
		for(int i=0; i < coreNum; i++)
				r2 += -(kernelQ->dist2c_wc(coreIdx[i],coreNum,coreIdx,outAlpha,0))/2;
		
		return r2;
	}
	double GetEta() const { return kernelQ->get_Eta(); }
	double GetKappa() const { return kernelQ->get_Kappa(); }
	
protected:
	void   _Init();	
	bool   _Create(double cvm_eps);
	double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx);
	double _maxDist(double maxDistance2, int &maxDistance2Idx);
	double _minDistFromCoreSet(double minDistance2, int &minDistance2Idx);
	double _maxDistFromCoreSet(double maxDistance, int &maxDistanceIdx){

	}
	double _most_inner_point_for_SWAP(double app_r2, double out_dist2,int out_idx, int &coreset_idx_in,double &alpha_swap, double &dist2_in, double &dist2_pts);
	double _best_point_for_SWAP(double app_r2, double out_dist2,int out_idx, int &coreset_idx_in,double &alpha_swap, double &dist2_in, double &dist2_pts);
	void   _UpdateCoreSet(int maxDistance2Idx);

private:
	int *classesNums;
	int **classesIdxs;
	int *cNums;
    int *label;
	int classTurn;
	int nclasses;	
	int *y;
	MCVC_Q *kernelQ;
	char  *chklist;
	double Eta;
};

//------------------------------------------------------------------------------------------------------------------
//
// Solver for CVDD
//
class Solver_CVDD : public Solver_Core
{
public:
	Solver_CVDD() {}
	~Solver_CVDD()
	{
		// free memory			
		delete [] chklist;
		delete [] coreIdx;
		delete solver;
		delete kernelQ;
	}
	bool   IsExitOnMaxIter() const { return (coreNum >= svm_commons::min(maxNumBasis,prob->l)); }
	double ComputeSolution(double *alpha, double Threshold);
	double ComputeRadius2() 
	{ 
		coreNorm2 = solver->computeCNorm();
		r2        = -2.0*solver->computeObj();
		return r2;
	}
	double GetEta() const { return kernelQ->get_Eta(); }
	double GetKappa() const { return kernelQ->get_Kappa(); }
	
protected:
	void   _Init();	
	bool   _Create(double cvm_eps);
	double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx);
	double _maxDist(double maxDistance2, int &maxDistance2Idx);
	void   _UpdateCoreSet(int maxDistance2Idx);

private:	
	CVDD_Q *kernelQ;
	Qfloat *QD;
	char  *chklist;	
};

//------------------------------------------------------------------------------------------------------------------
//
// Solver for CVR
//
class Solver_CVR : public Solver_Core
{
public:
	Solver_CVR() {}
	~Solver_CVR()
	{
		// free memory	
		delete [] posIdx;
		delete [] negIdx;
		delete [] chklist;
		delete [] coreIdx;
		delete solver;
		delete kernelQ;
	}
	bool   IsExitOnMaxIter() const { return (coreNum/2 >= svm_commons::min(maxNumBasis,prob->l)); }
	double ComputeSolution(double *alpha, double Threshold);
	double ComputeRadius2() 
	{ 
		coreNorm2 = solver->computeCNorm();
		r2        = -2.0*solver->computeObj();
		return r2;
	}
	double GetEta() const { return kernelQ->get_Eta(); }
	double GetKappa() const { return kernelQ->get_Kappa(); }
	
protected:
	void   _Init();	
	bool   _Create(double cvm_eps);
	double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx);
	double _maxDist(double maxDistance2, int &maxDistance2Idx);
	void   _UpdateCoreSet(int maxDistance2Idx);

private:
	int posNum;
	int negNum;	
	int *posIdx;
	int *negIdx;
	int pNum;
	int nNum;
	int numVar;
	bool posTurn;
	CVR_Q *kernelQ;
	Qfloat *QD;
	char  *chklist;
	double *LinCoef;
};



#endif /*MEBSOLVERS_H_*/
