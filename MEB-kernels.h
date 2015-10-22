#ifndef MEBKERNELS_H_
#define MEBKERNELS_H_

#include <assert.h>
#include "SVM-commons.h"
#include "Kernel.h"
#include "sCache.h"
using namespace svm_commons;

//Different type of kernels are defined to work with MEBs on feature spaces
//Beyond the logic of the kernel this classes provide functionaly to handle 
//the efficient storage of the kernel evaluations

//A problem-specific class is derived from the class Kernel provided in svm.h, svm.cpp

//
// Gram matrix of CVM
//

class CVC_Q : public Kernel
{
public:
	CVC_Q(const svm_problem* prob_, const svm_parameter* param_, schar *y_) : Kernel(prob_->l, prob_->x, *param_)
	{
		//init
		int i;
		prob  = prob_;
		param = param_;
		y     = y_;
		QD    = new Qfloat[prob->l];		
		C_inv = (Qfloat)(1.0/(param->C));

		//get diagonal
		if (Kernel::IsSelfConst(*param))
		{	
			Eta = (Qfloat)((this->*kernel_function)(0,0) + 1.0 + C_inv);
			for (i=0; i< prob->l; i++)
				QD[i] = Eta;
		}
		else
		{
			double tmp = 1.0 + C_inv;
			Eta        = 0.0;
			for (i=0; i< prob->l; i++)	
			{
				QD[i] = (Qfloat)((this->*kernel_function)(i, i) + tmp);
				if (QD[i] > Eta)
					Eta = QD[i];
			}	
		}		

		kernelCache = new sCache(param_, prob->l);
		//kernelCache = new Cache(prob_->l,50000);
		
		kernelEval  = 0;
	}

	~CVC_Q()
	{
		delete kernelCache;
		delete [] QD;		
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx) const
	{	
		
		//register the number of requested k-evals
		requested_kevals += basisNum;
		
		int numRet;
		Qfloat *Q = kernelCache->get_data(idx, basisNum, numRet);
		if (Q != NULL)
		{	
#ifdef COMP_STAT
			kernelEval += (basisNum - numRet);
#endif

			// fill remaining		 
			for(int i = numRet; i < basisNum; i++)
			{
				int idx2 = basisIdx[i];
				if (idx != idx2)		
					Q[i] = y[idx]*y[idx2]*(Qfloat)((this->*kernel_function)(idx, idx2) + 1.0);			
				else				
					Q[i] = QD[idx];
			}						
		}
		return Q;
	}
	
	double dist2c_wc(int idx, int basisNum, int* basisIdx, double *coeff, double norms)
	{
		double dist = 0.0;		
		Qfloat *Q_i = get_Q(idx, basisNum, basisIdx);			
		if (Q_i != NULL)
		{
			for (int j=0; j<basisNum; j++)
				dist += Q_i[j]*coeff[j];		
			dist  = norms - 2.0 * dist;				
		}		
		return dist;
	}
	
	Qfloat* get_QD() const { return QD; }	
	Qfloat get_Eta() const { return Eta; }
	Qfloat get_Kappa() const { return Eta-C_inv; }
	void swap_index(int i, int j) const { printf("CVC_Q::swap_index is not implemented!\n"); }
	
	Qfloat kernel_eval(int idx1, int idx2){
		
		requested_kevals++;
		Qfloat Q;
		
		if (idx1 != idx2)		
			Q = y[idx1]*y[idx2]*(Qfloat)((this->*kernel_function)(idx1, idx2) + 1.0);			
		else				
			Q = QD[idx1];
		
		return Q;
	}
	
private:
	const svm_parameter* param;
	const svm_problem* prob;	
	schar* y;
	Qfloat C_inv;

	Qfloat Eta;	
	Qfloat *QD;

	sCache *kernelCache;
	//Cache *kernelCache;
	
	mutable int kernelEval;	
};

//------------------------------------------------------------------------------------------------------------------

//
// Gram matrix of BVM
//
class BVM_Q: public Kernel
{
public:
	BVM_Q(const svm_problem* prob_, const svm_parameter* param_, schar *y_) : Kernel(prob_->l, prob_->x, *param_)
	{
		// init
		prob  = prob_;
		param = param_;
		y     = y_;
		kappa = (Qfloat)((this->*kernel_function)(0,0) + 1.0 + (1.0/(param->C)));

		if (!Kernel::IsSelfConst(*param))
		{
			printf("kernel: %d, BVM can work for isotropic kernels only!\n",param->kernel_type);
			exit(-1);
		}

		kernelCache = new sCache(param_, prob->l);
		kernelEval  = 0;
	}
	~BVM_Q() { delete kernelCache; }

	Qfloat *get_QD() const { return NULL; }
	Qfloat *get_Q(int idx, int basisNum, int* basisIdx) const
	{
		requested_kevals += basisNum;
		int numRet;
		Qfloat *Q = kernelCache->get_data(idx, basisNum, numRet);
		if (Q != NULL)
		{
#ifdef COMP_STAT
			kernelEval += (basisNum - numRet);
#endif

			// fill remaining
			for(int i = numRet; i < basisNum; i++)
			{
				int idx2 = basisIdx[i];
				if (idx != idx2)
					Q[i] = (Qfloat)y[idx]*y[idx2]*(Qfloat)((this->*kernel_function)(idx, idx2) + 1.0);
				else
					Q[i] = kappa;
			}
		}
		return Q;
	}
	double dot_c_wc(int idx, int basisNum, int* basisIdx, double *coeff, bool &depend, double thres = INF)
	{
		double dist = 0.0;
		depend      = false;
		Qfloat *Q_i = get_Q(idx, basisNum, basisIdx);
		if (Q_i != NULL)
		{
			for (int j=0; j<basisNum; j++)
				if (idx != basisIdx[j] && Q_i[j] >= thres)
				{
					depend = true;
					return INF;
				}
				else
					dist += Q_i[j]*coeff[basisIdx[j]];
		}
		return dist;
	}

	Qfloat getKappa() const { return kappa; }
	void swap_index(int i, int j) const { printf("CVC_Q::swap_index is not implemented!\n"); }

private:
	const svm_parameter* param;
	const svm_problem* prob;
	schar* y;
	Qfloat kappa;

	sCache *kernelCache;
	mutable int kernelEval;
};

//------------------------------------------------------------------------------------------------------------------
//
// Gram matrix of MCVM
//
class MCVC_Q : public Kernel
{
public:
	MCVC_Q(const svm_problem* prob_, const svm_parameter* param_, int *y_) : Kernel(prob_->l, prob_->x, *param_)
	{
		// init
		int i;
		prob  = prob_;
		param = param_;
		y     = y_;
		QD    = new Qfloat[prob->l];		
		C_inv = (Qfloat)(1.0/(param->C));

		// get diagonal
		if (Kernel::IsSelfConst(*param))
		{	
			Eta = (Qfloat)((param->sameclassYDP*((this->*kernel_function)(0,0))) + param->sameclassYDP + C_inv);
			for (i=0; i< prob->l; i++)
				QD[i] = Eta;
			printf("Initializing Kernel ... selfconsistent, k(i,i)=%g, C=%g \n",Eta,param->C);
		}
		else
		{
			double tmp = param->sameclassYDP + C_inv;
			Eta        = 0.0;
			for (i=0; i< prob->l; i++)	
			{
				QD[i] = (Qfloat)((param->sameclassYDP*((this->*kernel_function)(i, i))) + tmp);
				if (QD[i] > Eta)
					Eta = QD[i];
			}	
		}		

		kernelCache = new sCache(param_, prob->l);
		kernelEval  = 0;
	}

	~MCVC_Q()
	{
		delete kernelCache;
		delete [] QD;		
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx) const
	{
		//register the number of requested k-evals
		requested_kevals += basisNum;

		int numRet;
		Qfloat *Q = kernelCache->get_data(idx, basisNum, numRet);
		if (Q != NULL)
		{	
#ifdef COMP_STAT
			kernelEval += (basisNum - numRet);
#endif

			// fill remaining		 
			for(int i = numRet; i < basisNum; i++)
			{	
				double ydp_temp;
				int idx2 = basisIdx[i];
				if (idx != idx2){
					if (y[idx]==y[idx2])
				    	ydp_temp = param->sameclassYDP;
				    else
				    	ydp_temp = param->diffclassYDP;
				    		
					Q[i] = (Qfloat)ydp_temp*(Qfloat)((this->*kernel_function)(idx, idx2) + 1.0);			
				} else {				
					Q[i] = QD[idx];
				}	
			}						
		}
		return Q;
	}
	
	double dist2c_wc(int idx, int basisNum, int* basisIdx, double *coeff, double cNorm)
	{
		double dist = 0.0;		
		Qfloat *Q_i = get_Q(idx, basisNum, basisIdx);			
		if (Q_i != NULL)
		{
			for (int j=0; j<basisNum; j++)
				dist += Q_i[j]*coeff[j];		
			dist  = cNorm - 2.0 * dist;				
		}		
		return dist;
	}
	
	Qfloat* get_QD() const { return QD; }	
	Qfloat get_Eta() const { return Eta; }
	Qfloat get_Kappa() const { return Eta-C_inv; }
	void swap_index(int i, int j) const { printf("CVC_Q::swap_index is not implemented!\n"); }

private:
	const svm_parameter* param;
	const svm_problem* prob;	
	int* y;
	Qfloat C_inv;
	Qfloat Eta;	
	Qfloat *QD;
	sCache *kernelCache;
	mutable int kernelEval;	
};


//------------------------------------------------------------------------------------------------------------------

class CVDD_Q : public Kernel
{
public:
	CVDD_Q(const svm_problem* prob_, const svm_parameter* param_) : Kernel(prob_->l, prob_->x, *param_)
	{
		// init
		int i;
		prob  = prob_;
		param = param_;
		QD    = new Qfloat[prob->l];
		C_inv = (Qfloat)(1.0/(param->C));
		
		// get diagonal
		if (Kernel::IsSelfConst(*param))
		{	
			Eta = (Qfloat)((this->*kernel_function)(0,0) + C_inv);
			for (i=0; i< prob->l; i++)
				QD[i] = Eta;
		}
		else
		{	
			Eta = 0.0;
			for (i=0; i< prob->l; i++)	
			{
				QD[i] = (Qfloat)((this->*kernel_function)(i, i) + C_inv);
				if (QD[i] > Eta)
					Eta = QD[i];
			}	
		}		

		kernelCache = new sCache(param_, prob->l);
		kernelEval  = 0;
	}

	~CVDD_Q()
	{
		delete kernelCache;
		delete [] QD;		
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx) const
	{
		//register the number of requested k-evals
		requested_kevals += basisNum;
		
		int numRet;
		Qfloat *Q = kernelCache->get_data(idx, basisNum, numRet);
		if (Q != NULL)
		{	
#ifdef COMP_STAT
			kernelEval += (basisNum - numRet);
#endif

			// fill remaining		 
			for(int i = numRet; i < basisNum; i++)
			{
				int idx2 = basisIdx[i];
				if (idx != idx2)		
					Q[i] = (Qfloat)((this->*kernel_function)(idx, idx2));
				else				
					Q[i] = QD[idx];
			}						
		}
		return Q;
	}
	
	double dist2c_wc(int idx, int basisNum, int* basisIdx, double *coeff, double cNorm)
	{
		double dist = 0.0;		
		Qfloat *Q_i = get_Q(idx, basisNum, basisIdx);			
		if (Q_i != NULL)
		{
			for (int j=0; j<basisNum; j++)
				dist += Q_i[j]*coeff[j];		
			dist  = cNorm - 2.0 * dist;				
		}		
		return dist;
	}
	
	Qfloat* get_QD() const { return QD; }	
	Qfloat get_Eta() const { return Eta; }
	Qfloat get_Kappa() const { return Eta-C_inv; }
	void swap_index(int i, int j) const { printf("CVDD_Q::swap_index is not implemented!\n"); }

private:
	const svm_parameter* param;
	const svm_problem* prob;	
	Qfloat C_inv;

	Qfloat Eta;	
	Qfloat *QD;

	sCache *kernelCache;
	mutable int kernelEval;	
};
//------------------------------------------------------------------------------------------------------------------

//
// Gram matrix of CVR
//
class CVR_Q : public Kernel
{
public:
	CVR_Q(const svm_problem* prob_, const svm_parameter* param_) : Kernel(prob_->l, prob_->x, *param_)
	{
		// init
		int i;
		int numVar = 2*prob_->l;
		prob	   = prob_;
		param      = param_;
		QD         = new Qfloat[numVar];
		LinCoef    = new double[numVar];
		C_MU_inv   = (Qfloat)(param->mu*prob->l/param->C);	
		double tmp = 2.0/param->C;
		
		// get diagonal
		if (Kernel::IsSelfConst(*param))
		{
			Kappa         = (Qfloat)((this->*kernel_function)(0,0) + 1.0);		
			double Kappa2 = Kappa + C_MU_inv;
			Eta           = 0.0;
			for (i=0; i < prob->l; i++)
			{
				QD[i]              =  (Qfloat)Kappa2;
				QD[i+prob->l]      =  (Qfloat)Kappa2;
				double tmpLinCoef  =  prob->y[i]*tmp;				
				LinCoef[i]		   =  tmpLinCoef;
				LinCoef[i+prob->l] = -tmpLinCoef;
				Eta                =  svm_commons::max(Eta, Kappa2+tmpLinCoef);
				Eta                =  svm_commons::max(Eta, Kappa2-tmpLinCoef);
			}
		}
		else
		{
			double tmp = 1.0 + C_MU_inv;
			Eta        = 0.0;
			Kappa      = 0.0;
			for (i=0; i < prob->l; i++)	
			{
				double Kappa2	   =  (this->*kernel_function)(i, i) + tmp;
				QD[i]			   =  (Qfloat)Kappa2;
				QD[i+prob->l]	   =  (Qfloat)Kappa2;
				double tmpLinCoef  =  prob->y[i]*tmp;			
				LinCoef[i]		   =  tmpLinCoef;
				LinCoef[i+prob->l] = -tmpLinCoef;
				Eta                =  svm_commons::max(Eta, Kappa2+tmpLinCoef);
				Eta                =  svm_commons::max(Eta, Kappa2-tmpLinCoef);
				if (Kappa2 > Kappa)
					Kappa = (Qfloat)Kappa2;
			}	
			Kappa -= C_MU_inv;
		}		
		for (i=0; i < numVar; i++)	
			LinCoef[i] += Eta;
				
		buffer[0]   = new Qfloat[numVar];
		buffer[1]   = new Qfloat[numVar];
		next_buffer = 0;
		kernelCache = new sCache(param_, prob->l);
		kernelEval  = 0;
	}

	~CVR_Q()
	{
		delete kernelCache;		
		delete [] QD;		
		delete [] LinCoef;
		delete buffer[0];
		delete buffer[1];
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx) const
	{
		assert(basisNum % 2 == 0);
		int numCache = basisNum/2;
		int si       = 1;
		int real_idx = idx;		
		if (real_idx >= prob->l)
		{
			real_idx -= prob->l;
			si        = -1;
		}
		
		//register the number of requested k-evals
		requested_kevals += basisNum;
		
		int numRet;
		Qfloat *data = kernelCache->get_data(real_idx, numCache, numRet);
		if (data != NULL)
		{	
#ifdef COMP_STAT
			kernelEval += (numCache - numRet);
#endif
			// fill remaining		 
			int i;
			for(i = numRet; i < numCache; i++)
				data[i] = (Qfloat)((this->*kernel_function)(real_idx, basisIdx[2*i]) + 1.0);

			// reorder and copy
			Qfloat *buf = buffer[next_buffer];
			next_buffer = 1 - next_buffer;

			for(i=0; i < numCache; i++)
			{
				int bufIdx1  =  i*2;
				int bufIdx2  =  i*2+1;
				buf[bufIdx1] =  si*data[i];
				buf[bufIdx2] = -buf[bufIdx1];
				if (basisIdx[bufIdx1] == idx)
					buf[bufIdx1] = QD[idx];
				if (basisIdx[bufIdx2] == idx)
					buf[bufIdx2] = QD[idx];
			}					
			return buf;
		}
		else 
			return NULL;
	}
	
	double dot_c_wc(int idx, int basisNum, int* basisIdx, double *coeff)
	{
		assert(basisNum % 2 == 0);
		double dot   = 0.0;
		int numCache = basisNum/2;
		int si       = 1;
		int real_idx = idx;		
		if (real_idx >= prob->l)
		{
			real_idx -= prob->l;
			si        = -1;
		}		
		int numRet;
		Qfloat *data = kernelCache->get_data(real_idx, numCache, numRet);
		if (data != NULL)
		{	
#ifdef COMP_STAT
			kernelEval += (numCache - numRet);
#endif
			// fill remaining		 
			int i;
			for(i = numRet; i < numCache; i++)
				data[i] = (Qfloat)((this->*kernel_function)(real_idx, basisIdx[2*i]) + 1.0);

			for(i=0; i < numCache; i++)
				dot += data[i] * (coeff[i*2]-coeff[i*2+1]);			
			dot *= si;
		}		
		return dot;
	}
	
	double *get_LinCoef() const { return LinCoef; }
	Qfloat *get_QD() const { return QD; }	
	Qfloat get_Eta() const { return (Qfloat)Eta; }
	Qfloat get_Kappa() const { return Kappa; }
	void swap_index(int i, int j) const { printf("CVR_Q::swap_index is not implemented!\n"); }

private:
	const svm_parameter* param;
	const svm_problem* prob;	
	Qfloat C_MU_inv;

	double Eta;	
	Qfloat Kappa;
	Qfloat *QD;
	double *LinCoef;

	sCache *kernelCache;
	mutable int kernelEval;		
	mutable int next_buffer;		 // which buffer to fill	
	Qfloat *buffer[2];		 // sometimes, the outside program needs 2 columns (at the same time)	
};


#endif /*MEBKERNELS_H_*/
