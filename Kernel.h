#ifndef KERNEL_H_
#define KERNEL_H_

#include <math.h>
#include "SVM-commons.h"
#include "Cache.h"
using namespace svm_commons;


//Definition of the classes to compute and handle kernel evaluations
//Kernel is a base class from which specific Kernel handlers should be built
//The following implementations are defined:
//
//	    SVC_Q (for classification), 
//		ONE_CLASS_Q (for novelty detection)
//		SVR_Q (for regression)      
 
class QMatrix {
public:	
	virtual Qfloat *get_Q(int column, int len, int* indice = NULL) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	
	static unsigned long int real_kevals;
	static unsigned long int requested_kevals;
	
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();
	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	
	static unsigned long int get_real_kevals();
	static unsigned long int get_requested_kevals();
	
	static void reset_real_kevals();
	static void reset_requested_kevals();
	
	virtual Qfloat *get_Q(int column, int len, int* indice) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int i, int j) const;

	static double dot(const svm_node *px, const svm_node *py);
    static double distanceSq(const svm_node *x, const svm_node *y);

	bool IsSelfConst() const 
	{
		if (kernel_type == RBF || kernel_type == NORMAL_POLY || kernel_type == EXP || kernel_type == INV_DIST || kernel_type == INV_SQDIST)
			return true;
		else 
			return false;
	}

	static bool IsSelfConst(const svm_parameter& param)  
	{
		if (param.kernel_type == RBF || param.kernel_type == NORMAL_POLY || param.kernel_type == EXP || param.kernel_type == INV_DIST || param.kernel_type == INV_SQDIST)
			return true;
		else 
			return false;
	}

	
	 
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:

	
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	double kernel_linear(int i, int j) const;
	double kernel_poly(int i, int j) const;
	double kernel_rbf(int i, int j) const;
	double kernel_sigmoid(int i, int j) const;
	double kernel_precomputed(int i, int j) const;
    double kernel_exp(int i, int j) const;
    double kernel_normalized_poly(int i, int j) const;
	double kernel_inv_sqdist(int i, int j) const;
	double kernel_inv_dist(int i, int j) const;	    
};

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel
{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(int)(param.cache_size*(1<<20)));
		QD = new Qfloat[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i]= (Qfloat)(this->*kernel_function)(i,i);
	}

	
	Qfloat *get_Q(int i, int len, int* indice) const
	{
		Qfloat *data;
		int start;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(int j=start;j<len;j++)
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
		}
		return data;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	Qfloat *QD;
};

class ONE_CLASS_Q: public Kernel
{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		cache = new Cache(prob.l,(int)(param.cache_size*(1<<20)));
		QD = new Qfloat[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i]= (Qfloat)(this->*kernel_function)(i,i);
	}
	
	
	Qfloat *get_Q(int i, int len, int *indice) const
	{
		Qfloat *data;
		int start;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(int j=start;j<len;j++)
				data[j] = (Qfloat)(this->*kernel_function)(i,j);
		}
		return data;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		delete[] QD;
	}
private:
	Cache *cache;
	Qfloat *QD;
};

class SVR_Q: public Kernel
{ 
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new Cache(l,(int)(param.cache_size*(1<<20)));
		QD = new Qfloat[2*l];
		sign = new schar[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k]= (Qfloat)(this->*kernel_function)(k,k);
			QD[k+l]=QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int i, int j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}

		
	Qfloat *get_Q(int i, int len, int *indice) const
	{
		Qfloat *data;
		int real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
		{
			for(int j=0;j<l;j++)
				data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
		}

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(int j=0;j<len;j++)
			buf[j] = si * (Qfloat)sign[j] * data[index[j]];
		return buf;
	}

	Qfloat *get_QD() const
	{
		return QD;
	}

	~SVR_Q()
	{
		delete cache;
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
	}
private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	Qfloat *QD;
};


#endif /*KERNEL_H_*/
