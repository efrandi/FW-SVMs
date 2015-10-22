#ifndef SVM_COMMONS_H_
#define SVM_COMMONS_H_

#ifndef _MSC_VER
#include <sys/times.h>
#include <unistd.h>
#else
#include <time.h>
#endif


#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

// SVM_commons.h
// This file defines structures to solve SVM problems
// and some basic functions using in different implementations
// Some of the strutures defined here:
// svm_node, svm_problem, svm_parameter, svm_model, distrib_statistics 


//#define SHORT_INDEX 1
//#define INT_FEAT 1

#define TAU	1e-12
#define INF	HUGE_VAL	

#define INVALID_C -1		  // initial C

#ifdef SHORT_INDEX
    typedef short int INDEX_T;
#else
    typedef int INDEX_T;
#endif

#ifdef INT_FEAT
    typedef short int NODE_T;
#else
    typedef double NODE_T;
#endif

typedef double Qfloat;														// use single precision
typedef double Yfloat;														// use full precision
typedef double  Xfloat;														// use single precision
typedef double  Wfloat;														// use single precision
typedef double Afloat;														// use full precision
typedef signed char schar;													// for convenient


struct svm_node
{
	INDEX_T index;
	NODE_T  value;
};

struct svm_problem
{
	int l;
	int u;
	double *y;
	struct svm_node **x;
	int input_dim;
	struct SGraphStruct *graph;
};

enum { CVM, MCVM, C_SVC, PERCEPTRON, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR, CVDD, CVM_LS, CVR, BVM};	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED, EXP, NORMAL_POLY, INV_DIST, INV_SQDIST };   /* kernel_type */
enum { ONE_VS_ONE, ONE_VS_REST, C_AND_S };  /* mc_type */
enum { ADSVM, ASHARAF, ADNORM, ANN}; 
enum { BADOUCLARKSON, YILDRIM1, YILDRIM2, PARTAN, SWAP, lSWAP, BVMtsang,PANIGRAHY};
enum { RANDOM_MEB_INIT, YILDRIM_INIT, YILDRIM_SAMPLE_INIT};

struct svm_parameter
{
	int svm_type;
	int online_rule;    /* for online training */
	int kernel_type;
	int degree;         /* for poly */
	double gamma;       /* for poly/rbf/sigmoid */
	double coef0;       /* for poly/sigmoid */

	double reg_param;
	double scale_param;
	
	/* these are for training only */
	double cache_size;  /* in MB */
	double eps;	        /* stopping criteria */
	double C;           /* for C_SVC, EPSILON_SVR, BVM and NU_SVR */
	int nr_weight;	    /* for C_SVC */
	int *weight_label;  /* for C_SVC */
	double* weight;     /* for C_SVC */
	double nu;          /* for NU_SVC, ONE_CLASS, and NU_SVR */
	double mu;          /* for CVR */
	double p;           /* for EPSILON_SVR */
 	int mc_type;        /* for multiclass SVM/CVM/BVM */
	int shrinking;      /* use the shrinking heuristics */
	int probability;    /* do probability estimates */
	int sample_size;    /* size of probabilistic sampling in BVM */
	bool safe_stopping;
	int nsamplings_safe_stopping; 
	int nsamplings_iterations; 
	
	int num_basis;
	int knn;
	int weight_type;
	//for L2-SVMs
	int MEB_algorithm;
	bool cooling;
	bool randomized;
	bool stop_with_real_meb;
	int MEB_init_method;
	//for MCVM
	int mcvm_type;
	double diffclassYDP; /*ydot product for different y's*/ 
	double sameclassYDP; /*ydot product for equal y's*/
	int nrclasses;
	int exp_type;
	int* labels;
	//for online-dl2
	double online_nu;
	int budget;

	char* filename;
	int max_iterations;
	int frecuency_messages;

};

//
// svm_model
//
struct svm_model
{
	svm_parameter param;	// parameter
	int nr_class;		// number of classes, = 2 in regression/one class svm
	int l;			// total #SV
	int u;
	svm_node **SV;		// SVs (SV[l])
	double **sv_coef;	// coefficients for SVs in decision functions (sv_coef[n-1][l])
	double *rho;		// constants in decision functions (rho[n*(n-1)/2])
	double *cNorm;		// center Norm of decison functions (rho[n*(n-1)/2])
	double *probA;          // pariwise probability information
	double *probB;
	int nsubmodels;
	double *obj; 
	int *subsizes;
	
	// for classification only

	int *label;		// label of each class (label[n])
	int *nSV;		// number of SVs for each class (nSV[n])
				// nSV[0] + nSV[1] + ... + nSV[n-1] = l
	// XXX
	int free_sv;		// 1 if svm_model is created by svm_load_model
				// 0 if svm_model is created by svm_train		
	// for MCVM only
	int *ySV;
	double trainingTime;
	unsigned long int *smo_it; //for solvers using SMO
	int *greedy_it;//for greedy algorithms
	
};

//
// decision_function
//
struct decision_function
{
	double *alpha;
	double rho;	
	double cNorm;
	double obj;
	int nSV;
	int smo_it; //for solvers using SMO
	int greedy_it;//for greedy algorithms
};

//
// structure to handle statics in simulations
// of the distributed scenario

struct distrib_statistics
{					
	double trainingTime;
	double meanRemoteTime;
	double stdRemoteTime;
	double maxRemoteTime;
	double sumRemoteTime;
	
	double meanNodeCompression;
	double stdNodeCompression;
	double totalCompression;
	
	double N_kevals;
	int size_cent_model;
	double mean_remote_N_kevals;
	double std_remote_N_kevals;
	double max_remote_N_kevals;
	double sum_remote_N_kevals; 
};

namespace svm_commons{
#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
}


//#define INF HUGE_VAL
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void info(const char *fmt,...);
void info_flush();
double getRunTime();

inline double powi(double base, int times)
{
        double tmp = base, ret = 1.0;

    for(int t=times; t>0; t/=2)
	{
        if(t%2==1) ret*=tmp;
        tmp = tmp * tmp;
    }
    return ret;
}

#endif /*SVM_COMMONS_H_*/
