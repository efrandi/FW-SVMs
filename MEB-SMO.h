#ifndef MEB_SMO_H_
#define MEB_SMO_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "SVM-commons.h"
#include "CSVM-SMO.h"
#include "MEB-utilities.h"

//------------------------------------------------------------------------------------------------------------------
//
// Solver for the optimization problem associated to the MEB 
// Used to solve SVM classification and regression problems
// The solver is an implementation of Sequential Minimal Optimization
// For details of this approach see Fan, 2005. 
//
//
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) - 0.5(\alpha^T D)
//
//		e^T \alpha = 1
//		\alpha >= 0
//
// Given:
//
//	Q, D and an initial feasible point \alpha
//	num is the size of vectors and matrices
//
// solution will be put in \alpha
//
class MEB_SMO : public Solver
{
public:
	// constructor
	//
	// The index of coreset, core set size, gram matrix, linear coefficients, initial alpha
	MEB_SMO(int *CoreIdx, int numCore, const QMatrix& Q, const double* D, double *inAlpha, double Eps, double MinEps, int initSize = INITIAL_ALLOCATION_SIZE);
	~MEB_SMO();
		
	// The index of coreset, core set size, linear coefficients
	// return how many iteration used in SMO
	int Solve(int *CoreIdx, int numCore, const double* newD);
	double computeObj() const;
	double computeCNorm() const;
	const double* getGradient() const { return G; }	
	double *getAlpha() const { return alpha; }
    int get_allocated_size() { return allocatedSize; }
private:
	double* vec_d;								// vector d, linear objective part		
	int allocatedSize;							// allocated space for storage	
	int *_CoreIdx;
	Qfloat *CacheQ_i;
	double minEps;

	int select_working_set(int &i, int &j);
	void update_alpha_status2(int i)
	{
		if(alpha[i] <= 0.0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	double get_C(int i) { return Cp; }
	double calculate_rho() {printf("not necessary to compute rho\n"); exit(-1);}
	void do_shrinking() {printf("not implemented\n"); exit(-1);}	
};

//------------------------------------------------------------------------------------------------------------------
#endif /*MEB_SMO_H_*/
