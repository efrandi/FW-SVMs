#ifndef MEB_BASED_SVMS_H_
#define MEB_BASED_SVMS_H_

#include "random.h"
#include "SVM-commons.h"
#include "MEB-utilities.h"
#include "MEB-solvers.h"

#define RELEASE_VER 1
//#define COMP_STAT 1

// API for CVM 
void solve_cvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn);

void solve_fvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn);

int solve_cvm_warm(
	const svm_problem *prob, const svm_parameter* param, int* initial_coreset,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn, int size_new=1000);

// API for BVMs
void solve_bvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn);

// API for MCVM 
void solve_mcvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn);

// API for CVDD NOT YET IMPLEMENTED
void solve_cvdd(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si);

// API for CVR NOT YET IMPLEMENTED
void solve_cvr(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si);

#endif /*MEB_BASED_SVMS_H_*/
