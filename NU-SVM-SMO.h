#ifndef NU_SVM_SMO_H_
#define NU_SVM_SMO_H_

#include "SVM-commons.h"
#include "CSVM-SMO.h"


// Solver for nu-svm classification and regression
// Solves the same that CSVM_SMO but with an additional constraint:
// e^T \alpha = constant
//
class Solver_NU : public Solver
{
public:
	Solver_NU() {}
	void Solve(int l, const QMatrix& Q, const double *b, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
	{
		this->si = si;
		Solver::Solve(l,Q,b,y,alpha,Cp,Cn,eps,si,shrinking);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	void do_shrinking();
};

#endif /*NU_SVM_SMO_H_*/
