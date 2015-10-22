
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "MEB-based-SVMs.h"
#include "FW-based-SVMs.h"

void solve_cvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	// info
	info("**** L2 SVM model. N° Training Patterns = %ld\n", prob->l);
	
	// init		
	//srandom (0);
	srandom (time (0));

	for(int i = 0; i < prob->l; i++)
		alpha[i] = 0.0;

	// solve CVM
	Solver_CVM solver;
	solver.set_init_method(param->MEB_init_method);
	solver.Init(prob,param);
	
	double cvm_eps = param->eps;
	if (cvm_eps <= 0.0)
		cvm_eps = 0.00001;
	//(PREDICT_BND/(solver.GetKappa())+2.0/param->C/E_NUM_CV)/2.0/solver.GetEta();	
	
	//printf("Epsilon for MEB= %.8g\n",cvm_eps);
	//printf("Epsilon for SMO= %.8g\n",SMO_EPS);
	
	bool flag = solver.Create(cvm_eps);
	
	
	if (flag)
	{
		int coreNum;
		coreNum = solver.Solve(param->num_basis,cvm_eps, param->MEB_algorithm, param->cooling, param->randomized);
		// compute solution vector		
		double THRESHOLD = 1e-5/coreNum;

		double bias      = solver.ComputeSolution(alpha,TAU);
		double coreNorm2 = solver.GetCoreNorm2();

		// info in CVM
				
		si->obj    = 0.5*(solver.GetEta() - coreNorm2);
		si->rho    = -bias;
		si->margin = coreNorm2;
		//si->r2 = solver.GetEta() - coreNorm2;
		si->smo_it = solver.smo_it;
		si->greedy_it = solver.greedy_it;
	}
	
	//printf("End solve_cvm (cold).\n");	
	
}

void solve_fvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	// info
	info("**** L2 SVM model. N° Training Patterns = %ld\n", prob->l);
	info("**** Using Frank-Wolfe based Algorithms\n");
	
	// init		
	//srandom (0);
	srandom (time (0));

	for(int i = 0; i < prob->l; i++)
		alpha[i] = 0.0;
	
	// create a FW based solver for the corresponding L2-SVM 
	FW_L2SVM_Solver *FVMsolver = new FW_L2SVM_Solver(prob,param);

		
	if (FVMsolver != NULL)//If successfully created ...
	{
		FVMsolver->set_initialization_method(param->MEB_init_method);
	
		int coreNum;
		//compute the solution
		coreNum = FVMsolver->Solve(param->num_basis, param->eps, param->MEB_algorithm, param->cooling, param->randomized);
	
		double THRESHOLD = 1e-5/coreNum;

		double bias      = FVMsolver->ComputeSVMSolution(alpha,TAU);
		double objective = FVMsolver->GetObjective();
			
		si->obj    = objective;
		si->rho    = -bias;
		printf("BIAS: %g\n",bias);
		printf("SI.RHO: %g\n",si->rho);
		
		si->margin = 0.0;
		si->smo_it = FVMsolver->GetSMOIterations();
		si->greedy_it = FVMsolver->GetFWIterations();
	}

	delete FVMsolver;

}



int solve_cvm_warm(
	const svm_problem *prob, const svm_parameter* param, int* initial_coreset,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn, int size_new)
{
	// info
	info("CVM model. Warm-Start. N° Training Patterns = %ld\n", prob->l);
	
	// init		
	srandom (time (0));
	//srandom (0);
	Solver_CVM solver;
	

	printf("before init\n");
	
	solver.Init(prob,param,initial_coreset,alpha);
	// solve CVM
	
	double cvm_eps = param->eps;
	if (cvm_eps <= 0.0)
		cvm_eps = 0.00001;
	// (PREDICT_BND/(solver.GetKappa())+2.0/param->C/E_NUM_CV)/2.0/solver.GetEta();	
	
	// printf("Epsilon for MEB= %.8g\n",cvm_eps);
	// printf("Epsilon for SMO= %.8g\n",SMO_EPS);
	
	printf("before create\n");
	bool flag = solver.Create(cvm_eps);	
			
	
	if (flag)
	{	printf("beginning to solve\n");
		int coreNum = solver.Solve(param->num_basis,cvm_eps, param->MEB_algorithm, param->cooling, param->randomized);
			
		for(int i = 0; i < prob->l; i++)
			alpha[i] = 0.0;
			
		// compute solution vector		
		double THRESHOLD = 1e-5/coreNum;
		double bias      = solver.ComputeSolution(alpha, THRESHOLD);	
		double coreNorm2 = solver.GetCoreNorm2();

		// info in CVM		
		si->obj    = 0.5*coreNorm2;
		si->rho    = -bias;
		si->margin = coreNorm2;
		si->smo_it = solver.smo_it;
		si->greedy_it = solver.greedy_it;

	}
	
	//printf("End solve_cvm (warm-start).\n");

	return 0;
}


//---------------------------------------------------------------------------------------------------------------------

void solve_mcvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{
	// info
	info("MCVM model. Training patterns = %ld\n", prob->l);
	
	// init		
	srandom (time (0));
	//srandom (0);
	for(int i = 0; i < prob->l; i++)
		alpha[i] = 0.0;

	// solve MCVM
	
	Solver_MCVM solver;
	solver.Init(prob,param);
	
	double cvm_eps = param->eps;
	if (cvm_eps <= 0.0)
		cvm_eps = 0.00001; 
	//  (PREDICT_BND/(solver.GetKappa())+2.0/param->C/E_NUM_CV)/2.0/solver.GetEta();	
	//	printf("Epsilon for MEB = %.8g\n",cvm_eps);
	//	printf("Epsilon for SMO = %.8g\n",SMO_EPS);
	
	bool flag = solver.Create(cvm_eps);	
	if (flag)
	{
		
		int coreNum = solver.Solve(param->num_basis,cvm_eps);
		
	
		// compute solution vector		
		double THRESHOLD = 1e-6/coreNum;
		double bias      = solver.ComputeSolution(alpha, THRESHOLD);	
		double coreNorm2 = solver.GetCoreNorm2();

		// info in CVM		
		si->obj    = 0.5*coreNorm2;
		si->rho    = -bias;
		si->margin = coreNorm2;
	}
}

void solve_bvm(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn)
{

    // info
	info("solving bvm model, num pattern = %ld\n", prob->l);

	// init
	srandom (time (0));
	//srandom (0);
    for(int i = 0; i < prob->l; i++)
		alpha[i] = 0.0;

    // solve BVM
    Solver_BVM solver;
	solver.Init(prob,param,alpha);
	double bvm_eps = param->eps;
	if (bvm_eps <= 0.0)
		bvm_eps = 0.00001;

		//    bvm_eps = (4e-6/(solver.GetKappa()-1.0/param->C)+2.0/param->C/E_NUM_CV)/2.0/solver.GetKappa();
	printf("epsilon = %.9g\n",bvm_eps);
    int coreNum = solver.Solve(param->num_basis,bvm_eps, param->cooling, param->randomized);

    // compute solution vector
	double THRESHOLD = 1e-5/coreNum;
	double bias      = solver.ComputeSolution(alpha, THRESHOLD);
	double coreNorm2 = solver.GetCoreNorm2();

	// info in BVM
	si->obj    = 0.5*(solver.GetKappa() - coreNorm2);//0.5*coreNorm2;
	si->rho    = -bias;
	si->margin = coreNorm2;
	si->smo_it = solver.smo_it;
	si->greedy_it = solver.greedy_it;


}
