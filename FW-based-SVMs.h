
#include "MEB-SMO.h"
#include "MEB-kernels.h"

//Solver for a L2SVM based on FW Methods
//Problem:
//
//	min 0.5(\alpha^T Q \alpha)
//	s.t.	1^T \alpha = 1, \alpha >= 0
//
//Solution is put in outAlpha
class FW_L2SVM_Solver
{

	private:

		const svm_parameter *param;//SVM parameters
		const svm_problem   *prob;//Data
		schar *y;//labels
		CVC_Q *kernelQ; //class implementing the kernel function for this problem
		
		int init_method;//initialization method
		MEB_SMO *SMO_Solver;//Used to solve a nested or initial L2-SVM Problem using SMO.
	
		double 	objective;//objective function value
		int    	toward_vertex;//idx of the training point to move the solution towards 
		int    	away_vertex;//idx of the training point to move the solution away
		double  toward_gradient;
		double  away_gradient;

		int  	*coreIdx; //contains the "data ids" of active points, i.e. coreIdx[i]=j means that the i-th active point is the j-th example
		int     *inverted_coreIdx;//-1 if the point is not active, otherwise inverted_coreIdx[j]=i means that the j-th active point is the i-th example
		int   	coreNum; //number of active points

		int     maxNumBasis; //maximum allowed number of active points

		double  *outAlpha;//stores the last solution to the optimization problem
		double  *tmpAlpha;
		double  *tempD;//this is required because of the current form of MEB_SMO

 		int greedy_it; //FW iterations
    	unsigned long int smo_it; //SMO iterations
   
		int allocated_size_for_alpha;

		//cached things for computations
		Qfloat *gradientALLPoints; //gradient coordinates corresponding to the ALL points, always updated
		Qfloat *gradientActivePoints; //gradient coordinates corresponding to the active points (coreset points), always updated
	
		Qfloat *Q_actives_dot_toward; 
		Qfloat *Q_actives_dot_away; 
		Qfloat *previousQcolumn;

		//for the random sampling 
		int posNum;
		int negNum;	
		int *posIdx;
		int *negIdx;
		int pNum;
		int nNum;
		bool posTurn;	
		int nsamplings_randomized_iterations;

		int StandardFW(int num_basis, double convergence_eps, bool cooling, bool randomized);
		int MFW(int num_basis, double convergence_eps, bool cooling, bool randomized);
		int PartanFW(int num_basis, double convergence_eps, bool cooling, bool randomized);
		int SWAPFW(int num_basis, double convergence_eps, bool cooling, bool randomized);

		double TowardVertex(int &towardIdx); 
		double AwayVertex(int &awayIdx); 
		
		double ComputeGradientCoordinate(int idx, Qfloat** Qcolumn);
		int ChooseRandomIndex(bool balanced);

		bool Initialize();
		int Yildirim_Initialization();
		int RandomSet_Initialization();
		int FullSet_Initialization();

		bool AllocateMemoryForFW(int initial_size);
		bool CheckMemoryForFW();
		bool FreeMemoryForFW();

		double safe_stopping_check(double Objective, double &dualGap, int &towardIdx, double &towardGrad);

public:

	FW_L2SVM_Solver(const svm_problem *_prob, const svm_parameter* _param){

		prob  = _prob;
		param = _param;
		nsamplings_randomized_iterations = param->nsamplings_iterations;

		y       = new schar[prob->l];
		inverted_coreIdx = new int[prob->l];
		coreIdx = new int[prob->l];
		
		tempD = Malloc(double,2*INITIAL_CS);
		tmpAlpha = Malloc(double,2*INITIAL_CS);

		allocated_size_for_alpha = 2*INITIAL_CS;
		
		posIdx  = new int[prob->l];
		negIdx  = new int[prob->l];

		posNum	= 0;
		negNum	= 0;

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
			
			inverted_coreIdx[i] = -1;
			coreIdx[i] = -1;
		}


		outAlpha = &tmpAlpha[0];
		
		kernelQ  = new CVC_Q(prob, param, y);		
		
		pNum     = posNum;
		nNum     = negNum;	
	
		greedy_it = 0;
		smo_it = 0;
	}

	~FW_L2SVM_Solver(){

		delete [] posIdx;
		delete [] negIdx;
		delete [] y;
		delete [] coreIdx;
		delete [] inverted_coreIdx;
		delete kernelQ;
		delete SMO_Solver;
		free(outAlpha);
		free(tempD);
		
	}


	int Solve(int num_basis, double FW_eps, int method, bool cooling, bool randomized);

	double ComputeSVMSolution(double *alpha, double Threshold);
	double GetObjective(){
		return objective;
	}
	double GetSMOIterations(){
		return smo_it;
	}
	double GetFWIterations(){
		return greedy_it;
	}
	void set_initialization_method(int im){
			   this->init_method = im;//Enum RANDOM_MEB_INIT, YILDRIM_INIT, YILDRIM_SAMPLE_INIT
	}

};	