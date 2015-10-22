#ifndef MEB_H_
#define MEB_H_

#include "SVM-commons.h"
#include "MEB-utilities.h"
#include "MEB-SMO.h"

// Generic class for a MEB solver
// Currently there are available two methods to solve the problem 
// (i)  Badoiu-Clarkson's algorithm.
// (ii) Yildrim's algorithm (see A. Yildrim, "Two Algorithms for the MEB problem", SIAM Journal of Optimization, 2008)

// Technical comments:
// A lot of methods are virtual, that is, the class need to be derivated
// This happens because the MEB is used to cast for an specific problem:
// classification, novelty-detection, regression, etc.  


class Solver_Core
{
public:
	Solver_Core() {}
	virtual ~Solver_Core() { }
	bool showCoreSolverInfo;
	
	void Init(const svm_problem* prob, const svm_parameter* param);
	bool Create(double cvm_eps);
	
	int Solve(int num_basis, double cvm_eps, int method=BADOUCLARKSON, bool cooling = true, bool randomized = true);
	int Badou_Clarkson_Algorithm(int num_basis, double cvm_eps, bool cooling = true, bool randomized = true);
	int Yildrim_Algorithm(int num_basis, double cvm_eps, bool dropping, bool cooling = true, bool randomized = true);
	int Yildrim_Algorithm1(int num_basis, double cvm_eps, bool cooling = true, bool randomized = true);
	int Yildrim_Algorithm2(int num_basis, double cvm_eps, bool cooling = true, bool randomized = true);
	int SWAP_Algorithm(int num_basis, double cvm_eps, bool cooling = true, bool randomized = true, bool glotonous = true);

	double GetCoreNorm2 () const { return coreNorm2; } 
	virtual bool IsExitOnMaxIter() const = 0;
	virtual double ComputeRadius2() = 0;
	virtual double yildrim_computeRadius2(int iteration_type, double yfactor, double dist2){return 0;}
	virtual double update_radius_and_cache(int step, double new_radius2, double alpha, int core_idx_out, int core_idx_in){return 0;}
	virtual double cosa_computeRadius2(double maxDistance2, double maxCorePointDistance2, double beta){return 0;}
	virtual double ComputeSolution(double *alpha, double Threshold) = 0;
	
    	void show_memory_usage(char *header);
    	void process_mem_usage(double& vm_usage, double& resident_set);

    	int greedy_it; //for greedy algorithms
    	unsigned long int smo_it; //for algorithms using SMO
    	
protected:
	virtual void _Init() = 0;
	virtual bool _Create(double cvm_eps) = 0;
	
	virtual double _maxDist(double maxDistance2, int &maxDistance2Idx) = 0;
	virtual double _maxDistFromSampling(double maxDistance2, int &maxDistance2Idx) = 0;
	virtual double _minDistFromCoreSet(double minDistance2, int &minDistance2Idx) = 0;
	virtual double _maxDistFromCoreSet(double maxDistance, int &maxDistanceIdx) = 0;
	virtual double _most_inner_point_for_SWAP(double app_r2, double out_dist2,int out_idx, int &coreset_idx_in,double &alpha_swap, double &dist2_in, double &dist2_pts) = 0;
	virtual double _best_point_for_SWAP(double app_r2, double out_dist2,int out_idx, int &coreset_idx_in,double &alpha_swap, double &dist2_in, double &dist2_pts) = 0;

	virtual void _UpdateCoreSet(int maxDistance2Idx) = 0;	
	
	double _Yildrim_Update(double maxdist2, int maxdist2_tr_idx, bool dropping, double mindist2, int mindist2_idx, int &iteration_type);
			
	MEB_SMO          *solver;
	const svm_parameter *param;
	const svm_problem   *prob;

	int    maxNumBasis;
	int   *coreIdx;
	int    coreNum;
	double coreNorm2;
	double r2;	
	int id_fw1_it;
	int id_fw2_it;
	int id_swap_it;

	
	double  *outAlpha;
	double  *tmpAlpha;
	double  *tempD;
	
	Qfloat *Q_center_dot_out; //cache kevals between current center and last point included in the core-set
	Qfloat *Q_center_dot_in; //cache kevals between current center and a point of the center
	Qfloat *Q_furthest; //cache kevals between current center and a point of the center

	Qfloat center_dot_out;
	int furthest_coreidx;
	int closest_coreidx;
	int *inverted_coreIdx;
	int previous_violator_idx;
	Qfloat* previousQidx;
	Qfloat next_dot_violator_c;
	double previous_lambda;

	int allocated_size_for_weights;
	int actual_size_tmp_D;
	int actual_size_tmp_alpha;
	
	double Eta; 
	
};

#endif /*MEB_H_*/
