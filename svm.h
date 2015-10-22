#ifndef _LIBSVM_H
#define _LIBSVM_H

#include "SVM-commons.h"
#include "CSVM-SMO.h"

using namespace svm_commons;

#ifdef __cplusplus
extern "C" {
#endif

struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
struct svm_model *svm_train_warm(svm_model* model, const svm_problem *new_prob, int* check_list);

void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);
void svm_cross_validation_2(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target, double &meansvs, double &mean_real_kevals, double &mean_requested_kevals);

//struct svm_model *svm_simulate_distributed(const svm_problem *prob, const svm_parameter *param, Partition* partition, distrib_statistics *stats);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
double svm_get_svr_probability(const struct svm_model *model);

void svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_rank(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_destroy_model(struct svm_model *model);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void displayInfoAboutModel(const struct svm_model* model);
extern void (*svm_print_string) (const char *);

//not originally declared in the API
void svm_labels(const svm_problem *prob, int *nr_class_ret, int **label_ret);
void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm);
void svm_binary_svc_probability(const svm_problem *prob, const svm_parameter *param,double Cp, double Cn, double& probA, double& probB);
void solve_c_svc_warm(const svm_problem *prob, const svm_parameter* param, double *alpha, Solver::SolutionInfo* si, double Cp, double Cn);
void solve_c_svc(const svm_problem *prob, const svm_parameter* param, double *alpha, Solver::SolutionInfo* si, double Cp, double Cn);
	
#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
