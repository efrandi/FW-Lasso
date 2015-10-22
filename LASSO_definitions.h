#ifndef LASSO_DEFINITIONS_H_
#define LASSO_DEFINITIONS_H_

#include <stdio.h>
#include <stdlib.h>     
#include <string.h>

#define TAU	1e-12

typedef double Qfloat;
typedef signed char schar;												

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct data_node;
struct lasso_problem;
struct lasso_parameters;
struct lasso_model;

enum { BC, FW, MODFW, PARTAN, SWAP}; //training algorithms
enum { ZERO };//initialization methods
enum { PRIMAL, DUAL };//data representation model
enum { EXP_SINGLE_TRAIN, EXP_REGULARIZATION_PATH};
enum { STOPPING_WITH_DUAL_GAP, STOPPING_WITH_INF_NORM, STOPPING_WITH_OBJECTIVE};
enum { UNIFORM, BLOCKS};

struct data_node
{
	int index;
	double value;
};

struct lasso_stats
{
	double n_iterations;
	double n_performed_dot_products;
	double n_requested_dot_products;
	double physical_time;
	double time_upt_residuals;
	double time_towards_random;
	double time_towards_active;
	double time_cycle_weights_FW;

};

struct lasso_problem
{
	int l;//number of training points
	int input_dim ;//number of features
	double *y;
	struct data_node **x;
	data_node* x_space;
	int elements;
	int type;//data format:sparse or dense
	int representation;

	bool normalized;
	double mean_y;
	double inv_std_y;
	double* mean_predictors;
	double* inv_std_predictors;

};


struct lasso_parameters
{
	int exp_type;

	double reg_param;
	double reg_param_min;
	double reg_param_max;
	double reg_param_step;
	double eps_regularization_path;
	double n_steps_reg_path;

	bool computing_regularization_path;
	bool quick_stop_regularization_path;
	bool print_regularization_path;
	bool print_optimization_path;

	bool normalize;

	bool BORDER_WARM_START;
	bool ACTIVE_SET_HEURISTIC;
	double cache_size; 
	double eps;	

	int stopping_criterion;        
	bool safe_stopping;
	int nsamplings_safe_stopping; 
	int nsamplings_iterations; 
	int randomization_strategy;

	int training_algorithm;
	bool cooling;
	bool randomized;
	int sample_size;   
	int initialization_method;
	int max_iterations;

	int repetitions;
	int nfold;
	bool fixed_test;
	bool fixed_train;
	bool save_model;
	
	char* input_file_name;
	char* model_file_name;
	char* results_file_name;
	char* results_file_name_std;	
	char* path_file_name;
	char* summary_exp_file_name;
	int frecuency_messages;



};


struct lasso_model
{
	lasso_parameters* params;	
	data_node *weights;		
	double intercept;

	int n_predictors;

	double *obj; 
	unsigned long int *smo_it; //for solvers using SMO
	int *greedy_it;//for greedy algorithms
	double training_time;

	bool normalized;
	double mean_y;
	double inv_std_y;
	double* mean_predictors;
	double* inv_std_predictors;
	
};

template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
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

template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}


#endif /*LASSO_DEFINITIONS_H_*/
