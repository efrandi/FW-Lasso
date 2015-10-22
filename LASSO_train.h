#include "FW_based_LASSO.h"
#include "LASSO_definitions.h" 

class LASSO_train
{

public:


	LASSO_train(){
		FW_LASSO_solver=NULL;
	}

	lasso_model* train(lasso_problem* prob, lasso_parameters* params,lasso_stats* stats);
	double test(lasso_problem* prob, lasso_model* model);
	lasso_problem* readData(const char *filename);
	lasso_problem* dualRepresentation(lasso_problem* prob);
	lasso_problem* normalizeDual(lasso_problem* prob, lasso_model* model);
	void destroyProblem(lasso_problem* prob);
	void destroyModel(lasso_model* model);

	void compute_regularization_path(lasso_problem* prob, lasso_parameters* params,lasso_stats* stats);
	void compute_delta_range(lasso_problem* DUALproblem, lasso_parameters* params);

	void parse_command_line(lasso_parameters* params, int argc, char **argv, char *input_file_name, char *model_file_name, char* results_file_name, char* results_file_name_unstd, char* path_file_name);
	void exit_with_help();
	const char* getTextTrainingAlgorithm(int code);
	const char* getModalityAlgorithm(lasso_parameters* params);

	void printParams(FILE* file, lasso_parameters* params);
	void printStats(FILE* file, lasso_stats* stats);
	void showData(lasso_problem* prob);
	char* give_me_the_time();
	
private:

	FW_based_LASSO* FW_LASSO_solver;
	
	void count_pattern(FILE *fp, lasso_problem* prob, int &elements, int &type, int &dim);

};
