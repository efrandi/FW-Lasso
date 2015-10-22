#include "LASSO_train.h"
#include <stdio.h>
#include <cmath> 
#include <ctype.h>
#include <stdexcept>
#include <string.h>
#include <math.h>
#include <limits>
#include <stdexcept>

void LASSO_train::compute_delta_range(lasso_problem* DUALproblem, lasso_parameters* params){
	
	printf("DELTA RANGE\n");

	FW_based_LASSO* solver_ = new FW_based_LASSO(DUALproblem,params);

	printf("SOLVER CREATED\n");

	int old_stopping = params->stopping_criterion;
	double old_eps = params->eps;
	double old_eps_rp = params->eps_regularization_path;
	bool old_crp = params->computing_regularization_path;
	bool old_prp = params->print_regularization_path;
	bool old_qsrp = params->quick_stop_regularization_path;

	//params->stopping_criterion = STOPPING_WITH_INF_NORM;
	params->computing_regularization_path = true;
	params->quick_stop_regularization_path = true;
	params->print_regularization_path = false;

	params->eps = 0.001;
	params->eps_regularization_path = 0.001;

	params->reg_param = 10;
	params->reg_param_min = 10;
	params->reg_param_max = 200;
	params->reg_param_step = 10;
	params->n_steps_reg_path = 100.0;

	lasso_model* temp_model = new lasso_model();
	
	temp_model->params = params;
	temp_model->n_predictors = DUALproblem->l;
	
	temp_model->weights = NULL;

	printf("SOLVING\n");
	solver_->Solve(params->eps, params->training_algorithm, params->cooling, params->randomized);
	printf("END SOLVED\n");
	solver_->ComputeLASSOSolution(temp_model->weights, 0.0);
	
	int i=0;
	double l1norm = 0.0;
	while(temp_model->weights[i].index!=-1){
		//printf("WEIGHT: IDX=%d, VAL=%g\n",temp_model->weights[i].index,temp_model->weights[i].value);
		l1norm += std::abs(temp_model->weights[i].value);
		i++;
	}

	params->n_steps_reg_path = 100.0;
	params->reg_param_max = l1norm;
	params->reg_param_min = params->reg_param_max/params->n_steps_reg_path;
	params->reg_param = params->reg_param_min;
	params->reg_param_step = params->reg_param_max/params->n_steps_reg_path;
	
	//printf("DELTA MIN=%.10f, DELTA MAX=%.10f, DELTA STEP=%.10f\n",params->reg_param_min,params->reg_param_max,params->reg_param_step);
	
	params->stopping_criterion = old_stopping;
	params->computing_regularization_path = old_crp;
	params->eps = old_eps;
	params->eps_regularization_path = old_eps_rp;
	params->print_regularization_path = old_prp;
	params->quick_stop_regularization_path = old_qsrp;

	printf("DELTA RANGE END\n");
	
	//delete[] solver_;

}

void LASSO_train::compute_regularization_path(lasso_problem* prob, lasso_parameters* params,lasso_stats* stats){


	params->quick_stop_regularization_path = false;
	
	lasso_problem* dual_problem = this->dualRepresentation(prob);
	lasso_model* model = new lasso_model();
	
	model->params = params;
	model->n_predictors = dual_problem->l;
	
	if(params->normalize){
		dual_problem = normalizeDual(dual_problem, model);
		dual_problem->normalized = true;
		model->intercept = 0.0;
	} else
		dual_problem->normalized = false;

	model->weights = NULL;

	//CHECK PARAMETERS
	if((params->reg_param_max < 0.0) || (params->reg_param_min < 0.0))
		compute_delta_range(dual_problem, params);
	if(params->reg_param_min < 0.0)
		params->reg_param_min = params->eps;
	
	params->reg_param = params->reg_param_min;

	FW_LASSO_solver = new FW_based_LASSO(dual_problem,params);

	clock_t time_start = clock ();
	
	//TRAINING ..
	FW_LASSO_solver->Solve(params->eps, params->training_algorithm, params->cooling, params->randomized);
	//END TRAININ

	clock_t time_end = clock ();
	
	FW_LASSO_solver->getStats(stats);
	stats->physical_time = (double)(time_end - time_start)/CLOCKS_PER_SEC;

	destroyProblem(dual_problem);

}


lasso_model* LASSO_train::train(lasso_problem* prob, lasso_parameters* params,lasso_stats* stats){

	printf("Training the model ... \n");
	printf("Original problem is of size %d and dimension %d ... \n",prob->l, prob->input_dim);

	//CHECK PARAMETERS
	if(params->reg_param < 0.0)
		throw std::invalid_argument( "ERROR. Regularization parameter cannot be negative\n" );

	lasso_problem* dual_problem = this->dualRepresentation(prob);
	lasso_model* model = new lasso_model();
	
	model->params = params;
	model->n_predictors = dual_problem->l;
	
	if(params->normalize){
		dual_problem = normalizeDual(dual_problem, model);
		dual_problem->normalized = true;
		model->intercept = 0.0;
	} else
		dual_problem->normalized = false;

	model->weights = NULL;

	//TRAINING ...
	if(FW_LASSO_solver==NULL)
		FW_LASSO_solver = new FW_based_LASSO(dual_problem,params);

	clock_t time_start = clock ();
			
	FW_LASSO_solver->Solve(params->eps, params->training_algorithm, params->cooling, params->randomized);
				
	FW_LASSO_solver->ComputeLASSOSolution(model->weights, 0.0);
	model->intercept = FW_LASSO_solver->getIntercept();
	clock_t time_end = clock ();
	
	FW_LASSO_solver->getStats(stats);
	stats->physical_time = (double)(time_end - time_start)/CLOCKS_PER_SEC;

	int i=0;
	while(model->weights[i].index!=-1){
		printf("WEIGHT: IDX=%d, VAL=%g\n",model->weights[i].index,model->weights[i].value);
		i++;
	}

	// END TRAINING ...

	destroyProblem(dual_problem);
	return model;
}

//receives a problem in primal representation and computes the RMS
double LASSO_train::test(lasso_problem* prob,lasso_model* model){
	
	double RMS = 0.0;
	printf("TESTING ... \n");
	for(int i=0; i<prob->l; i++){
		
		data_node* px = prob->x[i];
		double y = prob->y[i];
		data_node* weights = model->weights;
		double dot = 0.0;
		double this_weight = 0.0;

		while(px->index != -1 && weights->index != -1)
		{
			if(px->index == weights->index)
			{
				this_weight = (double)weights->value;
				if(model->normalized){
					this_weight = this_weight*model->inv_std_predictors[px->index-1]/model->inv_std_y;
				}
				dot += (double)px->value * this_weight;
				++px;
				++weights;
			}
			else
			{
				if(px->index > weights->index)
					++weights;
				else
					++px;
			}			
		}

		double prediction = dot + model->intercept;

		RMS	+= (y-prediction)*(y-prediction);
		
	}

	return RMS/(double)prob->l;
}

//WARNING: it overrides the previous problem
lasso_problem* LASSO_train::normalizeDual(lasso_problem* prob, lasso_model* model){
	
	model->mean_predictors = new double[prob->l];
	model->inv_std_predictors = new double[prob->l];
	model->normalized = true;

	prob->mean_predictors = model->mean_predictors;
	prob->inv_std_predictors = model->inv_std_predictors;
	prob->normalized = true;

	//in dual representation, each point is a feature, so l = primal dim
	for(int i=0; i<prob->l; i++){
		
		data_node* x = prob->x[i];
		double sum_x = 0.0;
		double sum_x2 = 0.0;
		while(x->index != -1){
			sum_x +=  x->value;
			sum_x2 +=  (x->value)*(x->value);
			++x;
		}

		double av_x=sum_x/(double)prob->input_dim;
		double av_x2=sum_x2/(double)prob->input_dim;
		double std_x = std::sqrt((double)prob->input_dim*(av_x2 - (av_x*av_x)));	
		
		prob->mean_predictors[i] = av_x;
		if(std_x > 0.0){
			prob->inv_std_predictors[i] = (1.0/std_x);
		} else {
			prob->inv_std_predictors[i] = 1.0/std::sqrt((double)prob->input_dim);
		}
		//printf("NORMALIZING .... Dimension %d, mean %f, std %f\n",i+1,av_x,std_x);

	}

	//there are so many y as primal l i.e. as dual dims
	double mean_y=0.0;
	double sum_y2=0.0;

	for(int i=0; i<prob->input_dim; i++){
			mean_y += prob->y[i];
			sum_y2 += prob->y[i]*prob->y[i];
	}

 	mean_y = mean_y/(double)prob->input_dim;
 	double std_y = std::sqrt((double)prob->input_dim*((sum_y2/(double)prob->input_dim) - (mean_y*mean_y)));
 	//double std_y = sqrt(prob->input_dim*((sum_y2/prob->input_dim) - (mean_y*mean_y)));
 	std_y=0.0;
 	prob->mean_y = mean_y;
 	
 	if(std_y>0){
 		prob->inv_std_y = 1.0/std_y;
 	} else {
 		prob->inv_std_y = 1.0/std::sqrt((double)prob->input_dim);
 	}
 	model->mean_y = prob->mean_y;
 	model->inv_std_y = prob->inv_std_y;

 return prob;

}

lasso_problem* LASSO_train::dualRepresentation(lasso_problem* prob){

	lasso_problem* new_prob = Malloc(struct lasso_problem,1);
	new_prob->x_space  = NULL;
	new_prob->l = prob->input_dim;
	new_prob->input_dim = prob->l;
	new_prob->y  = Malloc(double,prob->l);
	new_prob->x  = Malloc(struct data_node*, new_prob->l);
	new_prob->elements = prob->elements;
	new_prob->type = prob->type;//sparse or dense
	new_prob->representation = DUAL;
	
	int* counters_by_dim = Malloc(int,new_prob->l);
	int* elements_by_dim = Malloc(int,new_prob->l);
	
	printf("changing representation ...\n");
	printf("new l: %d new d: %d ...\n",new_prob->l, new_prob->input_dim);

	for(int i=0; i<new_prob->l; i++){
		elements_by_dim[i]=0;
		counters_by_dim[i]=0;
	}

	for(int i=0; i<prob->l; i++){
		new_prob->y[i] = prob->y[i];
		data_node* old_x = prob->x[i];
		while(old_x->index != -1){
			int feature_idx = old_x->index;
			elements_by_dim[feature_idx-1]= elements_by_dim[feature_idx-1]+1;	
			old_x++;	
		}
	}


	for(int i=0; i<new_prob->l; i++){
		new_prob->x[i]= Malloc(struct data_node, elements_by_dim[i]+1);
	}
	

	for(int i=0; i<prob->l; i++){
			if(i%500==0){
				printf("%i datos procesados ...\n",i);
			}
			data_node* old_x = prob->x[i];
			while(old_x->index != -1){
				int feature_idx = old_x->index;
				double feature_value = old_x->value;	
				data_node* new_x = new_prob->x[feature_idx-1];
				new_x[counters_by_dim[feature_idx-1]].index = i+1;
				new_x[counters_by_dim[feature_idx-1]].value = feature_value;
				counters_by_dim[feature_idx-1]+=1;
				old_x++;	
			}
	}

	for(int i=0; i<new_prob->l; i++){
		data_node* new_x = new_prob->x[i];
		new_x[counters_by_dim[i]].index=-1;
		new_x[counters_by_dim[i]].value=0.0;
	}

	free(counters_by_dim);
	free(elements_by_dim);
	return new_prob;
}

lasso_problem* LASSO_train::readData(const char *filename){


	FILE *fp = fopen(filename,"r");
	lasso_problem* prob = Malloc(struct lasso_problem,1);
	prob->representation = PRIMAL;
	
	prob->l = 0;

	int elements, i, j;
	int type, dim;
	int max_index;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	
	elements = 0;
	type     = 0; // sparse format
	dim      = 0;
    
    count_pattern(fp, prob, elements, type, dim);
    	
	prob->y  = Malloc(double,prob->l);
	prob->x  = Malloc(struct data_node *,prob->l);
	prob->x_space  = Malloc(struct data_node,elements+prob->l);
	prob->elements = elements;
	prob->type = type;

	if (!prob->y || !prob->x || !prob->x_space) {
		fprintf(stdout, "ERROR: not enough memory!\n");
		prob->l = 0;
		return NULL;
	}

	max_index = 0;
	j         = 0;

 	for(int i=0; i<prob->l; i++)
	{	
		if(i%5000==0){
			printf("%i datos cargados...\n",i);
		}
		double label;
		prob->x[i] = &prob->x_space[j];
		if (type == 0) // sparse format
		{
			fscanf(fp,"%lf",&label);
			prob->y[i] = label;
		}

		int elementsInRow = 0;
		while(1)
		{	
			int c;
			
			do {
				c = getc(fp);	
				if(c=='\n') break;
			} while(isspace(c));
			
			if((c=='\n') || (c==EOF)) break;
			
			ungetc(c,fp);

			if (type == 0) // sparse format
			{

				fscanf(fp,"%d:%lf",&(prob->x_space[j].index),&(prob->x_space[j].value));
				++j;
			}
			else if ((type == 1) && (elementsInRow < dim)) // dense format, read a feature
			{
				prob->x_space[j].index = elementsInRow+1;
				elementsInRow++;
				fscanf(fp, "%lf,", &(prob->x_space[j].value));
				++j;
			}
			else if ((type == 1) && (elementsInRow >= dim)) // dense format, read the label
			{
                fscanf(fp,"%lf",&label);
				prob->y[i] = label;
			}
		}	

		if(j>=1 && prob->x_space[j-1].index > max_index)
			max_index = prob->x_space[j-1].index;
		prob->x_space[j++].index = -1;
		
	}
	printf("problem is of dimension: %d\n",max_index);
	printf("prob->l is: %d\n",prob->l);
	prob->input_dim = max_index;

	return prob;
}

void LASSO_train::destroyProblem(lasso_problem* prob){

	if(prob->representation == PRIMAL){
		free(prob->x_space);
		printf("\n\nDESTROYING PRIMAL ... \n");
	} else {//DUAL
		for(int i=0; i<prob->l; i++)
			free(prob->x[i]);
		printf("\n\nDESTROYING DUAL ... \n");
	} 

	free(prob->y);
	free(prob->x);

}
	
void LASSO_train::destroyModel(lasso_model* mod){
	
	// if(mod->mean_predictors!=NULL){
	// 	delete [] mod->mean_predictors;
	// } 

	// if(mod->std_predictors!=NULL){
	// 	delete [] mod->std_predictors;
	// } 

}

//detects data format and counts number of patterns and dimensions
void LASSO_train::count_pattern(FILE *fp, lasso_problem* prob, int &elements, int &type, int &dim)
{
    int c;
    do
    {
	    c = fgetc(fp);
	    switch(c)
	    {
		    case '\n':
		    	prob->l += 1;
		    	if ((type == 1) && (dim == 0)) // dense format
			        dim = elements;				    
			    break;

		    case ':':
			    ++elements;
			    break;

		    case ',':
			    ++elements;
			    type = 1;
			    break;

			case EOF:
		    	prob->l += 1;
		    	if ((type == 1) && (dim == 0)) // dense format
			        dim = elements;				    
			    break;

		    default:
			    ;
	    }
    } while  (c != EOF);
    printf(">>> COUNT PATTERN: l=%d, elements=%d\n",prob->l,elements);
    rewind(fp);
}

void LASSO_train::showData(lasso_problem* prob){

 FILE * pFile;
 pFile = fopen ("testData.txt","w");

	for(int i=0; i<prob->l; i++){
		data_node* old_x = prob->x[i];
		printf("\ndato %d\n",i);
		while(old_x->index != -1){
			int feature_idx = old_x->index;
			double feature_value = old_x->value;	
			fprintf(pFile, "%d:%f, ",feature_idx,feature_value);
			printf("%d:%f, ",feature_idx,feature_value);
			old_x++;
		}
		fprintf(pFile,"\n");
	}

}

void LASSO_train::parse_command_line(lasso_parameters* params, int argc, char **argv, char *input_file_name, char *model_file_name, char* results_file_name, char* results_file_name_std, char* path_file_name)
{
	int i;

	params->exp_type = EXP_SINGLE_TRAIN;
	params->cache_size   = 2000;	
	params->eps          = 1e-4;
	params->stopping_criterion = STOPPING_WITH_INF_NORM;
	params->sample_size  = 139;
	params->training_algorithm = FW;
	params->cooling = false;
	params->randomized = true;
	params->initialization_method = ZERO;
    params->save_model = false;
	params->max_iterations =  std::numeric_limits<int>::max();
	params->normalize = true;

	params->reg_param = -1.0;
	params->reg_param_min = -1.0;
	params->reg_param_max = -1.0;
	params->reg_param_step = -1.0;
	params->computing_regularization_path = false;
	params->quick_stop_regularization_path = false;
	params->print_regularization_path = false;
	params->print_optimization_path = false;

	params->safe_stopping = false;
	params->nsamplings_safe_stopping = 1;
	params->nsamplings_iterations = 1;
	params->n_steps_reg_path = 100;
	params->BORDER_WARM_START = true;
	params->ACTIVE_SET_HEURISTIC = true;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();

		switch(argv[i-1][1])
		{
			case 'E':
				params->exp_type = atoi(argv[i]);
				if(params->exp_type == EXP_REGULARIZATION_PATH)
					params->computing_regularization_path = true;
					params->print_regularization_path = true;
				break;
			case 'I':
				if(argv[i-1][2]=='M')
					params->initialization_method =  ZERO;
				if(argv[i-1][2]=='L')
					params->max_iterations = atoi(argv[i]);
				break;
			case 'N':
				if(argv[i-1][2]=='F'){
					params->nfold = atoi(argv[i]);
				} else if (argv[i-1][2]=='S'){
					params->repetitions = atoi(argv[i]);
				} else if (argv[i-1][2]=='M'){
					if(atof(argv[i])>0.0)
						params->normalize = true;
					else
						params->normalize = false;
				} 
				break;	
			case 'M':
				if(argv[i-1][2]=='A'){
					params->training_algorithm = atoi(argv[i]);
				}
				break;
			case 'B':
				if(argv[i-1][2]=='W'){
					if(atoi(argv[i])>0){
						params->BORDER_WARM_START  = true;
					} else {
						params->BORDER_WARM_START  = false;
					}
				}
				break;
			case 'A':
				if(argv[i-1][2]=='H'){
					if(atoi(argv[i])>0){
						params->ACTIVE_SET_HEURISTIC  = true;
					} else {
						params->ACTIVE_SET_HEURISTIC  = false;
					}
				}
				break;
			case 'F':
				if(argv[i-1][2]=='T'){
					params->fixed_test = true;
				}
				else if(argv[i-1][2]=='S'){
					params->fixed_train = true;
				} else if(argv[i-1][2]=='M'){
					params->frecuency_messages = atoi(argv[i]);
					printf(">> FM:%d\n",params->frecuency_messages);
					i=i+1;
				}
				i=i-1;
				break;
			case 'C':
				if (argv[i-1][2]=='O'){
					if(atoi(argv[i]) == 0)
						params->cooling = false;
					else 	
						params->cooling = true;
				}
				break;
			case 'm':
				params->cache_size = atof(argv[i]);
				break;
			case 'e':
				params->eps = atof(argv[i]);
				break;
			case 'R':
				if (argv[i-1][2]=='S'){
					if(atoi(argv[i]) == 0)
						params->randomized = false;
					else 	
						params->randomized = true;
				} else if(argv[i-1][2]=='P'){
						params->reg_param = atof(argv[i]);
				} else if(argv[i-1][2]=='M'){
						params->reg_param_min = atof(argv[i]);
				} else if(argv[i-1][2]=='X'){
						params->reg_param_max = atof(argv[i]);
				} else if(argv[i-1][2]=='D'){
						params->reg_param_step = atof(argv[i]);
				}
				break;	
			case 'S':
				if (argv[i-1][2]=='C'){
					if(atoi(argv[i]) == 1){
						params->stopping_criterion = STOPPING_WITH_INF_NORM;
					} else { 
						if (atoi(argv[i]) == 2)	
							params->stopping_criterion = STOPPING_WITH_DUAL_GAP;
						else
							params->stopping_criterion = STOPPING_WITH_OBJECTIVE;
					}	 
				} else if(argv[i-1][2]=='S'){
					//safe stopping
					params->safe_stopping = true;
					params->nsamplings_safe_stopping = atoi(argv[i]);
				} else if(argv[i-1][2]=='T'){
					if(atoi(argv[i]) == 0){
						params->randomization_strategy = UNIFORM;
					} else { 
						params->randomization_strategy = BLOCKS;
					}
				}  

				break;				 
			case 'a':
				params->sample_size = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1) 
	{
		strcpy(model_file_name,argv[i+1]);
		params->save_model = true;
	}

	char stamp[64];
	sprintf(stamp,"%s",give_me_the_time());

	sprintf(results_file_name,"%s.LASSO.RESULTS.%s.%.1E.%s.%d.txt",input_file_name,getTextTrainingAlgorithm(params->training_algorithm),params->eps,getModalityAlgorithm(params),stamp);
	sprintf(results_file_name_std,"%s.LASSO.STD.RESULTS.%s.%.1E.%s.%d.txt",input_file_name,getTextTrainingAlgorithm(params->training_algorithm),params->eps,getModalityAlgorithm(params),stamp);
	printf(">>> %s\n",results_file_name_std);
	sprintf(path_file_name,"%s.LASSO.ALG-PATH.%s.%.1E.%s.%d.txt",input_file_name,getTextTrainingAlgorithm(params->training_algorithm),params->eps,getModalityAlgorithm(params),stamp);
	params->summary_exp_file_name = new char[8192];
	sprintf(params->summary_exp_file_name,"%s.LASSO.SUMMARY.%s.%.1E.%s.%d.txt",input_file_name,getTextTrainingAlgorithm(params->training_algorithm),params->eps,getModalityAlgorithm(params),stamp);
	
	params->input_file_name = input_file_name;
	params->model_file_name = model_file_name;
	params->path_file_name = path_file_name;
	params->results_file_name = results_file_name;
	params->results_file_name_std = results_file_name_std;


}

void LASSO_train::printParams(FILE* file, lasso_parameters* params){
 
	if(params->stopping_criterion == STOPPING_WITH_DUAL_GAP)
		fprintf(file,"Stopping Criterion: STOPPING_WITH_DUAL_GAP\n");
	else if (params->stopping_criterion == STOPPING_WITH_INF_NORM)
		fprintf(file,"Stopping Criterion: STOPPING_WITH_INF_NORM\n");
	else if (params->stopping_criterion == STOPPING_WITH_OBJECTIVE)
		fprintf(file,"Stopping Criterion: STOPPING_WITH_OBJECTIVE\n");
	else
		fprintf(file,"Stopping Criterion: UNKNOWN\n");

	if(params->BORDER_WARM_START)
		fprintf(file,"Warm Start: BORDER (SCALING PREVIOUS SOLUTION)\n");
	else
		fprintf(file,"Warm Start: PREVIOUS SOLUTION (i.e. NOTHING SPECIAL)\n");

	if(params->ACTIVE_SET_HEURISTIC)
		fprintf(file,"Active Set Exploration in Toward Search: YES\n");
	else
		fprintf(file,"Active Set Exploration in Toward Search: NO\n");

	fprintf(file,"EPS Stopping: %g\n",params->eps);
	fprintf(file,"Cache Size (MB): %f\n",params->cache_size);

	fprintf(file,"Training Method: %s\n",getTextTrainingAlgorithm(params->training_algorithm));
	
	if(params->exp_type == EXP_SINGLE_TRAIN){
		fprintf(file,"Experiment Type: Single Trainining ...\n");
		fprintf(file,"Regularization Parameter (delta): %g\n",params->reg_param);
	} else if (params->exp_type == EXP_REGULARIZATION_PATH){
		fprintf(file,"Experiment Type: Regularization Path ...\n");
		fprintf(file,"MIN - Regularization Parameter (delta_min): %g\n",params->reg_param_min);
		fprintf(file,"MAX - Regularization Parameter (delta_max): %g\n",params->reg_param_max);
		fprintf(file,"STEP - Regularization Parameter (delta_step): %g\n",params->reg_param_step);
	}

	if(params->cooling)
		fprintf(file,"Cooling: YES\n");
	else
		fprintf(file,"Cooling: NO\n");

	if(params->randomized){
		
			fprintf(file, "Randomization: YES (%d points)\n", params->sample_size);
			
			if(params->randomization_strategy == UNIFORM){
				fprintf(file, "Randomization Strategy: SIMPLE RANDOM SAMPLE\n");
			} else if (params->randomization_strategy == BLOCKS) {
				fprintf(file, "Randomization Strategy: BLOCKS\n");
			}

			if(params->safe_stopping)
				fprintf(file, "Safe Stopping: YES (-SS %d)\n", params->nsamplings_safe_stopping);
			else
				fprintf(file, "Safe Stopping: NO\n");
	} else {
			fprintf(file, "Randomization: NO\n");
	}


	if(params->normalize)
		fprintf(file,"Normalize: YES\n");
	else
		fprintf(file,"Normalize: NO\n");

	if(params->max_iterations < std::numeric_limits<int>::max())
		fprintf(file,"Max Iterations: %d\n",params->max_iterations);

}

void LASSO_train::printStats(FILE* file, lasso_stats* stats){
 
 	fprintf(file,"** Performance ** \n");
	fprintf(file,"Iterations: %g\n",stats->n_iterations);
	fprintf(file,"Performed Dot Products: %g\n",stats->n_performed_dot_products);
	fprintf(file,"Requested Dot Products: %g\n",stats->n_requested_dot_products);
	fprintf(file,"Running Time (Secs): %g\n",stats->physical_time);
	fprintf(file,"Time FW Weights (Secs): %g\n",stats->time_cycle_weights_FW);
	fprintf(file,"Time Towards Random Part (Secs): %g\n",stats->time_towards_random);
	fprintf(file,"Time Towards Active Part (Secs): %g\n",stats->time_towards_active);
	fprintf(file,"Time Update Residuals (Secs): %g\n",stats->time_upt_residuals);

}

const char* LASSO_train::getModalityAlgorithm(lasso_parameters* params){
	if(params->randomized)
		return "RANDOMIZED";
	else
		return "DETERMINISTIC";
}

const char* LASSO_train::getTextTrainingAlgorithm(int code){
	switch(code)
		{
			case FW:
				return "FW";
				break;
			case MODFW:
				return "MFW";
				break;
			case PARTAN:
				return "PARTAN-FW";
				break;
			case SWAP:
				return "SWAP-FW";
				break;
		}
	return "OTHER (CHECK)";
}

char* LASSO_train::give_me_the_time(){

  time_t rawtime = time(0);
  tm *now = localtime(&rawtime);
  char timestamp[32];
  if(rawtime != -1){
     strftime(timestamp,sizeof(timestamp),"%Y-%m-%d-%Hhrs-%Mmins-%Ssecs",now);
  }
  return(timestamp);
}

void LASSO_train::exit_with_help()
{
	printf(
	"Usage: LASSO-train [options] training_set_file [model_file] \n"
	"VERSION 21.11.14. Options:\n"
	"-E : experiment-type (default 0)\n"
	"   0 -- simple train and test\n"
	"   1 -- find regularization path\n"
	"-MA : Train Algorithm: (0) FULLY-CORRECTIVE FW, (1)STANDARD FW (default),\n" 
	"                       (2) MFW (3) PARTAN (4) SWAP \n"
	"-CO : COOLING? (0) NO - default (1) yes\n"   
	"-RS : RANDOMIZED?: (0) no (1) yes - default\n" 
	"-RP value:: Regularization Parameter\n" 
	"-RM value:: Min value - Regularization Parameter (for EXP_TYPE = 1)\n" 
	"-RX value:: Max value - Regularization Parameter (for EXP_TYPE = 1)\n" 
	"-RD value:: Step value for chaning the Regularization Parameter (for EXP_TYPE = 1)\n" 	
	"-NM value: NORMALIZE PREDICTORS?: if value>0 yes (default) otherwise no\n" 
	"-IM : Initialization Method: (0) MEB of random sample (default)\n" 
	"-IL : Maximum number of Iterations\n" 
	"-SC : Stopping Criterion: (0) INFINITE NORM OF DIFFERENCE BETWEEN ITERATES (default), (1) DUALITY GAP, (2) IMPROVEMENT IN THE OBJECTIVE\n" 
	"-e value: epsilon tolerance of termination criterion\n"
	"-NS n : number of differents trainings (requires differentes files train/test) is set to n\n"
	"-SM: save the model(s) after training\n"
	"-m cachesize: set cache memory size in MB (default 200)\n"
	"-a size: sample size for probabilistic sampling (default 60)\n"
	);
	exit(1);
}

int main(int argc, char **argv){

	LASSO_train* fw = new LASSO_train();
	lasso_problem* problem = new lasso_problem();
	lasso_parameters* params = new lasso_parameters();
	char input_file_name[4096]; 
	char model_file_name[4096]; 
	char results_file_name[8192];
	char results_file_name_std[8192];  
	char path_file_name[8192]; 
	lasso_stats* stats = new lasso_stats();

	fw->parse_command_line(params,argc,argv,input_file_name,model_file_name,results_file_name,results_file_name_std,path_file_name);
	problem = fw->readData(input_file_name);
	FILE* summary_exp = fopen(params->summary_exp_file_name,"w");

	if(params->exp_type == EXP_SINGLE_TRAIN){

		fw->printParams(stdout,params);
		lasso_model* trained_model = fw->train(problem,params,stats);
		double error=fw->test(problem,trained_model);
		fw->printParams(summary_exp,params);
		fw->printStats(summary_exp,stats);

 		printf("\n\nRMS: %f\n",error);
 		fw->destroyProblem(problem);
	}
	
	if(params->exp_type == EXP_REGULARIZATION_PATH){
		fw->printParams(stdout,params);
		fw->compute_regularization_path(problem,params,stats);
		fw->printParams(summary_exp,params);
		fw->printStats(summary_exp,stats);
	}	
	
	return 1;

}


