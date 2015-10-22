#include "LASSO_kernel.h" 
#include <limits>
#include <cmath> 


LASSO_Q::LASSO_Q(const lasso_problem* prob_, const lasso_parameters* param_, double *y_) 
	{
		int i;
		prob  = prob_;
		param = param_;
		
		real_kevals = 0;
		requested_kevals = 0;

		kernelCache = new sCache(param_, prob->l);
		//kernel_function = &LASSO_Q::kernel_linear;
		clone(x,prob_->x,prob_->l);

		Sigmas = new Qfloat[prob_->l];
		Norms = new Qfloat[prob_->l];

		data_node* temp_yy = Malloc(struct data_node, prob_->input_dim+1);
		data_node* temp_y = temp_yy;
		for(int i=0; i < prob_->input_dim; i++){
			temp_y->index = i+1;
			temp_y->value = y_[i];	
			temp_y++;
		}

		temp_y->index = -1;//end of the sparse representation of y
		temp_y->value = 0.0;//end of the sparse representation of y

		maxAbsSigma = -INFINITY;

		for(int i=0; i < prob_->l; i++){
			if((param->normalize) && (prob->normalized))
				Sigmas[i] = dotCentered(temp_yy,x[i],prob->mean_y,prob->mean_predictors[i],prob->inv_std_y,prob->inv_std_predictors[i],prob->input_dim);
			else
				Sigmas[i] = dotUncentered(temp_yy,x[i]);
			
			if(std::abs(Sigmas[i])>maxAbsSigma)
				maxAbsSigma = std::abs(Sigmas[i]);
		}

		if((param->normalize) && (prob->normalized))
			y2norm = dotCentered(temp_yy,temp_yy,prob->mean_y,prob->mean_y,prob->inv_std_y,prob->inv_std_y,prob->input_dim);
		else
			y2norm = dotUncentered(temp_yy,temp_yy);

		free(temp_yy);

		for(int i=0; i < prob_->l; i++){
			Norms[i] = dot(i,i);
			//printf("Y2NORM: %g, NORM %d: %g, SIGMA=%g\n",y2norm,i,Norms[i],Sigmas[i]);
		}
		//printf("MAXSIGMA in ABSOLUTE VALUE=%g\n",maxAbsSigma);	
	}

Qfloat LASSO_Q::getSigma(int idx){
		
		return Sigmas[idx];

}


Qfloat LASSO_Q::getMaxSigma(){
		
		return maxAbsSigma;

}

Qfloat LASSO_Q::getNorm(int idx){
		//if((param->normalize) && (prob->normalized))
		//	return 1.0;
		return Norms[idx];

}

Qfloat LASSO_Q::kernel_eval(int idx1, int idx2){
		
		requested_kevals++;
		Qfloat Q;
		Q = (Qfloat)((this->kernel_linear)(idx1, idx2));			
		
		return Q;
	}

Qfloat* LASSO_Q::get_Q(int idx, int basisNum, int* basisIdx)
	{	
		
		requested_kevals += basisNum;
		
		int numRet;
		Qfloat *Q = kernelCache->get_data(idx, basisNum, numRet);
		if (Q != NULL)
		{	

			for(int i = numRet; i < basisNum; i++)
			{
				int idx2 = basisIdx[i];
				Q[i] = (Qfloat)((this->kernel_linear)(idx, idx2));	
					
		
			}						
		}
		return Q;
	}

double LASSO_Q::kernel_linear(int i, int j)
{
	real_kevals++;
	return dot(i,j);
}

double LASSO_Q::dot(int i, int j)
{
	
	if((param->normalize) && (prob->normalized))
		return dotCentered(x[i],x[j],prob->mean_predictors[i],prob->mean_predictors[j],prob->inv_std_predictors[i],prob->inv_std_predictors[j],prob->input_dim);
	else
		return dotUncentered(x[i],x[j]);
}

double LASSO_Q::scaleProducts(double constant, double* products, double &sum_products, double &scale_products){
	scale_products *= constant;
}

double LASSO_Q::updateProducts(double* products, int idxVarMod, double lambda, double delta_tilde, double &sum_products, double &scale_products){

	const data_node *px = x[idxVarMod];
	
	real_kevals++;
	requested_kevals++;


	double scalex, centerx;

	if((param->normalize) && (prob->normalized)){
		scalex = prob->inv_std_predictors[idxVarMod];
		centerx = prob->mean_predictors[idxVarMod];
	} else {
		scalex = 1.0;
		centerx = 0.0;
	}


	scale_products *= (1.0-lambda);

	if(std::abs(lambda-1.0)<TAU){
		scale_products = 1.0;
		for(int k=0; k < prob->input_dim; k++)
			products[k] = 0.0;
		sum_products = 0.0;	
	}

	while(px->index != -1){
		sum_products-= products[px->index-1];
		products[px->index-1] += (scalex*((double)px->value)*delta_tilde*lambda)/(scale_products);
		sum_products += products[px->index-1];
		px++;
	}
	
	
	return 1;
}

double LASSO_Q::getScalePredictor(int idx){
	return prob->inv_std_predictors[idx];
}

double LASSO_Q::getProductFeatureResiduals(int idx, double* products, double sum_products, double scale_products){

	const data_node *px = x[idx];
	double product = 0.0;
	double scalex, centerx;

	real_kevals++;
	requested_kevals++;

	if((param->normalize) && (prob->normalized)){
		scalex = prob->inv_std_predictors[idx];
		centerx = prob->mean_predictors[idx];
	} else {
		scalex = 1.0;
		centerx = 0.0;
	}
	

	while(px->index != -1){
		product += scale_products*scalex*((double)px->value)*products[px->index-1];
		px++;
	}

	if((param->normalize) && (prob->normalized)){
		product -= scale_products*scalex*centerx*sum_products;
	}

	return getSigma(idx) - product;

}

double LASSO_Q::dotCentered(const data_node *px, const data_node *py, double centerx, double centery, double factorx, double factory, int max_idx)
{
	double sum = 0.0;
	
	int previous_idx = 0;
	const data_node *pmin;
	double scale = factorx*factory;

	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += (double)px->value * (double)py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}

	sum = scale*(sum-(prob->input_dim*centerx*centery));
	return sum;
}

// double LASSO_Q::dotCentered(const data_node *px, const data_node *py, double centerx, double centery, double factorx, double factory, int max_idx)
// {
// 	double sum = 0.0;
	
// 	int previous_idx = 0;
// 	const data_node *pmin;
// 	double scale = factorx*factory;

// 	while(px->index != -1 && py->index != -1)
// 	{
// 		pmin = (px->index < py->index) ? px : py;  
		
// 		//for(int i=previous_idx+1;i<pmin->index;i++){
// 		//	sum += scale*centerx*centery;
// 		//}
// 		sum += ((double)pmin->index-(double)previous_idx-1.0)*scale*centerx*centery;
// 		previous_idx = pmin->index;	

// 		if(px->index == py->index)
// 		{
// 			sum += scale*((double)px->value - centerx)*((double)py->value - centery);
// 			++px;
// 			++py;
// 		}
// 		else
// 		{	
// 			if(pmin==px)
// 				sum += scale*((double)pmin->value - centerx)*(0.0 - centery);
// 			else 
// 				sum += scale*((double)pmin->value - centery)*(0.0 - centerx);

// 			if(px->index > py->index)
// 				++py;
// 			else
// 				++px;
// 		}
		
// 	}

// 	if(px->index != -1){
		
// 		while(px->index != -1){
// 			//for(int i=previous_idx+1;i<px->index;i++)
// 			//	sum += scale*centerx*centery;
// 			sum += ((double)px->index-(double)previous_idx-1.0)*scale*centerx*centery;
// 			sum += scale*((double)px->value - centerx)*(0.0 - centery);
// 			previous_idx = px->index;	
// 			++px;
// 		}
// 	}

// 	if(py->index != -1){
		
// 		while(py->index != -1){
// 			//for(int i=previous_idx+1;i<py->index;i++)
// 			//	sum += scale*centerx*centery;
// 			sum += ((double)py->index - (double)previous_idx - 1.0)*scale*centerx*centery;
// 			sum += scale*((double)py->value - centery)*(0.0 - centerx);
// 			previous_idx = py->index;	
// 			++py;
// 		}
// 	}

// 	//for(int i=previous_idx+1;i<=max_idx;i++){	
		
// 	//	sum += scale*centerx*centery;
// 	//}
// 	sum += ((double)max_idx-(double)previous_idx)*scale*centerx*centery;
// 	return sum;
// }


double LASSO_Q::dotUncentered(const data_node *px, const data_node *py)
{
	double sum = 0;
	
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += (double)px->value * (double)py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}


unsigned long int LASSO_Q::get_real_kevals(){
		return real_kevals;
	} 

unsigned long int LASSO_Q::get_requested_kevals(){
		return requested_kevals;
	} 
	
void LASSO_Q::reset_real_kevals(){
		real_kevals = 0;
	}

void LASSO_Q::reset_requested_kevals(){
		requested_kevals = 0;
	}

double LASSO_Q::distanceSq(const data_node *x, const data_node *y)
{
	double sum = 0.0;
	
    while(x->index != -1 && y->index !=-1)
	{
		if(x->index == y->index)
		{
			double d = (double)x->value - (double)y->value;
			sum += d*d;
			
			++x;
			++y;
		}
		else
		{
			if(x->index > y->index)
			{
				sum += ((double)y->value) * (double)y->value;
				++y;
			}
			else
			{
				sum += ((double)x->value) * (double)x->value;
				++x;
			}
		}
	}

	while(x->index != -1)
	{
		sum += ((double)x->value) * (double)x->value;
		++x;
	}

	while(y->index != -1)
	{
		sum += ((double)y->value) * (double)y->value;
		++y;
	}
	
	return (double)sum;
}

// int main(){

// 	printf("Hi");
// 	return 0;
// }
