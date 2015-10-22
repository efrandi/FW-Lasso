#ifndef LASSO_KERNEL_H_
#define LASSO_KERNEL_H_

#include "sCache.h" 
#include "LASSO_definitions.h" 

// Gram matrix for the LASSO
class LASSO_Q 
{
public:
	LASSO_Q(const lasso_problem* prob_, const lasso_parameters* param_, double *y_);

	~LASSO_Q()
	{
		delete kernelCache;	
		delete[] x;
		delete[] Sigmas;
		delete[] Norms;
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx);
	Qfloat kernel_eval(int idx1, int idx2);
	
	Qfloat* Sigmas;
	Qfloat* Norms;
	Qfloat maxAbsSigma;

	Qfloat getSigma(int idx);
	Qfloat getNorm(int idx);
	Qfloat getMaxSigma();

	double getY2Norm(){
		return y2norm;
	}
	double dot(int i, int j);

	double dotUncentered(const data_node *px, const data_node *py);
	double dotCentered(const data_node *px, const data_node *py, double centerx, double centery, double factorx, double factory, int max_idx);
    double distanceSq(const data_node *x, const data_node *y);

	unsigned long int real_kevals;
	unsigned long int requested_kevals;
	unsigned long int get_real_kevals();
	unsigned long int get_requested_kevals();
	void reset_real_kevals();
	void reset_requested_kevals();

	double getProductFeatureResiduals(int idx, double* products, double sum_products, double scale_products);
	double updateProducts(double* residuals, int idxVarMod, double oldAlpha, double newAlpha, double &sum_products, double &scale_products);
	double scaleProducts(double constant, double* products, double &sum_products, double &scale_products);
	double getScalePredictor(int idx);
	
private:

	const lasso_parameters* param;
	const lasso_problem* prob;	
	double y2norm;
	sCache *kernelCache;
	const data_node **x;
	
	double kernel_linear(int i, int j);

//protected:

//	double (LASSO_Q::*kernel_function)(int i, int j) const;
	

};



#endif /*LASSO_KERNEL_H_*/