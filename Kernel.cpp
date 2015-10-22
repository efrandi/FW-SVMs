
#include "Kernel.h"

unsigned long int Kernel::real_kevals;
unsigned long int Kernel::requested_kevals = 0;

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//

unsigned long int Kernel::get_real_kevals(){
		return real_kevals;
	} 

unsigned long int Kernel::get_requested_kevals(){
		return requested_kevals;
	} 
	
void Kernel::reset_real_kevals(){
		real_kevals = 0;
	}

void Kernel::reset_requested_kevals(){
		requested_kevals = 0;
	}

double Kernel::kernel_linear(int i, int j) const
{
	real_kevals++;
	return dot(x[i],x[j]);
}
double Kernel::kernel_poly(int i, int j) const
{
	real_kevals++;
	return powi(gamma*dot(x[i],x[j])+coef0,degree);
}
double Kernel::kernel_rbf(int i, int j) const
{
	real_kevals++;
	return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
}
double Kernel::kernel_sigmoid(int i, int j) const
{
	real_kevals++;
	return tanh(gamma*dot(x[i],x[j])+coef0);
}
double Kernel::kernel_precomputed(int i, int j) const
{
	real_kevals++;
	return x[i][(int)(x[j][0].value)].value;
}
double Kernel::kernel_normalized_poly(int i, int j) const
{
	real_kevals++;
	return pow((gamma*dot(x[i],x[j])+coef0) / sqrt((gamma*x_square[i]+coef0)*(gamma*x_square[j])+coef0),degree);
}
double Kernel::kernel_exp(int i, int j) const
{
	real_kevals++;
	double temp=gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j]));
	return exp(-sqrt(temp));
}
double Kernel::kernel_inv_sqdist(int i, int j) const
{	
	real_kevals++;
	return 1.0/(gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j]))+1.0);
}
double Kernel::kernel_inv_dist(int i, int j) const
{
	real_kevals++;
	double temp=gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j]));
	return 1.0/(sqrt(temp)+1.0);
}


void Kernel::swap_index(int i, int j) const	// no so const...
{
	swap(x[i],x[j]);
	if(x_square) swap(x_square[i],x_square[j]);
}

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
        case EXP:
            kernel_function = &Kernel::kernel_exp;
			break;
        case NORMAL_POLY:
            kernel_function = &Kernel::kernel_normalized_poly;
			break;
		case INV_DIST:
			kernel_function = &Kernel::kernel_inv_dist;
			break;
		case INV_SQDIST:
			kernel_function = &Kernel::kernel_inv_sqdist;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF || kernel_type == NORMAL_POLY || kernel_type == EXP || kernel_type == INV_DIST || kernel_type == INV_SQDIST)
	{
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel()
{
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py)
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

double Kernel::distanceSq(const svm_node *x, const svm_node *y)
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


double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	real_kevals++;
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
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
						sum += (double)y->value * (double)y->value;
						++y;
					}
					else
					{
						sum += (double)x->value * (double)x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += (double)x->value * (double)x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += (double)y->value * (double)y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
        case EXP:
		{
			double dist = param.gamma*distanceSq(x,y);			
            return exp(-sqrt(dist));
		}
		case INV_DIST:
		{
			double dist = param.gamma*distanceSq(x,y);			
            return 1.0/(sqrt(dist)+1.0);
		}
		case INV_SQDIST:
		{			
            return 1.0/(param.gamma*distanceSq(x,y)+1.0);
		}
        case NORMAL_POLY:        
	        return pow((param.gamma*dot(x,y)+param.coef0)/sqrt((param.gamma*dot(x,x)+param.coef0)*(param.gamma*dot(y,y)+param.coef0)),param.degree);
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;	/* Unreachable */
	}
}


