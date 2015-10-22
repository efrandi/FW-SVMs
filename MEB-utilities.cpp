#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#ifndef _MSC_VER
#include <sys/times.h>
#include <unistd.h>
#else
#include <time.h>
#endif

//----------------------------------------------------------------------------------
// Global helper functions

int delta(int x, int y)
{
	return ( (x==y)? (1) : (0) );
}

double dotProduct( const double* x1, const double* x2, int length )
{
	double sum = 0;
	for (int i=0; i<length; i++)
		sum += ( x1[i] * x2[i] );

	return sum;
}




