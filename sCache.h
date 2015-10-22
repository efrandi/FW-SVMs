#ifndef SCACHE_H_
#define SCACHE_H_

#include <stdio.h>
#include <stdlib.h>
#include "SVM-commons.h"
using namespace svm_commons;

#define CACHE_DELTA 10

// Sparse caching for kernel evaluations
//
// Design Details: 
//
// The cache is basically designed using two structures:
//
// (1) An array of pointers (head) -one for each element of the dataset- pointing to an array of doubles
//     which contain the kernel product of this point and the elements of the coreset. 
//     For example head[idx] is a pointer to the array data, such that and data[j] is the 
//     kernel product between idx and the j-th element of the core-set 
//  
// (2) A double linked list of pointers ordered according to a usage counter pointing to the elements of head
//     This structure is oly used when it is missing space, that is when the logical or the physical limit
//     to store things in the cache has been reached and it is not possible to accomodate more kernel-evals  
// 
// Procedure when requesting for kernel-products between idx and the current center
//  
// (a) the pointer to the data corcerning the point is recovered data = head[idx] 
// (b) it is determined if the number of inner products contained (len) in data coincides with the number 
//     of inner products requested. If this is not true, it means that new points has entered the coreset
//     and new kernel-products will be computed. Otherwise, it means that the coreset has not change and no
//     new inner products needs to be calculated.
//     *** The kernel products are computed out of this class. The get_data function only returns the difference
//     between the number of inner products requested and the number of inner products in memory for this point.
//     If the coreset has change and for example the first point is other than before, bad inner products can be obtained  
// (c) it is determined if the number of new kernel-products to be computed requires the allocation of new memory
// (d) if the memory to be allocated does not fall in the logical limits of the cache: it enters in play the double 
//     list of points ordered according to usage: the list is transversed from least-used to more-used until to have
//     enough space cancelling the registers which usage-counter = 0. If the list is completely transversed without
//     arriving to the amount of space requested, all the counters are decremented and the process is repeated.
// (e) if after cancelling points we have that the size of the cache logically allows the storage of the new 
//     kernel-products but physically this was not possible we continue cancelling using the same procedure as in (d)
//     if even using this procedure it is not possible to accomodate the requested kernel-evals an alarm is rise
//     and the pointer returned by the get_data function is NULL. 
//     ** a sucesfull application of procedure (e) always result in the recomputation of all the inner products 
//     corcerning the point for which the inner products has been requested. Why???? 
//       
//


class sCache
{
public:
	sCache(const svm_parameter* param_, int num);
	virtual ~sCache();

	Qfloat* get_data(int idx, int basisNum, int& numRet);
	bool has_data(int idx) { return (head[idx].len > 0); } 

protected:		
	struct shead_t
	{
		shead_t *prev, *next;	// a cicular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
		int max_len;
		int refcount;	
	};

	shead_t *head;
	shead_t lru_head;
	void lru_delete(shead_t *h);
	void lru_insert(shead_t *h);

	int numData;
	int maxItem;
		
};

#endif /*SCACHE_H_*/