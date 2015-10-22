#ifndef CACHE_H_
#define CACHE_H_

#include "SVM-commons.h"
using namespace svm_commons;

// Cache.h: caching for kernel evaluations
// Class Cache: 
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	//int get_data(const int index, Qfloat **data, int len);
	Qfloat* get_data(int idx,  int len, int& numRet);
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	// future_option
private:
	int l;
	int size;
	struct head_t
	{
		head_t *prev, *next;	// a cicular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};


#endif /*CACHE_H_*/
