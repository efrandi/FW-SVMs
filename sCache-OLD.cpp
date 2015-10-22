
#include "sCache.h"

//
//
// Sparse caching for kernel evaluations
//
// l is the number of total data items
// size is the cache size limit in bytes
//
 

sCache::sCache(const svm_parameter* param_, int num)
{
	//init cache and usage
	numData  = num;
	head     = (shead_t *) calloc(numData,sizeof(shead_t));	// initialized to 0	
	lru_head.next = lru_head.prev = &lru_head;	

	//compute maximal space
	maxItem = max((int)((max(param_->cache_size,4.0)*(1<<20))/sizeof(Qfloat)),numData*2); // cache must be large enough for two columns		
	printf("%d\n",(int)maxItem);
}

sCache::~sCache() 
{ 		
	for(shead_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);		
	free(head);
}

void sCache::lru_delete(shead_t *h)
{
	//delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void sCache::lru_insert(shead_t *h)
{
	//insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

Qfloat* sCache::get_data(int idx,  int len, int& numRet)
{	
	shead_t *h = &head[idx];
	h->refcount ++;

	if(h->len > 0) lru_delete(h);
	if(len > h->max_len)
	{
		int more   = len + CACHE_DELTA - h->max_len;
		h->max_len = h->max_len + more;
		
		// free old space
		bool deleted = false;
		while(maxItem < more) //requested space exceeds logical limit
		{
			shead_t *old = lru_head.next;
			lru_delete(old);
		
				free(old->data);
				old->data    = NULL;
				maxItem     += old->len;				
				old->len     = 0;	
				old->max_len = 0;
				deleted = true;
		}
	
		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*h->max_len);
		if (h->data == NULL)//requested space cannot be provided by physical limit
		{	
			while(h->data == NULL && lru_head.next != &lru_head)
			{
				shead_t *old = lru_head.next;
				lru_delete(old);

					free(old->data);
					old->data    = NULL;
					maxItem     += old->len;				
					old->len     = 0;	
					old->max_len = 0;
					h->data = (Qfloat *)calloc(h->max_len,sizeof(Qfloat));
	
			}
			h->len  = 0;
			if (h->data == NULL)
			{
				printf ("sCache cannot allocate memory!\n");
				return NULL;
			}
		}
		maxItem -= more;		
	}

	lru_insert(h);
	numRet = h->len;
	h->len = len;
	return h->data;
}


//---------------------------------------------------------------------------------------------------------------------
