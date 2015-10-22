#ifndef SCACHE_H_
#define SCACHE_H_

#include <stdio.h>
#include <stdlib.h>
#include "LASSO_definitions.h" 

#define CACHE_DELTA 10

class sCache
{
public:
	sCache(const lasso_parameters* param_, int num);
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