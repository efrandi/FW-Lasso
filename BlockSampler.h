#ifndef BLOCK_SAMPLER_H_
#define BLOCK_SAMPLER_H_

#include <math.h>
#include <cstdlib>
#include <ctime>
#include <stdexcept>


class BlockSampler
{

	private:

		int size_blocks;
		int size_last_block;
		int ndata;
		int nblocks;
		int* turns;
		int max_idx;
		int currentBlock;

	public:

	BlockSampler(int ndata_, int size_blocks_){
		
		if(ndata_%size_blocks_==0)
			nblocks = 1 + ((ndata_ - 1) / size_blocks_);
		else
			nblocks = ndata_/size_blocks_;

		if((nblocks<=0) || (size_blocks_<=0) || (ndata_<=0))
			throw std::invalid_argument("Something Wrong in BlockSampler Constructor\n" );
		

		ndata = ndata_;
		size_blocks = ndata/nblocks;
		size_last_block = ndata - (nblocks-1)*size_blocks;

		printf("NDATA=%d, NBLOCKS=%d, SIZE-BLOCKS=%d, SIZE-LAST=%d\n",ndata,nblocks,size_blocks,size_last_block);

		turns = new int[nblocks];
		for(int i=0; i < nblocks; i++)
			turns[i] = i;
		max_idx = nblocks-1;

		//std::srand(std::time(0));  
		

	}

	~BlockSampler(){
		delete [] turns;
	}

	int setRandomBlock(){

		int randIdx = std::rand()%(max_idx+1);
		int randBlock = turns[randIdx];
		turns[randIdx] = turns[max_idx];
		turns[max_idx] = randBlock;

		max_idx = max_idx - 1;

		if(max_idx<0)
			max_idx = nblocks-1;

		currentBlock = randBlock;
		return randBlock;
	}

	int getSizeCurrentBlock(){

		if(currentBlock == nblocks-1)
			return size_last_block;

		return size_blocks;

	}

	int getStartCurrentBlock(){

		return currentBlock*size_blocks;

	}

	int getEndCurrentBlock(){

		return (currentBlock*size_blocks)+getSizeCurrentBlock();

	}

	int getNBlocks(){
		return nblocks;
	}
	void reset(){
		max_idx = nblocks-1;
	}


};	


#endif /*BLOCK_SAMPLER_H_*/
