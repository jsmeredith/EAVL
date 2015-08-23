#ifndef EAVL_RAY_H
#define EAVL_RAY_H
#include "eavlArray.h"
/*

	Ray data in structure of array format

*/
class eavlRay
{

	public:
	    eavlFloatArray*		rayOriginX;   //ray Origins
	    eavlFloatArray*		rayOriginY;
	    eavlFloatArray*		rayOriginZ;
	    eavlFloatArray*		rayDirX;	  // ray directions
	    eavlFloatArray*		rayDirY;	 
	    eavlFloatArray*		rayDirZ;
	    eavlFloatArray*		alpha;	      //barycentric coordinates
	    eavlFloatArray*		beta;
	    eavlFloatArray* 	distance;     //distance to hit
	    eavlIntArray*		hitIdx;	      //index of primitive
	    size_t				numRays;	  //numbers of rays in this buffer
	    
	    eavlRay(const size_t &size)
	    {
	    	numRays = size;

	    	rayOriginX = new eavlFloatArray("",1,numRays);
	    	rayOriginY = new eavlFloatArray("",1,numRays);
	    	rayOriginZ = new eavlFloatArray("",1,numRays);

	    	rayDirX = new eavlFloatArray("",1,numRays);
	    	rayDirY = new eavlFloatArray("",1,numRays);
	    	rayDirZ = new eavlFloatArray("",1,numRays);

	    	hitIdx   = new eavlIntArray("",1,numRays);
	    	alpha    = new eavlFloatArray("",1,numRays);
	    	beta     = new eavlFloatArray("",1,numRays);
	    	distance = new eavlFloatArray("",1,numRays);
	    
	    }

	    ~eavlRay()
	    {
	    	delete rayOriginX;
	    	delete rayOriginY;
	    	delete rayOriginZ;
	    	delete rayDirX;
	    	delete rayDirY;
	    	delete rayDirZ;
	    	delete distance;
	    	delete hitIdx;
	    	delete alpha;
	    	delete beta;
	    }

	    virtual void resize(const size_t &newSize)
	    {
	    	if(newSize == numRays) return; //nothing to do
	    	 
	    	numRays = newSize;
	    	delete rayOriginX;
	    	delete rayOriginY;
	    	delete rayOriginZ;
	    	delete rayDirX;
	    	delete rayDirY;
	    	delete rayDirZ;
	    	delete distance;
	    	delete hitIdx;
	    	delete alpha;
	    	delete beta;

	    	rayOriginX = new eavlFloatArray("",1,numRays);
	    	rayOriginY = new eavlFloatArray("",1,numRays);
	    	rayOriginZ = new eavlFloatArray("",1,numRays);

	    	rayDirX = new eavlFloatArray("",1,numRays);
	    	rayDirY = new eavlFloatArray("",1,numRays);
	    	rayDirZ = new eavlFloatArray("",1,numRays);

	    	hitIdx   = new eavlIntArray("",1,numRays);
	    	alpha    = new eavlFloatArray("",1,numRays);
	    	beta     = new eavlFloatArray("",1,numRays);
	    	distance = new eavlFloatArray("",1,numRays);
			

	    }
};

class eavlFullRay : public eavlRay
{
  public:
  	//eavlFloatArray*     weight;	      		//contribution
  	eavlFloatArray*     normalX;	      	//intersection normal
  	eavlFloatArray*     normalY;	     
  	eavlFloatArray*     normalZ;
  	eavlFloatArray*     intersectionX;
  	eavlFloatArray*     intersectionY;
  	eavlFloatArray*     intersectionZ;
  	eavlFloatArray*		scalar;	     
  	//eavlByteArray*      hitType;      		//type of geometry hit
  	eavlFullRay(const size_t &size) : eavlRay(size) 
  	{
  		
  		normalX = new eavlFloatArray("",1,numRays);
  		normalY = new eavlFloatArray("",1,numRays);
  		normalZ = new eavlFloatArray("",1,numRays);
  		scalar  = new eavlFloatArray("",1,numRays);
  		//hitType = new eavlByteArray("",1, numRays);
  		//weight  = new eavlFloatArray("",1,numRays);
  		intersectionX = new eavlFloatArray("",1,numRays);
  		intersectionY = new eavlFloatArray("",1,numRays);
  		intersectionZ = new eavlFloatArray("",1,numRays);
  	}
  	virtual void resize(const size_t &newSize)
  	{
  		if(newSize == numRays) return;

  		//delete weight;
  		//delete hitType;
  		delete normalX;
  		delete normalY;
  		delete normalZ; 
  		delete scalar;
  		delete intersectionX;
  		delete intersectionY;
  		delete intersectionZ;
  		
  		normalX = new eavlFloatArray("",1,newSize);
  		normalY = new eavlFloatArray("",1,newSize);
  		normalZ = new eavlFloatArray("",1,newSize);
  		//hitType = new eavlByteArray("",1, newSize);
  		//weight  = new eavlFloatArray("",1,newSize);
  		scalar  = new eavlFloatArray("",1,newSize);
  		
  		intersectionX = new eavlFloatArray("",1,newSize);
  		intersectionY = new eavlFloatArray("",1,newSize);
  		intersectionZ = new eavlFloatArray("",1,newSize);
  		eavlRay::resize(newSize);
  	}
  	~eavlFullRay()
  	{
  		//delete weight;
  		//delete hitType;
  		delete normalX;
  		delete normalY;
  		delete normalZ; 
  		delete scalar;
  		delete intersectionX;
  		delete intersectionY;
  		delete intersectionZ;
  	}

};
#endif
