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

	    void resize(const size_t &newSize)
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
  	eavlFloatArray*     weight;	      //contribution
  	eavlFloatArray*     normalX;	      //intersection normal
  	eavlFloatArray*     normalY;	     
  	eavlFloatArray*     normalZ;
  	eavlFloatArray*     intersectionX;
  	eavlFloatArray*     intersectionY;
  	eavlFloatArray*     intersectionZ;	     
  	eavlByteArray*      hitType;      //type of geometry hit
  	eavlFullRay(const size_t &size) : eavlRay(size) 
  	{
  		weight  = new eavlFloatArray("",1,numRays);
  		normalX = new eavlFloatArray("",1,numRays);
  		normalY = new eavlFloatArray("",1,numRays);
  		normalZ = new eavlFloatArray("",1,numRays);
  		hitType = new eavlByteArray("",1, numRays);
  	}
  	void resize(const size_t &newSize)
  	{
  		eavlRay::resize(newSize);
  		if(newSize == numRays) return;
  		delete weight;
  		delete hitType;
  		delete normalX;
  		delete normalY;
  		delete normalZ; 
  		weight  = new eavlFloatArray("",1,numRays);
  		normalX = new eavlFloatArray("",1,numRays);
  		normalY = new eavlFloatArray("",1,numRays);
  		normalZ = new eavlFloatArray("",1,numRays);
  		hitType = new eavlByteArray("",1, numRays);
  	}
  	~eavlFullRay()
  	{
  		delete weight;
  		delete hitType;
  		delete normalX;
  		delete normalY;
  		delete normalZ; 
  	}

};
#endif