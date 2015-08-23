#ifndef EAVL_RAY_CAMERA_H
#define EAVL_RAY_CAMERA_H
#include "eavlVector3.h"
#include "eavlRay.h"
#include "eavlMatrix4x4.h"

#include "eavlRTUtil.h"

class eavlRayCamera
{
  public:
  	EAVL_HOSTONLY eavlRayCamera();
  	EAVL_HOSTONLY ~eavlRayCamera();
	EAVL_HOSTONLY void  printSummary();
	EAVL_HOSTONLY void  createRays(eavlRay* rays);
	EAVL_HOSTONLY void  createRaysSubset(eavlRay* rays, int &xmin, int &ymin, int &dx, int &dy);
	EAVL_HOSTONLY void  createJitterRays(eavlRay* rays, eavlIntArray *seeds, int sampleNum);
	EAVL_HOSTONLY void  createDOFRays(eavlRay* rays, eavlIntArray *seeds, int sampleNum, float apertureSize);
	EAVL_HOSTONLY void  setWidth(const int &_width);
	EAVL_HOSTONLY void  setHeight(const int &_height);
	EAVL_HOSTONLY int   getWidth();
	EAVL_HOSTONLY int   getHeight();

	EAVL_HOSTONLY void  setMortonSorting(bool on);
	EAVL_HOSTONLY bool  getMortonSorting();
	EAVL_HOSTONLY bool  getIsViewDirty();


	EAVL_HOSTONLY void  setFOVX(const float &_fovx);
	EAVL_HOSTONLY void  setFOVY(const float &_fovy);
	EAVL_HOSTONLY float getFOVX();
	EAVL_HOSTONLY float getFOVY();

	EAVL_HOSTONLY void  setCameraPosition (const float &_x, const float &_y,const float &_z);
	EAVL_HOSTONLY void  setCameraPositionX(const float &_x);
	EAVL_HOSTONLY void  setCameraPositionY(const float &_y);
	EAVL_HOSTONLY void  setCameraPositionZ(const float &_z);
	EAVL_HOSTONLY float getCameraPositionX();
	EAVL_HOSTONLY float getCameraPositionY();
	EAVL_HOSTONLY float getCameraPositionZ();

	EAVL_HOSTONLY void  setCameraUp (const float &_x, const float &_y,const float &_z);
	EAVL_HOSTONLY void  setCameraUpX(const float &_x);
	EAVL_HOSTONLY void  setCameraUpY(const float &_y);
	EAVL_HOSTONLY void  setCameraUpZ(const float &_z);
	EAVL_HOSTONLY float getCameraUpX();
	EAVL_HOSTONLY float getCameraUpY();
	EAVL_HOSTONLY float getCameraUpZ();


	EAVL_HOSTONLY void  setCameraLook (const float &_x, const float &_y,const float &_z);
	EAVL_HOSTONLY void  setCameraLookX(const float &_x);
	EAVL_HOSTONLY void  setCameraLookY(const float &_y);
	EAVL_HOSTONLY void  setCameraLookZ(const float &_z);
	EAVL_HOSTONLY float getCameraLookX();
	EAVL_HOSTONLY float getCameraLookY();
	EAVL_HOSTONLY float getCameraLookZ();
	
	EAVL_HOSTONLY void  setCameraZoom(const float &_zoom);

	EAVL_HOSTONLY void  lookAtPosition(const float &_x, const float &_y,const float &_z);

	EAVL_HOSTONLY eavlIntArray * getPixelIndexes();
  protected: 
	int 	height;
	int	 	width;
	int     size;
	float 	fovx;
	float   fovy;
	float 	zoom;
	bool    isViewDirty;
	bool    isResDirty;
	bool    mortonSort;
	bool 	lookAtSet;

	eavlVector3 look;     
    eavlVector3 up;
    eavlVector3 lookat;
    eavlVector3 position;

    eavlIntArray* pixelIndexes;
    EAVL_HOSTONLY void generatePixelIndexes();
};
#endif
