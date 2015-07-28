#ifndef EAVL_PATH_TRACER_H
#define EAVL_PATH_TRACER_H

#include <eavlRayTriangleIntersector.h>
#include <eavlRayTriangleGeometry.h>
#include <eavlTextureObject.h>
#include <eavlRayCamera.h>
#include <eavlRTScene.h>
#include <eavlRay.h>



class eavlPathTracer
{
protected:
	int 						numColors;
	eavlTextureObject<float> 	*colorMap;
	float 						*materials;
	eavlFloatArray				*eavlMaterials; //TODO get rid of this with scene refactor
	eavlRayTriangleGeometry 	*triGeometry;
	eavlFullRay 				*rays;
	eavlRayTriangleIntersector  *intersector;
	eavlFloatArray 			  	*frameBuffer;
	eavlByteArray				*rgbaPixels;
	eavlFloatArray 				*depthBuffer;
	eavlIntArray 				*inShadow;
	eavlFloatArray 				*ambPercentage;
	bool 						geometryDirty;
	int 						currentFrameSize;
	int 						numTriangles;

	eavlArrayIndexer      		*redIndexer;
    eavlArrayIndexer      		*greenIndexer;
    eavlArrayIndexer      		*blueIndexer;
    eavlArrayIndexer      		*alphaIndexer;
    eavlVector3					bgColor;
    //ambient occlusion
    bool 						occlusionOn;
    int 						numOccSamples;
    float 						occDistance;
    eavlFloatArray 				*shadowX;
    eavlFloatArray 				*shadowY;
    eavlFloatArray 				*shadowZ;
    eavlFloatArray 				*reflectX;
    eavlFloatArray 				*reflectY;
    eavlFloatArray 				*reflectZ;
    eavlIntArray 				*seeds;
    eavlArrayIndexer      		*indexer;

    eavlFloatArray  *rSurface;                  
    eavlFloatArray  *gSurface;
    eavlFloatArray  *bSurface;

    eavlFloatArray  *rCurrent;                  
    eavlFloatArray  *gCurrent;
    eavlFloatArray  *bCurrent;

    eavlFloatArray  *lred;                  
    eavlFloatArray  *lgreen;
    eavlFloatArray  *lblue;

    int numSamples;
    int rayDepth;

public:
	eavlRTScene					*scene;
	eavlRayCamera 				*camera;
	eavlVector3					lightPosition;
	eavlPathTracer();
	~eavlPathTracer();
	void setColorMap3f(float *, const int &);
	void setBackgroundColor(float, float, float);
	void setOcclusionSamples(int);
	void setOcclusionDistance(float);
	void setOcclusionOn(bool);
	void setNumberOfSamples(int);
	void setRayDepth(int);

	void render();
	void startScene();
	void setDefaultMaterial(const float&, const float&, const float&);
	eavlByteArray* getFrameBuffer();
	eavlFloatArray *getDepthBuffer(float proj22, float proj23, float proj32);
private:
	void init(int);
	void addColor();
};
#endif
