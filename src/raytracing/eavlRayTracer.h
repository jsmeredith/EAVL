#ifndef EAVL_RAY_TRACER_H
#define EAVL_RAY_TRACER_H

#include <eavlRayTriangleIntersector.h>
#include <eavlRayTriangleGeometry.h>
#include <eavlTextureObject.h>
#include <eavlRayCamera.h>
#include <eavlRTScene.h>
#include <eavlView.h>
#include <eavlRay.h>



class eavlRayTracer
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
    bool            shadowsOn;
    int 						numOccSamples;
    float 						occDistance;
    eavlFloatArray 				*occX;
    eavlFloatArray 				*occY;
    eavlFloatArray 				*occZ;
    eavlIntArray				*occHits;
    eavlFloatArray 				*ambientPct;
    eavlArrayIndexer      		*occIndexer;
    bool						occDirty;
    eavlView 					view;
    bool 						imageSubsetMode;
    int             subsetMinx;
    int             subsetMiny;
    int             subsetDx;
    int             subsetDy;

public:
	eavlRTScene					*scene;
	eavlRayCamera 				*camera;
	eavlVector3					lightPosition;
	eavlRayTracer();
	~eavlRayTracer();
	void setColorMap3f(float *, const int &);
	void setBackgroundColor(float, float, float);
	void setOcclusionSamples(int);
	void setOcclusionDistance(float);
	void setOcclusionOn(bool);
	void setShadowsOn(bool);
	void enableImageSubset(eavlView &);
	void render();
	void startScene();
	void setDefaultMaterial(const float&, const float&, const float&);
	eavlByteArray* getFrameBuffer();
	eavlFloatArray *getDepthBuffer(float proj22, float proj23, float proj32);
	void getImageSubsetDims(int *);
private:
	void init();
	void findImageExtent();
};
#endif
