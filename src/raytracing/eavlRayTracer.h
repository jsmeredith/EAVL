#ifndef EAVL_RAY_TRACER_H
#define EAVL_RAY_TRACER_H

#include <eavlRayTriangleIntersector.h>
#include <eavlRayTriangleGeometry.h>
#include <eavlTextureObject.h>
#include <eavlRayCamera.h>
#include <eavlRTScene.h>
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
	eavlFloatArray 				*inShadow;
	bool 						geometryDirty;
	int 						currentFrameSize;
	int 						numTriangles;

	eavlArrayIndexer      		*redIndexer;
    eavlArrayIndexer      		*greenIndexer;
    eavlArrayIndexer      		*blueIndexer;
    eavlArrayIndexer      		*alphaIndexer;

public:
	eavlRTScene					*scene;
	eavlRayCamera 				*camera;
	eavlVector3					lightPosition;
	eavlRayTracer();
	~eavlRayTracer();
	void setColorMap3f(float *, const int &);

	void render();
	void startScene();
	void setDefaultMaterial(const float&, const float&, const float&);
	eavlByteArray* getFrameBuffer();
	eavlFloatArray *getDepthBuffer(float proj22, float proj23, float proj32);
private:
	void init();
};
#endif