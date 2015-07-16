#ifndef EAVL_RAY_TRACER_H
#define EAVL_RAY_TRACER_H

#include <eavlTextureObject.h>
#include <eavlRayTriangleGeometry.h>
#include <eavlRayCamera.h>
#include <eavlRay.h>
#include <eavlRayTriangleIntersector.h>

class eavlRayTracer
{
protected:
	eavlTextureObject<float> 	*colorMap;
	eavlRayTriangleGeometry 	*triGeometry;
	eavlRayCamera 				*camera;
	eavlRay 					*rays;
	eavlRayTriangleIntersector  *intersector;
public:
	eavlRayTracer();
	~eavlRayTracer();
	void setColorMap3f(float *, const int &);

};
#endif