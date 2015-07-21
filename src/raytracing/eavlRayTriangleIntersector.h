#ifndef EAVL_RAY_TRIANGLE_INTERSECTOR
#define EAVL_RAY_TRIANGLE_INTERSECTOR

#include "eavlRay.h"
#include "eavlRayCamera.h"
#include "eavlRayTriangleGeometry.h"

class eavlRayTriangleIntersector{

	public:
		//Depth returns hit index and distance between ray origin and maxDistance.
		EAVL_HOSTONLY void intersectionDepth(const eavlRay *rays,
									  		 const int &maxDistance, 
									  		 const eavlRayTriangleGeometry *geometry);
		EAVL_HOSTONLY void intersectionDepth(const eavlRay *rays, 
									  		 eavlFloatArray *maxDistances, 
									  		 const eavlRayTriangleGeometry *geometry);
		EAVL_HOSTONLY void intersectionShadow(const eavlFullRay *rays, 
									  		 eavlIntArray *hits,
									  		 eavlVector3 &lightPosition,  
									  		 const eavlRayTriangleGeometry *geometry);
		EAVL_HOSTONLY void intersectionOcclusion(const eavlFullRay *rays, 
									  		 	 eavlFloatArray *occX,
									  		 	 eavlFloatArray *occY,
									  		 	 eavlFloatArray *occZ,
									  		 	 eavlIntArray *hits,
									  		 	 eavlArrayIndexer *occIndexer,
									  		 	 float maxDistance,  
									  		 	 const eavlRayTriangleGeometry *geometry);
		//Full returns hit index, distance, U, V, 
		//static void intersectionFull(const eavlRay &rays, const int &maxDistance, const eavlRayTriangleGeomtry &geometry);
		//static void intersectionFull(const eavlRay &rays, eavlFloatArray *maxDistances, const eavlRayTriangleGeomtry &geometry);

		EAVL_HOSTONLY void testIntersections(const eavlRay *rays, 
									  		 const int &maxDistance, 
									  		 const eavlRayTriangleGeometry *geometry,
									  		 const int &warmUpRounds,
									  		 const int &testRounds,
									  		 eavlRayCamera *cam);
};
#endif
