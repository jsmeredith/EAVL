#ifndef EAVL_VOLUME_RENDERER_SCENE_H
#define EAVL_VOLUME_RENDERER_SCENE_H

#include "eavlRTPrimitive.h"
#include <vector>
#include <limits>
using namespace std;

class eavlVRScene
{
	private:
		BBox				   sceneBox;
		float 				   surfaceArea;
		eavlFloatArray 		   *xverts;
		eavlFloatArray 		   *yverts;
		eavlFloatArray 		   *zverts;
		eavlFloatArray 		   *scalars;
		float 				   maxScalar;
		float 				   minScalar;
		float 				   scalarSpread;
		bool 				   normalizeScalars;
		int maxxtets ;
		EAVL_HOSTONLY inline float normalizeScalar(float);
	public: 
		EAVL_HOSTONLY inline eavlVRScene();
		EAVL_HOSTONLY inline ~eavlVRScene();
		EAVL_HOSTONLY inline int 			  getNumTets();
		EAVL_HOSTONLY inline eavlFloatArray** getEavlTetPtrs();
		EAVL_HOSTONLY inline eavlFloatArray*  getScalarPtr();
		EAVL_HOSTONLY inline void 		addTet(const eavlVector3 &v0, const eavlVector3 &v1, const eavlVector3 &v2, const eavlVector3 &v3,
									   const float &s0, const float &s1, const float &s2, const float &s3);
		EAVL_HOSTONLY inline void 		clear();
		EAVL_HOSTONLY inline float 		getSceneMagnitude();
		EAVL_HOSTONLY inline BBox 		getSceneBBox(){return sceneBox;};
		EAVL_HOSTONLY inline void 		normalizedScalars(bool on){normalizeScalars = on;};

};

EAVL_HOSTONLY inline eavlVRScene::eavlVRScene()
{
	xverts = new eavlFloatArray("",1,0);
	yverts = new eavlFloatArray("",1,0);
	zverts = new eavlFloatArray("",1,0);
	scalars = new eavlFloatArray("",1,0);
	maxScalar = std::numeric_limits<float>::min();
	minScalar = std::numeric_limits<float>::max();
	//maxxtets = 35000000;
	normalizeScalars = false;
}

EAVL_HOSTONLY inline eavlVRScene::~eavlVRScene()
{
	delete xverts;
	delete yverts;
	delete zverts;
	delete scalars;
}

EAVL_HOSTONLY inline int eavlVRScene::getNumTets()
{
	return xverts->GetNumberOfTuples() / 4;
}

EAVL_HOSTONLY inline void eavlVRScene::addTet(const eavlVector3 &v0, const eavlVector3 &v1, const eavlVector3 &v2, const eavlVector3 &v3,
						  const float &s0, const float &s1, const float &s2, const float &s3)
{
	//if(getNumTets() > maxxtets) return;
	VRTetrahedron t(v0,v1,v2,v3,s0,s1,s2,s3);
	minScalar = min(minScalar, s0);
	maxScalar = max(maxScalar, s0);
	minScalar = min(minScalar, s1);
	maxScalar = max(maxScalar, s1);
	minScalar = min(minScalar, s2);
	maxScalar = max(maxScalar, s2);
	minScalar = min(minScalar, s3);
	maxScalar = max(maxScalar, s3);
	sceneBox.expandToInclude(v0);
	sceneBox.expandToInclude(v1);
	sceneBox.expandToInclude(v2);
	sceneBox.expandToInclude(v3);
	xverts->AddValue(v0.x);
	yverts->AddValue(v0.y);
	zverts->AddValue(v0.z);
	xverts->AddValue(v1.x);
	yverts->AddValue(v1.y);
	zverts->AddValue(v1.z);
	xverts->AddValue(v2.x);
	yverts->AddValue(v2.y);
	zverts->AddValue(v2.z);
	xverts->AddValue(v3.x);
	yverts->AddValue(v3.y);
	zverts->AddValue(v3.z);
	scalars->AddValue(s0);
	scalars->AddValue(s1);
	scalars->AddValue(s2);
	scalars->AddValue(s3);
	//tets.push_back(t);
}

EAVL_HOSTONLY inline float eavlVRScene::normalizeScalar(float s)
{
	if(normalizeScalars) return (s - minScalar) / scalarSpread;
	else return s;
}

EAVL_HOSTONLY inline eavlFloatArray** eavlVRScene::getEavlTetPtrs()
{
     //std::sort(tets.begin(), tets.end(), less_than_key());
	eavlFloatArray **tetsarray =  (eavlFloatArray**)malloc(sizeof(eavlFloatArray*)*3) ;
	tetsarray[0] =  xverts;
	tetsarray[1] =  yverts;
	tetsarray[2] =  zverts;
	return tetsarray;
}



EAVL_HOSTONLY inline eavlFloatArray* eavlVRScene::getScalarPtr()
{
	return scalars;
}

EAVL_HOSTONLY inline void eavlVRScene::clear()
{
	delete xverts;
	delete yverts;
	delete zverts;
	delete scalars;
	xverts = new eavlFloatArray("",1,0);
	yverts = new eavlFloatArray("",1,0);
	zverts = new eavlFloatArray("",1,0);
	scalars = new eavlFloatArray("",1,0);
	maxScalar = std::numeric_limits<float>::min();
	minScalar = std::numeric_limits<float>::max();
}

EAVL_HOSTONLY inline float eavlVRScene::getSceneMagnitude()
{
	float mag = sceneBox.extent*sceneBox.extent;
	return sqrt(mag);
}
#endif