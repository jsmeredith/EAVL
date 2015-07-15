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
		vector<VRTetrahedron>  tets;
		float 				   maxScalar;
		float 				   minScalar;
		float 				   scalarSpread;
		bool 				   normalizeScalars;
		int maxxtets ;
		EAVL_HOSTONLY inline float normalizeScalar(float);
	public: 
		EAVL_HOSTONLY inline eavlVRScene();
		EAVL_HOSTONLY inline ~eavlVRScene();
		EAVL_HOSTONLY inline int 			 getNumTets();
		EAVL_HOSTONLY inline float* 		 getTetPtr();
		EAVL_HOSTONLY inline eavlFloatArray**			 getEavlTetPtrs();
		EAVL_HOSTONLY inline float*     	 getScalarPtr();
		EAVL_HOSTONLY inline void 		addTet(const eavlVector3 &v0, const eavlVector3 &v1, const eavlVector3 &v2, const eavlVector3 &v3,
									   const float &s0, const float &s1, const float &s2, const float &s3);
		EAVL_HOSTONLY inline void 		clear();
		EAVL_HOSTONLY inline float 		getSceneMagnitude();
		EAVL_HOSTONLY inline BBox 		getSceneBBox(){return sceneBox;};
		EAVL_HOSTONLY inline void 		normalizedScalars(bool on){normalizeScalars = on;};

};

EAVL_HOSTONLY inline eavlVRScene::eavlVRScene()
{
	maxScalar = std::numeric_limits<float>::min();
	minScalar = std::numeric_limits<float>::max();
	//maxxtets = 35000000;
	normalizeScalars = false;
}

EAVL_HOSTONLY inline eavlVRScene::~eavlVRScene()
{
	
}

EAVL_HOSTONLY inline int eavlVRScene::getNumTets()
{
	return tets.size();
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
	tets.push_back(t);
}

EAVL_HOSTONLY inline float eavlVRScene::normalizeScalar(float s)
{
	if(normalizeScalars) return (s - minScalar) / scalarSpread;
	else return s;
}
EAVL_HOSTONLY inline float* eavlVRScene::getTetPtr()
{
	int numTets = getNumTets();
	if (numTets == 0 ) return NULL;
	scalarSpread = maxScalar - minScalar;
	//cout<<"Scalar range ("<<minScalar<<","<<maxScalar<<")"<<endl;
	float * tets_raw = new float [numTets*16]; /* 4*( xyz + scalar) */
	for (int i=0;  i < numTets; i++)
	{
		tets_raw[i*16+ 0] = tets.at(i).verts[0].x;
		tets_raw[i*16+ 1] = tets.at(i).verts[0].y; 
		tets_raw[i*16+ 2] = tets.at(i).verts[0].z;
		tets_raw[i*16+ 3] = normalizeScalar(tets.at(i).scalars[0]);

		tets_raw[i*16+ 4] = tets.at(i).verts[1].x;
		tets_raw[i*16+ 5] = tets.at(i).verts[1].y; 
		tets_raw[i*16+ 6] = tets.at(i).verts[1].z;
		tets_raw[i*16+ 7] = normalizeScalar(tets.at(i).scalars[1]);

		tets_raw[i*16+ 8] = tets.at(i).verts[2].x;
		tets_raw[i*16+ 9] = tets.at(i).verts[2].y; 
		tets_raw[i*16+10] = tets.at(i).verts[2].z;
		tets_raw[i*16+11] = normalizeScalar(tets.at(i).scalars[2]);

		tets_raw[i*16+12] = tets.at(i).verts[3].x;
		tets_raw[i*16+13] = tets.at(i).verts[3].y; 
		tets_raw[i*16+14] = tets.at(i).verts[3].z;
		tets_raw[i*16+15] = normalizeScalar(tets.at(i).scalars[3]);


	}

	return tets_raw;
}

EAVL_HOSTONLY inline eavlFloatArray** eavlVRScene::getEavlTetPtrs()
{
     //std::sort(tets.begin(), tets.end(), less_than_key());

	int numTets = getNumTets();
	if (numTets == 0 ) return NULL;
	scalarSpread = maxScalar - minScalar;
	cout<<"Scalar range ("<<minScalar<<","<<maxScalar<<")"<<endl;
	
	eavlFloatArray * xtets = new eavlFloatArray("",1, numTets*4);
	eavlFloatArray * ytets = new eavlFloatArray("",1, numTets*4);
	eavlFloatArray * ztets = new eavlFloatArray("",1, numTets*4);
	for (int i=0;  i < numTets; i++)
	{
		xtets->SetValue(i*4 + 0, tets.at(i).verts[0].x);
		ytets->SetValue(i*4 + 0, tets.at(i).verts[0].y); 
		ztets->SetValue(i*4 + 0, tets.at(i).verts[0].z);
		

		xtets->SetValue(i*4 + 1, tets.at(i).verts[1].x);
		ytets->SetValue(i*4 + 1, tets.at(i).verts[1].y); 
		ztets->SetValue(i*4 + 1, tets.at(i).verts[1].z);
		

		xtets->SetValue(i*4 + 2, tets.at(i).verts[2].x);
		ytets->SetValue(i*4 + 2, tets.at(i).verts[2].y); 
		ztets->SetValue(i*4 + 2, tets.at(i).verts[2].z);
		

		xtets->SetValue(i*4 + 3, tets.at(i).verts[3].x);
		ytets->SetValue(i*4 + 3, tets.at(i).verts[3].y); 
		ztets->SetValue(i*4 + 3, tets.at(i).verts[3].z);
		

		
	}
	eavlFloatArray **tetsarray =  (eavlFloatArray**)malloc(sizeof(eavlFloatArray*)*3) ;
	tetsarray[0] =  xtets;
	tetsarray[1] =  ytets;
	tetsarray[2] =  ztets;
	return tetsarray;
}



EAVL_HOSTONLY inline float* eavlVRScene::getScalarPtr()
{
	int numTets = getNumTets();
	if (numTets == 0 ) return NULL;
	
	float * scalars_raw = new float [numTets*4]; 
	
	for (int i=0;  i < numTets; i++)
	{

		scalars_raw[i*4+ 0] = normalizeScalar(tets.at(i).scalars[0]);
		scalars_raw[i*4+ 1] = normalizeScalar(tets.at(i).scalars[1]);
		scalars_raw[i*4+ 2] = normalizeScalar(tets.at(i).scalars[2]);
		scalars_raw[i*4+ 3] = normalizeScalar(tets.at(i).scalars[3]);
		
	}

	return scalars_raw;
}

EAVL_HOSTONLY inline void eavlVRScene::clear()
{
	tets.clear();
}

EAVL_HOSTONLY inline float eavlVRScene::getSceneMagnitude()
{
	float mag = sceneBox.extent*sceneBox.extent;
	return sqrt(mag);
}
#endif