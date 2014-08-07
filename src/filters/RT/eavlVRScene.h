#ifndef EAVL_VOLUME_RENDERER_SCENE_H
#define EAVL_VOLUME_RENDERER_SCENE_H

#include "eavlRTPrimitive.h"
#include <vector>

using namespace std;

class eavlVRScene
{
	private:
		BBox				   sceneBox;
		float 				   surfaceArea;
		vector<VRTetrahedron>  tets;
	public: 
		EAVL_HOSTONLY inline eavlVRScene();
		EAVL_HOSTONLY inline ~eavlVRScene();
		EAVL_HOSTONLY inline int 		getNumTets();
		EAVL_HOSTONLY inline float* 	getTetPtr();
		EAVL_HOSTONLY inline float*     getScalarPtr();
		EAVL_HOSTONLY inline void 		addTet(const eavlVector3 &v0, const eavlVector3 &v1, const eavlVector3 &v2, const eavlVector3 &v3,
									   const float &s0, const float &s1, const float &s2, const float &s3);
		EAVL_HOSTONLY inline void 		clear();
		EAVL_HOSTONLY inline float 		getSceneMagnitude();

};

EAVL_HOSTONLY inline eavlVRScene::eavlVRScene()
{

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
	VRTetrahedron t(v0,v1,v2,v3,s0,s1,s2,s3);
	sceneBox.expandToInclude(v0);
	sceneBox.expandToInclude(v1);
	sceneBox.expandToInclude(v2);
	sceneBox.expandToInclude(v3);
	tets.push_back(t);
}

EAVL_HOSTONLY inline float* eavlVRScene::getTetPtr()
{
	int numTets = getNumTets();
	if (numTets == 0 ) return NULL;
	
	float * tets_raw = new float [numTets*16]; /* 4*( xyz + scalar) */
	for (int i=0;  i < numTets; i++)
	{
		tets_raw[i*16+ 0] = tets.at(i).verts[0].x;
		tets_raw[i*16+ 1] = tets.at(i).verts[0].y; 
		tets_raw[i*16+ 2] = tets.at(i).verts[0].z;
		tets_raw[i*16+ 3] = tets.at(i).scalars[0];

		tets_raw[i*16+ 4] = tets.at(i).verts[1].x;
		tets_raw[i*16+ 5] = tets.at(i).verts[1].y; 
		tets_raw[i*16+ 6] = tets.at(i).verts[1].z;
		tets_raw[i*16+ 7] = tets.at(i).scalars[1];

		tets_raw[i*16+ 8] = tets.at(i).verts[2].x;
		tets_raw[i*16+ 9] = tets.at(i).verts[2].y; 
		tets_raw[i*16+10] = tets.at(i).verts[2].z;
		tets_raw[i*16+11] = tets.at(i).scalars[2];

		tets_raw[i*16+12] = tets.at(i).verts[3].x;
		tets_raw[i*16+13] = tets.at(i).verts[3].y; 
		tets_raw[i*16+14] = tets.at(i).verts[3].z;
		tets_raw[i*16+15] = tets.at(i).scalars[3];

	}

	return tets_raw;
}

EAVL_HOSTONLY inline float* eavlVRScene::getScalarPtr()
{
	int numTets = getNumTets();
	if (numTets == 0 ) return NULL;
	
	float * scalars_raw = new float [numTets*4]; 
	for (int i=0;  i < numTets; i++)
	{

		scalars_raw[i*4+ 0] = tets.at(i).scalars[0];
		scalars_raw[i*4+ 1] = tets.at(i).scalars[1];
		scalars_raw[i*4+ 2] = tets.at(i).scalars[2];
		scalars_raw[i*4+ 3] = tets.at(i).scalars[3];

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