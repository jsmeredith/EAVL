#ifndef EAVL_RAY_TRACER_SCENE_H
#define EAVL_RAY_TRACER_SCENE_H
#include "eavlRTPrimitive.h"
#include <vector>
#include <map>
#include <string>
#include "objloader.h"
using namespace std;

class eavlRTScene
{
	private:

		BBox 				sceneBbox;
		float 				surfaceArea; 
		vector<RTMaterial>* mats;
		vector<RTTriangle>* tris;					// Optimization idea: have a triangle class keep a struct of arrays and preformat the data as it is inserted. Just get pointer to the vector data
		vector<RTSphere>* 	spheres;
		map<string, int> 	matMap;
		int 				numMats;
		string 				filename;
		RTMaterial			defaultMat;
	public:
		EAVL_HOSTONLY inline eavlRTScene(RTMaterial defualtMaterial= RTMaterial());
		EAVL_HOSTONLY inline ~eavlRTScene();
		EAVL_HOSTONLY inline const  eavlVector3 getSceneExtent();
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
													const  float &scalarV0, const float &scalarV1, const float &scalarV2,  string matName="default");
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
													const  float &scalarV0, const float &scalarV1, const float &scalarV2,  const int &matId);
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
													const eavlVector3 &n0 , const eavlVector3 &n1, const eavlVector3 &n2,
													const  float &scalarV0, const float &scalarV1, const float &scalarV2,  string matName="default");
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
													const eavlVector3 &n0 , const eavlVector3 &n1, const eavlVector3 &n2,
													const  float &scalarV0, const float &scalarV1, const float &scalarV2,  const int &matId);
		EAVL_HOSTONLY inline void 		addSphere(const float &radius, const float &centerX, const float &centerY, const float &centerZ, string matName="default");
		EAVL_HOSTONLY inline void 		addSphere(const float &radius, const float &centerX, const float &centerY, const float &centerZ, const int &matId);
		EAVL_HOSTONLY inline void 		setDefaultMaterial(const RTMaterial &_mat);
		EAVL_HOSTONLY inline int 		addMaterial( RTMaterial _mat, string matName);
		EAVL_HOSTONLY inline int 		addMaterial( RTMaterial _mat);
		EAVL_HOSTONLY inline int 		getNumTriangles(){ return tris->size(); };
		EAVL_HOSTONLY inline int 		getNumMaterials(){ return mats->size(); };
		EAVL_HOSTONLY inline int 		getNumSpheres(){ return spheres->size(); };
		EAVL_HOSTONLY inline int 		getTotalPrimitives();
		EAVL_HOSTONLY inline void       loadObjFile(const char* filename);
		EAVL_HOSTONLY inline void       clear();   												/* clears primitives */
		EAVL_HOSTONLY inline float*     getTrianglePtr();
		EAVL_HOSTONLY inline float*     getTriangleNormPtr();
		EAVL_HOSTONLY inline float*     getSpherePtr();
	    EAVL_HOSTONLY inline int*       getSphrMatIdxPtr();
		EAVL_HOSTONLY inline float*     getMatsPtr();
		EAVL_HOSTONLY inline int*       getTriMatIdxsPtr();
		EAVL_HOSTONLY inline float 		getSceneExtentMagnitude();
		EAVL_HOSTONLY inline RTMaterial getDefaultMaterial(){ return defaultMat;};

};


EAVL_HOSTONLY inline eavlRTScene::eavlRTScene(RTMaterial defaultMaterial)
{
	defaultMat = defaultMaterial;
	mats = new vector<RTMaterial>();
	mats->push_back(defaultMaterial);
	matMap.insert(pair<string, int> ("default", 0));
	numMats = 1;

	tris= new vector<RTTriangle>();
	spheres= new vector<RTSphere>();
}

EAVL_HOSTONLY inline eavlRTScene::~eavlRTScene()
{
	delete mats;
	delete tris;
	delete spheres;
}

EAVL_HOSTONLY inline void eavlRTScene::setDefaultMaterial(const RTMaterial &_mat)
{
	mats->at(0)=_mat;
	defaultMat=_mat;
}

EAVL_HOSTONLY inline int eavlRTScene::addMaterial(RTMaterial _mat, string matName)
{
	int idx=numMats;
	matMap.insert(pair<string, int>(matName, numMats));
	mats->push_back(_mat);
	numMats++;
	return idx;
}

EAVL_HOSTONLY inline int eavlRTScene::addMaterial( RTMaterial _mat)
{
	int idx=numMats;
	mats->push_back(_mat);
	numMats++;
	return idx;
}

EAVL_HOSTONLY inline const  eavlVector3 eavlRTScene::getSceneExtent() { return sceneBbox.extent; }

EAVL_HOSTONLY inline void eavlRTScene::addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
										    const float &scalarV0 , const float &scalarV1, const float &scalarV2,  string matName)
{

	int matId=0;
	if(matName!="default")
	{
		if ( matMap.find(matName) != matMap.end() ) 
		{
  			matId=matMap[matName];
		} 
	}

	RTTriangle t(v0, v1, v2, scalarV0, scalarV1, scalarV2, matId);
	tris->push_back( t );
	sceneBbox.expandToInclude(t.getBBox());
}

EAVL_HOSTONLY inline void eavlRTScene::addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
										    const float &scalarV0 , const float &scalarV1, const float &scalarV2, const int &matId)
{
	RTTriangle t(v0, v1, v2, scalarV0, scalarV1, scalarV2, matId);
	tris->push_back( t );
	sceneBbox.expandToInclude(t.getBBox());
}

EAVL_HOSTONLY inline void eavlRTScene::addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
											const eavlVector3 &n0 , const eavlVector3 &n1, const eavlVector3 &n2,
											const float &scalarV0 , const float &scalarV1, const float &scalarV2,  string matName)
{

	int matId=0;
	if(matName!="default")
	{
		if ( matMap.find(matName) != matMap.end() ) 
		{
  			matId=matMap[matName];
		} 
	}
	RTTriangle t(v0, v1, v2, n0, n1 ,n2, scalarV0, scalarV1, scalarV2, matId);
	tris->push_back( t );
	sceneBbox.expandToInclude(t.getBBox());
}

EAVL_HOSTONLY  inline void eavlRTScene::addSphere(const float &radius, const float &centerX, const float &centerY, const float &centerZ, string matName)
{
	int matId=0;
	if(matName!="default")
	{
		if ( matMap.find(matName) != matMap.end() ) 
		{
  			matId=matMap[matName];
		} 
	}

	RTSphere t ( radius, eavlVector3(centerX, centerY, centerZ),matId );
	spheres->push_back( t );
	sceneBbox.expandToInclude(t.getBBox());
}

EAVL_HOSTONLY  inline void eavlRTScene::addSphere(const float &radius, const float &centerX, const float &centerY, const float &centerZ, const int &matId)
{
	

	RTSphere t ( radius, eavlVector3(centerX, centerY, centerZ),matId );
	spheres->push_back( t );
	sceneBbox.expandToInclude(t.getBBox());
}

EAVL_HOSTONLY inline void eavlRTScene::loadObjFile(const char * _filename)
{
	ObjReader *objreader= new ObjReader(_filename);
	float *v;  	//verts 
    float *n;	//norms
    int *mIdx;
    int numTris=objreader->totalTriangles;
    float *mats;
    int matCount;
    objreader->getRawData(v,n,mats,mIdx,matCount);
    map<int,int> newMatidxs; 				//create a mapping of indexs
    int matId=0;
    cout<<matCount<<endl;
    for (int i=0;i<matCount;i++)
	{   		
		matId=addMaterial( RTMaterial(eavlVector3(mats[i*12   ], mats[i*12+1 ], mats[i*12+2 ]),
					 				  eavlVector3(mats[i*12+3 ], mats[i*12+4 ], mats[i*12+5 ]),
					 				  eavlVector3(mats[i*12+6 ], mats[i*12+7 ], mats[i*12+8 ]), mats[i*12+9], .3f));
	    newMatidxs.insert(pair<int,int>(i,matId));
	}

	for( int i=0; i<numTris;i++)
	{
		addTriangle(eavlVector3( v[i*9  ],v[i*9+1],v[i*9+2] ),
					eavlVector3( v[i*9+3],v[i*9+4],v[i*9+5] ),
					eavlVector3( v[i*9+6],v[i*9+7],v[i*9+8]),
					0.f, 0.f, 0.f, newMatidxs[mIdx[i]]	);
	}

	delete objreader;
	delete[] mats;
	delete[] mIdx;
	delete[] v;
	delete[] n;


}

void inline eavlRTScene::clear()
{
	tris->clear();
	spheres->clear();
	mats->clear();
	matMap.clear();
	/* Load the defualt materials back in*/
	mats->push_back(defaultMat);
	matMap.insert(pair<string, int> ("default", 0));
}

EAVL_HOSTONLY  inline float*  eavlRTScene::getTrianglePtr()
{
	if(getNumTriangles() < 1) return NULL;
	cout<<"Num Tris "<<getNumTriangles()<<endl;
	int n = getNumTriangles();
	float *tris_raw = new float[getNumTriangles()*12];
	for(int i=0; i<n;i++)
	{

	 	tris_raw[i*12   ] = tris->at(i).verts[0].x;
      	tris_raw[i*12+1 ] = tris->at(i).verts[0].y;
      	tris_raw[i*12+2 ] = tris->at(i).verts[0].z;
      	tris_raw[i*12+3 ] = tris->at(i).verts[1].x;
      	tris_raw[i*12+4 ] = tris->at(i).verts[1].y;
      	tris_raw[i*12+5 ] = tris->at(i).verts[1].z;
      	tris_raw[i*12+6 ] = tris->at(i).verts[2].x;
      	tris_raw[i*12+7 ] = tris->at(i).verts[2].y;
      	tris_raw[i*12+8 ] = tris->at(i).verts[2].z;
      	tris_raw[i*12+9 ] = tris->at(i).scalars[0]; 						
      	tris_raw[i*12+10] = tris->at(i).scalars[1];
      	tris_raw[i*12+11] = tris->at(i).scalars[2];
  	}

	
	return tris_raw;
}

EAVL_HOSTONLY  inline float* eavlRTScene::getTriangleNormPtr()
{
	if(getNumTriangles()==0) return NULL;
	int n = getNumTriangles();
	float *tris_norms_raw  = new float[getNumTriangles()*9];
	for(int i=0; i<n;i++)
	{
      	tris_norms_raw[i*9  ] = tris->at(i).norms[0].x;
      	tris_norms_raw[i*9+1] = tris->at(i).norms[0].y;
      	tris_norms_raw[i*9+2] = tris->at(i).norms[0].z;
      	tris_norms_raw[i*9+3] = tris->at(i).norms[1].x;
      	tris_norms_raw[i*9+4] = tris->at(i).norms[1].y;
      	tris_norms_raw[i*9+5] = tris->at(i).norms[1].z;
      	tris_norms_raw[i*9+6] = tris->at(i).norms[2].x;
      	tris_norms_raw[i*9+7] = tris->at(i).norms[2].y;
      	tris_norms_raw[i*9+8] = tris->at(i).norms[2].z;
  	}
	return tris_norms_raw;
}

EAVL_HOSTONLY inline float* eavlRTScene::getSpherePtr()
{
	if(getNumSpheres()==0) return NULL;
	float * spheres_raw;
 	int numSpheres= getNumSpheres();
	spheres_raw= new float[numSpheres*4];
	for(int i=0; i< numSpheres ; i++)
	{
		spheres_raw[i*4  ] = spheres->at(i).data[0];
		spheres_raw[i*4+1] = spheres->at(i).data[1];
		spheres_raw[i*4+2] = spheres->at(i).data[2];
		spheres_raw[i*4+3] = spheres->at(i).data[3];
	}
			
 	return spheres_raw;
};

EAVL_HOSTONLY inline int*   eavlRTScene::getSphrMatIdxPtr()
{ 
	if(getNumSpheres()==0) return NULL;
	int* sphrMatIdx;
 	int numSpheres= getNumSpheres();
	sphrMatIdx = new int[numSpheres];
	for(int i=0; i< numSpheres ; i++)
	{
		sphrMatIdx [i]= spheres->at(i).getMatIndex();
	}
	return sphrMatIdx;
};

EAVL_HOSTONLY inline float* eavlRTScene::getMatsPtr()
{
	if(getNumMaterials() ==0) return NULL;
	float* mats_raw;
	int numMats=mats->size();
	
	mats_raw= new float[12*numMats];
	cout<<"Dumping mats "<<numMats<<endl;
	for(int i=0;i<numMats;i++)
	{
		mats_raw[i*12   ] = mats->at(i).ka.x;
		mats_raw[i*12+1 ] = mats->at(i).ka.y;
		mats_raw[i*12+2 ] = mats->at(i).ka.z;
		mats_raw[i*12+3 ] = mats->at(i).kd.x;
		mats_raw[i*12+4 ] = mats->at(i).kd.y;
		mats_raw[i*12+5 ] = mats->at(i).kd.z;
		mats_raw[i*12+6 ] = mats->at(i).ks.x;
		mats_raw[i*12+7 ] = mats->at(i).ks.y;
		mats_raw[i*12+8 ] = mats->at(i).ks.z;
		mats_raw[i*12+9 ] = mats->at(i).shiny;
		mats_raw[i*12+10] = mats->at(i).rs;
		mats_raw[i*12+11] = 0.f;

	}
	cout<<"After mat dump"<<endl;


	return mats_raw;
}

EAVL_HOSTONLY inline int* eavlRTScene::getTriMatIdxsPtr()
{
	if(getNumTriangles()==0) return NULL;
	int n = getNumTriangles();
	int* trisMatIdxs	= new int[getNumTriangles()];
	cout<<"Dumping verts and Mats "<<mats->size()<<endl;
	for(int i=0; i<n;i++)
	{
		trisMatIdxs[i] = tris->at(i).getMatIndex();
  	}
	return trisMatIdxs;
}

EAVL_HOSTONLY inline float eavlRTScene::getSceneExtentMagnitude()
{
	return sqrt(sceneBbox.extent.x*sceneBbox.extent.x+sceneBbox.extent.y*sceneBbox.extent.y+sceneBbox.extent.z*sceneBbox.extent.z);
}

EAVL_HOSTONLY  inline int  eavlRTScene::getTotalPrimitives()
{
	return getNumTriangles()+getNumSpheres();
}
#endif
