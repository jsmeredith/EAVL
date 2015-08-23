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
    bool        useMaterials;
		BBox 				sceneBbox;
		float 				surfaceArea; 
		eavlFloatArray		*mats;
		eavlFloatArray		*trisVerts;
		eavlFloatArray		*triMatIds;
		eavlFloatArray		*triScalars;
		eavlFloatArray		*triNormals;
		vector<RTSphere>* 	spheres;
		vector<RTCyl>* 	    cyls;
		map<string, int> 	matMap;
		int 				numMats;
		string 				filename;
		RTMaterial			defaultMat;
	public:
		EAVL_HOSTONLY inline eavlRTScene(bool _useMats = true, RTMaterial defualtMaterial= RTMaterial());
		EAVL_HOSTONLY inline ~eavlRTScene();
		EAVL_HOSTONLY inline const  eavlVector3 getSceneExtent();
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
													const  float &scalarV0, const float &scalarV1, const float &scalarV2,  string matName="default");
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,  string matName="default");
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
													const  float &scalarV0, const float &scalarV1, const float &scalarV2,  const int &matId);
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
													const eavlVector3 &n0 , const eavlVector3 &n1, const eavlVector3 &n2,
													const  float &scalarV0, const float &scalarV1, const float &scalarV2,  string matName="default");
		EAVL_HOSTONLY inline void 		addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
													const eavlVector3 &n0 , const eavlVector3 &n1, const eavlVector3 &n2,
													const  float &scalarV0, const float &scalarV1, const float &scalarV2,  const int &matId);
		EAVL_HOSTONLY inline void 		addSphere(const float &radius, const float &centerX, const float &centerY, const float &centerZ, const float _scalar , string matName="default");
		EAVL_HOSTONLY inline void 		addSphere(const float &radius, const float &centerX, const float &centerY, const float &centerZ, const float _scalar , const int &matId);
		EAVL_HOSTONLY inline void 		addLine(const float &radius, const eavlVector3 &p0, const float &s0, const eavlVector3 &p1, const float &s1, string matName="default");
		EAVL_HOSTONLY inline void 		addLine(const float &radius, const eavlVector3 &p0, const float &s0, const eavlVector3 &p1, const float &s1, const int &matId);
		EAVL_HOSTONLY inline void 		setDefaultMaterial(const RTMaterial &_mat);
		EAVL_HOSTONLY inline int 		addMaterial( RTMaterial _mat, string matName);
		EAVL_HOSTONLY inline int 		addMaterial( RTMaterial _mat);
		EAVL_HOSTONLY inline int 		getNumTriangles(){ return trisVerts->GetNumberOfTuples() / 9; };
		EAVL_HOSTONLY inline int 		getNumMaterials(){ return mats->GetNumberOfTuples() / 12; };
		EAVL_HOSTONLY inline int 		getNumSpheres(){ return spheres->size(); };
		EAVL_HOSTONLY inline int 		getNumCyls(){ return cyls->size(); };
		EAVL_HOSTONLY inline int 		getTotalPrimitives();
		EAVL_HOSTONLY inline void       loadObjFile(const char* filename);
		EAVL_HOSTONLY inline void       clear();   												/* clears primitives */
		EAVL_HOSTONLY inline float*     getSpherePtr();
		EAVL_HOSTONLY inline float*     getSphereScalarPtr();
	    EAVL_HOSTONLY inline int*       getSphrMatIdxPtr();
	    EAVL_HOSTONLY inline float*     getCylPtr();
		EAVL_HOSTONLY inline float*     getCylScalarPtr();
	    EAVL_HOSTONLY inline int*       getCylMatIdxPtr();
		EAVL_HOSTONLY inline float 		getSceneExtentMagnitude();
		EAVL_HOSTONLY inline RTMaterial getDefaultMaterial(){ return defaultMat;};
		EAVL_HOSTONLY inline eavlFloatArray* getTrianglePtr();
		EAVL_HOSTONLY inline eavlFloatArray* getTriangleNormPtr();
		EAVL_HOSTONLY inline eavlFloatArray* getTriangleScalarsPtr();
		EAVL_HOSTONLY inline eavlFloatArray* getTriMatIdxsPtr();
		EAVL_HOSTONLY inline eavlFloatArray* getMatsPtr();
		EAVL_HOSTONLY inline BBox*			 getBBox(){ return &sceneBbox; }

};


EAVL_HOSTONLY inline eavlRTScene::eavlRTScene(bool _useMats, RTMaterial defaultMaterial)
{
  useMaterials = _useMats;
	defaultMat = defaultMaterial;
	mats = new eavlFloatArray("",1,0);
	//mats->AddValue(defaultMaterial);
	if(useMaterials)
	{
	  mats->AddValue(defaultMat.ka.x);
	  mats->AddValue(defaultMat.ka.y);
	  mats->AddValue(defaultMat.ka.z);
	  mats->AddValue(defaultMat.kd.x);
	  mats->AddValue(defaultMat.kd.y);
	  mats->AddValue(defaultMat.kd.z);
	  mats->AddValue(defaultMat.ks.x);
	  mats->AddValue(defaultMat.ks.y);
	  mats->AddValue(defaultMat.ks.z);
	  mats->AddValue(defaultMat.shiny);
	  mats->AddValue(defaultMat.rs);
	  mats->AddValue(0.f);
	  matMap.insert(pair<string, int> ("default", 0));
	  numMats = 1;
	 }
	

	trisVerts    = new eavlFloatArray("",1,0);
	triMatIds   = new eavlFloatArray("",1,0);
	triScalars   = new eavlFloatArray("",1,0);
	triNormals   = new eavlFloatArray("",1,0);
	spheres = new vector<RTSphere>();
	cyls    = new vector<RTCyl>();
}

EAVL_HOSTONLY inline eavlRTScene::~eavlRTScene()
{
	delete mats;
	delete trisVerts;
	delete triMatIds;
	delete triScalars;
	delete triNormals;
	delete spheres;
	delete cyls;
}

EAVL_HOSTONLY inline void eavlRTScene::setDefaultMaterial(const RTMaterial &_mat)
{
  if(useMaterials)
  {
	  mats->SetValue(0,_mat.ka.x);
	  mats->SetValue(1,_mat.ka.y);
	  mats->SetValue(2,_mat.ka.z);
	  mats->SetValue(3,_mat.kd.x);
	  mats->SetValue(4,_mat.kd.y);
	  mats->SetValue(5,_mat.kd.z);
	  mats->SetValue(6,_mat.ks.x);
	  mats->SetValue(7,_mat.ks.y);
	  mats->SetValue(8,_mat.ks.z);
	  mats->SetValue(9,_mat.shiny);
	  mats->SetValue(10,_mat.rs);
  }
  else cerr<<"Material has no effect"<<endl;

	defaultMat=_mat;
}

EAVL_HOSTONLY inline int eavlRTScene::addMaterial(RTMaterial _mat, string matName)
{
  if(useMaterials)
  {
	  int idx=numMats;
	  matMap.insert(pair<string, int>(matName, numMats));
	  mats->AddValue(_mat.ka.x);
	  mats->AddValue(_mat.ka.y);
	  mats->AddValue(_mat.ka.z);
	  mats->AddValue(_mat.kd.x);
	  mats->AddValue(_mat.kd.y);
	  mats->AddValue(_mat.kd.z);
	  mats->AddValue(_mat.ks.x);
	  mats->AddValue(_mat.ks.y);
	  mats->AddValue(_mat.ks.z);
	  mats->AddValue(_mat.shiny);
	  mats->AddValue(_mat.rs);
	  mats->AddValue(0.f);
	  numMats++;
	  return idx;
	}
  else cerr<<"Material has no effect"<<endl;
  return 0;
}

EAVL_HOSTONLY inline int eavlRTScene::addMaterial( RTMaterial _mat)
{
	if(useMaterials)
  {
	  int idx=numMats;
	  mats->AddValue(_mat.ka.x);
	  mats->AddValue(_mat.ka.y);
	  mats->AddValue(_mat.ka.z);
	  mats->AddValue(_mat.kd.x);
	  mats->AddValue(_mat.kd.y);
	  mats->AddValue(_mat.kd.z);
	  mats->AddValue(_mat.ks.x);
	  mats->AddValue(_mat.ks.y);
	  mats->AddValue(_mat.ks.z);
	  mats->AddValue(_mat.shiny);
	  mats->AddValue(_mat.rs);
	  mats->AddValue(0.f);
	  numMats++;
	  return idx;
	}
  else cerr<<"Material has no effect"<<endl;
  return 0;
}

EAVL_HOSTONLY inline const  eavlVector3 eavlRTScene::getSceneExtent() { return sceneBbox.extent; }

EAVL_HOSTONLY inline void eavlRTScene::addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
										    const float &scalarV0 , const float &scalarV1, const float &scalarV2,  string matName)
{
  
	int matId=0;
	
	if(useMaterials && matName!="default")
	{
		if ( matMap.find(matName) != matMap.end() ) 
		{
  			matId=matMap[matName];
		} 
	}
	trisVerts->AddValue(v0.x);
	trisVerts->AddValue(v0.y);
	trisVerts->AddValue(v0.z);
	trisVerts->AddValue(v1.x);
	trisVerts->AddValue(v1.y);
	trisVerts->AddValue(v1.z);
	trisVerts->AddValue(v2.x);
	trisVerts->AddValue(v2.y);
	trisVerts->AddValue(v2.z);
	eavlVector3 normal = (v1 - v0) % (v2 - v0);
	for (int i = 0; i < 3; ++i)
	{
		triNormals->AddValue(normal.x);
		triNormals->AddValue(normal.y);
		triNormals->AddValue(normal.z);
	}
	triScalars->AddValue(scalarV0);
	triScalars->AddValue(scalarV1);
	triScalars->AddValue(scalarV2);
	if(useMaterials) triMatIds->AddValue(matId);
	BBox bbox;
	bbox.expandToInclude(v0);
	bbox.expandToInclude(v1);
	bbox.expandToInclude(v2);
	sceneBbox.expandToInclude(bbox);  //TODO: get rid of this
}

EAVL_HOSTONLY inline void eavlRTScene::addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2, string matName)
{

	int matId=0;
	if(useMaterials)
	  if(matName!="default")
	  {
		  if ( matMap.find(matName) != matMap.end() ) 
		  {
    			matId=matMap[matName];
		  } 
	  }

	trisVerts->AddValue(v0.x);
	trisVerts->AddValue(v0.y);
	trisVerts->AddValue(v0.z);
	trisVerts->AddValue(v1.x);
	trisVerts->AddValue(v1.y);
	trisVerts->AddValue(v1.z);
	trisVerts->AddValue(v2.x);
	trisVerts->AddValue(v2.y);
	trisVerts->AddValue(v2.z);
	eavlVector3 normal = (v1 - v0) % (v2 - v0);
	for (int i = 0; i < 3; ++i)
	{
		triNormals->AddValue(normal.x);
		triNormals->AddValue(normal.y);
		triNormals->AddValue(normal.z);
	}
	triScalars->AddValue(0.f);
	triScalars->AddValue(0.f);
	triScalars->AddValue(0.f);
	if(useMaterials) triMatIds->AddValue(matId);
	BBox bbox;
	bbox.expandToInclude(v0);
	bbox.expandToInclude(v1);
	bbox.expandToInclude(v2);
	sceneBbox.expandToInclude(bbox);  //TODO: get rid of this
}

EAVL_HOSTONLY inline void eavlRTScene::addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
										    const float &scalarV0 , const float &scalarV1, const float &scalarV2, const int &matId)
{
	trisVerts->AddValue(v0.x);
	trisVerts->AddValue(v0.y);
	trisVerts->AddValue(v0.z);
	trisVerts->AddValue(v1.x);
	trisVerts->AddValue(v1.y);
	trisVerts->AddValue(v1.z);
	trisVerts->AddValue(v2.x);
	trisVerts->AddValue(v2.y);
	trisVerts->AddValue(v2.z);
	eavlVector3 normal = (v1 - v0) % (v2 - v0);
	for (int i = 0; i < 3; ++i)
	{
		triNormals->AddValue(normal.x);
		triNormals->AddValue(normal.y);
		triNormals->AddValue(normal.z);
	}
	triScalars->AddValue(scalarV0);
	triScalars->AddValue(scalarV1);
	triScalars->AddValue(scalarV2);
	if(useMaterials) triMatIds->AddValue(matId);
	BBox bbox;
	bbox.expandToInclude(v0);
	bbox.expandToInclude(v1);
	bbox.expandToInclude(v2);
	sceneBbox.expandToInclude(bbox);  //TODO: get rid of this
}

EAVL_HOSTONLY inline void eavlRTScene::addTriangle(const eavlVector3 &v0 , const eavlVector3 &v1, const eavlVector3 &v2,
											const eavlVector3 &n0 , const eavlVector3 &n1, const eavlVector3 &n2,
											const float &scalarV0 , const float &scalarV1, const float &scalarV2,  string matName)
{

	int matId=0;
	if(useMaterials)
	  if(matName!="default")
	  {
		  if ( matMap.find(matName) != matMap.end() ) 
		  {
    			matId=matMap[matName];
		  } 
	  }
	trisVerts->AddValue(v0.x);
	trisVerts->AddValue(v0.y);
	trisVerts->AddValue(v0.z);
	trisVerts->AddValue(v1.x);
	trisVerts->AddValue(v1.y);
	trisVerts->AddValue(v1.z);
	trisVerts->AddValue(v2.x);
	trisVerts->AddValue(v2.y);
	trisVerts->AddValue(v2.z);
	triNormals->AddValue(n0.x);
	triNormals->AddValue(n0.y);
	triNormals->AddValue(n0.z);
	triNormals->AddValue(n1.x);
	triNormals->AddValue(n1.y);
	triNormals->AddValue(n1.z);
	triNormals->AddValue(n2.x);
	triNormals->AddValue(n2.y);
	triNormals->AddValue(n2.z);
	if(useMaterials) triMatIds->AddValue(matId);
	triScalars->AddValue(scalarV0);
	triScalars->AddValue(scalarV1);
	triScalars->AddValue(scalarV2);
	BBox bbox;
	bbox.expandToInclude(v0);
	bbox.expandToInclude(v1);
	bbox.expandToInclude(v2);
	sceneBbox.expandToInclude(bbox);  //TODO: get rid of this
}

EAVL_HOSTONLY  inline void eavlRTScene::addSphere(const float &radius, const float &centerX, const float &centerY, const float &centerZ, const float _scalar, string matName)
{
	int matId=0;
	if(matName!="default")
	{
		if ( matMap.find(matName) != matMap.end() ) 
		{
  			matId=matMap[matName];
		} 
	}

	RTSphere t ( radius, eavlVector3(centerX, centerY, centerZ),_scalar,matId );
	spheres->push_back( t );
	sceneBbox.expandToInclude(t.getBBox());
}

EAVL_HOSTONLY  inline void eavlRTScene::addSphere(const float &radius, const float &centerX, const float &centerY, const float &centerZ , const float _scalar, const int &matId)
{
	

	RTSphere t ( radius, eavlVector3(centerX, centerY, centerZ),_scalar, matId );
	spheres->push_back( t );
	sceneBbox.expandToInclude(t.getBBox());
}

EAVL_HOSTONLY inline void eavlRTScene::addLine(const float &radius, const eavlVector3 &p0, const float &s0, const eavlVector3 &p1, const float &s1, string matName)
{
	int matId=0;
	if(matName!="default")
	{
		if ( matMap.find(matName) != matMap.end() ) 
		{
  			matId=matMap[matName];
		} 
	}

	eavlVector3 axis = p1 - p0;
 	float h = sqrt(axis * axis);
	RTCyl t(radius, p0, h, axis, s0, s1, matId);
	cyls->push_back( t );
	sceneBbox.expandToInclude(t.getBBox());
}

EAVL_HOSTONLY inline void eavlRTScene::addLine(const float &radius, const eavlVector3 &p0, const float &s0, const eavlVector3 &p1, const float &s1, const int & matId)
{
	eavlVector3 axis = p1 - p0;
 	float h = sqrt(axis * axis);
	RTCyl t(radius, p0, h, axis, s0, s1, matId);
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
	delete trisVerts;
	delete triMatIds;
	delete triScalars;
	delete triNormals;
	delete mats;
	mats = new eavlFloatArray("",1,0);
	trisVerts    = new eavlFloatArray("",1,0);
	triMatIds   = new eavlFloatArray("",1,0);
	triScalars   = new eavlFloatArray("",1,0);
	triNormals   = new eavlFloatArray("",1,0);
	mats->AddValue(defaultMat.ka.x);
	mats->AddValue(defaultMat.ka.y);
	mats->AddValue(defaultMat.ka.z);
	mats->AddValue(defaultMat.kd.x);
	mats->AddValue(defaultMat.kd.y);
	mats->AddValue(defaultMat.kd.z);
	mats->AddValue(defaultMat.ks.x);
	mats->AddValue(defaultMat.ks.y);
	mats->AddValue(defaultMat.ks.z);
	mats->AddValue(defaultMat.shiny);
	mats->AddValue(defaultMat.rs);
	mats->AddValue(0.f);
	spheres->clear();
	matMap.clear();
	/* Load the defualt materials back in*/
	matMap.insert(pair<string, int> ("default", 0));
}

EAVL_HOSTONLY inline eavlFloatArray* eavlRTScene::getTrianglePtr()
{
	return trisVerts;
}

EAVL_HOSTONLY inline eavlFloatArray* eavlRTScene::getTriangleScalarsPtr()
{
	return triScalars;
}

EAVL_HOSTONLY  inline eavlFloatArray* eavlRTScene::getTriangleNormPtr()
{
	return triNormals;
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

EAVL_HOSTONLY inline float* eavlRTScene::getCylPtr()
{
	if(getNumCyls()==0) return NULL;
	float * cyls_raw;
 	int numCyls = getNumCyls();
	cyls_raw= new float[numCyls*8];
	for(int i=0; i< numCyls ; i++)
	{
		cyls_raw[i*8  ] = cyls->at(i).data[0]; //BasePoint
		cyls_raw[i*8+1] = cyls->at(i).data[1];
		cyls_raw[i*8+2] = cyls->at(i).data[2];
		cyls_raw[i*8+3] = cyls->at(i).data[3]; //radius
		cyls_raw[i*8+4] = cyls->at(i).data[4]; //axis
		cyls_raw[i*8+5] = cyls->at(i).data[5];
		cyls_raw[i*8+6] = cyls->at(i).data[6];
		cyls_raw[i*8+7] = cyls->at(i).data[7]; //height
	}
			
 	return cyls_raw;
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

EAVL_HOSTONLY inline int*   eavlRTScene::getCylMatIdxPtr()
{ 
	if(getNumCyls()==0) return NULL;
	int* cylMatIdx;
 	int numCyls= getNumCyls();
	cylMatIdx = new int[numCyls];
	for(int i=0; i< numCyls ; i++)
	{
		cylMatIdx [i]= cyls->at(i).getMatIndex();
	}
	return cylMatIdx;
};

EAVL_HOSTONLY inline eavlFloatArray* eavlRTScene::getMatsPtr()
{
  if(useMaterials) return mats;
  else return NULL;
}

EAVL_HOSTONLY inline eavlFloatArray* eavlRTScene::getTriMatIdxsPtr()
{
	return triMatIds;
}

EAVL_HOSTONLY inline float eavlRTScene::getSceneExtentMagnitude()
{
	return sqrt(sceneBbox.extent.x*sceneBbox.extent.x+sceneBbox.extent.y*sceneBbox.extent.y+sceneBbox.extent.z*sceneBbox.extent.z);
}

EAVL_HOSTONLY  inline int  eavlRTScene::getTotalPrimitives()
{
	return getNumTriangles()+getNumSpheres()+getNumCyls();
}

EAVL_HOSTONLY inline float* eavlRTScene::getSphereScalarPtr()
{
	if(getNumSpheres()==0) return NULL;
	float* sphrScalars;
 	int numSpheres= getNumSpheres();
	sphrScalars = new float[numSpheres];
	for(int i=0; i< numSpheres ; i++)
	{
		sphrScalars [i]= spheres->at(i).scalar;
	}
	return sphrScalars;
}

EAVL_HOSTONLY inline float* eavlRTScene::getCylScalarPtr()
{
	if(getNumCyls()==0) return NULL;
	float* cylScalars;
 	int numCyls= getNumCyls();
	cylScalars = new float[numCyls*2];
	for(int i=0; i< numCyls ; i++)
	{
		cylScalars [i*2  ]= cyls->at(i).scalar[0];
		cylScalars [i*2+1]= cyls->at(i).scalar[1];
		//cout<<"S "<<cyls->at(i).scalar[0]<<" "<<cyls->at(i).scalar[1]<<endl;
	}
	return cylScalars;
}
#endif
