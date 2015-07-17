#ifndef EAVL_RAY_TRAINGLE_GEOMETRY
#define EAVL_RAY_TRAINGLE_GEOMETRY
#include "eavlRayGeometry.h"
#include "eavlRTUtil.h"
#include "string.h"
class eavlRayTriangleGeometry : public eavlRayGeometry
{

	protected:
	  bool hasVertexNormals;
	  bool woopify;
	  bool useBVHCache;
	  string cacheName;

	public:
	  eavlTextureObject<float>    *normals;
	  eavlRayTriangleGeometry() 
	  {
	  	geometryType = EAVL_TRI;
	  	hasVertexNormals = false;
	  	woopify 	= true;
	  	normals 	= NULL;
	  	useBVHCache = false;
	  }
	  void woopifyVerts(eavlFloatArray * _vertices, const int &_size);

	  void setBVHCacheName(string _cacheName)
	  {
	  	cacheName = _cacheName + ".bvh";
	  	//useBVHCache = true;
	  }

	  void setVertices(eavlFloatArray *_vertices,const int &_size)
	  {
	  	if(_size > 0) size = _size;
	  	else THROW(eavlException,"Cannot set vertices with size 0");
	  	bool cacheExists  = false;
    	bool writeCache   = false;
		int  innerSize    = 0;
		int  leafSize     = 0;
		float *inner_raw;
	    float *leafs_raw;	  	
		if(useBVHCache)
	  	{
	  		if(useBVHCache)
		    {
		        cacheExists = readBVHCache(inner_raw, innerSize, leafs_raw, leafSize, cacheName.c_str());
		        if(cacheExists) writeCache = false;
		    }
		    else 
		    {
		        writeCache = false;
		    }

		    if(cacheExists)
		    {
		    	bvhInnerNodes = new eavlTextureObject<float4>(innerSize / 4, (float4*)inner_raw, true);
				bvhLeafNodes = new eavlTextureObject<float>(leafSize, leafs_raw, true);
		    }
	  	}
	  	if(!cacheExists)
	  	{
	  		if(fastBuild)
			{   cout<<"Fast build\n";
				MortonBVHBuilder *mortonBVH = new MortonBVHBuilder(_vertices, _size, TRIANGLE);
				mortonBVH->setVerbose(3);
				mortonBVH->build();
				eavlFloatArray *inner = mortonBVH->getInnerNodes();
				eavlFloatArray *leafs = mortonBVH->getLeafNodes(); 
				innerSize = inner->GetNumberOfTuples();
				leafSize = leafs->GetNumberOfTuples();
				bvhInnerNodes = new eavlTextureObject<float4>(innerSize / 4, inner, true);
				bvhLeafNodes = new eavlTextureObject<float>(leafSize , leafs, true);
				delete mortonBVH;
			}
			else
			{
				float *fptr = (float*)_vertices->GetHostArray();
				SplitBVH *sbvh= new SplitBVH(fptr, size, TRIANGLE);

				sbvh->getFlatArray(innerSize, leafSize, inner_raw, leafs_raw);
				bvhInnerNodes = new eavlTextureObject<float4>(innerSize / 4, (float4*)inner_raw, true);
				bvhLeafNodes = new eavlTextureObject<float>(leafSize, leafs_raw, true);  
				delete sbvh;
			}
	  	}
		
		if(writeCache)
		{
			writeBVHCache(inner_raw, innerSize, leafs_raw, leafSize, cacheName.c_str());
		}
        
        if(woopify) woopifyVerts(_vertices, size);
	  }

	  void setScalars(eavlFloatArray * _scalars, const int &size)
	  {
	  	scalars = new eavlTextureObject<float>(size * 3, _scalars, true);
	  }

	  void setNormals(eavlFloatArray * _normals, const int &size)
	  {
	  	normals = new eavlTextureObject<float>(size * 9, _normals, true);
	  }
	  ~eavlRayTriangleGeometry()
	  {
	  	if(normals != NULL) 	delete normals;
	  }
};
#endif