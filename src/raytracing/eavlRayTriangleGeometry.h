#ifndef EAVL_RAY_TRAINGLE_GEOMETRY
#define EAVL_RAY_TRAINGLE_GEOMETRY
#include "eavlRayGeometry.h"
#include "eavlRTUtil.h"
#include "string.h"
class eavlRayTriangleGeometry : public eavlRayGeometry
{

	protected:
	  bool hasVertexNormals;
	  
	  bool useBVHCache;
	  string cacheName;
	  

	public:
	  eavlTextureObject<float>    *normals;
	  eavlTextureObject<float>  *verticesActual;
	  bool woopify;
	  eavlRayTriangleGeometry() 
	  {
	  	geometryType = EAVL_TRI;
	  	hasVertexNormals = false;
	  	woopify 	= false;
	  	normals 	= NULL;
	  	useBVHCache = false;
	  	verticesActual = NULL;
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
	  	if(verticesActual != NULL) delete verticesActual;
	  	if(bvhInnerNodes  != NULL) delete bvhInnerNodes;
	  	if(bvhLeafNodes   != NULL) delete bvhLeafNodes;
	  	bool cacheExists  = false;
    	bool writeCache   = false;
		  int  innerSize    = 0;
		  int  leafSize     = 0;
		  float *inner_raw;
	    int   *leafs_raw;	  	
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
			  bvhLeafNodes = new eavlTextureObject<int>(leafSize, leafs_raw, true);
	      }
    	}
	  	if(!cacheExists)
	  	{
	  		if(fastBuild)
			{  
				MortonBVHBuilder *mortonBVH = new MortonBVHBuilder(_vertices, _size, TRIANGLE);
				mortonBVH->setVerbose(0);
				mortonBVH->build();
				eavlFloatArray *inner = mortonBVH->getInnerNodes();
				eavlIntArray *leafs = mortonBVH->getLeafNodes(); 
				innerSize = inner->GetNumberOfTuples();
				leafSize = leafs->GetNumberOfTuples();
				bvhInnerNodes = new eavlTextureObject<float4>(innerSize / 4, inner, true);
				bvhLeafNodes = new eavlTextureObject<int>(leafSize , leafs, true);
				delete mortonBVH;
			}
			else
			{
				float *fptr = (float*)_vertices->GetHostArray();
				SplitBVH *sbvh= new SplitBVH(fptr, size, TRIANGLE);

				sbvh->getFlatArray(innerSize, leafSize, inner_raw, leafs_raw);
				bvhInnerNodes = new eavlTextureObject<float4>(innerSize / 4, (float4*)inner_raw, true);
				bvhLeafNodes = new eavlTextureObject<int>(leafSize, leafs_raw, true);  
				delete sbvh;
			}
	  	}
		
		if(writeCache)
		{
			writeBVHCache(inner_raw, innerSize, leafs_raw, leafSize, cacheName.c_str());
		}
        
        if(woopify) woopifyVerts(_vertices, size);
        else 
        { 
          verticesActual = new eavlTextureObject<float>(_size * 9, _vertices, false);
        }
	  }

	  void setScalars(eavlFloatArray * _scalars, const int &size)
	  {
	  	if(scalars != NULL) delete scalars;
	  	scalars = new eavlTextureObject<float>(size * 3, _scalars, false);
	  }
	  
	  void setMaterialIds(eavlFloatArray *_matIds, const int &size)
	  {
	  	if(materialIds != NULL) delete materialIds;
	  	materialIds = new eavlTextureObject<int>(size, _matIds, false);
	  }

	  void setNormals(eavlFloatArray * _normals, const int &size)
	  {
	  	if(normals != NULL) delete normals;
	  	normals = new eavlTextureObject<float>(size * 9, _normals, false);
	  }
	  ~eavlRayTriangleGeometry()
	  {
	  	if(normals != NULL) 	delete normals;
	  }
};
#endif
