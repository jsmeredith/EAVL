#ifndef EAVL_RAY_GEOMETRY_H
#define EAVL_RAY_GEOMETRY_H

#include "eavlTextureObject.h"
#include "MortonBVHBuilder.h"
#include "SplitBVH.h"
#include "eavlCell.h"
class eavlRayGeometry
{
	protected:
		int  size;
		bool isDirty;
		bool fastBuild;
		eavlCellShape geometryType;
	public:
		eavlTextureObject<float4>  *vertices;
		eavlTextureObject<int>     *materialIds;
		eavlTextureObject<float>   *scalars;
		eavlTextureObject<float4>  *bvhInnerNodes;
		eavlTextureObject<int>     *bvhLeafNodes;
		eavlRayGeometry()
		{
			isDirty 		= true;
			fastBuild 		= true;
			vertices 		= NULL;
			materialIds 	= NULL;
			bvhLeafNodes 	= NULL;
			bvhInnerNodes 	= NULL;
			scalars   		= NULL;
			size = 0;
		}
		void setVertices(eavlFloatArray *_vertices,const int &_size);
		void setMaterialIds(eavlFloatArray *_matIds,const int &_size);
		void setScalars(eavlFloatArray *_scalars, const int &_size);
		void setFastBuild(bool on)
		{
			fastBuild = on;
		}
		~eavlRayGeometry()
		{
			if(vertices != NULL) 		delete vertices;
			if(materialIds != NULL) 	delete materialIds;
			if(bvhInnerNodes != NULL) 	delete bvhInnerNodes;
			if(bvhLeafNodes != NULL) 	delete bvhLeafNodes;
		}
};
#endif
