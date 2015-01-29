// Copyright 2010-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RAY_QUERY_MUTATOR_H
#define EAVL_RAY_QUERY_MUTATOR_H
#include "eavlFilter.h"
#include "eavlRTScene.h"
#include "eavlRTUtil.h"
#include "eavlConstTextureArray.h"

class eavlRayQueryMutator : public eavlMutator
{
  public:
    eavlRayQueryMutator();
    
    void SetField(const string &name)
    {
        fieldname = name;
    }

    void setQuerySize(const int _size)
    {
      if(size != _size) 
      {
        sizeDirty = true;
        size = _size;
        allocateArrays();
      }
      
    }

    void setVerbose(bool on)
    {
        verbose = on;
    }

    void useBVHCache(bool on, const string fname)
    {
        cacheBVH = on;
        cacheName = fname;
    }

    void resetWritePtr() { writePtr = 0; }

    void setRay(float dirx, float diry, float dirz, float ox, float oy, float oz)
    {
        if(writePtr < size && size > 0)
        {
            rayDirX->SetValue(writePtr, dirx);
            rayDirY->SetValue(writePtr, diry);
            rayDirZ->SetValue(writePtr, dirz);
            rayOriginX->SetValue(writePtr, ox);
            rayOriginY->SetValue(writePtr, oy);
            rayOriginZ->SetValue(writePtr, oz);
            writePtr++;
        }
        else
        {
            cout<<"Ray buffer already full. Ray not added.\n";
        }
    }

    void setRayAt(int idx,float dirx, float diry, float dirz, float ox, float oy, float oz)
    {
        if(idx < size && size > 0 && idx > -1)
        {
            rayDirX->SetValue(idx, dirx);
            rayDirY->SetValue(idx, diry);
            rayDirZ->SetValue(idx, dirz);
            rayOriginX->SetValue(idx, ox);
            rayOriginY->SetValue(idx, oy);
            rayOriginZ->SetValue(idx, oz);
        }
        else
        {
            cout<<"Ray buffer already full. Ray not added.\n";
        }
    }

    void getIntersectDataAt(int idx, int &hitIdx, float &distance, float &_u, float &_v)
    {
        if(idx < size && size > 0 && idx > -1)
        {
            hitIdx = indexes->GetValue(idx);
            distance = zBuffer->GetValue(idx);
            _u = u->GetValue(idx);
            _v = v->GetValue(idx);
        }
    }

    void getHitAt(int idx, int &hitIdx)
    {
        if(idx < size && size > 0 && idx > -1)
        {
            hitIdx = indexes->GetValue(idx);
            
        }
    }

   
    void addTriangle(const eavlVector3 &v0, const eavlVector3 &v1, const eavlVector3 &v2)
    {

        scene.addTriangle(v0,v1,v2);
    }


    void clearMesh()
    {

        scene.clear();
        geomDirty = true;
    }

    
    void runTest();

    
  virtual void Execute();
     eavlRTScene         scene;
  protected:
    string fieldname;
    bool    cpu;
    bool    cacheBVH;
    string  cacheName;
    int 	size;
    int 	numTris;
    bool	sizeDirty;
    bool 	geomDirty;
    bool    verbose;
    int     writePtr;    /*where the next ray will be added in the array*/

    eavlFloatArray*		rayOriginX;
    eavlFloatArray*		rayOriginY;
    eavlFloatArray*		rayOriginZ;
    eavlFloatArray*		rayDirX;
    eavlFloatArray*		rayDirY;
    eavlFloatArray*		rayDirZ;
    eavlFloatArray*     u;
    eavlFloatArray*     v;

    eavlFloatArray*     zBuffer;
    eavlIntArray*		indexes;

    eavlFloatArray      *tempFloat;
   

    float     *tri_verts_raw;
    float     *tri_bvh_in_raw;            /* BVH broken up into inner nodes and leaf nodes */
    float     *tri_bvh_lf_raw;

    void init();
    void allocateArrays();
    void extractGeometry();
    void createRays();
    void freeRaw();
    void freeTextures();
    void scatter();
    void clearOutput();


};
#endif
