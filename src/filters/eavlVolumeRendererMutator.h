#ifndef EAVL_VOLUME_RENDERER_MUTATOR_H
#define EAVL_VOLUME_RENDERER_MUTATOR_H
#include "eavlFilter.h"
struct Camera
{
	eavlVector3 	look;
    eavlVector3 	lookat;
    eavlVector3		position;
    eavlVector3		up;
    float 			fovx;
    float 			fovy;
    float			zoom;
};

class eavlVolumeRendererMutator : public eavlMutator
{
  public:
    eavlVolumeRendererMutator();
    void SetField(const string &name)
    {
        fieldname = name;
    }
    void setFOVx(const float d)
    {
      camera.fovx=d; 
    }

    void setFOVy(const float d)
    {
      camera.fovy=d;
    }

    void setResolution(const int h, const int w)
    {
      if(h!=height || width !=w) 
      {
        sizeDirty = true;
      }
      height = h;
      width = w;
    }

    
    virtual void Execute();
  protected:
    string fieldname;
    
    int 	height;
    int 	width;
    int 	size;

    int 	numTets;

    bool	sizeDirty;
    bool 	geomDirty;
    bool    verbose;

    Camera  camera;

    eavlFloatArray*		rayOriginX;
    eavlFloatArray*		rayOriginY;
    eavlFloatArray*		rayOriginZ;
    eavlFloatArray*		rayDirX;
    eavlFloatArray*		rayDirY;
    eavlFloatArray*		rayDirZ;
    eavlIntArray*		indexes;
    eavlIntArray*		mortonIndexes;

    void init();
    void allocateArrays();
    void extractGeometry();
    void createRays();


};
#endif