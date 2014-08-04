#ifndef EAVL_VOLUME_RENDERER_MUTATOR_H
#define EAVL_VOLUME_RENDERER_MUTATOR_H
#include "eavlFilter.h"
#include "RT/eavlVRScene.h"
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

    void setZoom(float factor)
    {
        camera.zoom = factor;
    }

    void setLookAtPos(float x, float y, float z)
    {
        camera.lookat.x = x;
        camera.lookat.y = y;
        camera.lookat.z = z;
    }

    void setCameraPos(float x, float y, float z)
    {
        camera.position.x = x;
        camera.position.y = y;
        camera.position.z = z;
    }

    void setUp(float x, float y, float z)
    {
        camera.up.x = x;
        camera.up.y = y;
        camera.up.z = z;
    }
    void addTetrahedron(const eavlVector3 &v0, const eavlVector3 &v1, const eavlVector3 &v2, const eavlVector3 &v3,
                                      const float &s0, const float &s1, const float &s2, const float &s3)
    {

        scene.addTet(v0,v1,v2,v3,s0,s1,s2,s3);
    }

    void startScene()
    {

        scene.clear();
        geomDirty = true;
    }

    void setGPU(bool onGPU)
    {
        gpu = onGPU;
    }

    eavlFloatArray* getFrameBuffer() { return frameBuffer; }

    void setSampleDelta(float delta)
    {
        sampleDelta = delta;
    }
    void setColorMap3f(float* cmap,int size);
    void setDefaultColorMap();
    
    virtual void Execute();
     eavlVRScene         scene;
  protected:
    string fieldname;
    bool    gpu;
    int 	height;
    int 	width;
    int 	size;
    float   sampleDelta;
    int 	numTets;
    int     colorMapSize;
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
    eavlFloatArray*     r;
    eavlFloatArray*     g;
    eavlFloatArray*     b;
    eavlFloatArray*     a;
    eavlFloatArray*     frameBuffer;
    eavlIntArray*		indexes;
    eavlIntArray*		mortonIndexes;

    eavlArrayIndexer      *redIndexer;
    eavlArrayIndexer      *greenIndexer;
    eavlArrayIndexer      *blueIndexer;
    eavlArrayIndexer      *alphaIndexer;
    eavlFloatArray        *tempFloat;
   

    float     *tet_verts_raw;
    float     *tet_bvh_in_raw;            /* BVH broken up into inner nodes and leaf nodes */
    float     *tet_bvh_lf_raw;
    float     *color_map_raw;

    void init();
    void allocateArrays();
    void extractGeometry();
    void createRays();
    void freeRaw();
    void freeTextures();
    void clearFrameBuffer(eavlFloatArray *r,eavlFloatArray *g,eavlFloatArray *b, eavlFloatArray* a);
    void scatter();


};
#endif