#ifndef EAVL_RAY_TRACE_MUTATOR_H
#define EAVL_RAY_TRACE_MUTATOR_H
#include "eavlFilter.h"
#include "eavlNewIsoTables.h"   //const Array
#include "eavlVector3.h"
#include "eavlVector3i.h"
#include "eavlMatrix4x4.h"
#include "eavlRTScene.h"
#include "eavlRTUtil.h"
#include "eavlConstTextureArray.h"

class eavlRayTracerMutator : public eavlMutator
{
  public:
    bool cpu;
    eavlRTScene* scene;

    eavlRayTracerMutator();
    void printMemUsage();

    void SetField(const string &name)  
    {
      fieldname = name;
    }

    eavlByteArray* getFrameBuffer() { return frameBuffer; }
    eavlFloatArray* getDepthBuffer(float, float, float);

    void setDefaultMaterial(float ka, float kd, float ks)
    {
      float old_a=scene->getDefaultMaterial().ka.x;
      float old_s=scene->getDefaultMaterial().ka.x;
      float old_d=scene->getDefaultMaterial().ka.x;
      if(old_a==ka && old_d == kd && old_s == ks) return;     //no change, do nothing
      scene->setDefaultMaterial(RTMaterial(eavlVector3(ka,ka,ka),
                                           eavlVector3(kd,kd,kd),
                                           eavlVector3(ks,ks,ks), 10.f,1));
      defaultMatDirty=true;
      cout<<"Changing Default mats"<<endl;
    }
    void setResolution(const int h, const int w)
    {
      if(h!=height || width !=w) 
      {
        cout<<h<<" "<<height<<" "<<w<<" "<<width<<endl;
        sizeDirty=true;
      }
      height=h;
      width=w;
    }

    void setBackgroundColor(float r, float g, float b)
    {
      bgColor.x = min(max(r,0.f), 1.f);
      bgColor.y = min(max(g,0.f), 1.f);
      bgColor.z = min(max(b,0.f), 1.f);
    } 

    void setBackgroundColor(int r, int g, int b)
    {
      bgColor.x = min(max(r/255.f, 0.f), 1.f);
      bgColor.y = min(max(g/255.f, 0.f), 1.f);
      bgColor.z = min(max(b/255.f, 0.f), 1.f);
    } 

    void setDepth(const int d)
    {
      depth = d;
    }

    void setFOVx(const float d)
    {
      fovx = d; 
    }

    void setFOVy(const float d)
    {
      fovy = d;
    }

    void setBVHBuildFast(bool fast)
    {
      fastBVHBuild = true;
    }

    void setBVHCache(bool on)
    {
      useBVHCache=on;
    }

    void setShadowsOn(bool on)
    {
      shadowsOn=on;
    }

    void rotateLight(float xRadians)
    {
      eavlMatrix4x4 rot;
      rot.CreateRotateY(xRadians);
      light=rot*light;
    }

    void setCameraPos(const eavlVector3 pos)
    {
      if( eye.x!=pos.x  || eye.y!=pos.y || eye.z!=pos.z) cameraDirty=true;
      eye=pos;

    }
    void setCameraPos(float x, float y, float z)
    {
      if ( eye.x!=x || eye.y!=y || eye.z!=z ) cameraDirty=true;
      eye.x=x;
      eye.y=y;
      eye.z=z;

    }

    void setUp(float x, float y, float z)
    {
      if ( up.x!=x || up.y!=y || up.z!=z ) cameraDirty=true;
      up.x=x;
      up.y=y;
      up.z=z;

    }
    void setZoom(float _zoom)
    {
      if(zoom != _zoom) cameraDirty = true;
      zoom = _zoom;
    }

    
    void rotateCamera(float xRadians)
    {
      eavlMatrix4x4 rot;
      rot.CreateRotateX(xRadians);
      eye=rot*eye;
      cameraDirty=true;
    }

    void lookAtPos(float x, float y, float z)
    {
      if ( lookat.x!=x || lookat.y!=y || lookat.z!=z ) cameraDirty=true;
      lookat.x=x;
      lookat.y=y;
      lookat.z=z;
    }

    /* there could be a settings dirty, but just set size dirty=true */
    void setAO(bool on)
    {
      if(isOccusionOn!=on) {sizeDirty=true; cout<<"occ dirty"<<endl; }
      isOccusionOn=on;
    }

    void setCompactOp(bool on)
    {
      if (compactOp!=on)
      {
        sizeDirty=true;
        cout<<"compact dirty"<<endl;
      }
      compactOp=on;
    }

    void setVerbose(bool on)
    {
      verbose=on;
    }

    void setAOMax(float m)
    {
      aoMax=m;
    }

    void setOccSamples(int num)
    {
      if(occSamples!=num) sizeDirty=true;
      occSamples=num; 
    }
    
    void setOutputFilename(char * name)
    {
      outfilename=name;
    }


    void setLightParams(float x, float y, float z, float intensity, float _lightCoConst, float _lightCoLinear, float _lightCoExponent)
    {
      light.x=x;
      light.y=y;
      light.z=z;
      lightIntensity=intensity;
      lightCoConst    = _lightCoConst;   //used as coefficients for light attenuation
      lightCoLinear   = _lightCoLinear;
      lightCoExponent = _lightCoExponent;
    }

    void setBVHCacheName(const string &name)
    {
      bvhCacheName = name+".bvh";
    }

    void setAntiAlias(bool isOn)
    {

      //antiAlias=isOn;
    }

    eavlVector3 getCameraPos()
    {

      eavlVector3 camPos=eye;
      return camPos;
    }

    void visualRayTrace(int rayId, const char * outfile);
    void writeSubtree(int);
    void traversalTest(int warmupRounds, int testRounds);
    //void traversalTestISPC(int warmupRounds, int testRounds);
    void fpsTest(int warmupRounds, int testRounds);

    ~eavlRayTracerMutator()
    {
      deleteClassPtr(rayDirX);
      deleteClassPtr(rayDirY);
      deleteClassPtr(rayDirZ);

      deleteClassPtr(rayOriginX);
      deleteClassPtr(rayOriginY);
      deleteClassPtr(rayOriginZ);

      deleteClassPtr(r);
      deleteClassPtr(g);
      deleteClassPtr(b);

      deleteClassPtr(alphas);
      deleteClassPtr(betas);

      deleteClassPtr(interX);
      deleteClassPtr(interY);
      deleteClassPtr(interZ);

      deleteClassPtr(normX);
      deleteClassPtr(normY);
      deleteClassPtr(normZ);

      deleteClassPtr(hitIdx);
      deleteClassPtr(indexes);
      deleteClassPtr(mortonIndexes);
      deleteClassPtr(shadowHits);

      deleteClassPtr(r2);
      deleteClassPtr(b2);
      deleteClassPtr(g2);

      deleteClassPtr(ambPct);
      deleteClassPtr(zBuffer);
      deleteClassPtr(frameBuffer);
      deleteClassPtr(scalars);
      deleteClassPtr(primitiveTypeHit);
      deleteClassPtr(minDistances);

      deleteClassPtr(compactTempInt);
      deleteClassPtr(compactTempFloat);

      deleteClassPtr(redIndexer);
      deleteClassPtr(blueIndexer);
      deleteClassPtr(greenIndexer);
      deleteClassPtr(alphaIndexer);

          //what happens when someone turns these on and off??? 
          //1. we could just always do it and waste memory
          //2 call allocation if we detect dirty settings/dirtySize <-this is what is being done;
      if(antiAlias)
      {
          
          deleteClassPtr(rOut);
          deleteClassPtr(gOut);
          deleteClassPtr(bOut);
      }

      if(isOccusionOn)
      {
           deleteClassPtr(occX);
           deleteClassPtr(occY);
           deleteClassPtr(occZ);
           deleteClassPtr(localHits);
           deleteClassPtr(tempAmbPct);
           deleteClassPtr(occIndexer);
      }
      if (compactOp)
      {
           deleteClassPtr(mask);
           deleteClassPtr(count); 
           deleteClassPtr(indexScan);
      }


      /*Raw arrays*/
      freeRaw();
      //conditional deletes
      freeTextures(); 
    }
    virtual void Execute();
    void setColorMap3f(float*,int);
    void startScene()
    {
      scene->clear();
      geomDirty=true;
    }
  protected:
    string fieldname;

  private:
    int       height;
    int       width;  
    float     zoom;         
    float     fovy;           /*half vertical field of view in degrees*/
    float     fovx;           /*half horizontal field of view in degrees**/
    int       depth;          /*Number of ray bounces*/
    int       size;           /*Size of the ray arrays h*w*/
    int       occSamples;     /*Number of ambient occlusion samples per intersection*/
    int       colorMapSize;   /*Number of values if the color map lookup table*/
    bool      isOccusionOn;   /*True if ambient occlusion is on*/
    int       currentSize;    /*Current working set size if array compaction is on*/
    int       numTriangles;   /*number of triangles*/
    int       numSpheres;
    int       numCyls;
    bool      compactOp;      /*True if array compaction is on*/
    bool      antiAlias;      /*True if anti-aliasing is on*/
    bool      cameraDirty;    /*True is camera parameters are dirty. Used to accumulate AO values when the view is the same*/
    bool      geomDirty;      /*Geometry is Dirty. Rebuild the BVH*/
    bool      sizeDirty;      /*Image size is dirty. Resize the ray arrays*/
    bool      defaultMatDirty;/*Default Material is dirty.*/
    bool      verbose;        /*Turn on print statements*/
    bool      useBVHCache;    /*Turn on print statements*/
    bool      shadowsOn;      /*use shadows*/
    bool      fastBVHBuild;

    float     aoMax;          /* Maximum ambient occulsion ray length*/
    int       sampleCount;    /* keeps a running total of the number of re-usable ambient occlusion samples ie., the camera is unchanged */
    int       numMats;        /* current number of materials*/
    /*Light Distance values*/
    float     lightIntensity; //light attenuation coefficients
    float     lightCoConst;
    float     lightCoLinear;
    float     lightCoExponent;
    
    eavlVector3 bgColor;

    std::ostringstream oss;
    string fileprefix;
    string filetype;
    string bvhCacheName;
    int frameCounter;
    string scounter;
    char* outfilename;

    float *debugPtr;


    //camera vars
    eavlVector3 look;     //Todo: shove this is a struct or class( could  share it with path tracer)
    eavlVector3 up;
    eavlVector3 lookat;
    eavlVector3 eye;
    
    eavlVector3 light;
    eavlVector3 movement;

    eavlFloatArray  *rayDirX;
    eavlFloatArray  *rayDirY;
    eavlFloatArray  *rayDirZ;

    eavlFloatArray  *rayOriginX;
    eavlFloatArray  *rayOriginY;
    eavlFloatArray  *rayOriginZ;

    eavlFloatArray  *reflectX;
    eavlFloatArray  *reflectY;
    eavlFloatArray  *reflectZ;
    eavlFloatArray  *shadowHits;

    eavlFloatArray  *r;
    eavlFloatArray  *g;
    eavlFloatArray  *b;


    eavlFloatArray  *r2;                    /*cmpressed color arrays*/         
    eavlFloatArray  *g2;            
    eavlFloatArray  *b2;

    eavlFloatArray  *rOut;                  /*todo: get rid of these if possible */
    eavlFloatArray  *gOut;
    eavlFloatArray  *bOut;

    eavlFloatArray  *alphas;                /*barycentric coeffients of triangle hit for lerping*/
    eavlFloatArray  *betas;

    eavlFloatArray  *interX;  
    eavlFloatArray  *interY;
    eavlFloatArray  *interZ;

    eavlFloatArray  *normX;                 /*Lerped normals of the hit*/ 
    eavlFloatArray  *normY;
    eavlFloatArray  *normZ;

    /*Ambient occlusion arrays */
    eavlFloatArray  *occX;  
    eavlFloatArray  *occY;
    eavlFloatArray  *occZ;
    eavlFloatArray  *localHits;
    eavlFloatArray  *ambPct;                /*percenatage of ambient light reaching the hit */
    eavlFloatArray  *tempAmbPct;            /*used to accumulate ambient occlusion in multiple passes */ 

    /* Compact Arrays*/
    eavlIntArray    *mask;
    eavlIntArray    *indexScan;
    eavlIntArray    *count;

    eavlIntArray    *hitIdx;                /*index of traingle hit*/
    eavlIntArray    *primitiveTypeHit;      /*type of primitive that was hit*/
    eavlFloatArray  *minDistances;          /*distance to ray hit */
    eavlIntArray    *indexes;               /*pixel  index corresponding to the ray */
    eavlIntArray    *mortonIndexes;         /*indexes of primiary rays sorted in morton order */
    eavlIntArray    *compactTempInt;        /*temp arrays for misc usage */
    eavlFloatArray  *compactTempFloat;
    eavlFloatArray  *zBuffer;
    eavlByteArray   *frameBuffer;           /* RGBARGBA..*/

    /*eavl Array indexers*/
    eavlArrayIndexer      *occIndexer;
    eavlArrayIndexer      *redIndexer;
    eavlArrayIndexer      *greenIndexer;
    eavlArrayIndexer      *blueIndexer;
    eavlArrayIndexer      *alphaIndexer;

    eavlFloatArray        *scalars;         /*lerped intersection scalars */ 
    eavlConstArray<float> *mats;

    eavlConstArray<int>   *tri_matIdx;
    eavlConstArray<float> *tri_norms;

    eavlConstArray<int>   *sphr_matIdx;
    /*  Raw Data Arrays used for eavlConst and eavlConstV2 */
    float     *colorMap_raw;
    float     *tri_verts_raw;                   /* Triangle verts, currenly scalars are stored with the verts*/
    float     *tri_norms_raw;
    float     *tri_bvh_in_raw;            /* BVH broken up into inner nodes and leaf nodes */
    float     *tri_bvh_lf_raw;
    int       *tri_matIdx_raw;

    float     *sphr_verts_raw;
    float     *sphr_bvh_in_raw;            /* BVH broken up into inner nodes and leaf nodes */
    float     *sphr_bvh_lf_raw;
    float     *sphr_scalars_raw;
    int       *sphr_matIdx_raw;

    float     *cyl_verts_raw;
    float     *cyl_bvh_in_raw;            /* BVH broken up into inner nodes and leaf nodes */
    float     *cyl_bvh_lf_raw;
    float     *cyl_scalars_raw;
    int       *cyl_matIdx_raw;

    float     *mats_raw;

    void Init();
    void setDefaultColorMap();
    int  compact();
    void compactFloatArray(eavlFloatArray*& input, eavlIntArray* reverseIndex, int nitems);
    void compactIntArray(eavlIntArray*& input, eavlIntArray* reverseIndex, int nitems);
    void extractGeometry();
    void setCompact(bool);
    void clearFrameBuffer(eavlFloatArray *r,eavlFloatArray *g,eavlFloatArray *b);
    void sort();
    void createRays();
    void allocateArrays();
    void freeTextures();
    void freeRaw();
    void intersect();                        /*find the closest intersection point         */
    void shadowIntersect();                  /*Find any hit between intersect and lights   */
    void occlusionIntersect();               /*Ambient occulsion intersection              */
    void reflect();                          /*Find relfections, lerped normals and scalars*/ 
};
#endif