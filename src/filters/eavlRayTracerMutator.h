#ifndef EAVL_RAY_TRACE_MUTATOR_H
#define EAVL_RAY_TRACE_MUTATOR_H
#include "eavlFilter.h"
#include "eavlNewIsoTables.h"   //const Array
#include "eavlVector3.h"
#include "eavlMatrix4x4.h"
#include "RT/eavlRTScene.h"

enum primitive_t { TRIANGLE=0, SPHERE=1 };


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

    eavlFloatArray* getFrameBuffer() { return frameBuffer; }
    eavlFloatArray* getDepthBuffer() { return zBuffer; }

    void setDefaultMaterial(float ka, float kd, float ks)
    {
      float old_a=scene->getDefaultMaterial().ka.x;
      float old_s=scene->getDefaultMaterial().ka.x;
      float old_d=scene->getDefaultMaterial().ka.x;
      if(old_a==ka && old_d == kd && old_s == ks) return;     //no change, do nothing
      scene->setDefaultMaterial(RTMaterial(eavlVector3(ka,ka,ka),
                                           eavlVector3(kd,kd,kd),
                                           eavlVector3(ks,ks,ks), 10.f,.3));
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

    void setDepth(const int d)
    {
      depth=d;
    }

    void setFOVx(const float d)
    {
      fovx=d; 
    }

    void setFOVy(const float d)
    {
      fovy=d;
    }

    void setBVHCache(bool on)
    {
      useBVHCache=on;
    }

    void setRawData(float *v, float *n, int numTri, float* _materials, int * _matIndex, int _numMats)
    {
      //tri_verts_raw=v;
      tri_norms_raw=n;
      numTriangles=numTri;
      mats_raw=_materials;
      tri_matIdx_raw=_matIndex;
      numMats=_numMats;
      //convert the verts into aligned memory
      
      //if there are no normals add them
      if(true)
      {
        cout<<"No norms: extracting"<<endl;
        tri_norms_raw= new float [numTri*9];
        tri_verts_raw= new float [numTri*12];
        //eavlVector3* vec3Ptr=(eavlVector3*)&tri_verts_raw[0];
        for (int i=0; i<numTri; i++)
        {
          eavlVector3 a(v[i*9  ],v[i*9+1],v[i*9+2]);
          eavlVector3 b(v[i*9+3],v[i*9+4],v[i*9+5]);
          eavlVector3 c(v[i*9+6],v[i*9+7],v[i*9+8]);

          tri_verts_raw[i*12   ]=v[i*9];
          tri_verts_raw[i*12+1 ]=v[i*9+1];
          tri_verts_raw[i*12+2 ]=v[i*9+2];
          tri_verts_raw[i*12+3 ]=v[i*9+3];
          tri_verts_raw[i*12+4 ]=v[i*9+4];
          tri_verts_raw[i*12+5 ]=v[i*9+5];
          tri_verts_raw[i*12+6 ]=v[i*9+6];
          tri_verts_raw[i*12+7 ]=v[i*9+7];

          tri_verts_raw[i*12+8 ]=v[i*9+8];
          tri_verts_raw[i*12+9 ]=0; 
          tri_verts_raw[i*12+10]=0;
          tri_verts_raw[i*12+11]=0;

          eavlVector3 norm;
          norm = (b-a)%(c-a);
          //cout<<norm<<endl;
          //if(i==2) cout<<a<<b<<c<<endl;
          tri_norms_raw[i*9  ]=norm.x;
          tri_norms_raw[i*9+1]=norm.y;
          tri_norms_raw[i*9+2]=norm.z;
          tri_norms_raw[i*9+3]=norm.x;
          tri_norms_raw[i*9+4]=norm.y;
          tri_norms_raw[i*9+5]=norm.z;
          tri_norms_raw[i*9+6]=norm.x;
          tri_norms_raw[i*9+7]=norm.y;
          tri_norms_raw[i*9+8]=norm.z;
        }
      }
      //exit(0);
      delete v;
      //debugPtr=v;
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
    void fpsTest(int warmupRounds, int testRounds);

    ~eavlRayTracerMutator()
    {

      delete scene;

      delete rayDirX;
      delete rayDirY;
      delete rayDirZ;

      delete  rayOriginX;
      delete  rayOriginY;
      delete  rayOriginZ; 

      delete    r;
      delete    g;
      delete    b;

      delete    r2;
      delete    g2;
      delete    b2;

      delete  mats;

      delete  tri_norms;

      delete  alphas;
      delete  betas;

      delete  interX;
      delete  interY;
      delete  interZ;

      delete  normX;
      delete  normY;
      delete  normZ;

      delete  hitIdx;
      delete  indexes;
      delete  mortonIndexes;

      delete shadowHits;
      delete ambPct;

      delete frameBuffer;

      delete[] tri_verts_raw;
      delete[] tri_norms_raw;
      delete[] tri_bvh_in_raw;
      delete   zBuffer;

      delete redIndexer;
      delete blueIndexer;
      delete greenIndexer;

      delete primitiveTypeHit;
      delete scalars;
      delete tri_bvh_lf_raw;
      //delete bvhFlatArray;
      //conditional deletes
      if (antiAlias)
      {
         //Not using this
         //delete rOut;
         //delete gOut;
         //delete bOut;
      }
      if(isOccusionOn)
      {
        delete localHits;      
        delete occX;
        delete occY;
        delete occZ;
        delete occIndexer;
        delete tempAmbPct;
      }

      if (compactOp)
      {
          delete mask;
          delete count;
          delete indexScan;
      }
      cleanUp();
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
    float     fovy;           /*half vertical field of view in degrees*/
    float     fovx;           /*half horizontal field of view in degrees**/
    int       depth;          /*Number of ray bounces*/
    int       size;           /*Size of the ray arrays h*w*/
    int       occSamples;     /*Number of ambient occlusion samples per intersection*/
    int       colorMapSize;   /*Number of values if the color map lookup table*/
    bool      isOccusionOn;   /*True if ambient occlusion is on*/
    int       currentSize;    /*Current working set size if array compaction is on*/
    int       numTriangles;   /*number of triangles*/
    bool      compactOp;      /*True if array compaction is on*/
    bool      antiAlias;      /*True if anti-aliasing is on*/
    bool      cameraDirty;    /*True is camera parameters are dirty. Used to accumulate AO values when the view is the same*/
    bool      geomDirty;      /*Geometry is Dirty. Rebuild the BVH*/
    bool      sizeDirty;      /*Image size is dirty. Resize the ray arrays*/
    bool      defaultMatDirty;/*Default Material is dirty.*/
    bool      verbose;        /*Turn on print statements*/
    bool      useBVHCache;    /*Turn on print statements*/

    float     aoMax;          /* Maximum ambient occulsion ray length*/
    int       sampleCount;    /* keeps a running total of the number of re-usable ambient occlusion samples ie., the camera is unchanged */
    int       numMats;        /* current number of materials*/
    /*Light Distance values*/
    float     lightIntensity; //light attenuation coefficients
    float     lightCoConst;
    float     lightCoLinear;
    float     lightCoExponent;
    

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
    eavlFloatArray  *frameBuffer;           /* RGBRGB..*/

    /*eavl Array indexers*/
    eavlArrayIndexer      *occIndexer;
    eavlArrayIndexer      *redIndexer;
    eavlArrayIndexer      *greenIndexer;
    eavlArrayIndexer      *blueIndexer;

    eavlFloatArray        *scalars;         /*lerped intersection scalars */ 
    eavlConstArray<float> *mats;
    eavlConstArray<int>   *tri_matIdx;
    eavlConstArray<float> *tri_norms;

    /*  Raw Data Arrays used for eavlConst and eavlConstV2 */
    float     *colorMap_raw;
    float     *tri_verts_raw;                   /* Triangle verts, currenly scalars are stored with the verts*/
    float     *tri_norms_raw;
    float     *tri_bvh_in_raw;            /* BVH broken up into inner nodes and leaf nodes */
    float     *tri_bvh_lf_raw;
    int       *tri_matIdx_raw;
    float     *mats_raw;

    void Init();
    void setDefaultColorMap();
    void writeBMP(int, int, eavlFloatArray*,eavlFloatArray*,eavlFloatArray*,const char*);
    int  compact();
    void compactFloatArray(eavlFloatArray*& input, eavlIntArray* reverseIndex, int nitems);
    void compactIntArray(eavlIntArray*& input, eavlIntArray* reverseIndex, int nitems);
    void extractGeometry();
    void setCompact(bool);
    void clearFrameBuffer(eavlFloatArray *r,eavlFloatArray *g,eavlFloatArray *b);
    void sort();
    void createRays();
    void allocateArrays();
    void cleanUp();
    void intersect();                        /*find the closest intersection point         */
    void shadowIntersect();                  /*Find any hit between intersect and lights   */
    void occlusionIntersect();               /*Ambient occulsion intersection              */
    void reflect();                          /*Find relfections, lerped normals and scalars*/ 
};
#endif