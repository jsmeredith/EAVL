#ifndef EAVL_RAY_TRACE_MUTATOR_H
#define EAVL_RAY_TRACE_MUTATOR_H
#include "eavlFilter.h"
#include "eavlNewIsoTables.h"   //const Array
#include "eavlVector3.h"
#include "eavlMatrix4x4.h"
#include "RT/eavlRTScene.h"




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


    void setResolution(const int h, const int w)
    {
      if(h!=height || width !=height) 
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

    void setCameraPos(float x, float y, float z)
    {

      eye.x=x;
      eye.y=y;
      eye.z=z;

    }
    void setUp(float x, float y, float z)
    {

      up.x=x;
      up.y=y;
      up.z=z;

    }

    void setRawData(float *v, float *n, int numTri, float* _materials, int * _matIndex, int _numMats)
    {
      //verts_raw=v;
      norms_raw=n;
      numTriangles=numTri;
      mats_raw=_materials;
      matIdx_raw=_matIndex;
      numMats=_numMats;
      //convert the verts into aligned memory
      
      //if there are no normals add them
      if(true)
      {
        cout<<"No norms: extracting"<<endl;
        norms_raw= new float [numTri*9];
        verts_raw= new float [numTri*12];
        //eavlVector3* vec3Ptr=(eavlVector3*)&verts_raw[0];
        for (int i=0; i<numTri; i++)
        {
          eavlVector3 a(v[i*9  ],v[i*9+1],v[i*9+2]);
          eavlVector3 b(v[i*9+3],v[i*9+4],v[i*9+5]);
          eavlVector3 c(v[i*9+6],v[i*9+7],v[i*9+8]);

          verts_raw[i*12   ]=v[i*9];
          verts_raw[i*12+1 ]=v[i*9+1];
          verts_raw[i*12+2 ]=v[i*9+2];
          verts_raw[i*12+3 ]=v[i*9+3];
          verts_raw[i*12+4 ]=v[i*9+4];
          verts_raw[i*12+5 ]=v[i*9+5];
          verts_raw[i*12+6 ]=v[i*9+6];
          verts_raw[i*12+7 ]=v[i*9+7];

          verts_raw[i*12+8 ]=v[i*9+8];
          verts_raw[i*12+9 ]=0; 
          verts_raw[i*12+10]=0;
          verts_raw[i*12+11]=0;

          eavlVector3 norm;
          norm = (b-a)%(c-a);
          //cout<<norm<<endl;
          //if(i==2) cout<<a<<b<<c<<endl;
          norms_raw[i*9  ]=norm.x;
          norms_raw[i*9+1]=norm.y;
          norms_raw[i*9+2]=norm.z;
          norms_raw[i*9+3]=norm.x;
          norms_raw[i*9+4]=norm.y;
          norms_raw[i*9+5]=norm.z;
          norms_raw[i*9+6]=norm.x;
          norms_raw[i*9+7]=norm.y;
          norms_raw[i*9+8]=norm.z;
        }
      }
      //exit(0);
      delete v;
      //debugPtr=v;
    }



    void rotateCamera(float xRadians)
    {
      eavlMatrix4x4 rot;
      rot.CreateRotateX(xRadians);
      eye=rot*eye;
    }

    void rotateLight(float xRadians)
    {
      eavlMatrix4x4 rot;
      rot.CreateRotateY(xRadians);
      light=rot*light;
    }

    void setCameraPos(const eavlVector3 pos)
    {

      eye=pos;

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

    void lookAtPos(float x, float y, float z)
    {
      lookat.x=x;
      lookat.y=y;
      lookat.z=z;
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
      
      delete  verts;
      delete  norms;

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

      delete[] verts_raw;
      delete[] norms_raw;
      delete[] bvhFlatArray_raw;
      delete   zBuffer;

      delete redIndexer;
      delete blueIndexer;
      delete greenIndexer;
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
      
  protected:
    string fieldname;

  private:
    int       height;
    int       width;
    int       occSamples;
    float     fovy;
    float     fovx;
    bool      geomDirty;
    bool      sizeDirty;
    bool      verbose;
    float     isoVal;
    int       depth;
    int       size;
    bool      isOccusionOn;
    int       currentSize;
    int       numTriangles;
    bool      compactOp;
    bool      antiAlias;
    float     aoMax;

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
    eavlVector3 look;
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

    eavlFloatArray    *r;
    eavlFloatArray    *g;
    eavlFloatArray    *b;


    eavlFloatArray    *r2;      //cmpressed color arrays         
    eavlFloatArray    *g2;            
    eavlFloatArray    *b2;

    eavlFloatArray    *rOut;
    eavlFloatArray    *gOut;
    eavlFloatArray    *bOut;

    eavlFloatArray  *alphas;  //barycentric coeffients of triangle hit for lerping
    eavlFloatArray  *betas;

    eavlFloatArray  *interX;  
    eavlFloatArray  *interY;
    eavlFloatArray  *interZ;
    eavlFloatArray  *normX;  
    eavlFloatArray  *normY;
    eavlFloatArray  *normZ;


    eavlFloatArray  *occX;  
    eavlFloatArray  *occY;
    eavlFloatArray  *occZ;
    eavlFloatArray  *localHits;
    eavlFloatArray  *ambPct;

    eavlIntArray    *mask;
    eavlIntArray    *indexScan;
    eavlIntArray    *count;

    eavlIntArray    *hitIdx;    //index of traingle hit
    eavlIntArray    *indexes; //pixels index corresponding to the ray
    eavlIntArray    *mortonIndexes; //
    eavlIntArray    *compactTempInt;
    eavlFloatArray  *compactTempFloat;
    eavlFloatArray  *zBuffer;

    eavlArrayIndexer      *occIndexer;
    eavlConstArray<float> *verts;

    eavlConstArray<float> *norms;
    eavlFloatArray *       frameBuffer;
    eavlArrayIndexer*      redIndexer;
    eavlArrayIndexer*      greenIndexer;
    eavlArrayIndexer*      blueIndexer;
    //mats and colors
    int numMats;
    eavlConstArray<float> *mats;
    eavlConstArray<int> *matIdx;
    int * matIdx_raw;
    float * mats_raw;
    eavlConstArray<float> *bvhFlatArray;

    float *verts_raw;
    float *norms_raw;
    float *bvhFlatArray_raw;

    void Init();
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

};
#endif