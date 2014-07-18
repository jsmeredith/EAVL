#include "eavlException.h"
#include "eavlExecutor.h"

#include "eavlMapOp.h"
#include "eavlFilter.h"
#include "eavlTimer.h" 
#include <string.h>
#include <sys/time.h>
#include <ctime> 
#include <cstdlib>
 
#include "eavlExecutor.h"

#include "eavlRayTracerMutator.h"
#include "eavlNewIsoTables.h" // need this for eavl const array

#include "eavlGatherOp.h"
#include "eavlScatterOp.h"
#include "eavlReduceOp_1.h"
#include "eavlMapOp.h"

#include "eavlPrefixSumOp_1.h"
#include "eavlSimpleReverseIndexOp.h"

#include <list>

//#include "eavlAliasStencilOp.h" not using aa right now
#include "eavl1toNScatterOp.h"
#include "eavlNto1GatherOp.h"
#include "RT/SplitBVH.h"

#include <sstream>
#include <iostream>
#include <fstream>


#define TOLERANCE   0.00001
#define BARY_TOLE   0.0001f
#define EPSILON     0.0001f
#define PI          3.14159265359f
#define INFINITE    1000000
#define END_FLAG    -1000000000





#define FILE_LEAF -100001

//declare the texture reference even if we are not using texture memory
texture<float4> bvhInnerTexRef;
texture<float4> vertsTexRef;
texture<float>  bvhLeafTexRef;

#define USE_TEXTURE_MEM
template<class T>
class eavlConstArrayV2
{
  public:
    T *host;
    T *device;
  public:
    eavlConstArrayV2(T *from, int N, texture<T> &g_textureRef)
    {
        host = from;

#ifdef HAVE_CUDA
        int nbytes = N * sizeof(T);
        cudaMalloc((void**)&device, nbytes);
        CUDA_CHECK_ERROR();
        cudaMemcpy(device, &(host[0]),
                   nbytes, cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();
#ifdef USE_TEXTURE_MEM
        cudaBindTexture(0, g_textureRef,device,nbytes);
        CUDA_CHECK_ERROR();
#endif

#endif
    }
#ifdef __CUDA_ARCH__
#ifdef USE_TEXTURE_MEM
    EAVL_DEVICEONLY  const T getValue(texture<T> g_textureRef, int index) const
    {
        return tex1Dfetch(g_textureRef, index);
    }
#else
    EAVL_DEVICEONLY const T &getValue(texture<T> g_textureRef, int index) const
    {
        return device[index];
    }
#endif
#else
    EAVL_HOSTONLY const T &getValue(texture<T> g_textureRef, int index) const
    {
        return host[index];
    }
#endif
};

eavlConstArrayV2<float4>* bvhTex;
eavlConstArrayV2<float4>* vertsTex;
eavlConstArrayV2<float>*  bvhLeafsTex;

void writeBVHCache(const float *innerNodes, const int innerSize, const float * leafNodes, const int leafSize, const char* filename )
{
    cout<<"Writing BVH to cache"<<endl;
    ofstream bvhcache(filename, ios::out |ios::binary);
    if(bvhcache.is_open())
    {
        bvhcache.write((char*)&innerSize, sizeof(innerSize));
        bvhcache.write((const char*)innerNodes, sizeof(float)*innerSize);

        bvhcache.write((char*)&leafSize, sizeof(leafSize));
        bvhcache.write((const char*)leafNodes, sizeof(float)*leafSize);
    }
    else
    {
        cerr<<"Error. Could not open file "<<filename<<" for storing bvh cache."<<endl;
    }
    bvhcache.close();
    
}

bool readBVHCache(float *&innerNodes, int &innerSize, float *&leafNodes, int &leafSize, const char* filename )
{
    ifstream bvhcache(filename, ios::in |ios::binary);
    if(bvhcache.is_open())
    {
        cout<<"Reading BVH Cache"<<endl;
        bvhcache.read((char*)&innerSize, sizeof(innerSize));
        if(innerSize<0) 
        {
            cerr<<"Invalid inner node array size "<<innerSize<<endl;
            bvhcache.close();
            return false;
        }
        innerNodes= new float[innerSize];
        bvhcache.read((char*)innerNodes, sizeof(float)*innerSize);

        bvhcache.read((char*)&leafSize, sizeof(leafSize));
        if(leafSize<0) 
        {
            cerr<<"Invalid leaf array size "<<leafSize<<endl;
            bvhcache.close();
            delete innerNodes;
            return false;
        }

        leafNodes= new float[leafSize];
        bvhcache.read((char*)leafNodes, sizeof(float)*leafSize);
    }
    else
    {
        cerr<<"Could not open file "<<filename<<" for reading bvh cache. Rebuilding..."<<endl;
        bvhcache.close();
        return false;
    }

    bvhcache.close();
    return true;
}


EAVL_HOSTDEVICE float rcp(float f){ return 1.0f/f;}
EAVL_HOSTDEVICE float rcp_safe(float f) { return rcp((fabs(f) < 1e-8f) ? 1e-8f : f); }

int test;
eavlRayTracerMutator::eavlRayTracerMutator()
{
    scene= new eavlRTScene(RTMaterial());
    height  =1080;         //set up some defaults
    width   =1920;
    depth   =0;
 
    fovy    =30;
    fovx    =50;
    srand (time(NULL));   //currently not used 
    look.x  =0;
    look.y  =0;
    look.z  =-1;

    lookat.x=.001;
    lookat.y=0;
    lookat.z=-30;

    eye.x   =0;
    eye.y   =0;
    eye.z   =-20;

    light.x =0;
    light.y =0;
    light.z =-20;

    compactOp   =false;
    geomDirty   =true;
    antiAlias   =false;
    sizeDirty   =true;

    fileprefix  ="output";
    filetype    =".bmp";
    outfilename ="output.bmp";
    frameCounter=0;
    isOccusionOn=false;
    occSamples  =4;
    aoMax       =1.f;
    verbose     =false;
    rayDirX     =NULL;
    redIndexer  = new eavlArrayIndexer(3,0);
    greenIndexer= new eavlArrayIndexer(3,1);
    blueIndexer = new eavlArrayIndexer(3,2);
    cout<<"Construtor Done. Dirty"<<endl;
}



void eavlRayTracerMutator::setCompact(bool comp)
{
    compactOp=comp;
}



struct RNG
{
   unsigned long x, y, z;

    EAVL_HOSTDEVICE RNG(long seed)
     {
        x=123456789*seed;
        y=362436069;
        z=521288629;
     }
     EAVL_HOSTDEVICE float rand(void) {          //period 2^96-1
        unsigned long t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

       t = x;
       x = y;
       y = z;
       z = t ^ x ^ y;

      return (float) x/((~0UL>>1)*1.0) - 1.0;
    }
};


EAVL_HOSTDEVICE float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}
struct OccRayGenFunctor
{   
    OccRayGenFunctor()
        
    {}

    EAVL_FUNCTOR tuple<float,float,float,float,float,float,int> operator()(tuple<float,float,float,float,float,float,int>input, int seed){
        RNG rng(get<0>(input)+seed);
        eavlVector3 normal(get<0>(input),get<1>(input),get<2>(input));
        eavlVector3 dir;
        //rng warm up
        float out=rng.rand();
        for(int i=0; i<25;i++) out=rng.rand();
        
        while(true)
        {
            dir.x=rng.rand();
            dir.y=rng.rand();
            dir.z=rng.rand();

            //dir.x=RandomFloat(-1,1);
            //dir.y=RandomFloat(-1,1);
            //dir.z=RandomFloat(-1,1);

            if(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z >1) continue;
            if(dir*normal<0) continue;
            dir.normalize();
            return tuple<float,float,float,float,float,float,int>(dir.x,dir.y,dir.z,get<3>(input),get<4>(input),get<5>(input),get<6>(input));
        }



        
    }

   

};


/* Next two functions use code adapted from NVIDIA rayGen kernels see headers in RT/bvh/ for full copyright information */
EAVL_HOSTDEVICE void jenkinsMix(unsigned int & a, unsigned int & b, unsigned int & c)
{
    a -= b; a -= c; a ^= (c>>13);
    b -= c; b -= a; b ^= (a<<8);
    c -= a; c -= b; c ^= (b>>13);
    a -= b; a -= c; a ^= (c>>12);
    b -= c; b -= a; b ^= (a<<16);
    c -= a; c -= b; c ^= (b>>5);
    a -= b; a -= c; a ^= (c>>3);
    b -= c; b -= a; b ^= (a<<10);
    c -= a; c -= b; c ^= (b>>15);   
}
struct OccRayGenFunctor2
{   
    OccRayGenFunctor2(){}

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<float,float,float,float,float,float,int>input, int seed, int sampleNum){
        int hitIdx=get<6>(input);
        if(hitIdx==-1) tuple<float,float,float>(0.f,0.f,0.f);
        eavlVector3 normal(get<0>(input),get<1>(input),get<2>(input));
        
        //rng warm up
        eavlVector3 absNormal;
        absNormal.x=abs(normal.x);
        absNormal.y=abs(normal.y);
        absNormal.z=abs(normal.z);
        
        float maxN=max(max(absNormal.x,absNormal.y),absNormal.z);
        eavlVector3 perp=eavlVector3(normal.y,-normal.x,0.f);
        if(maxN==absNormal.z)  
        {
            perp.x=0.f;
            perp.y=normal.z;
            perp.z=-normal.y;
        }
        else if (maxN==absNormal.x)
        {
            perp.x=-normal.z;
            perp.y=0.f;
            perp.z=normal.x;
        }
        perp.normalize(); 

        eavlVector3 biperp= normal%perp;

        unsigned int hashA =6371625+seed;
        unsigned int hashB =0x9e3779b9u;
        unsigned int hashC =0x9e3779b9u;
        jenkinsMix(hashA, hashB, hashC);
        jenkinsMix(hashA, hashB, hashC);
        float angle=2.f*PI*(float)hashC*exp2(-32.f);
        eavlVector3 t0= perp*cosf(angle)+biperp*sinf(angle);
        eavlVector3 t1= perp* -sinf(angle)+biperp*cosf(angle);

        float  x = 0.0f;
        float  xadd = 1.0f;
        unsigned int hc2 = 1+sampleNum;
        while (hc2 != 0)
        {
            xadd *= 0.5f;
            if ((hc2 & 1) != 0)
                x += xadd;
            hc2 >>= 1;
        }

        float  y = 0.0f;
        float  yadd = 1.0f;
        int hc3 = 1+sampleNum;
        while (hc3 != 0)
        {
            yadd *= 1.0f / 3.0f;
            y += (float)(hc3 % 3) * yadd;
            hc3 /= 3;
        }


        float angle2 = 2.0f * PI * y;
        float r = sqrtf(x);
        x = r * cosf(angle2);
        y = r * sinf(angle2);
        float z = sqrtf(1.0f - x * x - y * y);
        eavlVector3 dir=eavlVector3( t0*x  +  t1*y +  normal*z);
        dir.normalize();
        //if(1087534==get<7>(input)) cout<<"DIR "<<dir<<" "<<t0<<" "<<x<<" "<<t1<<" "<<y<<" "<<angle<<" "<<normal<<endl;
        return tuple<float,float,float>(dir.x,dir.y,dir.z);
    }

   

};

#define INIT(TYPE, TABLE, COUNT)            \
{                                           \
TABLE = new TYPE(TABLE ## _raw, COUNT);     \
}



struct RayGenFunctor
{
    int w;
    int h; 
    eavlVector3 nlook;// normalized look
    eavlVector3 delta_x;
    eavlVector3 delta_y;
    RayGenFunctor(int width, int height, float half_fovX, float half_fovY, eavlVector3 look, eavlVector3 up)
        : h(height), w(width)
    {
        float thx=tan(half_fovX*PI/180);
        float thy=tan(half_fovY*PI/180);

        //eavlVector3 ru= look%up;// % == cross
        eavlVector3 ru= up%look;
        ru.normalize();
        //eavlVector3 rv= look%ru;
        eavlVector3 rv= ru%look;
        rv.normalize();

        delta_x=ru*(2*thx/(float)w);
        delta_y=rv*(2*thy/(float)h);

        nlook.x=look.x;
        nlook.y=look.y;
        nlook.z=look.z;
        nlook.normalize();

    }

    EAVL_FUNCTOR tuple<float,float, float> operator()(int idx){
        int i=idx%w;
        int j=idx/w;

        eavlVector3 ray_dir=nlook+delta_x*((2*i-w)/2.0f)+delta_y*((2*j-h)/2.0f);
        ray_dir.normalize();

        return tuple<float,float,float>(ray_dir.x,ray_dir.y,ray_dir.z);

    }

};
/*----------------------Utility Functors---------------------------------- */
// 1 value to 3 arrays this uses a dummy array as input;
struct FloatMemsetFunctor1to3
{
    float value;
    FloatMemsetFunctor1to3(const float v)
        : value(v)
    {}

    EAVL_FUNCTOR tuple<float,float,float> operator()(float r){
        return tuple<float,float,float>(value,value,value);
    }

   

};

struct FloatMemsetFunctor
{
    float value;
    FloatMemsetFunctor(const float v)
        : value(v)
    {}

    EAVL_FUNCTOR tuple<float> operator()(float r){
        return tuple<float>(value);
    }

   

};

struct IntMemsetFunctor
{
    const int value;
    IntMemsetFunctor(const int v)
        : value(v)
    {}

    EAVL_FUNCTOR tuple<int> operator()(int dummy){
        return tuple<int>(value);
    }

   

};

//three values to three arrays
struct FloatMemsetFunctor3to3
{
    const float value1;
    const float value2;
    const float value3;
    FloatMemsetFunctor3to3(const float v1,const float v2, const float v3)
        : value1(v1),value2(v2),value3(v3)
    {}

    EAVL_FUNCTOR tuple<float,float,float> operator()(float r){
        return tuple<float,float,float>(value1,value2,value3);
    }

   

};

struct FloatMemcpyFunctor3to3
{
    FloatMemcpyFunctor3to3(){}

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<float,float,float> input){
        return tuple<float,float,float>(get<0>(input),get<1>(input),get<2>(input));
    }

   

};

struct FloatMemcpyFunctor1to1
{
    FloatMemcpyFunctor1to1(){}

    EAVL_FUNCTOR tuple<float> operator()(tuple<float> input){
        return tuple<float>(get<0>(input));
    }
};
struct IntMemcpyFunctor1to1
{
    IntMemcpyFunctor1to1(){}

    EAVL_FUNCTOR tuple<int> operator()(tuple<int> input){
        return tuple<int>(get<0>(input));
    }
};



struct AccFunctor3to3
{
    AccFunctor3to3(){}

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<float,float,float,float,float,float> input){
        eavlVector3 color1(get<0>(input),get<1>(input),get<2>(input));
        eavlVector3 color2(get<3>(input),get<4>(input),get<5>(input));
        color1=color1+color2;
        return tuple<float,float,float>(min(color1.x,1.0f),min(color1.y,1.0f),min(color1.z,1.0f));
    }

};
struct AccFunctor1to1
{
    AccFunctor1to1(){}

    EAVL_FUNCTOR tuple<float> operator()(tuple<float,float> input){
        float color=get<0>(input)+get<1>(input);
        return tuple<float>(min(color,1.0f));
    }

};


/*----------------------End Utility Functors---------------------------------- */
//if this is called we know that the there is a hit
EAVL_HOSTDEVICE void triangleIntersectionDistance(const eavlVector3 ray,const eavlVector3 rayOrigin,const eavlVector3 a, const eavlVector3 b, const eavlVector3 c, float &tempDist)
{
    eavlVector3 intersect,normal;
    float d,dot;//,area;
    normal=(b-a)%(c-a);                                
    dot=normal*ray;
    d=normal*a; 
    tempDist=(d-normal*rayOrigin)/dot; 
}

//if this is called we know that the there is a hit
EAVL_HOSTDEVICE eavlVector3 triangleIntersectionABG(const eavlVector3 ray,const eavlVector3 rayOrigin,const eavlVector3 a, const eavlVector3 b, const eavlVector3 c, int index, float &alpha,float &beta)
{

    eavlVector3 intersect,normal;
    float tempDistance,d,dot,area;
    normal=(b-a)%(c-a);                                  //**I would think that the normal will already be calculated, otherwise add this to preprocessing step
    dot=normal*ray;
    //if(dot<TOLERANCE && dot >-TOLERANCE) return eavlVector3(-9999,-9999,-9999);    //traingle is parallel to the ray
    d=normal*a; //solve for d using any point on the plane
    tempDistance=(d-normal*rayOrigin)/dot; //**this could be preprocessed as well, but could be cost prohibitive if a lot of triangles are parallel to the ray( not likely i think)
    intersect=rayOrigin+ray*tempDistance;
    //inside test
    alpha=((c-b)%(intersect-b))*normal; //angles between the intersect point and edges
    beta =((a-c)%(intersect-c))*normal;
    // this is for the barycentric coordinates for color, normal lerping.
    area=normal*normal;
    alpha=alpha/area;
    beta =beta/area;
    return intersect;
}



EAVL_HOSTDEVICE int getIntersectionTri(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstArrayV2<float4> &bvh,const eavlConstArrayV2<float> &bvhLeafs,eavlConstArrayV2<float4> &verts,const float &maxDistance)
{


    float minDistance =maxDistance;
    int   minIndex    =-1;
    
    float dirx=rayDir.x;
    float diry=rayDir.y;
    float dirz=rayDir.z;

    float invDirx=rcp_safe(dirx);
    float invDiry=rcp_safe(diry);
    float invDirz=rcp_safe(dirz);
    int currentNode;
  
    int todo[64]; //num of nodes to process
    int stackptr = 0;
    int barrier=(int)END_FLAG;
    currentNode=0;

    todo[stackptr] = barrier;

    float ox=rayOrigin.x;
    float oy=rayOrigin.y;
    float oz=rayOrigin.z;
    float odirx=ox*invDirx;
    float odiry=oy*invDiry;
    float odirz=oz*invDirz;

    while(currentNode!=END_FLAG) {
        

        
        if(currentNode>-1)
        {

            float4 n1=bvh.getValue(bvhInnerTexRef, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
            float4 n2=bvh.getValue(bvhInnerTexRef, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3=bvh.getValue(bvhInnerTexRef, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
            
            float txmin0 =   n1.x* invDirx -odirx;       
            float tymin0 =   n1.y* invDiry -odiry;         
            float tzmin0 =   n1.z* invDirz -odirz;
            float txmax0 =   n1.w* invDirx -odirx;
            float tymax0 =   n2.x* invDiry -odiry;
            float tzmax0 =   n2.y* invDirz -odirz;
           
            float tmin0=max(max(max(min(tymin0,tymax0),min(txmin0,txmax0)),min(tzmin0,tzmax0)),0.f);
            float tmax0=min(min(min(max(tymin0,tymax0),max(txmin0,txmax0)),max(tzmin0,tzmax0)), minDistance);
            
            bool traverseChild0=(tmax0>=tmin0);

             
            float txmin1 =   n2.z* invDirx -odirx;       
            float tymin1 =   n2.w* invDiry -odiry;
            float tzmin1 =   n3.x* invDirz -odirz;
            float txmax1 =   n3.y* invDirx -odirx;
            float tymax1 =   n3.z* invDiry- odiry;
            float tzmax1 =   n3.w* invDirz -odirz;
            float tmin1=max(max(max(min(tymin1,tymax1),min(txmin1,txmax1)),min(tzmin1,tzmax1)),0.f);
            float tmax1=min(min(min(max(tymin1,tymax1),max(txmin1,txmax1)),max(tzmin1,tzmax1)), minDistance);
            
            bool traverseChild1=(tmax1>=tmin1);

        if(!traverseChild0 && !traverseChild1)
        {

            currentNode=todo[stackptr]; //go back put the stack
            stackptr--;
        }
        else
        {
            float4 n4=bvh.getValue(bvhInnerTexRef, currentNode+3); //(leftChild, rightChild, pad,pad)
            int leftChild =(int)n4.x;
            int rightChild=(int)n4.y;

            currentNode= (traverseChild0) ? leftChild : rightChild;
            if(traverseChild1 && traverseChild0)
            {
                if(tmin0>tmin1)
                {

                   
                    currentNode=rightChild;
                    stackptr++;
                    todo[stackptr]=leftChild;
                }
                else
                {   
                    stackptr++;
                    todo[stackptr]=rightChild;
                }


            }
        }
        }
        
        if(currentNode < 0 && currentNode != barrier)//check register usage
        {
            

            currentNode=-currentNode; //swap the neg address 
            int numTri=(int)bvhLeafs.getValue(bvhLeafTexRef,currentNode)+1;

            for(int i=1;i<numTri;i++)
            {        
                    int triIndex=(int)bvhLeafs.getValue(bvhLeafTexRef,currentNode+i);
                   
                    float4 a4=verts.getValue(vertsTexRef, triIndex*3);
                    float4 b4=verts.getValue(vertsTexRef, triIndex*3+1);
                    float4 c4=verts.getValue(vertsTexRef, triIndex*3+2);
                    eavlVector3 e1( a4.w-a4.x , b4.x-a4.y, b4.y-a4.z ); 
                    eavlVector3 e2( b4.z-a4.x , b4.w-a4.y, c4.x-a4.z );


                    eavlVector3 p;
                    p.x=diry*e2.z-dirz*e2.y;
                    p.y=dirz*e2.x-dirx*e2.z;
                    p.z=dirx*e2.y-diry*e2.x;
                    float dot=e1*p;
                    if(dot!=0.f)
                    {
                        dot=1.f/dot;
                        eavlVector3 t;
                        t.x=ox-a4.x;
                        t.y=oy-a4.y;
                        t.z=oz-a4.z;

                        float u=(t*p)*dot;
                        if(u>=0.f && u<=1.f)
                        {
                            eavlVector3 q=t%e1;
                            float v=(dirx*q.x+diry*q.y+dirz*q.z)*dot;
                            if(v>=0.f && v<=1.f)
                            {
                                float dist=(e2*q)*dot;
                                if((dist>EPSILON && dist<minDistance) && !(u+v>1) )
                                {
                                    minDistance=dist;
                                    minIndex=triIndex;
                                    if(occlusion) return minIndex;//or set todo to -1
                                }
                            }
                        }

                    }
                   
            }
            currentNode=todo[stackptr];
            stackptr--;
        }

    }
 return minIndex;
}

EAVL_HOSTDEVICE float getIntersectionDepth(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstArrayV2<float4> &bvh,const eavlConstArrayV2<float> &bvhLeafs,eavlConstArrayV2<float4> &verts,const float &maxDistance)
{


    float minDistance=maxDistance;
    int minIndex=-1;
    
    float dirx=rayDir.x;
    float diry=rayDir.y;
    float dirz=rayDir.z;

    float invDirx=rcp_safe(dirx);
    float invDiry=rcp_safe(diry);
    float invDirz=rcp_safe(dirz);
    int currentNode;
  

    int todo[64]; //num of nodes to process
    int stackptr = 0;
    int barrier=(int)END_FLAG;
    currentNode=0;

    todo[stackptr] = barrier;

    float ox=rayOrigin.x;
    float oy=rayOrigin.y;
    float oz=rayOrigin.z;
    float odirx=ox*invDirx;
    float odiry=oy*invDiry;
    float odirz=oz*invDirz;

    while(currentNode!=END_FLAG) {
        
        
        if(currentNode>-1)
        {
            float4 n1=bvh.getValue(bvhInnerTexRef, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
            //cout<<n1.x<<" "<<n1.y<<" "<<n1.z<<" "<<n1.w<<endl;
            float4 n2=bvh.getValue(bvhInnerTexRef, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3=bvh.getValue(bvhInnerTexRef, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
            
            float txmin0 =   n1.x* invDirx -odirx;       
            float tymin0 =   n1.y* invDiry -odiry;         
            float tzmin0 =   n1.z* invDirz -odirz;
            float txmax0 =   n1.w* invDirx -odirx;
            float tymax0 =   n2.x* invDiry -odiry;
            float tzmax0 =   n2.y* invDirz -odirz;
           
            float tmin0=max(max(max(min(tymin0,tymax0),min(txmin0,txmax0)),min(tzmin0,tzmax0)),0.f);
            float tmax0=min(min(min(max(tymin0,tymax0),max(txmin0,txmax0)),max(tzmin0,tzmax0)), minDistance);
            
            bool traverseChild0=(tmax0>=tmin0);
            //test if re-using the same variables makes a difference with instruction level parallism
             
            float txmin1 =   n2.z* invDirx -odirx;       
            float tymin1 =   n2.w* invDiry -odiry;
            float tzmin1 =   n3.x* invDirz -odirz;
            float txmax1 =   n3.y* invDirx -odirx;
            float tymax1 =   n3.z* invDiry- odiry;
            float tzmax1 =   n3.w* invDirz -odirz;
            float tmin1=max(max(max(min(tymin1,tymax1),min(txmin1,txmax1)),min(tzmin1,tzmax1)),0.f);
            float tmax1=min(min(min(max(tymin1,tymax1),max(txmin1,txmax1)),max(tzmin1,tzmax1)), minDistance);
            
            bool traverseChild1=(tmax1>=tmin1);

        if(!traverseChild0 && !traverseChild1)
        {
            currentNode=todo[stackptr]; //go back put the stack
            stackptr--;
        }
        else
        {
            float4 n4=bvh.getValue(bvhInnerTexRef, currentNode+3); //(leftChild, rightChild, pad,pad)
            int leftChild =(int)n4.x;
            int rightChild=(int)n4.y;
            currentNode= (traverseChild0) ? leftChild : rightChild;
            if(traverseChild1 && traverseChild0)
            {
                if(tmin0>tmin1)
                {

                   
                    currentNode=rightChild;
                    stackptr++;
                    todo[stackptr]=leftChild;
                }
                else
                {   
                    stackptr++;
                    todo[stackptr]=rightChild;
                }


            }
        }
        }
        
        if(currentNode<0&&currentNode!=barrier)//check register usage
        {
            

            currentNode=-currentNode; //swap the neg address 
            int numTri=(int)bvhLeafs.getValue(bvhLeafTexRef,currentNode)+1;

            for(int i=1;i<numTri;i++)
            {        
                    int triIndex=(int)bvhLeafs.getValue(bvhLeafTexRef,currentNode+i);
                    float4 a4=verts.getValue(vertsTexRef, triIndex*3);
                    float4 b4=verts.getValue(vertsTexRef, triIndex*3+1);
                    float4 c4=verts.getValue(vertsTexRef, triIndex*3+2);
                    eavlVector3 e1( a4.w-a4.x , b4.x-a4.y, b4.y-a4.z ); 
                    eavlVector3 e2( b4.z-a4.x , b4.w-a4.y, c4.x-a4.z );

                    eavlVector3 p;
                    p.x=diry*e2.z-dirz*e2.y;
                    p.y=dirz*e2.x-dirx*e2.z;
                    p.z=dirx*e2.y-diry*e2.x;
                    float dot=e1*p;
                    if(dot!=0.f)
                    {
                        dot=1.f/dot;
                        eavlVector3 t;
                        t.x=ox-a4.x;
                        t.y=oy-a4.y;
                        t.z=oz-a4.z;

                        float u=(t*p)*dot;
                        if(u>=0.f && u<=1.f)
                        {
                            eavlVector3 q=t%e1;
                            float v=(dirx*q.x+diry*q.y+dirz*q.z)*dot;
                            if(v>=0.f && v<=1.f)
                            {
                                float dist=(e2*q)*dot;
                                if((dist>EPSILON && dist<minDistance) && !(u+v>1) )
                                {
                                    minDistance=dist;
                                    minIndex=triIndex;
                                    if(occlusion) return minIndex;//or set todo to -1
                                }
                            }
                        }

                    }
                   
            }
            currentNode=todo[stackptr];
            stackptr--;
        }

    }

 if(minIndex==-1) return INFINITE; //what value does a no hit need to be?
 else  return minDistance;
}

struct RayIntersectFunctor{


    eavlConstArrayV2<float4> verts;
    //eavlConstArray<float> bvh;
    eavlConstArrayV2<float4> bvh;
    eavlConstArrayV2<float> bvhLeafs;

    RayIntersectFunctor(eavlConstArrayV2<float4> *_verts, eavlConstArrayV2<float4> *theBvh,eavlConstArrayV2<float> *theBvhLeafs)
        :verts(*_verts),
         bvh(*theBvh),
         bvhLeafs(*theBvhLeafs)
    {

        
    }                                                     //order a b c clockwise
    EAVL_HOSTDEVICE tuple<int> operator()( tuple<float,float,float,float,float,float,int> rayTuple){
       
        
        int deadRay=get<6>(rayTuple);
        if(deadRay==-1) return tuple<int>(-1);

        int minHit=-1; 

        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));
        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        minHit= getIntersectionTri(ray, rayOrigin, false,bvh,bvhLeafs, verts,INFINITE);
    
        if(minHit!=-1) return tuple<int>(minHit);
        else           return tuple<int>(-1);
    }
};

struct RayIntersectFunctorDepth{


    eavlConstArrayV2<float4> verts;
    eavlConstArrayV2<float4> bvh;
    eavlConstArrayV2<float>  bvhLeafs;

    RayIntersectFunctorDepth(eavlConstArrayV2<float4> *_verts, eavlConstArrayV2<float4> *theBvh,eavlConstArrayV2<float> *theBvhLeafs)
        :verts(*_verts),
         bvh(*theBvh),
         bvhLeafs(*theBvhLeafs)
    {

        
    }                                                    
    EAVL_HOSTDEVICE tuple<float> operator()( tuple<float,float,float,float,float,float,int> rayTuple){
       
        
        int deadRay=get<6>(rayTuple);
        if(deadRay==-1) return tuple<int>(-1);

        float depth=-1; 

        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));
        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        depth= getIntersectionDepth(ray, rayOrigin, false,bvh,bvhLeafs, verts,INFINITE);
    
        return tuple<float>(depth);
        
    }
};

struct ReflectFunctor{

    eavlConstArrayV2<float4> verts;
    eavlConstArray<float> norms;


    ReflectFunctor(eavlConstArrayV2<float4> *_verts,eavlConstArray<float> *xnorm)
        :verts(*_verts),
         norms(*xnorm)
    {
        
    }                                                    //order a b c clockwise
    EAVL_FUNCTOR tuple<float,float,float,float,float, float,float,float, float,float,float> operator()( tuple<float,float,float,float,float,float,int> rayTuple){
       
        
        int hitIndex=get<6>(rayTuple);//hack for now
        if(hitIndex==-1) return tuple<float,float,float,float,float, float,float,float, float,float,float>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
        
        eavlVector3 intersect;

        eavlVector3 reflection(0,0,0);
        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));

        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        float alpha=0, beta=0, gamma=0;

        float4 a4=verts.getValue(vertsTexRef, hitIndex*3);
        float4 b4=verts.getValue(vertsTexRef, hitIndex*3+1);
        float4 c4=verts.getValue(vertsTexRef, hitIndex*3+2);
        eavlVector3 a(a4.x,a4.y,a4.z);
        eavlVector3 b(a4.w,b4.x,b4.y);
        eavlVector3 c(b4.z,b4.w,c4.x);
        intersect= triangleIntersectionABG(ray, rayOrigin, a,b,c,0.0f,alpha,beta);
        gamma=1-alpha-beta;
        eavlVector3 normal(999999.f,0,0);

        eavlVector3* normalPtr=(eavlVector3*)(&norms[0]+hitIndex*9);
        eavlVector3 aNorm=normalPtr[0];
        eavlVector3 bNorm=normalPtr[1];
        eavlVector3 cNorm=normalPtr[2];
        normal=(-aNorm)*alpha+(-bNorm)*beta+(-cNorm)*gamma;
        //reflect the ray
        ray.normalize();
        normal.normalize();
        if ((normal*ray) > 0.0f) normal = -normal;
        reflection=ray-normal*2.f*(normal*ray);
        reflection.normalize();
        intersect=intersect+(-ray*BARY_TOLE);


        return tuple<float,float,float,float,float,float,float,float,float,float,float>(intersect.x, intersect.y,intersect.z,reflection.x,reflection.y,reflection.z,normal.x,normal.y,normal.z,alpha,beta);
    }
};


struct DepthFunctor{

    eavlConstArrayV2<float4> verts;


    DepthFunctor(eavlConstArrayV2<float4> *_verts)
        :verts(*_verts)
    {
        
    }                                                    
    EAVL_FUNCTOR tuple<float> operator()( tuple<float,float,float,float,float,float,int> rayTuple){
       
        
        int hitIndex=get<6>(rayTuple);
        if(hitIndex==-1) return tuple<float>(INFINITE);
        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));
        eavlVector3       ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));

        float4 a4=verts.getValue(vertsTexRef, hitIndex*3);
        float4 b4=verts.getValue(vertsTexRef, hitIndex*3+1);
        float4 c4=verts.getValue(vertsTexRef, hitIndex*3+2);
        eavlVector3 a(a4.x,a4.y,a4.z);
        eavlVector3 b(a4.w,b4.x,b4.y);
        eavlVector3 c(b4.z,b4.w,c4.x);
        float depth;
        triangleIntersectionDistance(ray, rayOrigin, a,b,c,depth);
        return tuple<float>(depth);
    }
};

struct ReflectFunctorBasic{

    eavlConstArray<float> verts;
    eavlConstArray<float> norms;

    ReflectFunctorBasic(eavlConstArray<float> *_verts,eavlConstArray<float> *xnorm)
        :verts(*_verts),
         norms(*xnorm)
    {
        
    }                                                    //order a b c clockwise
    EAVL_FUNCTOR tuple<float,float, float,float,float> operator()( tuple<float,float,float,float,float,float,int> rayTuple){
       
        
        int hitIndex=get<6>(rayTuple);//hack for now
        if(hitIndex==-1) { return tuple<float,float, float,float,float>(0.0,0.0,0.0,0.0,0.0);}
        
        eavlVector3 intersect;


        eavlVector3 eyePos(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));//change this name, only eye for the first pass

        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        float alpha=0, beta=0;
        eavlVector3* ptr=(eavlVector3*)(&verts[0]+hitIndex*9);
        eavlVector3 a=ptr[0];
        eavlVector3 b=ptr[1];
        eavlVector3 c=ptr[2];
        
        //triangleIntersectionABG(const eavlVector3 ray,const eavlVector3 rayOrigin,const eavlVector3 a, const eavlVector3 b, const eavlVector3 c, int index, float &alpha,float &beta,float gamma)
        intersect= triangleIntersectionABG(ray, eyePos, a,b,c,0.0f,alpha,beta);
        //reflect the ray
        ray.normalize();
        intersect=intersect+(-ray*BARY_TOLE);

        return tuple<float,float,float,float,float>(intersect.x, intersect.y,intersect.z,alpha,beta);
    }
};

struct occIntersectFunctor{

    float           maxDistance;
    eavlConstArrayV2<float4> verts;
    // /eavlConstArray<float> bvh;
    eavlConstArrayV2<float4> bvh;
    eavlConstArrayV2<float> bvhLeafs;


    occIntersectFunctor(eavlConstArrayV2<float4> *_verts, eavlConstArrayV2<float4> *theBvh, eavlConstArrayV2<float> *theBvhLeafs, float max)
        :verts(*_verts),
         bvh(*theBvh),
         bvhLeafs(*theBvhLeafs)
    {

        maxDistance=max;
        
    }                                                    //order a b c clockwise
    EAVL_FUNCTOR tuple<float> operator()( tuple<float,float,float,float,float,float,int> rayTuple){
        
        int deadRay=get<6>(rayTuple);//hack for now leaving this in for CPU
        if(deadRay==-1) return tuple<float>(0.0f);
        int minHit=-1;   
        eavlVector3 intersect(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));//change this name, only eye for the first pass
        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        minHit= getIntersectionTri(ray, intersect, true,bvh, bvhLeafs, verts,maxDistance);

        if(minHit!=-1) return tuple<float>(0.0f);
        else return tuple<float>(1.0f);
    }
};



struct ShadowRayFunctor
{
    eavlConstArrayV2<float4> verts;
    //eavlConstArray<float> bvh;
    eavlConstArrayV2<float4> bvh;
    eavlConstArrayV2<float> bvhLeafs;
    eavlVector3 light;

    ShadowRayFunctor(eavlVector3 theLight,eavlConstArrayV2<float4> *_verts,eavlConstArrayV2<float4> *theBvh,eavlConstArrayV2<float> *theBvhLeafs)
        :verts(*_verts),
         bvh(*theBvh),
         bvhLeafs(*theBvhLeafs),
         light(theLight)
    {}

    EAVL_FUNCTOR tuple<int> operator()(tuple<float,float,float,int> input)
    {
        int hitIdx=get<3>(input);
        if(hitIdx==-1) return tuple<int>(1);// primary ray never hit anything.
        //float alpha,beta,gamma,d,tempDistance;
        eavlVector3 rayOrigin(get<0>(input),get<1>(input),get<2>(input));
        
        eavlVector3 shadowRay=light-rayOrigin;
        float lightDistance=sqrt(shadowRay.x*shadowRay.x+shadowRay.y*shadowRay.y+shadowRay.z*shadowRay.z);
        shadowRay.normalize();
        int minHit;
        minHit= getIntersectionTri(shadowRay, rayOrigin, true,bvh, bvhLeafs, verts,lightDistance);
        if(minHit!=-1) return tuple<int>(1);//in shadow
        else return tuple<int>(0);//clear view of the light

    }


};


struct ShaderFunctor
{
    eavlVector3     light;
    eavlVector3     lightDiff;
    eavlVector3     lightSpec;
    float           lightIntensity;
    float           lightCoLinear;
    float           lightCoConst;
    float           lightCoExponent;
    eavlVector3     eye; 
    float           depth;
    int             size;
    eavlConstArray<float> norms;
    eavlConstArray<int>   matIds;
    eavlConstArray<float> mats;

    ShaderFunctor(int numTris,eavlVector3 theLight,eavlVector3 eyePos,eavlConstArray<float> *xnorm, int dpth,eavlConstArray<int> *_matIds,eavlConstArray<float> *_mats,
                  int _lightIntensity, float _lightCoConst, float _lightCoLinear, float _lightCoExponent)
        :norms(*xnorm), matIds(*_matIds), mats(*_mats)

    {
        depth=(float)dpth;
        light=theLight;
        size=numTris;

        eye=eyePos;

        lightDiff=eavlVector3(.5,.5,.5);
        lightSpec=eavlVector3(.5,.5,.5);
        lightIntensity  = _lightIntensity;
        lightCoConst    = _lightCoConst;   //used as coefficients for light attenuation
        lightCoLinear   = _lightCoLinear;
        lightCoExponent = _lightCoExponent;
    }

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<int,int,float,float,float,float,float,float,float,float,float,float,float,float> input)
    {
        int hitIdx=get<0>(input);
        int hit=get<1>(input);
        float specConst;


        if(hitIdx==-1 ) return tuple<float,float,float>(0,0,0);// primary ray never hit anything.

        eavlVector3 normal(get<8>(input), get<9>(input), get<10>(input));
        eavlVector3 rayInt(get<2>(input),get<3>(input),get<4>(input));
        eavlVector3 rayOrigin(get<11>(input),get<12>(input),get<13>(input));
        eavlVector3 abg(get<5>(input),get<6>(input),1.f); //alpha beta gamma
        abg.z=abg.z-abg.x-abg.y; // get gamma
        eavlVector3 lightDir=light-rayInt;
        eavlVector3 viewDir=eye-rayOrigin;
        float dist=sqrt(lightDir.x*lightDir.x+lightDir.y*lightDir.y+lightDir.z*lightDir.z); 
        dist+=sqrt(viewDir.x*viewDir.x+viewDir.y*viewDir.y+viewDir.z*viewDir.z);
        lightDir=lightDir/dist;
        dist=lightIntensity/(lightCoConst+lightCoLinear*dist+lightCoExponent*dist*dist);
        float ambPct=get<7>(input);
        eavlVector3 pixel(0,0,0);

        int id=matIds[hitIdx];
        eavlVector3* matPtr=(eavlVector3*)(&mats[0]+id*12);
        eavlVector3 ka=matPtr[0];//these could be lerped if it is possible that a single tri could be made of several mats
        eavlVector3 kd=matPtr[1];
        eavlVector3 ks=matPtr[2];
        float matShine=matPtr[3].x;

//********************************************************
        //pixel=aColor*abg.x+bColor*abg.y+cColor*abg.z;
        float red=0;
        float green=0;
        float blue=0;
 
        float cosTheta=normal*lightDir; //for diffuse
        cosTheta=min(max(cosTheta,0.f),1.f); //clamp this to [0,1]
        
        eavlVector3 halfVector=viewDir+lightDir;
        halfVector.normalize();
        
        viewDir.normalize();
        
        float cosPhi=normal*halfVector;
        
        specConst=pow(max(cosPhi,0.0f),matShine);
        //cosTheta=.5;
        //cout<<specConst<<endl;
        //ambPct=0;
        float shadowHit= (hit==1) ? 0.f : 1.f;
        //red  =ambPct;//ka.x*ambPct+ (kd.x*lightDiff.x*max(cosTheta,0.0f)+ks.x*lightSpec.x*specConst)*shadowHit;
        //green=ambPct;//ka.y*ambPct+ (kd.y*lightDiff.y*max(cosTheta,0.0f)+ks.y*lightSpec.y*specConst)*shadowHit;
        //blue =ambPct;//ka.z*ambPct+ (kd.z*lightDiff.z*max(cosTheta,0.0f)+ks.z*lightSpec.z*specConst)*shadowHit;
        
        red  =ka.x*ambPct+ (kd.x*lightDiff.x*cosTheta+ks.x*lightSpec.x*specConst)*shadowHit*dist;
        green=ka.y*ambPct+ (kd.y*lightDiff.y*cosTheta+ks.y*lightSpec.y*specConst)*shadowHit*dist;
        blue =ka.z*ambPct+ (kd.z*lightDiff.z*cosTheta+ks.z*lightSpec.z*specConst)*shadowHit*dist;
        //cout<<kd<<ks<<id<<endl;
        //cout<<cosTheta<<endl;
        //red  =ka.x*ambPct+ (ks.x*lightSpec.x*specConst)*shadowHit;
        //green=ka.y*ambPct+ (ks.y*lightSpec.y*specConst)*shadowHit;
        //blue =ka.z*ambPct+ (ks.z*lightSpec.z*specConst)*shadowHit;

        //red  =ka.x*ambPct+ (kd.x*lightDiff.x*cosTheta)*shadowHit;
        //green=ka.y*ambPct+ (kd.y*lightDiff.y*cosTheta)*shadowHit;
        //blue =ka.z*ambPct+ (kd.z*lightDiff.z*cosTheta)*shadowHit;


        float reflectionCo=1;
        if(depth==0) reflectionCo=1;
        else reflectionCo=pow(.3f,depth);
        pixel.x=red  *reflectionCo;
        pixel.y=green*reflectionCo;
        pixel.z=blue *reflectionCo;
        //if(specConst==0.0) pixel.z=1;

        //}
        return tuple<float,float,float>(min(pixel.x,1.0f),min(pixel.y,1.0f),min(pixel.z,1.0f));

    }


};

void eavlRayTracerMutator::writeBMP(int _height, int _width, eavlFloatArray *r, eavlFloatArray *g, eavlFloatArray *b,const char*fname)
{
    FILE *f;
    int size=_height*_width;
    unsigned char *img = NULL;
    int filesize = 54 + 3*_width*_height;  //w is your image width, h is image height, both int
    if( img )
        free( img );
    img = (unsigned char *)malloc(3*_width*_height);
    memset(img,0,sizeof(img));

    for(int j=size;j>-1;j--)
    {
        img[j*3  ]= (unsigned char) (int)(b->GetValue(j)*255);
        img[j*3+1]= (unsigned char) (int)(g->GetValue(j)*255);
        img[j*3+2]= (unsigned char) (int)(r->GetValue(j)*255);

    }

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       _width    );
    bmpinfoheader[ 5] = (unsigned char)(       _width>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       _width>>16);
    bmpinfoheader[ 7] = (unsigned char)(       _width>>24);
    bmpinfoheader[ 8] = (unsigned char)(       _height    );
    bmpinfoheader[ 9] = (unsigned char)(       _height>> 8);
    bmpinfoheader[10] = (unsigned char)(       _height>>16);
    bmpinfoheader[11] = (unsigned char)(       _height>>24);

    f = fopen(fname,"wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(int i=0; i<height; i++)
    {
        fwrite(img+(_width*(_height-i-1)*3),3,_width,f);
        fwrite(bmppad,1,(4-(_width*3)%4)%4,f);
    }
    fclose(f);
    free(img);
}

void eavlRayTracerMutator::allocateArrays()
{
    if(rayDirX!=NULL)
    {   cout<<"Deleting"<<rayDirX<<endl;
        delete  rayDirX;
        delete  rayDirY;
        delete  rayDirZ;

        delete  rayOriginX;
        delete  rayOriginY;
        delete  rayOriginZ;

        delete  r;
        delete  g;
        delete  b;

        delete  alphas;
        delete  betas;

        delete  interX;
        delete  interY;
        delete  interZ;

        delete  normX;
        delete  normY;
        delete  normZ;

        delete hitIdx;
        delete indexes;
        delete mortonIndexes;
        delete shadowHits;

        delete r2;
        delete b2;
        delete g2;

        delete ambPct;
        delete zBuffer;
        delete frameBuffer;
        //what happens when someone turns these on and off??? 
        //1. we could just always do it and waste memory
        //2 call allocation if we detect dirty settings/dirtySize <-this is what is being done;
        if(antiAlias)
        {
            
           delete rOut;
           delete gOut;
           delete bOut;
        }

        if(isOccusionOn)
        {
            delete occX;
            delete occY;
            delete occZ;
            delete localHits;

            delete occIndexer;
        }
        if (compactOp)
        {
            delete mask;
            delete count;
            delete indexScan;
        }

    }
    cout<<"alloc "<<size<<endl;
    /*Temp arrays for compact*/
    compactTempInt  = new eavlIntArray("temp",1,size);
    compactTempFloat= new eavlFloatArray("temp",1,size);
    mortonIndexes   = new eavlIntArray("mortonIdxs",1,size);

    rayDirX        = new eavlFloatArray("x",1,size);
    rayDirY        = new eavlFloatArray("y",1,size);
    rayDirZ        = new eavlFloatArray("z",1,size);

    rayOriginX     = new eavlFloatArray("x",1,size);
    rayOriginY     = new eavlFloatArray("y",1,size);
    rayOriginZ     = new eavlFloatArray("z",1,size);


    r              = new eavlFloatArray("r",1,size);
    g              = new eavlFloatArray("g",1,size);
    b              = new eavlFloatArray("b",1,size);

    r2             = new eavlFloatArray("",1,size);
    g2             = new eavlFloatArray("",1,size);
    b2             = new eavlFloatArray("",1,size);

   
    alphas         = new eavlFloatArray("aplha",1,size);
    betas          = new eavlFloatArray("beta",1,size);

    interX         = new eavlFloatArray("interX",1,size);
    interY         = new eavlFloatArray("interY",1,size);
    interZ         = new eavlFloatArray("interZ",1,size);
    normX          = new eavlFloatArray("normX",1,size);
    normY          = new eavlFloatArray("normY",1,size);
    normZ          = new eavlFloatArray("normZ",1,size);

    hitIdx         = new eavlIntArray("hitIndex",1,size);
    indexes        = new eavlIntArray("indexes",1,size);
    shadowHits     = new eavlFloatArray("",1,size);
    ambPct         = new eavlFloatArray("",1, size);
    zBuffer        = new eavlFloatArray("",1, size);
    frameBuffer    = new eavlFloatArray("",1, width*height*3);
    //compact arrays
    if(compactOp)
    {
        mask       = new eavlIntArray("mask",1,currentSize);        //array to store the mask
        indexScan  = new eavlIntArray("indexScan",1,currentSize);
        count      = new eavlIntArray("count",1,1);
    }

#if 0
    if(antiAlias)
    {
        rOut=new eavlFloatArray("",1,(height)*(width));
        gOut=new eavlFloatArray("",1,(height)*(width));
        bOut=new eavlFloatArray("",1,(height)*(width));
    }
#endif
    if(isOccusionOn)
    {
        occX      = new eavlFloatArray("",1, size*occSamples);//not so good to hav all these //also is there a way to filter out the misses?
        occY      = new eavlFloatArray("",1, size*occSamples);//
        occZ      = new eavlFloatArray("",1, size*occSamples);
        localHits = new eavlFloatArray("",1, size*occSamples);
        
        occIndexer= new eavlArrayIndexer(occSamples,1e9, 1, 0);
    }
    else //set amient light percentage to whatever the lights are
    {
       cout<<"setting amb"<<endl;
       eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(ambPct),
                                                eavlOpArgs(ambPct),
                                                FloatMemsetFunctor(1.0f)),
                                                "force");
       eavlExecutor::Go();
    }

    sizeDirty=false;
    cout<<"dirty end should be false "<<sizeDirty<<endl;
}

void eavlRayTracerMutator::Init()
{   if(verbose) cerr<<"INIT"<<endl;
    size=height*width;
#if 0    
    if(antiAlias) size=(width+1)*(height+1);
#endif
    currentSize=size; //for compact
    cout<<"Here 1 dirty="<<sizeDirty<<endl;
    if(sizeDirty) 
    {
        cout<<"Here 2"<<endl;
        allocateArrays();
        cout<<"Here 3"<<endl;
        createRays(); //creates the morton ray indexes
    }
    cout<<"Here 4"<<endl;
    //fill the const arrays with vertex and normal data
    
    /* Set ray origins to the eye */

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes), //dummy arg
                                             eavlOpArgs(rayOriginX,rayOriginY,rayOriginZ),
                                             FloatMemsetFunctor3to3(eye.x,eye.y,eye.z)),
                                             "init");
    eavlExecutor::Go();

    /* Copy morton indexes into idxs*/ 
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(mortonIndexes),
                                             eavlOpArgs(indexes),
                                             FloatMemcpyFunctor1to1()),
                                             "cpy");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(hitIdx),
                                             eavlOpArgs(hitIdx),
                                             IntMemsetFunctor(0)),
                                             "init");
    eavlExecutor::Go();
    cout<<"Here 4"<<endl;
    if(geomDirty) extractGeometry();
}

void eavlRayTracerMutator::extractGeometry()
{
    
    
    //sort();
    //list<Triangle> triangles; 
    if(verbose) cerr<<"Extracting Geometry"<<endl;
    
    scene->createRawData();
    cout<<"raw data created"<<endl;
    numTriangles    = scene->getNumTriangles();
    numMats         = scene->getNumMaterials();
    verts_raw       = scene->getTrianglePtr();
    norms_raw       = scene->getTriangleNormPtr();
    matIdx_raw      = scene->getTriMatIdxsPtr();
    mats_raw        = scene->getMatsPtr();
     cout<<"got pointers"<<endl;
    int bvhsize=0;
    int bvhLeafSize=0;
    float *bvhLeafs;
    bool cacheExists=false;
    bool writeCache=true;

    //if(bvhCacheName!="")
    //{
    //    cacheExists=readBVHCache(bvhFlatArray_raw, bvhsize, bvhLeafs, bvhLeafSize, bvhCacheName.c_str());
    //}
    //else 
    //{
    //    writeCache=false;
//
    //
    //if(!cacheExists)
    //{
        SplitBVH *testSplit= new SplitBVH((eavlVector3*)verts_raw, numTriangles); 
        testSplit->getFlatArray(bvhsize, bvhLeafSize, bvhFlatArray_raw, bvhLeafs);
        //if( writeCache) writeBVHCache(bvhFlatArray_raw, bvhsize, bvhLeafs, bvhLeafSize, bvhCacheName.c_str());
        delete testSplit;
    //}
    

    if(numMats==0) { cerr<<"NO MATS bailing"<<endl; exit(0); }

    if(verbose) cerr<<"BVH Size : " <<bvhsize<<endl;
   
    


    bvhTex     = new eavlConstArrayV2<float4>((float4*)bvhFlatArray_raw, bvhsize/4,bvhInnerTexRef);
    bvhLeafsTex= new eavlConstArrayV2<float>(bvhLeafs, bvhLeafSize,bvhLeafTexRef);
    vertsTex   = new eavlConstArrayV2<float4>((float4*)verts_raw,numTriangles*3, vertsTexRef);


    INIT(eavlConstArray<float>, mats,numMats*12);
    INIT(eavlConstArray<float>,norms,numTriangles*9);
    INIT(eavlConstArray<int>, matIdx,numTriangles);
    if(verbose) cerr<<"after allocation"<<endl;
    
    geomDirty=false;
    
}


void eavlRayTracerMutator::Execute()
{
    
    if(verbose) cerr<<"Executing Before Init"<<endl;

    Init();
    if(verbose) cerr<<"Executing After Init"<<endl;
   
    if(verbose) cerr<<"Number of triangles "<<numTriangles<<endl;
    //Extract the triangles and normals from the isosurface
   

    clearFrameBuffer(r,g,b);//may not needed if this isn't going to be supported

    //light=light+movement;
    look=lookat-eye;
    if(verbose) cerr<<"Look "<<look<<endl;
    if(verbose) cerr<<"Eye"<<eye<<endl;
    if(verbose) cerr<<"Light"<<light<<endl;
   
    int th = eavlTimer::Start();
    //init camera rays
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes),
                                             eavlOpArgs(rayDirX,rayDirY,rayDirZ),
                                             RayGenFunctor(width,height,fovx,fovy,look,up)),
                                             "ray gen");
    eavlExecutor::Go();


    for(int i=0; i<depth;i++) 
    {
        int tintersect = eavlTimer::Start();

        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx),
                                                 eavlOpArgs(hitIdx),
                                                 RayIntersectFunctor(vertsTex,bvhTex,bvhLeafsTex)),
                                                                                                    "intersect");
        eavlExecutor::Go();
        /* Get the depth buffer. This can only happen on the first bounce. */
        if(i==0)
        {
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx),
                                                     eavlOpArgs(zBuffer),
                                                     DepthFunctor(vertsTex)),
                                                                                                     "Depth");
            eavlExecutor::Go(); 
        }

        cerr << "intersect RUNTIME: "<<eavlTimer::Stop(tintersect,"intersect")<<endl;
        if(compactOp) 
        {
            int tcompact = eavlTimer::Start();
            currentSize=compact();
            cerr << "compact RUNTIME: "<<eavlTimer::Stop(tcompact,"intersect")<<endl;
        }
        int treflect = eavlTimer::Start();
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx),
                                                 eavlOpArgs(interX, interY,interZ,rayDirX,rayDirY,rayDirZ,normX,normY,normZ,alphas,betas),
                                                 ReflectFunctor(vertsTex,norms)),
                                                                                                     "reflect");
        eavlExecutor::Go();                        
        if(verbose) cerr<<"Reflect "<<eavlTimer::Stop(treflect,"rf")<<endl;
        /*if(currentSize==0)
        {

           cerr<<"No more rays"<<endl;
            break;
        }*/ 
        /*********************************AMB OCC***********************************/
        if(isOccusionOn){
            
            int toccGen = eavlTimer::Start(); 
            eavlExecutor::AddOperation(new_eavl1toNScatterOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(normX),
                                                     eavlIndexable<eavlFloatArray>(normY),
                                                     eavlIndexable<eavlFloatArray>(normZ),
                                                     eavlIndexable<eavlFloatArray>(interX),
                                                     eavlIndexable<eavlFloatArray>(interY),
                                                     eavlIndexable<eavlFloatArray>(interZ),
                                                     eavlIndexable<eavlIntArray>  (hitIdx)),
                                          eavlOpArgs(occX,occY,occZ),OccRayGenFunctor2(),
                                          occSamples),
                                          "occ scatter");
            eavlExecutor::Go();
            cerr << "occRayGen RUNTIME: "<<eavlTimer::Stop(toccGen,"occGen")<<endl;
            int toccInt = eavlTimer::Start(); 
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(occX),
                                                                eavlIndexable<eavlFloatArray>(occY),
                                                                eavlIndexable<eavlFloatArray>(occZ),
                                                                eavlIndexable<eavlFloatArray>(interX, *occIndexer),
                                                                eavlIndexable<eavlFloatArray>(interY, *occIndexer),
                                                                eavlIndexable<eavlFloatArray>(interZ, *occIndexer),
                                                                eavlIndexable<eavlIntArray>  (hitIdx, *occIndexer)),
                                                     eavlOpArgs(localHits),
                                                     occIntersectFunctor(vertsTex,bvhTex,bvhLeafsTex,aoMax)),
                                                     "occIntercept");
            
            eavlExecutor::Go();
    
            cerr << "occInt RUNTIME: "<<eavlTimer::Stop(toccGen,"occGen")<<endl;

            eavlExecutor::AddOperation(new_eavlNto1GatherOp(eavlOpArgs(localHits),
                                                            eavlOpArgs(ambPct), // now using this space(normX) to store bent normal of ave light direction
                                                            occSamples),
                                                            "gather");
            eavlExecutor::Go();

            
        }
        /*********************************END AMB OCC***********************************/
        
        
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(r2),
                                                eavlOpArgs(r2,g2,b2),
                                                FloatMemsetFunctor1to3(0.0f)),
                                                "memset");
        
        eavlExecutor::Go();

        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(interX,interY,interZ,hitIdx),
                                                 eavlOpArgs(shadowHits),
                                                 ShadowRayFunctor(light,vertsTex,bvhTex,bvhLeafsTex)),
                                                 "shadowRays");
        eavlExecutor::Go();
        if(verbose) cerr<<"After Shadow Rays"<<endl;
        int shade = eavlTimer::Start();
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(hitIdx,shadowHits,interX,interY,interZ,alphas,betas,ambPct,normX,normY,normZ,rayOriginX,rayOriginY,rayOriginZ),
                                                 eavlOpArgs(r2,g2,b2),
                                                 ShaderFunctor(numTriangles,light,eye,norms,i,matIdx,mats,lightIntensity, lightCoConst, lightCoLinear, lightCoExponent)),
                                                 "shader");
        eavlExecutor::Go();
        cerr<<"\nShading : "<<eavlTimer::Stop(shade,"")<<endl;
        //if(verbose) cerr<<"after delete amb"<<endl;


        if(!compactOp)
        {
            if(verbose) cerr<<"before ADD"<<endl;
            //these RGB values are in morton order and must be scattered 
            //into the correct order before they are written.
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(r2,g2,b2,r,g,b),
                                                     eavlOpArgs(r,g,b),
                                                     AccFunctor3to3()),
                                                     "add");
            eavlExecutor::Go();
            if(verbose) cerr<<"After Add"<<endl;
        }
        else
        {
            //zero out the array so bad values don't get accumulated.
            //The compact arrays must be scattered back out to the original size
            //After the scatter, RGB values are in the correct order(not morton order)
            int tscatterAcc = eavlTimer::Start();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(compactTempFloat),
                                             eavlOpArgs(compactTempFloat),
                                             FloatMemsetFunctor(0)),
                                            "memset");
            eavlExecutor::Go();

            eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(r2),
                                                 eavlOpArgs(compactTempFloat),
                                                 eavlOpArgs(indexes),
                                                 currentSize),
                                                "scatter");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(compactTempFloat,r),
                                                     eavlOpArgs(r),
                                                     AccFunctor1to1()),
                                                     "add");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(g2),
                                                 eavlOpArgs(compactTempFloat),
                                                 eavlOpArgs(indexes),
                                                 currentSize),
                                                "scatter");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(compactTempFloat,g),
                                                     eavlOpArgs(g),
                                                     AccFunctor1to1()),
                                                     "add");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(b2),
                                                 eavlOpArgs(compactTempFloat),
                                                 eavlOpArgs(indexes),
                                                 currentSize),
                                                "scatter");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(compactTempFloat,b),
                                                     eavlOpArgs(b),
                                                     AccFunctor1to1()),
                                                     "add");
            eavlExecutor::Go();
            cerr<<"\nScatter Acc : "<<eavlTimer::Stop(tscatterAcc,"")<<endl;
        }


        /* Copy intersections to origins if there are more bounces*/
        if(i<depth-1) 
        {
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(interX, interY, interZ),
                                                     eavlOpArgs(rayOriginX, rayOriginY, rayOriginZ),
                                                     FloatMemcpyFunctor3to3()),
                                                     "memcopy");
            eavlExecutor::Go();
        }

    }
    //todo fix this
    int ttest = eavlTimer::Start();
    if(!compactOp)
    {
        //Non-compacted RGB values are in morton order anbd must be scattered
        //back into the correct order
        eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(r),
                                                     eavlOpArgs(r2),
                                                     eavlOpArgs(mortonIndexes)),
                                                    "scatter");
        eavlExecutor::Go();
    
   
        eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(g),
                                                     eavlOpArgs(g2),
                                                     eavlOpArgs(mortonIndexes)),
                                                     "scatter");
        eavlExecutor::Go();
        eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(b),
                                                     eavlOpArgs(b2),
                                                     eavlOpArgs(mortonIndexes)),
                                                     "scatter");
        eavlExecutor::Go();
    }
    else
    {
        //arrays are already scattered into the correct RGB order
        //out so just switch the pointers around
        eavlFloatArray* tmpPtr;

        tmpPtr=r2;
        r2=r;
        r=tmpPtr;
        
        tmpPtr=g2;
        g2=g;
        g=tmpPtr;
        
        tmpPtr=b2;
        b2=b;
        b=tmpPtr;

    }
    if(verbose) cerr<<"\nscat : "<<eavlTimer::Stop(ttest,"")<<endl;

#if 0
    if(antiAlias)
    {   
        int talias=eavlTimer::Start();
        //This will align the rgb arrays as if they were smaller
        eavlExecutor::AddOperation(new_eavlAliasStencilOp(eavlOpArgs(r2,g2,b2),
                                                          eavlOpArgs(rOut,gOut,bOut),
                                                          width),
                                                          "add");
        eavlExecutor::Go();
        if(verbose) cerr<<"\nAlias : "<<eavlTimer::Stop(talias,"")<<endl; 
        if(verbose) cerr << "TOTAL     RUNTIME: "<<eavlTimer::Stop(th,"raytrace")<<endl;
        oss.str("");
        oss<<fileprefix<<frameCounter<<filetype;
        scounter=oss.str();
        writeBMP(height-1,width-1,rOut,gOut,bOut,outfilename);
        for (int i=0;i<width*height;i++) 
        {
            //cout<<"("<<r->GetValue(i)<<","<<g->GetValue(i)<<")"<<endl;
        }
    }
#else
    //else
    //{
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(r2, g2, b2),
                                                 eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                            eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                            eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer)),
                                                 FloatMemcpyFunctor3to3()),
                                                 "memcopy");
        eavlExecutor::Go();

        if(verbose) cerr << "TOTAL     RUNTIME: "<<eavlTimer::Stop(th,"raytrace")<<endl;
        //writeBMP(height,width,r2,g2,b2,"notA.bmp");
#endif
    //} 
    
    
    //writeBMP(height,width,r,g,b,(char*)scounter.c_str()); 
    
    
    cout<<"leaving execute"<<endl;
}
/*
void eavlRayTracerMutator::printMemUsage()
{
    size_t free_byte ;

    size_t total_byte ;

    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;

    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

}
*/




struct ThreshFunctor
{
    int thresh;
    ThreshFunctor(int t)
        : thresh(t)
    {}

    EAVL_FUNCTOR tuple<int> operator()(int in){
        int out;
        if(in<thresh) out=0;
        else out=1;
        return tuple<int>(out);
    }

};


int  eavlRayTracerMutator::compact()
{
    cout<<"COMPACTING"<<endl;


    int outputSize=0;

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(hitIdx),
                                             eavlOpArgs(mask),
                                             ThreshFunctor(0.0f)),
                                             "thresh");
    eavlExecutor::AddOperation(new eavlPrefixSumOp_1(mask,indexScan,false), //inclusive==true exclusive ==false
                                                     "create indexes");
    eavlExecutor::AddOperation(new eavlReduceOp_1<eavlAddFunctor<int> >
                              (mask,
                               count,
                               eavlAddFunctor<int>()),
                               "count output");
    eavlExecutor::Go();

    outputSize=count->GetValue(0);

    if (outputSize==currentSize || outputSize == 0) cout<<"Bailing out of Compact, nothing to do"<<endl;
    //else cout<<"Bailing out of Compact, no rays hit"<<endl;
    //return outputSize; //nothing to do here
    //cout<<"output size "<<outputSize<<endl;
    eavlIntArray   *reverseIndex    = new eavlIntArray ("reverseIndex",1,outputSize);
    eavlExecutor::AddOperation(new eavlSimpleReverseIndexOp(mask,
                                                            indexScan,
                                                            reverseIndex),
                                                            "generate reverse lookup");

    eavlExecutor::Go();
    cout<<"after reverse"<<endl;
    //Now start compacting
    compactIntArray(indexes , reverseIndex, outputSize);
    
    cout<<"First Compact"<<endl;
    compactFloatArray(rayDirX, reverseIndex, outputSize);
    compactFloatArray(rayDirY, reverseIndex, outputSize);
    compactFloatArray(rayDirZ, reverseIndex, outputSize);

    compactFloatArray(rayOriginX, reverseIndex, outputSize);
    compactFloatArray(rayOriginY, reverseIndex, outputSize);
    compactFloatArray(rayOriginZ, reverseIndex, outputSize);
   
    
    /* Must memset this to retain -1 for misses in the */
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(compactTempInt),
                                             eavlOpArgs(compactTempInt),
                                             IntMemsetFunctor(-1)),
                                            "force");
    eavlExecutor::Go();
    compactIntArray(hitIdx, reverseIndex, outputSize);//special
    delete reverseIndex;

    return outputSize;
}

void eavlRayTracerMutator::compactFloatArray(eavlFloatArray*& input, eavlIntArray* reverseIndex, int nitems)
{
    eavlFloatArray* tempPtr;
    eavlExecutor::AddOperation(new_eavlGatherOp(eavlOpArgs(input),
                                                eavlOpArgs(compactTempFloat),
                                                eavlOpArgs(reverseIndex),
                                                nitems),
                                                "compact array");
    eavlExecutor::Go();
    tempPtr=input;
    input=compactTempFloat;
    compactTempFloat=tempPtr;
}

void eavlRayTracerMutator::compactIntArray(eavlIntArray*& input, eavlIntArray* reverseIndex, int nitems)
{

    eavlIntArray* tempPtr;
    cout<<"Before"<<endl;
    eavlExecutor::AddOperation(new_eavlGatherOp(eavlOpArgs(input),
                                                eavlOpArgs(compactTempInt),
                                                eavlOpArgs(reverseIndex),
                                                nitems),
                                                "compact array");
    eavlExecutor::Go();
    cout<<"after"<<endl;

    tempPtr=input;
    input=compactTempInt;
    compactTempInt=tempPtr;
}

void eavlRayTracerMutator::clearFrameBuffer(eavlFloatArray *r,eavlFloatArray *g,eavlFloatArray *b)
{
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(r),
                                            eavlOpArgs(r,g,b),
                                            FloatMemsetFunctor1to3(0)),
                                            "memset");
    eavlExecutor::Go();
}
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

unsigned int morton2D(float x, float y)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    return xx*4  + yy*2 ;
}
struct raySort{
    int id;
    unsigned int mortonCode;

}; 
struct tri
{
    float x[3];
    float y[3];
    float z[3];

    float xNorm[3];
    float yNorm[3];
    float zNorm[3];

    unsigned int mortonCode;

    float unitCentroid[3];

};



bool spacialCompare(const raySort &lhs,const raySort &rhs)
{
    return lhs.mortonCode < rhs.mortonCode;
}
  


//creates the morton indexes that is used to create the ray directions per frame.
//only needs to be done once everytime a resolution is specified. Every other frame
//the morton indexes can just be recopied to the indexes array. Two seperate arrays
//are needed to support compaction. One day this will be done in parallel when a 
//sort is created. 
void eavlRayTracerMutator::createRays()
{
    float fwidth=(float)width;
    float fheight=(float)height;
    float  w,h;

    raySort *rayArray= new raySort[size]; // since this is happening every frame it should not be allocated and deleted constantly.
                                          //Fix: this only has to be generated once, then just memset the indexes every frame.
    for(int i=0; i<size;i++)
    {
        rayArray[i].id=i;
        w = (float)(i%width)/fwidth;
        h = (float) (i/width)/fheight;
        rayArray[i].mortonCode=morton2D(w,h);
    }
    std::sort(rayArray,rayArray+size,spacialCompare);
    cout<<endl;
    for(int i=0; i<size;i++)
    {
        mortonIndexes->SetValue(i, rayArray[i].id);
    }
    delete[] rayArray; 
} 

void eavlRayTracerMutator::traversalTest(int warmupRounds, int testRounds)
{
    
    Init();
    eavlIntArray *dummy= new eavlIntArray("",1,size);
    look=lookat-eye;
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes),
                                             eavlOpArgs(rayDirX,rayDirY,rayDirZ),
                                             RayGenFunctor(width,height,fovx,fovy,look,up)),
                                             "ray gen");
    eavlExecutor::Go();

    cout<<"Warming up "<<warmupRounds<<" rounds."<<endl;
    int warm = eavlTimer::Start(); //Dirs, origins
    for(int i=0; i<warmupRounds;i++)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx),
                                                 eavlOpArgs(dummy),
                                                 RayIntersectFunctor(vertsTex,bvhTex,bvhLeafsTex)),
                                                                                                    "intersect");
        eavlExecutor::Go();
    }

    float rayper=size/(eavlTimer::Stop(warm,"warm")/(float)warmupRounds);
    cout << "Warm up "<<rayper/1000000.f<< " Mrays/sec"<<endl;

    int test = eavlTimer::Start(); //Dirs, origins
    for(int i=0; i<testRounds;i++)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx),
                                                 eavlOpArgs(dummy),
                                                 RayIntersectFunctor(vertsTex,bvhTex,bvhLeafsTex)),
                                                                                                    "intersect");
        eavlExecutor::Go();
    }
    rayper=size/(eavlTimer::Stop(test,"test")/(float)testRounds);
    cout << "# "<<rayper/1000000.f<<endl;


    //verify output
    eavlFloatArray *depthBuffer= new eavlFloatArray("",1,size);
    eavlFloatArray *d= new eavlFloatArray("",1,size);

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx),
                                                 eavlOpArgs(depthBuffer),
                                                 RayIntersectFunctorDepth(vertsTex,bvhTex,bvhLeafsTex)),
                                                                                                    "intersect");
    eavlExecutor::Go();
    float maxDepth=0;
    float minDepth=INFINITE;

    for(int i=0; i< size; i++)
    {
        maxDepth= max(depthBuffer->GetValue(i), maxDepth);  
        minDepth= min(minDepth,depthBuffer->GetValue(i));
    } 
    //for(int i=0; i< size; i++) cout<<depthBuffer->GetValue(i)<<" ";
    maxDepth=maxDepth-minDepth;
    for(int i=0; i< size; i++) depthBuffer->SetValue(i, (depthBuffer->GetValue(i)-minDepth)/maxDepth);
    
    eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(depthBuffer),
                                                 eavlOpArgs(d),
                                                 eavlOpArgs(indexes)),
                                                "scatter");
    eavlExecutor::Go();

    writeBMP(height,width,d,d,d,"depth.bmp");
    delete depthBuffer;
    delete dummy;
    delete d;
    
}

void eavlRayTracerMutator::fpsTest(int warmupRounds, int testRounds)
{
    Init();
    look=lookat-eye;
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes),
                               eavlOpArgs(rayDirX,rayDirY,rayDirZ),
                               RayGenFunctor(width,height,fovx,fovy,look,up)),
                               "ray gen");
    eavlExecutor::Go();

    eavlFloatArray *eyex= new eavlFloatArray("x",1,size);
    eavlFloatArray *eyey= new eavlFloatArray("y",1,size);
    eavlFloatArray *eyez= new eavlFloatArray("z",1,size);

    for(int i=0;i<size;i++) 
    {
        eyex->SetValue(i,eye.x);
        eyey->SetValue(i,eye.y);
        eyez->SetValue(i,eye.z);
    }
    
    cout<<"Warming up "<<warmupRounds<<" rounds."<<endl;
    int warm = eavlTimer::Start(); //Dirs, origins
    for(int i=0; i<warmupRounds;i++)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,eyex,eyey,eyez,hitIdx),
                                                 eavlOpArgs(hitIdx),
                                                 RayIntersectFunctor(vertsTex,bvhTex, bvhLeafsTex)),
                                                 "intersect");

        eavlExecutor::Go();

        //eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,interX,interY,interZ,hitIdx),
        //                                         eavlOpArgs(interX, interY,interZ,alphas,betas),
        //                                         ReflectFunctorBasic(vertsTex,norms)),
        //                                         "reflectBasic");
        //eavlExecutor::Go();

        //eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(hitIdx,shadowHits,interX,interY,interZ,alphas,betas,ambPct),
        //                                         eavlOpArgs(r,g,b),
        //                                         ShaderFunctor(numTriangles,light,eye,norms,0,matIdx,mats)),
        //                                         "shader");
        //eavlExecutor::Go();
        cout<<"Warning: shading disabled"<<endl;
    }

    float rayper=size/(eavlTimer::Stop(warm,"warm")/(float)warmupRounds);
    cout << "Warm up "<<rayper/1000000.f<< " Mrays/sec"<<endl;

    int test = eavlTimer::Start(); //Dirs, origins
    for(int i=0; i<testRounds;i++)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,eyex,eyey,eyez,hitIdx),
                                                 eavlOpArgs(hitIdx),
                                                 RayIntersectFunctor(vertsTex,bvhTex, bvhLeafsTex)),
                                                 "intersect");

        eavlExecutor::Go();

        //eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,interX,interY,interZ,hitIdx),
        //                                         eavlOpArgs(interX, interY,interZ,alphas,betas),
        //                                         ReflectFunctorBasic(vertsTex,norms)),
        //                                         "reflectBasic");
        //eavlExecutor::Go();

        //eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(hitIdx,shadowHits,interX,interY,interZ,alphas,betas,ambPct),
        //                                         eavlOpArgs(r,g,b),
        //                                         ShaderFunctor(numTriangles,light,eye,norms,0,matIdx,mats)),
        //                                         "shader");
        //eavlExecutor::Go();
        cout<<"Warning: shading disabled BIG PROBLEMS FIX LATER"<<endl;
        if(!cpu)
        {
                eavlExecutor::AddOperation(
                    new_eavlScatterOp(eavlOpArgs(r),
                                eavlOpArgs(rOut),
                                eavlOpArgs(indexes)),
                                "scatter");
                eavlExecutor::Go();

                eavlExecutor::AddOperation(
                    new_eavlScatterOp(eavlOpArgs(g),
                                eavlOpArgs(gOut),
                                eavlOpArgs(indexes)),
                                "scatter");
                eavlExecutor::Go();
                eavlExecutor::AddOperation(
                    new_eavlScatterOp(eavlOpArgs(b),
                                eavlOpArgs(bOut),
                                eavlOpArgs(indexes)),
                                "scatter");
                eavlExecutor::Go();
            }
    }
    rayper=(float)testRounds/(eavlTimer::Stop(test,"test"));
    cout << "# "<<rayper<<endl;
    eavlExecutor::AddOperation(
                    new_eavlScatterOp(eavlOpArgs(r),
                                eavlOpArgs(rOut),
                                eavlOpArgs(indexes)),
                                "scatter");
                eavlExecutor::Go();

                eavlExecutor::AddOperation(
                    new_eavlScatterOp(eavlOpArgs(g),
                                eavlOpArgs(gOut),
                                eavlOpArgs(indexes)),
                                "scatter");
                eavlExecutor::Go();
                eavlExecutor::AddOperation(
                    new_eavlScatterOp(eavlOpArgs(b),
                                eavlOpArgs(bOut),
                                eavlOpArgs(indexes)),
                                "scatter");
                eavlExecutor::Go();

    writeBMP(height,width,rOut,gOut,bOut,outfilename);
    exit(0);
}


void eavlRayTracerMutator::cleanUp()
{
    delete bvhLeafsTex;
    delete bvhTex;
    delete vertsTex;
}

