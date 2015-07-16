#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlFilter.h"
#include "eavlTimer.h" 

#include <sys/time.h>
#include <string.h>
#include <cstdlib>
#include <ctime> 

#include "eavlRayTracerMutator.h"
#include "eavlNewIsoTables.h" // need this for eavl const array

//Operations
#include "eavlSimpleReverseIndexOp.h"
#include "eavlPrefixSumOp_1.h"
#include "eavl1toNScatterOp.h"
#include "eavlNto1GatherOp.h"
#include "eavlReduceOp_1.h"
#include "eavlScatterOp.h"
#include "eavlGatherOp.h"
#include "eavlMapOp.h"
//RT
#include "MortonBVHBuilder.h"
#include "eavlRTUtil.h"
#include "SplitBVH.h"


//declare the texture reference even if we are not using texture memory

/* Triangle textures */
texture<float4> tri_bvh_in_tref;            /* BVH inner nodes */
texture<float4> tri_verts_tref;             /* vert+ scalar data */
texture<float>  tri_bvh_lf_tref;            /* BVH leaf nodes */
texture<float>  tri_norms_tref;
/*Sphere Textures */
texture<float4> sphr_verts_tref;
texture<float4> sphr_bvh_in_tref;
texture<float>  sphr_bvh_lf_tref;
texture<float>  sphr_scalars_tref;
/*Cylinder textures */
texture<float4> cyl_bvh_in_tref;         
texture<float4> cyl_verts_tref;             
texture<float>  cyl_bvh_lf_tref;            
texture<float>  cyl_scalars_tref;

/*material Index textures*/
texture<int>    tri_matIdx_tref;
texture<int>    sphr_matIdx_tref;
texture<int>    cyl_matIdx_tref;

/*color map texture */
texture<float4> color_map_tref;

eavlConstTexArray<float4>* tri_bvh_in_array;
eavlConstTexArray<float4>* tri_verts_array;
eavlConstTexArray<float>*  tri_bvh_lf_array;
eavlConstTexArray<float>*  tri_norms_array;
eavlConstTexArray<int>*    tri_matIdx_array;    

eavlConstTexArray<float4>* sphr_bvh_in_array;
eavlConstTexArray<float4>* sphr_verts_array;
eavlConstTexArray<float>*  sphr_bvh_lf_array;
eavlConstTexArray<float>*  sphr_scalars_array;
eavlConstTexArray<int>*    sphr_matIdx_array;

eavlConstTexArray<float4>* cyl_bvh_in_array;
eavlConstTexArray<float4>* cyl_verts_array;
eavlConstTexArray<float>*  cyl_bvh_lf_array;
eavlConstTexArray<float>*  cyl_scalars_array;
eavlConstTexArray<int>*    cyl_matIdx_array;

eavlConstTexArray<float4>* cmap_array;

eavlRayTracerMutator::eavlRayTracerMutator()
{
    
    //eavlExecutor::SetExecutionMode(eavlExecutor::ForceCPU);
    if(eavlExecutor::GetExecutionMode() == eavlExecutor::ForceCPU ) cpu = true;
    else cpu = false;

    scene= new eavlRTScene(RTMaterial());
    height  = 1080;         //set up some defaults
    width   = 1920;
    depth   = 0;
    zoom    = 1.f;
    fovy    = 30;
    fovx    = 50;
    srand (time(NULL));   
    look.x  = 0;
    look.y  = 0;
    look.z  = -1;

    lookat.x= .001;
    lookat.y= 0;
    lookat.z= -30;

    eye.x   = 0;
    eye.y   = 0;
    eye.z   = -20;

    light.x = 0;
    light.y = 0;
    light.z = -20;
    
    bgColor.x = 0;
    bgColor.y = 0;
    bgColor.z = 0;

    compactOp   = false;
    geomDirty   = true;
    antiAlias   = false;
    sizeDirty   = true;
    cameraDirty = true;
    useBVHCache = false;
    shadowsOn   = true;

    fileprefix  = "output";
    filetype    = ".bmp";
    outfilename = "output.bmp";
    frameCounter= 0;
    isOccusionOn= false;
    occSamples  = 4;
    aoMax       = 1.f;
    verbose     = false;
    rayDirX     = NULL;

    redIndexer   = new eavlArrayIndexer(4,0);
    greenIndexer = new eavlArrayIndexer(4,1);
    blueIndexer  = new eavlArrayIndexer(4,2);
    alphaIndexer = new eavlArrayIndexer(4,3);
    
    cmap_array   = NULL;
    colorMap_raw = NULL;

    /*texture array Ptrs */
    tri_bvh_in_array    = NULL;
    tri_verts_array     = NULL;
    tri_bvh_lf_array    = NULL;
    tri_matIdx_array    = NULL;
    tri_norms_array     = NULL;

    sphr_bvh_in_array   = NULL;
    sphr_verts_array    = NULL;
    sphr_bvh_lf_array   = NULL;
    sphr_scalars_array  = NULL;
    sphr_matIdx_array   = NULL;

    cyl_bvh_in_array   = NULL;
    cyl_verts_array    = NULL;
    cyl_bvh_lf_array   = NULL;
    cyl_scalars_array  = NULL;
    cyl_matIdx_array   = NULL;

    /* Raw arrays */
    tri_verts_raw    = NULL;
    tri_norms_raw    = NULL;
    tri_matIdx_raw   = NULL;
    tri_bvh_in_raw   = NULL;
    tri_bvh_lf_raw   = NULL;

    sphr_verts_raw   = NULL;
    sphr_matIdx_raw  = NULL;
    sphr_bvh_in_raw  = NULL;
    sphr_bvh_lf_raw  = NULL;
    sphr_scalars_raw = NULL;

    cyl_verts_raw   = NULL;
    cyl_matIdx_raw  = NULL;
    cyl_bvh_in_raw  = NULL;
    cyl_bvh_lf_raw  = NULL;
    cyl_scalars_raw = NULL;

    mats_raw         = NULL;




    rayDirX = NULL;
    rayDirY = NULL;
    rayDirZ = NULL;

    rayOriginX = NULL;
    rayOriginY = NULL;
    rayOriginZ = NULL;

    r = NULL;
    g = NULL;
    b = NULL;

    alphas = NULL;
    betas = NULL;

    interX = NULL;
    interY = NULL;
    interZ = NULL;

    normX = NULL;
    normY = NULL;
    normZ = NULL;
 
    hitIdx = NULL;
    indexes = NULL;
    mortonIndexes = NULL;
    shadowHits = NULL;

    r2 = NULL;
    b2 = NULL;
    g2 = NULL;
    ambPct = NULL;
    zBuffer = NULL;
    frameBuffer = NULL;
    scalars = NULL;
    primitiveTypeHit = NULL;
    minDistances = NULL;

    compactTempInt = NULL;
    compactTempFloat = NULL;

    rOut = NULL;
    gOut = NULL;
    bOut = NULL;
    occX = NULL;
    occY = NULL;
    occZ = NULL;
    localHits = NULL;
    tempAmbPct = NULL;
    occIndexer = NULL;
    mask = NULL;
    count = NULL;
    indexScan = NULL;
    
    fastBVHBuild = true; //Use the Morton LBVH Builder

    setDefaultColorMap();
    if(verbose) cout<<"Constructor finished\n";
}

void eavlRayTracerMutator::setColorMap3f(float* cmap, int size)
{
    colorMapSize=size;
    if (verbose) cout<<"Setting color map size "<<colorMapSize<<endl;

    if(cmap_array != NULL)
    {
        cmap_array->unbind(color_map_tref);

        delete cmap_array;
       
        cmap_array = NULL;
    }
    if(colorMap_raw != NULL)
    {
        delete[] colorMap_raw;
        colorMap_raw = NULL;
    }
    colorMap_raw = new float[size*4];
    
    for(int i = 0; i < size; i++)
    {
        colorMap_raw[i*4  ] = cmap[i*3  ];
        colorMap_raw[i*4+1] = cmap[i*3+1];
        colorMap_raw[i*4+2] = cmap[i*3+2];
        colorMap_raw[i*4+3] = 0;
    }
    cmap_array = new eavlConstTexArray<float4>((float4*)colorMap_raw, colorMapSize, color_map_tref, cpu);
}
void eavlRayTracerMutator::setDefaultColorMap()
{
    if(verbose) cout<<"Setting default color map"<<endl;
    if(cmap_array!=NULL)
    {
        cmap_array->unbind(color_map_tref);
        delete cmap_array;
        cmap_array = NULL;
    }
    if(colorMap_raw!=NULL)
    {
        delete[] colorMap_raw;
        colorMap_raw = NULL;
    }
    //two values all 1s
    colorMapSize=2;
    colorMap_raw= new float[8];
    for(int i=0;i<8;i++) colorMap_raw[i]=1.f;
    cmap_array = new eavlConstTexArray<float4>((float4*)colorMap_raw, colorMapSize, color_map_tref, cpu);

}

void eavlRayTracerMutator::setCompact(bool comp)
{
    compactOp = comp;
}

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
        int hitIdx = get<6>(input);
        if(hitIdx == -1) tuple<float,float,float>(0.f,0.f,0.f);
        eavlVector3 normal(get<0>(input),get<1>(input),get<2>(input));
        
        //rng warm up
        eavlVector3 absNormal;
        absNormal.x = abs(normal.x);
        absNormal.y = abs(normal.y);
        absNormal.z = abs(normal.z);
        
        float maxN = max(max(absNormal.x,absNormal.y),absNormal.z);
        eavlVector3 perp = eavlVector3(normal.y,-normal.x,0.f);
        if(maxN == absNormal.z)  
        {
            perp.x = 0.f;
            perp.y = normal.z;
            perp.z = -normal.y;
        }
        else if (maxN == absNormal.x)
        {
            perp.x = -normal.z;
            perp.y = 0.f;
            perp.z = normal.x;
        }
        perp.normalize(); 

        eavlVector3 biperp = normal % perp;

        unsigned int hashA = 6371625 + seed;
        unsigned int hashB = 0x9e3779b9u;
        unsigned int hashC = 0x9e3779b9u;
        jenkinsMix(hashA, hashB, hashC);
        jenkinsMix(hashA, hashB, hashC);
        float angle = 2.f * PI * (float)hashC * exp2(-32.f);
        eavlVector3 t0 = perp * cosf(angle) + biperp * sinf(angle);
        eavlVector3 t1 = perp * -sinf(angle) + biperp * cosf(angle);

        float  x = 0.0f;
        float  xadd = 1.0f;
        unsigned int hc2 = 1 + sampleNum;
        while (hc2 != 0)
        {
            xadd *= 0.5f;
            if ((hc2 & 1) != 0)
                x += xadd;
            hc2 >>= 1;
        }

        float  y = 0.0f;
        float  yadd = 1.0f;
        int hc3 = 1 + sampleNum;
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
        return tuple<float,float,float>(dir.x,dir.y,dir.z);
    }

   

};

#define INIT(TYPE, TABLE, COUNT)            \
{                                           \
TABLE = new TYPE(TABLE ## _raw, COUNT);     \
}

//if this is called we know that the there is a hit
EAVL_HOSTDEVICE void triangleIntersectionDistance(const eavlVector3 ray,const eavlVector3 rayOrigin,const eavlVector3 a, const eavlVector3 b, const eavlVector3 c, float &tempDist)
{
    eavlVector3 intersect,normal;
    float d,dot;//,area;
    normal = (b-a)%(c-a);                                
    dot = normal * ray;
    d = normal * a; 
    tempDist = (d - normal*rayOrigin) / dot; 
}

//if this is called we know that the there is a hit
EAVL_HOSTDEVICE eavlVector3 triangleIntersectionABG(const eavlVector3 ray,const eavlVector3 rayOrigin,const eavlVector3 a, const eavlVector3 b, const eavlVector3 c, int index, float &alpha,float &beta)
{

    eavlVector3 intersect , normal;
    float tempDistance, d, dot, area;
    normal = (b-a) % (c-a);
    dot = normal * ray;
    //if(dot<TOLERANCE && dot >-TOLERANCE) return eavlVector3(-9999,-9999,-9999);    //traingle is parallel to the ray
    d = normal * a; //solve for d using any point on the plane
    tempDistance = (d - normal*rayOrigin) / dot; //**this could be preprocessed as well, but could be cost prohibitive if a lot of triangles are parallel to the ray( not likely i think)
    intersect=rayOrigin + ray * tempDistance;
    //inside test
    alpha= ((c-b) % (intersect-b)) * normal; //angles between the intersect point and edges
    beta = ((a-c) % (intersect-c)) * normal;
    // this is for the barycentric coordinates for color, normal lerping.
    area = normal * normal;
    alpha = alpha / area;
    beta = beta / area;
    return intersect;
}



EAVL_HOSTDEVICE int getIntersectionTri(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstTexArray<float4> *bvh,
                                       const eavlConstTexArray<float> *tri_bvh_lf_raw, const eavlConstTexArray<float4> *verts,const float &maxDistance, float &distance)
{


    float minDistance = maxDistance;
    int   minIndex    = -1;
    
    float dirx = rayDir.x;
    float diry = rayDir.y;
    float dirz = rayDir.z;

    float invDirx = rcp_safe(dirx);
    float invDiry = rcp_safe(diry);
    float invDirz = rcp_safe(dirz);
    int currentNode;
  
    int todo[64]; //num of nodes to process
    int stackptr = 0;
    int barrier = (int)END_FLAG;
    currentNode = 0;

    todo[stackptr] = barrier;

    float ox = rayOrigin.x;
    float oy = rayOrigin.y;
    float oz = rayOrigin.z;
    float odirx = ox * invDirx;
    float odiry = oy * invDiry;
    float odirz = oz * invDirz;

    while(currentNode != END_FLAG) {

        if(currentNode>-1)
        {

            float4 n1 = bvh->getValue(tri_bvh_in_tref, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
            float4 n2 = bvh->getValue(tri_bvh_in_tref, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3 = bvh->getValue(tri_bvh_in_tref, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
            
            float txmin0 = n1.x * invDirx - odirx;       
            float tymin0 = n1.y * invDiry - odiry;         
            float tzmin0 = n1.z * invDirz - odirz;
            float txmax0 = n1.w * invDirx - odirx;
            float tymax0 = n2.x * invDiry - odiry;
            float tzmax0 = n2.y * invDirz - odirz;
           
            float tmin0 = max(max(max(min(tymin0,tymax0),min(txmin0,txmax0)),min(tzmin0,tzmax0)),0.f);
            float tmax0 = min(min(min(max(tymin0,tymax0),max(txmin0,txmax0)),max(tzmin0,tzmax0)), minDistance);
            
            bool traverseChild0 = (tmax0 >= tmin0);

             
            float txmin1 = n2.z * invDirx - odirx;       
            float tymin1 = n2.w * invDiry - odiry;
            float tzmin1 = n3.x * invDirz - odirz;
            float txmax1 = n3.y * invDirx - odirx;
            float tymax1 = n3.z * invDiry-  odiry;
            float tzmax1 = n3.w * invDirz - odirz;
            float tmin1 = max(max(max(min(tymin1,tymax1),min(txmin1,txmax1)),min(tzmin1,tzmax1)),0.f);
            float tmax1 = min(min(min(max(tymin1,tymax1),max(txmin1,txmax1)),max(tzmin1,tzmax1)), minDistance);
            
            bool traverseChild1 = (tmax1 >= tmin1);

        if(!traverseChild0 && !traverseChild1)
        {

            currentNode = todo[stackptr]; //go back put the stack
            stackptr--;
        }
        else
        {
            float4 n4 = bvh->getValue(tri_bvh_in_tref, currentNode+3); //(leftChild, rightChild, pad,pad)
            int leftChild = (int)n4.x;
            int rightChild = (int)n4.y;

            currentNode = (traverseChild0) ? leftChild : rightChild;
            if(traverseChild1 && traverseChild0)
            {
                if(tmin0 > tmin1)
                {

                   
                    currentNode = rightChild;
                    stackptr++;
                    todo[stackptr] = leftChild;
                }
                else
                {   
                    stackptr++;
                    todo[stackptr] = rightChild;
                }


            }
        }
        }
        
        if(currentNode < 0 && currentNode != barrier)//check register usage
        {
            

            currentNode = -currentNode - 1; //swap the neg address 
            int numTri = (int)tri_bvh_lf_raw->getValue(tri_bvh_lf_tref,currentNode)+1;

            for(int i = 1; i < numTri; i++)
            {        
                    int triIndex = (int)tri_bvh_lf_raw->getValue(tri_bvh_lf_tref,currentNode+i);
                   
                    float4 a4 = verts->getValue(tri_verts_tref, triIndex*3);
                    float4 b4 = verts->getValue(tri_verts_tref, triIndex*3+1);
                    float4 c4 = verts->getValue(tri_verts_tref, triIndex*3+2);
                    eavlVector3 e1( a4.w - a4.x , b4.x - a4.y, b4.y - a4.z ); 
                    eavlVector3 e2( b4.z - a4.x , b4.w - a4.y, c4.x - a4.z );


                    eavlVector3 p;
                    p.x = diry * e2.z - dirz * e2.y;
                    p.y = dirz * e2.x - dirx * e2.z;
                    p.z = dirx * e2.y - diry * e2.x;
                    float dot = e1.x * p.x + e1.y * p.y + e1.z * p.z;
                    if(dot != 0.f)
                    {
                        dot = 1.f/dot;
                        eavlVector3 t;
                        t.x = ox - a4.x;
                        t.y = oy - a4.y;
                        t.z = oz - a4.z;

                        float u = (t.x* p.x + t.y * p.y + t.z * p.z) * dot;
                        if(u >= (0.f - EPSILON) && u <= (1.f + EPSILON))
                        {
                            eavlVector3 q; // = t % e1;
                            q.x = t.y * e1.z - t.z * e1.y;
                            q.y = t.z * e1.x - t.x * e1.z;
                            q.z = t.x * e1.y - t.y * e1.x;
                            float v = (dirx * q.x + diry * q.y + dirz * q.z) * dot;
                            if(v >= (0.f - EPSILON) && v <= (1.f + EPSILON))
                            {
                                float dist = (e2.x * q.x + e2.y * q.y + e2.z * q.z) * dot;
                                if((dist > EPSILON && dist < minDistance) && !(u + v > 1) )
                                {
                                    minDistance = dist;
                                    minIndex = triIndex;
                                    if(occlusion) return minIndex;//or set todo to -1
                                }
                            }
                        }

                    }
                   
            }
            currentNode = todo[stackptr];
            stackptr--;
        }

    }
 distance = minDistance;
 return minIndex;
}


EAVL_HOSTDEVICE int getIntersectionSphere(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstTexArray<float4> *bvh,
                                          const eavlConstTexArray<float> *bvh_lf,const eavlConstTexArray<float4> *verts, const float &maxDistance, float &distance)
{


    float minDistance = maxDistance;
    int   minIndex    = -1;
    
    float dirx = rayDir.x;
    float diry = rayDir.y;
    float dirz = rayDir.z;

    float invDirx = rcp_safe(dirx);
    float invDiry = rcp_safe(diry);
    float invDirz = rcp_safe(dirz);
    int currentNode;
  
    int todo[64]; //num of nodes to process
    int stackptr = 0;
    int barrier = (int)END_FLAG;
    currentNode = 0;

    todo[stackptr] = barrier;

    float ox = rayOrigin.x;
    float oy = rayOrigin.y;
    float oz = rayOrigin.z;
    float odirx = ox*invDirx;
    float odiry = oy*invDiry;
    float odirz = oz*invDirz;

    while(currentNode != END_FLAG) {

        if(currentNode > -1)
        {

            float4 n1 = bvh->getValue(sphr_bvh_in_tref, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
            float4 n2 = bvh->getValue(sphr_bvh_in_tref, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3 = bvh->getValue(sphr_bvh_in_tref, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
            
            float txmin0 = n1.x* invDirx - odirx;       
            float tymin0 = n1.y* invDiry - odiry;         
            float tzmin0 = n1.z* invDirz - odirz;
            float txmax0 = n1.w* invDirx - odirx;
            float tymax0 = n2.x* invDiry - odiry;
            float tzmax0 = n2.y* invDirz - odirz;
           
            float tmin0 = max(max(max(min(tymin0,tymax0),min(txmin0,txmax0)),min(tzmin0,tzmax0)),0.f);
            float tmax0 = min(min(min(max(tymin0,tymax0),max(txmin0,txmax0)),max(tzmin0,tzmax0)), minDistance);
            
            bool traverseChild0 = (tmax0 >= tmin0);

             
            float txmin1 = n2.z* invDirx - odirx;       
            float tymin1 = n2.w* invDiry - odiry;
            float tzmin1 = n3.x* invDirz - odirz;
            float txmax1 = n3.y* invDirx - odirx;
            float tymax1 = n3.z* invDiry-  odiry;
            float tzmax1 = n3.w* invDirz - odirz;
            float tmin1 = max(max(max(min(tymin1,tymax1),min(txmin1,txmax1)),min(tzmin1,tzmax1)),0.f);
            float tmax1 = min(min(min(max(tymin1,tymax1),max(txmin1,txmax1)),max(tzmin1,tzmax1)), minDistance);
            
            bool traverseChild1 = (tmax1 >= tmin1);

            if(!traverseChild0 && !traverseChild1)
            {

                currentNode = todo[stackptr]; //go back put the stack
                stackptr--;
            }
            else
            {
                float4 n4 = bvh->getValue(sphr_bvh_in_tref, currentNode+3); //(leftChild, rightChild, pad,pad)
                int leftChild = (int)n4.x;
                int rightChild = (int)n4.y;

                currentNode = (traverseChild0) ? leftChild : rightChild;
                if(traverseChild1 && traverseChild0)
                {
                    if(tmin0 > tmin1)
                    {

                       
                        currentNode = rightChild;
                        stackptr++;
                        todo[stackptr] = leftChild;
                    }
                    else
                    {   
                        stackptr++;
                        todo[stackptr] = rightChild;
                    }


                }
            }
        }
        
        if(currentNode < 0 && currentNode != barrier)//check register usage
        {
            currentNode = -currentNode - 1; //swap the neg address
            int numSheres = (int)bvh_lf->getValue(sphr_bvh_lf_tref,currentNode)+1;
 
            for(int i = 1; i < numSheres; i++)
            {        
                int sphereIndex = (int)bvh_lf->getValue(sphr_bvh_lf_tref,currentNode+i);
                
                float4 data = verts->getValue(sphr_verts_tref, sphereIndex);

                float lx = data.x-ox;
                float ly = data.y-oy;
                float lz = data.z-oz;

                float dot1 = lx*dirx+ly*diry+lz*dirz;
                if(dot1 >= 0)
                {
                    float d  = lx*lx + ly*ly + lz*lz - dot1*dot1;
                    float r2 = data.w * data.w;
                    if(d <= r2 )
                    {
                        float tch = sqrt(r2 - d);
                        float t0  = dot1 - tch;
                        //float t1 = dot1+tch; /* if t1 is > 0 and t0<0 then the ray is inside the sphere.

                        if( t0 < minDistance && t0 > 0)
                        {
                            minIndex = sphereIndex;
                            minDistance = t0;
                            if(occlusion) return minIndex;
                        }
                        
                    }
                }

            }
            currentNode = todo[stackptr];
            stackptr--;
        }

    }
 distance = minDistance;
 return minIndex;
}

EAVL_HOSTDEVICE int getIntersectionCyl(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstTexArray<float4> *bvh,
                                       const eavlConstTexArray<float> *cyl_bvh_lf_raw, const eavlConstTexArray<float4> *verts,const float &maxDistance, float &distance)
{


    float minDistance = maxDistance;
    int   minIndex    = -1;
    
    float dirx = rayDir.x;
    float diry = rayDir.y;
    float dirz = rayDir.z;

    float invDirx = rcp_safe(dirx);
    float invDiry = rcp_safe(diry);
    float invDirz = rcp_safe(dirz);
    int currentNode;
  
    int todo[64]; //num of nodes to process
    int stackptr = 0;
    int barrier = (int)END_FLAG;
    currentNode = 0;

    todo[stackptr] = barrier;

    float ox = rayOrigin.x;
    float oy = rayOrigin.y;
    float oz = rayOrigin.z;
    float odirx = ox * invDirx;
    float odiry = oy * invDiry;
    float odirz = oz * invDirz;

    while(currentNode != END_FLAG) {

        if(currentNode>-1)
        {

            float4 n1 = bvh->getValue(cyl_bvh_in_tref, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
            float4 n2 = bvh->getValue(cyl_bvh_in_tref, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3 = bvh->getValue(cyl_bvh_in_tref, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
            
            float txmin0 = n1.x * invDirx - odirx;       
            float tymin0 = n1.y * invDiry - odiry;         
            float tzmin0 = n1.z * invDirz - odirz;
            float txmax0 = n1.w * invDirx - odirx;
            float tymax0 = n2.x * invDiry - odiry;
            float tzmax0 = n2.y * invDirz - odirz;
           
            float tmin0 = max(max(max(min(tymin0,tymax0),min(txmin0,txmax0)),min(tzmin0,tzmax0)),0.f);
            float tmax0 = min(min(min(max(tymin0,tymax0),max(txmin0,txmax0)),max(tzmin0,tzmax0)), minDistance);
            
            bool traverseChild0 = (tmax0 >= tmin0);

             
            float txmin1 = n2.z * invDirx - odirx;       
            float tymin1 = n2.w * invDiry - odiry;
            float tzmin1 = n3.x * invDirz - odirz;
            float txmax1 = n3.y * invDirx - odirx;
            float tymax1 = n3.z * invDiry-  odiry;
            float tzmax1 = n3.w * invDirz - odirz;
            float tmin1 = max(max(max(min(tymin1,tymax1),min(txmin1,txmax1)),min(tzmin1,tzmax1)),0.f);
            float tmax1 = min(min(min(max(tymin1,tymax1),max(txmin1,txmax1)),max(tzmin1,tzmax1)), minDistance);
            
            bool traverseChild1 = (tmax1 >= tmin1);

        if(!traverseChild0 && !traverseChild1)
        {

            currentNode = todo[stackptr]; //go back put the stack
            stackptr--;
        }
        else
        {
            float4 n4 = bvh->getValue(cyl_bvh_in_tref, currentNode+3); //(leftChild, rightChild, pad,pad)
            int leftChild = (int)n4.x;
            int rightChild = (int)n4.y;

            currentNode = (traverseChild0) ? leftChild : rightChild;
            if(traverseChild1 && traverseChild0)
            {
                if(tmin0 > tmin1)
                {

                   
                    currentNode = rightChild;
                    stackptr++;
                    todo[stackptr] = leftChild;
                }
                else
                {   
                    stackptr++;
                    todo[stackptr] = rightChild;
                }


            }
        }
        }
        
        if(currentNode < 0 && currentNode != barrier)//check register usage
        {

            currentNode = -currentNode - 1; //swap the neg address
            int numCyl = (int)cyl_bvh_lf_raw->getValue(cyl_bvh_lf_tref,currentNode)+1;
            eavlVector3 dir(dirx, diry, dirz);
            eavlVector3 o(ox, oy, oz);
            for(int i = 1; i < numCyl; i++)
            {        
                int cylIndex = (int)cyl_bvh_lf_raw->getValue(cyl_bvh_lf_tref,currentNode+i);
               
                float4 a4 = verts->getValue(cyl_verts_tref, cylIndex*2);   /*basePoint + radius*/ 
                float4 b4 = verts->getValue(cyl_verts_tref, cylIndex*2+1); /*axis + height*/
                
                eavlVector3 basePoint(a4.x, a4.y, a4.z);
                eavlVector3 axis(b4.x, b4.y, b4.z);
                /* project vectors onto the plane defined by the axis*/
                eavlVector3 pdir = dir - (axis * dir) * axis;
                eavlVector3 po = o - (axis * o) * axis;
                eavlVector3 pc = basePoint - (axis * basePoint) * axis;
                /* get quadratic values*/
                eavlVector3 L = po - pc;
                float a = pdir * pdir;
                float b = 2 * pdir * L;
                float c = L * L - a4.w * a4.w; 

                float t0 = INFINITE;
                float t1 = INFINITE;

                solveQuadratic(a, b, c, t0,t1);
                float dist1 = INFINITE;
                float dist2 = INFINITE; //TODO consolidate t0-dist1
                
                if(t0 > 0)
                {
                    eavlVector3 hit = o + t0 * dir;
                    eavlVector3 hp = hit - basePoint; 
                    float dot = hp * axis;
                    if(dot > 0 && dot < b4.w) /*b4.w == height of cylinder */
                    {
                        dist1 = t0;
                    }
                }

                if(t1 > 0)
                {
                    eavlVector3 hit = o + t1 * dir;
                    eavlVector3 hp = hit - basePoint; 
                    float dot = hp * axis;
                    if(dot > 0 && dot < b4.w)
                    {
                        dist2 = t1;
                    }
                }
    
                dist1 = min(dist1, dist2);

                if(dist1 >EPSILON && dist1 > 0 && dist1 != INFINITE)
                {
                    minDistance = dist1;
                    minIndex = cylIndex;
                    if(occlusion) return minIndex;
                }


            }
            currentNode = todo[stackptr];
            stackptr--;
        }

    }
 distance = minDistance;
 return minIndex;
}

/* it is known that there is an intersection */
EAVL_HOSTDEVICE float intersectSphereDist(eavlVector3 rayDir, eavlVector3 rayOrigin, float4 sphere)
{
    eavlVector3 center(sphere.x, sphere.y,sphere.z);
    eavlVector3 l = center - rayOrigin;
    //cout<<l<<(l*l)<<endl;
    float dot = l * rayDir;
    float d = l*l - dot*dot;
    float r2 = sphere.w * sphere.w;
    float tch = 0;
    if(d < r2) tch = sqrt(r2 - d); /* Floating point precision is effecting this. Should already be a intersection.*/
    return dot-tch;

}

EAVL_HOSTDEVICE float getIntersectionDepth(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstTexArray<float4> &bvh,const eavlConstTexArray<float> &tri_bvh_lf_raw,eavlConstTexArray<float4> &verts,const float &maxDistance)
{


    float minDistance = maxDistance;
    int minIndex = -1;
    
    float dirx = rayDir.x;
    float diry = rayDir.y;
    float dirz = rayDir.z;

    float invDirx = rcp_safe(dirx);
    float invDiry = rcp_safe(diry);
    float invDirz = rcp_safe(dirz);
    int currentNode;
  

    int todo[64]; //num of nodes to process
    int stackptr = 0;
    int barrier = (int)END_FLAG;
    currentNode = 0;

    todo[stackptr] = barrier;

    float ox = rayOrigin.x;
    float oy = rayOrigin.y;
    float oz = rayOrigin.z;
    float odirx = ox * invDirx;
    float odiry = oy * invDiry;
    float odirz = oz * invDirz;

    while(currentNode != END_FLAG) {
        
        
        if(currentNode > -1)
        {
            float4 n1 = bvh.getValue(tri_bvh_in_tref, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0
            float4 n2=bvh.getValue(tri_bvh_in_tref, currentNode+1);   //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3=bvh.getValue(tri_bvh_in_tref, currentNode+2);   //(tzmin1, txmax1, tymax1, tzmax1)
            
            float txmin0 =   n1.x* invDirx - odirx;       
            float tymin0 =   n1.y* invDiry - odiry;         
            float tzmin0 =   n1.z* invDirz - odirz;
            float txmax0 =   n1.w* invDirx - odirx;
            float tymax0 =   n2.x* invDiry - odiry;
            float tzmax0 =   n2.y* invDirz - odirz;
           
            float tmin0 = max(max(max(min(tymin0,tymax0),min(txmin0,txmax0)),min(tzmin0,tzmax0)),0.f);
            float tmax0 = min(min(min(max(tymin0,tymax0),max(txmin0,txmax0)),max(tzmin0,tzmax0)), minDistance);
            
            bool traverseChild0 = (tmax0 >= tmin0);
            //test if re-using the same variables makes a difference with instruction level parallism
             
            float txmin1 =   n2.z* invDirx - odirx;       
            float tymin1 =   n2.w* invDiry - odiry;
            float tzmin1 =   n3.x* invDirz - odirz;
            float txmax1 =   n3.y* invDirx - odirx;
            float tymax1 =   n3.z* invDiry-  odiry;
            float tzmax1 =   n3.w* invDirz - odirz;
            float tmin1 = max(max(max(min(tymin1,tymax1),min(txmin1,txmax1)),min(tzmin1,tzmax1)),0.f);
            float tmax1 = min(min(min(max(tymin1,tymax1),max(txmin1,txmax1)),max(tzmin1,tzmax1)), minDistance);
            
            bool traverseChild1 = (tmax1 >= tmin1);

        if(!traverseChild0 && !traverseChild1)
        {
            currentNode = todo[stackptr]; //go back put the stack
            stackptr--;
        }
        else
        {
            float4 n4 = bvh.getValue(tri_bvh_in_tref, currentNode+3); //(leftChild, rightChild, pad,pad)
            int leftChild  = (int)n4.x;
            int rightChild = (int)n4.y;
            currentNode = (traverseChild0) ? leftChild : rightChild;
            if(traverseChild1 && traverseChild0)
            {
                if(tmin0 > tmin1)
                {

                   
                    currentNode = rightChild;
                    stackptr++;
                    todo[stackptr] = leftChild;
                }
                else
                {   
                    stackptr++;
                    todo[stackptr] = rightChild;
                }


            }
        }
        }
        
        if(currentNode < 0 && currentNode != barrier)//check register usage
        {
            

            currentNode = -currentNode - 1; //swap the neg address
            int numTri = (int)tri_bvh_lf_raw.getValue(tri_bvh_lf_tref,currentNode)+1;

            for(int i = 1; i < numTri; i++)
            {        
                    int triIndex = (int)tri_bvh_lf_raw.getValue(tri_bvh_lf_tref,currentNode+i);
                    float4 a4 = verts.getValue(tri_verts_tref, triIndex*3);
                    float4 b4 = verts.getValue(tri_verts_tref, triIndex*3+1);
                    float4 c4 = verts.getValue(tri_verts_tref, triIndex*3+2);
                    eavlVector3 e1( a4.w-a4.x , b4.x-a4.y, b4.y-a4.z ); 
                    eavlVector3 e2( b4.z-a4.x , b4.w-a4.y, c4.x-a4.z );

                    eavlVector3 p;
                    p.x = diry*e2.z-dirz*e2.y;
                    p.y = dirz*e2.x-dirx*e2.z;
                    p.z = dirx*e2.y-diry*e2.x;
                    float dot = e1*p;
                    if(dot != 0.f)
                    {
                        dot = 1.f / dot;
                        eavlVector3 t;
                        t.x = ox - a4.x;
                        t.y = oy - a4.y;
                        t.z = oz - a4.z;

                        float u = (t * p) * dot;
                        if(u >= 0.f && u <= 1.f)
                        {
                            eavlVector3 q = t % e1;
                            float v = (dirx*q.x + diry*q.y + dirz*q.z) * dot;
                            if(v >= 0.f && v <= 1.f)
                            {
                                float dist = (e2 * q) * dot;
                                if((dist > EPSILON && dist < minDistance) && !(u + v > 1) )
                                {
                                    minDistance = dist;
                                    minIndex = triIndex;
                                    if(occlusion) return minIndex;//or set todo to -1
                                }
                            }
                        }

                    }
                   
            }
            currentNode = todo[stackptr];
            stackptr--;
        }

    }

 if(minIndex == -1) return INFINITE; 
 else  return minDistance;
}

struct RayIntersectFunctor{


    const eavlConstTexArray<float4> *verts;
    const eavlConstTexArray<float4> *bvh;
    const eavlConstTexArray<float>  *bvh_inner;
    primitive_t                     primitiveType;

    RayIntersectFunctor(const eavlConstTexArray<float4> *_verts, const eavlConstTexArray<float4> *theBvh,
                        const eavlConstTexArray<float> *_bvh_inner, primitive_t _primitveType)
        :verts(_verts),
         bvh(theBvh),
         bvh_inner(_bvh_inner),
         primitiveType(_primitveType)
    {}                                                 
    EAVL_HOSTDEVICE tuple<int,float,int> operator()( tuple<float,float,float,float,float,float,int, int, float> rayTuple){
       
        
        int hitIdx = get<6>(rayTuple);
        if(hitIdx == -1) return tuple<int,float,int>(-1, INFINITE, -1);

        int   minHit = -1; 
        float distance;
        float maxDistance = get<8>(rayTuple);
        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));
        eavlVector3       ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        if(primitiveType == TRIANGLE)
        {
            minHit = getIntersectionTri(ray, rayOrigin, false,bvh,bvh_inner, verts,maxDistance,distance);
        } 
        else if(primitiveType == SPHERE)
        {
            
            minHit = getIntersectionSphere(ray, rayOrigin, false,bvh,bvh_inner, verts,maxDistance,distance);
        }
        else if(primitiveType == CYLINDER)
        {
            
            minHit = getIntersectionCyl(ray, rayOrigin, false,bvh,bvh_inner, verts,maxDistance,distance);
        }
        
        if(minHit!=-1) return tuple<int,float,int>(minHit, distance, primitiveType);
        else           return tuple<int,float,int>(hitIdx, INFINITE, get<7>(rayTuple));
    }
};


struct ReflectTriFunctor{

    const eavlConstTexArray<float4> *verts;
    const eavlConstTexArray<float>  *norms;


    ReflectTriFunctor(const eavlConstTexArray<float4> *_verts,const eavlConstTexArray<float>  *_norms)
        :verts(_verts),
         norms(_norms)
    {
        
    }                                                    //order a b c clockwise
    EAVL_FUNCTOR tuple<float,float,float,float,float, float,float,float, float,float,float, float> operator()( tuple<float,float,float,float,float,float,int, int> rayTuple){
       
        
        int hitIndex=get<6>(rayTuple);//hack for now
        int primitiveType=get<7>(rayTuple);
        if(hitIndex == -1 || primitiveType != TRIANGLE) return tuple<float,float,float,float,float, float,float,float, float,float,float, float>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
        
        eavlVector3 intersect;

        eavlVector3 reflection(0,0,0);
        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));

        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        float alpha = 0, beta = 0, gamma = 0;

        float4 a4 = verts->getValue(tri_verts_tref, hitIndex * 3);
        float4 b4 = verts->getValue(tri_verts_tref, hitIndex * 3 + 1);
        float4 c4 = verts->getValue(tri_verts_tref, hitIndex * 3 + 2); //scalars are stored in c.yzw
        eavlVector3 a(a4.x,a4.y,a4.z);
        eavlVector3 b(a4.w,b4.x,b4.y);
        eavlVector3 c(b4.z,b4.w,c4.x);
        intersect = triangleIntersectionABG(ray, rayOrigin, a,b,c,0.0f,alpha,beta);
        gamma = 1 - alpha - beta;
        eavlVector3 normal(999999.f,0,0);

        eavlVector3 aNorm, bNorm, cNorm;
        aNorm.x = norms->getValue(tri_norms_tref, hitIndex * 9 + 0);
        aNorm.y = norms->getValue(tri_norms_tref, hitIndex * 9 + 1);
        aNorm.z = norms->getValue(tri_norms_tref, hitIndex * 9 + 2);
        bNorm.x = norms->getValue(tri_norms_tref, hitIndex * 9 + 3);
        bNorm.y = norms->getValue(tri_norms_tref, hitIndex * 9 + 4);
        bNorm.z = norms->getValue(tri_norms_tref, hitIndex * 9 + 5);
        aNorm.x = norms->getValue(tri_norms_tref, hitIndex * 9 + 6);
        aNorm.y = norms->getValue(tri_norms_tref, hitIndex * 9 + 7);
        aNorm.z = norms->getValue(tri_norms_tref, hitIndex * 9 + 8);

        normal = aNorm*alpha + bNorm*beta + cNorm*gamma;
        float lerpedScalar = c4.y*alpha + c4.z*beta + c4.w*gamma;
        //reflect the ray
        ray.normalize();
        normal.normalize();
        if ((normal * ray) > 0.0f) normal = -normal; //flip the normal if we hit the back side
        reflection = ray - normal*2.f*(normal*ray);
        reflection.normalize();
        intersect = intersect+(-ray * BARY_TOLE);


        return tuple<float,float,float,float,float,float,float,float,float,float,float,float>(intersect.x, intersect.y,intersect.z,reflection.x,reflection.y,reflection.z,normal.x,normal.y,normal.z,alpha,beta, lerpedScalar);
    }
};

struct ReflectSphrFunctor{

    const eavlConstTexArray<float4> *verts;
    const eavlConstTexArray<float>  *sphr_scalars;


    ReflectSphrFunctor(const eavlConstTexArray<float4> *_verts, const eavlConstTexArray<float> *_sphr_scalars )
        :verts(_verts), sphr_scalars(_sphr_scalars)
    {
        
    }                                                
    EAVL_FUNCTOR tuple<float,float,float,float,float, float,float,float, float,float,float, float> operator()( tuple<float,float,float,float,float,float,int, int> rayTuple){
       
        
        int hitIndex = get<6>(rayTuple);//hack for now
        int primitiveType = get<7>(rayTuple);
        if(hitIndex == -1 || primitiveType != SPHERE) return tuple<float,float,float,float,float, float,float,float, float,float,float, float>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
        
        eavlVector3 intersect;

        eavlVector3 reflection(0,0,0);
        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));

        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        float alpha = 0, beta = 0;
        ray.normalize();
        float4 data = verts->getValue(sphr_verts_tref, hitIndex);
        float distance = intersectSphereDist(ray,rayOrigin, data);
        intersect = rayOrigin+distance*ray;
        eavlVector3 normal;
        normal.x = intersect.x - data.x;
        normal.y = intersect.y - data.y;
        normal.z = intersect.z - data.z;
        //reflect the ray
        normal.normalize();
        reflection = ray - normal*2.f*(normal*ray);
        reflection.normalize();
        intersect = intersect + (-ray*BARY_TOLE);
        float scalar = sphr_scalars->getValue(sphr_scalars_tref, hitIndex);

        return tuple<float,float,float,float,float,float,float,float,float,float,float,float>(intersect.x, intersect.y,intersect.z,reflection.x,reflection.y,reflection.z,normal.x,normal.y,normal.z,alpha,beta,scalar);
    }
};

struct ReflectCylFunctor{

    const eavlConstTexArray<float4> *verts;
    const eavlConstTexArray<float>  *cyl_scalars;


    ReflectCylFunctor(const eavlConstTexArray<float4> *_verts, const eavlConstTexArray<float> *_cyl_scalars )
        :verts(_verts), cyl_scalars(_cyl_scalars)
    {
        
    }                                                
    EAVL_FUNCTOR tuple<float,float,float,float,float, float,float,float, float,float,float, float> operator()( tuple<float,float,float,float,float,float,int, int, float> rayTuple){
       
        
        int hitIndex = get<6>(rayTuple);//hack for now
        int primitiveType = get<7>(rayTuple);
        if(hitIndex == -1 || primitiveType != CYLINDER) return tuple<float,float,float,float,float, float,float,float, float,float,float, float>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
        float distance = get<8>(rayTuple);
        eavlVector3 intersect;

        eavlVector3 reflection(0,0,0);
        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));

        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        float alpha=0, beta=0;
        ray.normalize();
        float4 a4 = verts->getValue(cyl_verts_tref, hitIndex * 2    );
        float4 b4 = verts->getValue(cyl_verts_tref, hitIndex * 2 + 1);
        eavlVector3 basePoint(a4.x, a4.y, a4.z);
        eavlVector3 axis(b4.x, b4.y, b4.z);
        intersect = rayOrigin + distance * ray;
        eavlVector3 v1 = intersect - basePoint;
        eavlVector3 v2 = axis.project(v1);
        eavlVector3 normal = v1 - v2;

        //reflect the ray
        
        normal.normalize();
        if ((normal*ray) > 0.0f) normal = -normal; //flip the normal if we hit the back side
        reflection = ray - normal * 2.f * (normal * ray);
        reflection.normalize();
        intersect = intersect+(-ray*BARY_TOLE);
        
        float scalar1 = cyl_scalars->getValue(cyl_scalars_tref, hitIndex * 2    );
        float scalar2 = cyl_scalars->getValue(cyl_scalars_tref, hitIndex * 2 + 1);
        eavlVector3 top = basePoint + b4.w * axis;
        
        float t = sqrt(v2 * v2) / sqrt (top * top); 
        float scalar = lerp(scalar1, scalar2, t);
        


        return tuple<float,float,float,float,float,float,float,float,float,float,float,float>(intersect.x, intersect.y,intersect.z,reflection.x,reflection.y,reflection.z,normal.x,normal.y,normal.z,alpha,beta,scalar);
    }
};


struct DepthFunctor{

    const eavlConstTexArray<float4> *verts;


    DepthFunctor(const eavlConstTexArray<float4> *_verts)
        :verts(_verts)
    {
        
    }                                                    
    EAVL_FUNCTOR tuple<float> operator()( tuple<float,float,float,float,float,float,int> rayTuple){
       
        
        int hitIndex = get<6>(rayTuple);
        if(hitIndex == -1) return tuple<float>(INFINITE);
        eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));
        eavlVector3       ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));

        float4 a4 = verts->getValue(tri_verts_tref, hitIndex * 3);
        float4 b4 = verts->getValue(tri_verts_tref, hitIndex * 3 + 1);
        float4 c4 = verts->getValue(tri_verts_tref, hitIndex * 3 + 2);
        eavlVector3 a(a4.x,a4.y,a4.z);
        eavlVector3 b(a4.w,b4.x,b4.y);
        eavlVector3 c(b4.z,b4.w,c4.x);
        float depth;
        triangleIntersectionDistance(ray, rayOrigin, a,b,c,depth);
        return tuple<float>(depth);
    }
};

struct occIntersectFunctor{

    float           maxDistance;
    const eavlConstTexArray<float4> *verts;
    const eavlConstTexArray<float4> *bvh;
    const eavlConstTexArray<float>  *bvh_lf;
    primitive_t                     primitiveType;


    occIntersectFunctor(const eavlConstTexArray<float4> *_verts, eavlConstTexArray<float4> *theBvh, eavlConstTexArray<float> *_bvh_lf, float max, primitive_t _primitveType)
        :verts(_verts),
         bvh(theBvh),
         bvh_lf(_bvh_lf),
         primitiveType(_primitveType)
    {

        maxDistance = max;
        
    }                                                   
    EAVL_FUNCTOR tuple<float> operator()( tuple<float,float,float,float,float,float,int, float> rayTuple){
        
        int deadRay = get<6>(rayTuple);//hack for now leaving this in for CPU
        bool alreadyOccluded = get<7>(rayTuple) > 0 ? true : false;
        if(deadRay == -1)     return tuple<float>(0.0f);
        if(alreadyOccluded)   return tuple<float>(1.f);
        int minHit = -1;   
        float distance;
        eavlVector3 intersect(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple));
        eavlVector3 ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));
        if(primitiveType ==  TRIANGLE )
        {
            minHit= getIntersectionTri(ray, intersect, true,bvh, bvh_lf, verts,maxDistance, distance);
        }
        else if(primitiveType == SPHERE)
        {
            minHit= getIntersectionSphere(ray, intersect, true,bvh, bvh_lf, verts,maxDistance, distance);
        }
        else if(primitiveType == CYLINDER)
        {
            minHit= getIntersectionCyl(ray, intersect, true,bvh, bvh_lf, verts,maxDistance, distance);
        }

        if(minHit!=-1) return tuple<float>(0.0f);
        else return tuple<float>(1.0f);
    }
};



struct ShadowRayFunctor
{
    const eavlConstTexArray<float4> *verts;
    const eavlConstTexArray<float4> *bvh;
    const eavlConstTexArray<float>  *bvh_lf;
    eavlVector3 light;
    primitive_t type;

    ShadowRayFunctor(eavlVector3 theLight,const eavlConstTexArray<float4> *_verts, const eavlConstTexArray<float4> *theBvh,
                     const eavlConstTexArray<float> *_bvh_lf, primitive_t _type)
        :verts(_verts),
         bvh(theBvh),
         bvh_lf(_bvh_lf),
         light(theLight),
         type(_type)
    {}

    EAVL_FUNCTOR tuple<int> operator()(tuple<float,float,float,int,int> input)
    {
        int hitIdx           = get<3>(input);
        bool alreadyOccluded = get<4>(input) == 0 ? false : true;
        if( hitIdx == -1 || alreadyOccluded ) return tuple<int>(1);// primary ray never hit anything.

        //float alpha,beta,gamma,d,tempDistance;
        eavlVector3 rayOrigin(get<0>(input),get<1>(input),get<2>(input));
        
        eavlVector3 shadowRay=light-rayOrigin;
        float lightDistance=sqrt(shadowRay.x*shadowRay.x+shadowRay.y*shadowRay.y+shadowRay.z*shadowRay.z);
        shadowRay.normalize();
        int minHit;
        float distance;
        if(type ==  TRIANGLE )
        {
            minHit= getIntersectionTri(shadowRay, rayOrigin, true,bvh, bvh_lf, verts,lightDistance, distance);
        }
        else if(type == SPHERE)
        {
            minHit= getIntersectionSphere(shadowRay, rayOrigin, true,bvh, bvh_lf, verts,lightDistance, distance);
        }
        else if(type == CYLINDER)
        {
            minHit= getIntersectionCyl(shadowRay, rayOrigin, true,bvh, bvh_lf, verts,lightDistance, distance);
        }
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
    int             colorMapSize;
    eavlVector3     bgColor;
    float4          defaultColor;

    const eavlConstTexArray<int>      *matIds; //tris
    const eavlConstTexArray<int>      *sphr_matIds;
    eavlConstArray<float>             mats;
    const eavlConstTexArray<float4>   *colorMap;
    

    ShaderFunctor(int numTris,eavlVector3 theLight, eavlVector3 eyePos, int dpth, const eavlConstTexArray<int> *_matIds, 
                  eavlConstArray<float> *_mats, int _lightIntensity, float _lightCoConst, float _lightCoLinear, 
                  float _lightCoExponent, const eavlConstTexArray<float4> *_colorMap, int _colorMapSize, 
                  const eavlConstTexArray<int>* _sphr_matIds, int numTri, int numSpheres, eavlVector3 _bgColor)
        : mats(*_mats), colorMap(_colorMap), bgColor(_bgColor)

    {


        matIds = _matIds;
        sphr_matIds = _sphr_matIds;
        
        light = theLight;
        size = numTris;
        colorMapSize = _colorMapSize;
        eye = eyePos;

        depth = (float)dpth;
        lightDiff=eavlVector3(.9,.9,.9);
        lightSpec=eavlVector3(.9,.9,.9);
        lightIntensity  = _lightIntensity;
        lightCoConst    = _lightCoConst;   //used as coefficients for light attenuation
        lightCoLinear   = _lightCoLinear;
        lightCoExponent = _lightCoExponent;
        defaultColor.x = .5f;
        defaultColor.y = .5f;
        defaultColor.z = .5f;
        defaultColor.w = 1.f;
    }

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<int,int,float,float,float,float,float,float,float,float,float,float,float,float,float,float > input)
    {
        //return tuple<float,float,float>(1,0,0);
        int hitIdx = get<0>(input);
        int hit = get<1>(input);
        float specConst;


        if(hitIdx == -1 ) return tuple<float,float,float>(bgColor.x, bgColor.y,bgColor.z);// primary ray never hit anything.
        //return tuple<float,float,float>(0,0,1);
        int primitiveType = get<15>(input);
        

        eavlVector3 normal(get<8>(input), get<9>(input), get<10>(input));
        eavlVector3 rayInt(get<2>(input), get<3>(input), get<4 >(input));
        eavlVector3 rayOrigin(get<11>(input),get<12>(input),get<13>(input));
        eavlVector3 abg(get<5>(input),get<6>(input),1.f); //alpha beta gamma
        //return tuple<float,float,float>(1,0,0);
        abg.z = abg.z - abg.x - abg.y; // get gamma
        eavlVector3 lightDir    = light - rayInt;
        eavlVector3 viewDir     = eye - rayOrigin;
        float dist = sqrt( lightDir.x*lightDir.x + lightDir.y*lightDir.y + lightDir.z*lightDir.z ); 
        dist += sqrt( viewDir.x*viewDir.x + viewDir.y*viewDir.y + viewDir.z*viewDir.z );
        lightDir = lightDir/dist;
        dist = lightIntensity/(lightCoConst + lightCoLinear*dist + lightCoExponent*dist*dist);
        float ambPct=get<7>(input);
        int id = 0;

        if(primitiveType == TRIANGLE)       id = matIds->getValue(tri_matIdx_tref, hitIdx);
        else if(primitiveType == SPHERE ) id = sphr_matIds->getValue(sphr_matIdx_tref, hitIdx);

        eavlVector3* matPtr = (eavlVector3*)(&mats[0]+id*12);
        eavlVector3 ka = matPtr[0];//these could be lerped if it is possible that a single tri could be made of several mats
        eavlVector3 kd = matPtr[1];
        eavlVector3 ks = matPtr[2];
        float matShine = matPtr[3].x;

//********************************************************
        //pixel=aColor*abg.x+bColor*abg.y+cColor*abg.z;
        float red   = 0;
        float green = 0;
        float blue  = 0;
 
        float cosTheta = normal*lightDir; //for diffuse
        cosTheta = min(max(cosTheta,0.f),1.f); //clamp this to [0,1]
        
        eavlVector3 halfVector = viewDir+lightDir;
        halfVector.normalize();
        
        viewDir.normalize();
        
        float cosPhi = normal * halfVector;
        
        specConst = pow(max(cosPhi,0.0f),matShine);

        float shadowHit = (hit==1) ? 0.f : 1.f;
        //red  =ambPct;
        //green=ambPct;
        //blue =ambPct;
    
        red   = ka.x * ambPct+ (kd.x * lightDiff.x * cosTheta + ks.x * lightSpec.x * specConst) * shadowHit*dist;
        green = ka.y * ambPct+ (kd.y * lightDiff.y * cosTheta + ks.y * lightSpec.y * specConst) * shadowHit*dist;
        blue  = ka.z * ambPct+ (kd.z * lightDiff.z * cosTheta + ks.z * lightSpec.z * specConst) * shadowHit*dist;
        
        /*Color map*/
        float scalar   = get<14>(input);
        int   colorIdx = max(min(colorMapSize-1, (int)floor(scalar*colorMapSize)), 0); 

        float4 color = (colorMapSize != 2) ? colorMap->getValue(color_map_tref, colorIdx) : defaultColor; 
        
        red   *= color.x;
        green *= color.y;
        blue  *= color.z;

        float reflectionCo=1;
        if(depth == 0) reflectionCo = 1;
        else reflectionCo = pow(.3f,depth);
        red   = red  *reflectionCo;
        green = green*reflectionCo;
        blue  = blue *reflectionCo;



        normal.normalize();

        return tuple<float,float,float>(min(red,1.0f),min(green,1.0f),min(blue,1.0f));

    }


};


void eavlRayTracerMutator::allocateArrays()
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
         //deleteClassPtr(occIndexer);
    }
    if (compactOp)
    {
         deleteClassPtr(mask);
         deleteClassPtr(count); 
         deleteClassPtr(indexScan);
    }

    /*Temp arrays for compact*/
    compactTempInt   = new eavlIntArray("temp",1,size);
    compactTempFloat = new eavlFloatArray("temp",1,size);
    mortonIndexes    = new eavlIntArray("mortonIdxs",1,size);

    rayDirX          = new eavlFloatArray("x",1,size);
    rayDirY          = new eavlFloatArray("y",1,size);
    rayDirZ          = new eavlFloatArray("z",1,size);

    rayOriginX       = new eavlFloatArray("x",1,size);
    rayOriginY       = new eavlFloatArray("y",1,size);
    rayOriginZ       = new eavlFloatArray("z",1,size);


    r                = new eavlFloatArray("r",1,size);
    g                = new eavlFloatArray("g",1,size);
    b                = new eavlFloatArray("b",1,size);

    r2               = new eavlFloatArray("",1,size);
    g2               = new eavlFloatArray("",1,size);
    b2               = new eavlFloatArray("",1,size);

   
    alphas           = new eavlFloatArray("aplha",1,size);
    betas            = new eavlFloatArray("beta",1,size);

    interX           = new eavlFloatArray("interX",1,size);
    interY           = new eavlFloatArray("interY",1,size);
    interZ           = new eavlFloatArray("interZ",1,size);
    normX            = new eavlFloatArray("normX",1,size);
    normY            = new eavlFloatArray("normY",1,size);
    normZ            = new eavlFloatArray("normZ",1,size);

    hitIdx           = new eavlIntArray("hitIndex",1,size);
    indexes          = new eavlIntArray("indexes",1,size);
    shadowHits       = new eavlFloatArray("",1,size);
    ambPct           = new eavlFloatArray("",1, size);
    zBuffer          = new eavlFloatArray("",1, size);
    frameBuffer      = new eavlByteArray("",1, width*height*4);
    scalars          = new eavlFloatArray("",1,size);
    primitiveTypeHit = new eavlIntArray("primitiveType",1,size);
    minDistances     = new eavlFloatArray("",1,size);
    //compact arrays
    if(compactOp)
    {
        mask         = new eavlIntArray("mask",1,currentSize);        //array to store the mask
        indexScan    = new eavlIntArray("indexScan",1,currentSize);
        count        = new eavlIntArray("count",1,1);
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
        tempAmbPct= new eavlFloatArray("",1, size);
        occIndexer= new eavlArrayIndexer(occSamples,1e9, 1, 0);
    }
    else //set amient light percentage to whatever the lights are
    {
       eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(ambPct),
                                                eavlOpArgs(ambPct),
                                                FloatMemsetFunctor(1.0f)),
                                                "force");
       eavlExecutor::Go();
    }

    sizeDirty=false;

}

void eavlRayTracerMutator::Init()
{   if(verbose) cerr<<"INIT"<<endl;
    size = height*width;
#if 0    
    if(antiAlias) size=(width+1)*(height+1);
#endif
    currentSize=size; //for compact
    if(sizeDirty) 
    {
        allocateArrays();
        createRays(); //creates the morton ray indexes
        cameraDirty=true; //in this case the camera isn't really dirty, but for accumulating occ samples we need this. maybe rename it
    }

    if(cameraDirty)
    {
        //cout<<"Camera Dirty."<<endl;
        sampleCount=0;
        cameraDirty=false;
    }
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
                                             IntMemsetFunctor(-2)),
                                             "init");
    eavlExecutor::Go();

    if( !shadowsOn )
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(shadowHits),
                                             eavlOpArgs(shadowHits),
                                             FloatMemsetFunctor(0.0)),
                                             "init");
        eavlExecutor::Go();
    }

    if(geomDirty) extractGeometry();
    /*Need to start seperating functions that are not geometry. This allows the materials to be updated */
    if(defaultMatDirty)
    {
        numMats         = scene->getNumMaterials();
        mats_raw        = scene->getMatsPtr();
        if(mats!=NULL)
        {
            delete mats;
        }
        INIT(eavlConstArray<float>, mats,numMats*12);
        defaultMatDirty=false;
    }
}

void eavlRayTracerMutator::extractGeometry()
{
    

    if(verbose) cerr<<"Extracting Geometry"<<endl;

    freeRaw();
    if(verbose) cout<<"Before freeTextures"<<endl;
    freeTextures();

    numTriangles     = scene->getNumTriangles();
    numSpheres       = scene->getNumSpheres();
    numCyls          = scene->getNumCyls();
    tri_verts_raw    = scene->getTrianglePtr();
    tri_norms_raw    = scene->getTriangleNormPtr();
    tri_matIdx_raw   = scene->getTriMatIdxsPtr();
    sphr_verts_raw   = scene->getSpherePtr();
    sphr_matIdx_raw  = scene->getSphrMatIdxPtr();
    sphr_scalars_raw = scene->getSphereScalarPtr();
    cyl_verts_raw    = scene->getCylPtr();
    cyl_matIdx_raw   = scene->getCylMatIdxPtr();
    cyl_scalars_raw  = scene->getCylScalarPtr();
    numMats          = scene->getNumMaterials();
    mats_raw         = scene->getMatsPtr();
    

    int tri_bvh_in_size   = 0;
    int tri_bvh_lf_size   = 0;
    int sphr_bvh_in_size  = 0;
    int sphr_bvh_lf_size  = 0;
    int cyl_bvh_in_size   = 0;
    int cyl_bvh_lf_size   = 0;


    bool cacheExists  =false;
    bool writeCache   =true;
     

    if(useBVHCache)
    {
        cacheExists=readBVHCache(tri_bvh_in_raw, tri_bvh_in_size, tri_bvh_lf_raw, tri_bvh_lf_size, bvhCacheName.c_str());
    }
    else 
    {
        writeCache=false;
    }

    if(numTriangles > 0)
    {

        if(!cacheExists)
        {  
            cout<<"Building BVH....Triangles"<<endl;
            if(fastBVHBuild)
            {
                MortonBVHBuilder *mortonBVH = new MortonBVHBuilder(tri_verts_raw, numTriangles, TRIANGLE);
                mortonBVH->build();
                tri_bvh_in_raw  = mortonBVH->getInnerNodes(tri_bvh_in_size);
                tri_bvh_lf_raw  = mortonBVH->getLeafNodes(tri_bvh_lf_size);
                delete mortonBVH;
            }
            else
            {
                SplitBVH *sbvh= new SplitBVH(tri_verts_raw, numTriangles, TRIANGLE); // 0=triangle
                sbvh->getFlatArray(tri_bvh_in_size, tri_bvh_lf_size, tri_bvh_in_raw, tri_bvh_lf_raw);  
                delete sbvh;
            }
            
            
            if( writeCache) writeBVHCache(tri_bvh_in_raw, tri_bvh_in_size, tri_bvh_lf_raw, tri_bvh_lf_size, bvhCacheName.c_str());
        }

        tri_bvh_in_array   = new eavlConstTexArray<float4>( (float4*)tri_bvh_in_raw, tri_bvh_in_size/4, tri_bvh_in_tref, cpu);
        tri_bvh_lf_array   = new eavlConstTexArray<float>( tri_bvh_lf_raw, tri_bvh_lf_size, tri_bvh_lf_tref, cpu);
        tri_verts_array    = new eavlConstTexArray<float4>( (float4*)tri_verts_raw,numTriangles*3, tri_verts_tref, cpu);
        tri_matIdx_array   = new eavlConstTexArray<int>( tri_matIdx_raw, numTriangles, tri_matIdx_tref, cpu );       
        tri_norms_array = new eavlConstTexArray<float>(tri_norms_raw, numTriangles * 9, tri_norms_tref, cpu);

    }
    if(verbose) cout<<"NUM SPHERES "<<numSpheres<<endl;
    if(numSpheres > 0)
    {

        //TODO: caching
        if(fastBVHBuild)
        {
            MortonBVHBuilder *mortonBVH = new MortonBVHBuilder(sphr_verts_raw, numSpheres, SPHERE);
            mortonBVH->build();
            sphr_bvh_in_raw  = mortonBVH->getInnerNodes(sphr_bvh_in_size);
            sphr_bvh_lf_raw  = mortonBVH->getLeafNodes(sphr_bvh_lf_size);
            delete mortonBVH;
        }
        else
        {
            SplitBVH *sbvh= new SplitBVH(sphr_verts_raw, numSpheres, SPHERE); 
            sbvh->getFlatArray(sphr_bvh_in_size, sphr_bvh_lf_size, sphr_bvh_in_raw, sphr_bvh_lf_raw);
            delete sbvh;
        }

        sphr_bvh_in_array   = new eavlConstTexArray<float4>( (float4*)sphr_bvh_in_raw, sphr_bvh_in_size/4, sphr_bvh_in_tref, cpu);
        sphr_bvh_lf_array   = new eavlConstTexArray<float>( sphr_bvh_lf_raw, sphr_bvh_lf_size, sphr_bvh_lf_tref, cpu);
        sphr_verts_array    = new eavlConstTexArray<float4>( (float4*)sphr_verts_raw,numSpheres, sphr_verts_tref, cpu);
        sphr_matIdx_array   = new eavlConstTexArray<int>( sphr_matIdx_raw, numSpheres,  sphr_matIdx_tref, cpu );
        sphr_scalars_array  = new eavlConstTexArray<float>(sphr_scalars_raw, numSpheres, sphr_scalars_tref, cpu);
        /*no need for normals, trivial calculation */
    }
    if(verbose) cout<<"Num lines "<<numCyls<<endl;
    if(numCyls > 0)
    {
        //TODO: cache
        if(fastBVHBuild)
        {
            MortonBVHBuilder *mortonBVH = new MortonBVHBuilder(cyl_verts_raw, numCyls, CYLINDER);
            mortonBVH->build();
            cyl_bvh_in_raw  = mortonBVH->getInnerNodes(cyl_bvh_in_size);
            cyl_bvh_lf_raw  = mortonBVH->getLeafNodes(cyl_bvh_lf_size);
            delete mortonBVH;
        }
        else
        {
            SplitBVH *sbvh= new SplitBVH(cyl_verts_raw, numCyls, CYLINDER); 
            sbvh->getFlatArray(cyl_bvh_in_size, cyl_bvh_lf_size, cyl_bvh_in_raw, cyl_bvh_lf_raw);
            delete sbvh;
        }

        cyl_bvh_in_array   = new eavlConstTexArray<float4>( (float4*)cyl_bvh_in_raw, cyl_bvh_in_size/4, cyl_bvh_in_tref, cpu);
        cyl_bvh_lf_array   = new eavlConstTexArray<float>( cyl_bvh_lf_raw, cyl_bvh_lf_size, cyl_bvh_lf_tref, cpu);
        cyl_verts_array    = new eavlConstTexArray<float4>( (float4*)cyl_verts_raw,numCyls*2, cyl_verts_tref, cpu);
        cyl_matIdx_array   = new eavlConstTexArray<int>( cyl_matIdx_raw, numCyls,  cyl_matIdx_tref, cpu );
        cyl_scalars_array  = new eavlConstTexArray<float>(cyl_scalars_raw, numCyls*2, cyl_scalars_tref, cpu);
    }
    
    
    if(numMats == 0) { cerr<<"NO MATS bailing"<<endl; exit(0); }

    INIT(eavlConstArray<float>, mats,numMats*12);
    geomDirty = false;
    defaultMatDirty = false;
    
}

void eavlRayTracerMutator::intersect()
{
    /* Ideas : 
               Order of intersections : determine order of intersections by a combination of total surface area and which bounding box is hit first.
               On multiple bounces    : reduce based on primitve types and hit indexes to see if it is even needed to intersect with them.

               Also could change primitive type to a usnigned char to save on data movement
    */
    /*set the minimum distances to INFINITE */

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes), //dummy arg
                                             eavlOpArgs(minDistances),
                                             FloatMemsetFunctor(INFINITE)),
                                             "init");
    eavlExecutor::Go();

    if(numTriangles>0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx,primitiveTypeHit,minDistances),
                                                 eavlOpArgs(hitIdx, minDistances, primitiveTypeHit),
                                                 RayIntersectFunctor(tri_verts_array,tri_bvh_in_array,tri_bvh_lf_array,TRIANGLE)),
                                                                                                        "intersect");
        eavlExecutor::Go();
    }
    if(numSpheres>0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx,primitiveTypeHit,minDistances),
                                                 eavlOpArgs(hitIdx, minDistances, primitiveTypeHit),
                                                 RayIntersectFunctor(sphr_verts_array,sphr_bvh_in_array,sphr_bvh_lf_array,SPHERE)),
                                                                                                        "intersect");
        eavlExecutor::Go();

    }
    if(numCyls>0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx,primitiveTypeHit,minDistances),
                                                 eavlOpArgs(hitIdx, minDistances, primitiveTypeHit),
                                                 RayIntersectFunctor(cyl_verts_array,cyl_bvh_in_array,cyl_bvh_lf_array,CYLINDER)),
                                                                                                        "intersect");
        eavlExecutor::Go();

    }
    //for (int i=0 ; i<size; i++) cout<< hitIdx->GetValue(i)<<" "<<minDistances->GetValue(i)<<" | ";

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(hitIdx),                /*On primary ray: hits some in as -2, and leave as -1 if it misses everything*/
                                             eavlOpArgs(hitIdx),                /*This allows dead rays to be filtered out on multiple bounces */
                                             HitFilterFunctor()),
                                             "Hit Filter");
    eavlExecutor::Go();
}

void eavlRayTracerMutator::occlusionIntersect()
{
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(localHits), //dummy arg
                                             eavlOpArgs(localHits),
                                             FloatMemsetFunctor(0.f)),
                                             "memset");
    eavlExecutor::Go();
    if(numTriangles > 0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(occX),
                                                            eavlIndexable<eavlFloatArray>(occY),
                                                            eavlIndexable<eavlFloatArray>(occZ),
                                                            eavlIndexable<eavlFloatArray>(interX, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(interY, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(interZ, *occIndexer),
                                                            eavlIndexable<eavlIntArray>  (hitIdx, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(localHits)),
                                                 eavlOpArgs(localHits),
                                                 occIntersectFunctor(tri_verts_array,tri_bvh_in_array,tri_bvh_lf_array,aoMax, TRIANGLE)),
                                                 "occIntercept");
                
        eavlExecutor::Go();
    }
    if(numSpheres > 0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(occX),
                                                            eavlIndexable<eavlFloatArray>(occY),
                                                            eavlIndexable<eavlFloatArray>(occZ),
                                                            eavlIndexable<eavlFloatArray>(interX, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(interY, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(interZ, *occIndexer),
                                                            eavlIndexable<eavlIntArray>  (hitIdx, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(localHits)),
                                                 eavlOpArgs(localHits),
                                                 occIntersectFunctor(sphr_verts_array,sphr_bvh_in_array,sphr_bvh_lf_array,aoMax, SPHERE)),
                                                 "occIntercept");
                
        eavlExecutor::Go();
    }
    if(numCyls > 0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(occX),
                                                            eavlIndexable<eavlFloatArray>(occY),
                                                            eavlIndexable<eavlFloatArray>(occZ),
                                                            eavlIndexable<eavlFloatArray>(interX, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(interY, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(interZ, *occIndexer),
                                                            eavlIndexable<eavlIntArray>  (hitIdx, *occIndexer),
                                                            eavlIndexable<eavlFloatArray>(localHits)),
                                                 eavlOpArgs(localHits),
                                                 occIntersectFunctor(cyl_verts_array,cyl_bvh_in_array,cyl_bvh_lf_array,aoMax, CYLINDER)),
                                                 "occIntercept");
                
        eavlExecutor::Go();
    }
}

void eavlRayTracerMutator::reflect()
{
    if(numTriangles > 0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx, primitiveTypeHit),
                                                 eavlOpArgs(interX, interY,interZ,rayDirX,rayDirY,rayDirZ,normX,normY,normZ,alphas,betas,scalars),
                                                 ReflectTriFunctor(tri_verts_array,tri_norms_array)),
                                                 "reflect");
        eavlExecutor::Go(); 
    }
    if(numSpheres > 0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx, primitiveTypeHit),
                                                 eavlOpArgs(interX, interY,interZ,rayDirX,rayDirY,rayDirZ,normX,normY,normZ,alphas,betas,scalars),
                                                 ReflectSphrFunctor(sphr_verts_array, sphr_scalars_array)),
                                                 "reflect");
        eavlExecutor::Go();
    } 
    if(numCyls > 0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx, primitiveTypeHit, minDistances),
                                                      eavlOpArgs(interX, interY,interZ,rayDirX,rayDirY,rayDirZ,normX,normY,normZ,alphas,betas,scalars),
                                                      ReflectCylFunctor(cyl_verts_array, cyl_scalars_array)),
                                                      "reflect");
        eavlExecutor::Go();
    }     

   
}

void eavlRayTracerMutator::shadowIntersect()
{
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(shadowHits), //dummy arg
                                             eavlOpArgs(shadowHits),
                                             FloatMemsetFunctor(0.f)),
                                             "memset");
    eavlExecutor::Go();
    
    

    

    if(numTriangles > 0)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(interX,interY,interZ,hitIdx, shadowHits),
                                                 eavlOpArgs(shadowHits),
                                                 ShadowRayFunctor(light,tri_verts_array,tri_bvh_in_array,tri_bvh_lf_array, TRIANGLE)),
                                                 "shadowRays");
        eavlExecutor::Go();
    }
    if(numSpheres > 0)
    {   
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(interX,interY,interZ,hitIdx, shadowHits),
                                                 eavlOpArgs(shadowHits),
                                                 ShadowRayFunctor(light,sphr_verts_array,sphr_bvh_in_array,sphr_bvh_lf_array, SPHERE)),
                                                 "shadowRays");
        eavlExecutor::Go(); 
    }  
    if(numCyls > 0)
    {   
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(interX,interY,interZ,hitIdx, shadowHits),
                                                 eavlOpArgs(shadowHits),
                                                 ShadowRayFunctor(light, cyl_verts_array, cyl_bvh_in_array, cyl_bvh_lf_array, CYLINDER)),
                                                 "shadowRays");
        eavlExecutor::Go(); 
    }       
}

void eavlRayTracerMutator::Execute()
{
    //cudaSetDevice(0);
    int th ;
    int tinit;
    if(verbose) tinit = eavlTimer::Start();
    if(verbose) th = eavlTimer::Start();

    Init();
    /*if there are no primitives, just return black*/
    if(scene->getTotalPrimitives()==0) return;
    //if(verbose) cerr<<"Executing After Init"<<endl;
   
    if(verbose) cerr<<"Number of triangles: "<<numTriangles<<endl;
    if(verbose) cerr<<"Number of Spheres: "<<numSpheres<<endl;
    if(verbose) cerr<<"Number of Cylinders: "<<numCyls<<endl;
    

    clearFrameBuffer(r,g,b);

    //light=light+movement;
    look=lookat-eye;
    if(verbose) cerr<<"Look "<<look<<" Eye"<<eye<<"Light"<<light<<endl;
   
   
    //init camera rays
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes),
                                             eavlOpArgs(rayDirX ,rayDirY, rayDirZ),
                                             RayGenFunctor(width, height, fovx, fovy, look, up, zoom)),
                                             "ray gen");
    eavlExecutor::Go();

    if(verbose) cout<<"init       RUNTIME: "<<eavlTimer::Stop(tinit,"intersect")<<endl;
    for(int i=0; i<depth;i++) 
    {
        int tintersect;
        if(verbose) tintersect = eavlTimer::Start();

        intersect();
        /* Get the depth buffer. This can only happen on the first bounce. */
        if(i==0)
        {
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(minDistances),
                                             eavlOpArgs(zBuffer),
                                             FloatMemcpyFunctor1to1()),
                                             "cpy");
            eavlExecutor::Go();
        }

        if(verbose) cout<<"intersect   RUNTIME: "<<eavlTimer::Stop(tintersect,"intersect")<<endl;
        

        if(compactOp) 
        {
            int tcompact = eavlTimer::Start();
            currentSize=compact();
            cout << "compact     RUNTIME: "<<eavlTimer::Stop(tcompact,"intersect")<<endl;
        }

        int treflect ;
        if(verbose) treflect = eavlTimer::Start();
        reflect();
                           
        if(verbose) cout<<"Reflect     RUNTIME: "<<eavlTimer::Stop(treflect,"rf")<<endl;
        /*if(currentSize==0)
        {

           cerr<<"No more rays"<<endl;
            break;
        }*/ 
        /*********************************AMB OCC***********************************/
        if(isOccusionOn){
            
            int toccGen ;
            if(verbose) toccGen = eavlTimer::Start(); 
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
            if(verbose) cout << "occRayGen   RUNTIME: "<<eavlTimer::Stop(toccGen,"occGen")<<endl;

            int toccInt;
            if(verbose) toccInt = eavlTimer::Start(); 
            occlusionIntersect();
            if(verbose) cout<<"occInt      RUNTIME: "<<eavlTimer::Stop(toccInt,"occGen")<<endl;

            eavlExecutor::AddOperation(new_eavlNto1GatherOp(eavlOpArgs(localHits),
                                                            eavlOpArgs(tempAmbPct), 
                                                            occSamples),
                                                            "gather");
            eavlExecutor::Go();

            /*Only do this when depth is set to zero reflections*/
            if(!cameraDirty&& depth==1)
            {
                eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(ambPct,tempAmbPct),
                                                         eavlOpArgs(ambPct),
                                                         WeightedAccFunctor1to1(sampleCount,occSamples)),
                                                         "weighted average");
                eavlExecutor::Go();
                sampleCount+=occSamples;
            }
            
        }
        /*********************************END AMB OCC***********************************/
        
        
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(r2),
                                                eavlOpArgs(r2,g2,b2),
                                                FloatMemsetFunctor1to3(0.0f)),
                                                "memset");
        
        eavlExecutor::Go();
        int tshadow ;
        if( shadowsOn )
        {
            if(verbose) tshadow = eavlTimer::Start();
            shadowIntersect();
            if(verbose) cout<<  "Shadow      RUNTIME: "<<eavlTimer::Stop(tshadow,"")<<endl;
        }
        
        int shade ;
        if(verbose) shade = eavlTimer::Start();
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(hitIdx,shadowHits,interX,interY,interZ,alphas,betas,ambPct,normX,normY,normZ,rayOriginX,rayOriginY,rayOriginZ, scalars, primitiveTypeHit),
                                                 eavlOpArgs(r2,g2,b2),
                                                 ShaderFunctor(numTriangles,light,eye ,i, tri_matIdx_array, mats, lightIntensity,
                                                               lightCoConst, lightCoLinear, lightCoExponent, cmap_array,
                                                               colorMapSize, sphr_matIdx_array, numTriangles, numSpheres, bgColor)),
                                                 "shader");
        eavlExecutor::Go();
        if(verbose) cout<<  "Shading     RUNTIME: "<<eavlTimer::Stop(shade,"")<<endl;


        if(!compactOp)
        {

            //these RGB values are in morton order and must be scattered 
            //into the correct order before they are written.
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(r2,g2,b2,r,g,b),
                                                     eavlOpArgs(r,g,b),
                                                     AccFunctor3to3()),
                                                     "add");
            eavlExecutor::Go();
        }
        else
        {
            //zero out the array so bad values don't get accumulated.
            //The compact arrays must be scattered back out to the original size
            //After the scatter, RGB values are in the correct order(not morton order)
            int tscatterAcc;
            if(verbose) tscatterAcc = eavlTimer::Start();
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
            if(verbose) cout<<"ScatterACC  RUNTIME : "<<eavlTimer::Stop(tscatterAcc,"")<<endl;

        }


        /* Copy intersections to origins if there are more bounces*/
        if(i<depth-1) //depth -2?
        {
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(interX, interY, interZ),
                                                     eavlOpArgs(rayOriginX, rayOriginY, rayOriginZ),
                                                     FloatMemcpyFunctor3to3()),
                                                     "memcopy");
            eavlExecutor::Go();
        }

    }
    //todo fix this
    int ttest;
    if(verbose) ttest = eavlTimer::Start();

    if(!compactOp)
    {
        //Non-compacted RGB values are in morton order and must be scattered
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
        eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(zBuffer),
                                                     eavlOpArgs(compactTempFloat),
                                                     eavlOpArgs(mortonIndexes)),
                                                     "scatter");
        eavlExecutor::Go();

        eavlFloatArray *tmpPtr=zBuffer;
        zBuffer=compactTempFloat;
        compactTempFloat=tmpPtr;

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
    if(verbose) cout<<"scatter     RUNTIME: "<<eavlTimer::Stop(ttest,"")<<endl;

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
        cout<<"Transfering to fb"<<endl;
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(r2, g2, b2, b2/*dummy*/),
                                                 eavlOpArgs(eavlIndexable<eavlByteArray>(frameBuffer,*redIndexer),
                                                            eavlIndexable<eavlByteArray>(frameBuffer,*greenIndexer),
                                                            eavlIndexable<eavlByteArray>(frameBuffer,*blueIndexer),
                                                            eavlIndexable<eavlByteArray>(frameBuffer,*alphaIndexer)),
                                                 CopyFrameBuffer()),
                                                 "memcopy");
        eavlExecutor::Go();

        if(verbose) cout<<"TOTAL       RUNTIME: "<<eavlTimer::Stop(th,"raytrace")<<endl;
        
        //writeBMP(height,width,r2,g2,b2,"notA.bmp");
#endif
    //} 
    
    
    //writeBMP(height,width,r,g,b,(char*)scounter.c_str()); 
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

    tempPtr = input;
    input = compactTempInt;
    compactTempInt = tempPtr;
}

void eavlRayTracerMutator::clearFrameBuffer(eavlFloatArray *r,eavlFloatArray *g,eavlFloatArray *b)
{
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(r),
                                            eavlOpArgs(r,g,b),
                                            FloatMemsetFunctor1to3(0)),
                                            "memset");
    eavlExecutor::Go();
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
    float w,h;

    raySort *rayArray= new raySort[size]; // since this is happening every frame it should not be allocated and deleted constantly.
                                          
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
    //cudaSetDevice(0);
    Init();
    eavlIntArray    *dummy= new eavlIntArray("",1,size);
    eavlFloatArray  *dummyFloat= new eavlFloatArray("",1,size);
    look=lookat-eye;
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes),
                                             eavlOpArgs(rayDirX,rayDirY,rayDirZ),
                                             RayGenFunctor(width,height,fovx,fovy,look,up, zoom)),
                                             "ray gen");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes), //dummy arg
                                             eavlOpArgs(minDistances),
                                             FloatMemsetFunctor(INFINITE)),
                                             "init");
    eavlExecutor::Go();

    cout<<"Warming up "<<warmupRounds<<" rounds."<<endl;
    int warm = eavlTimer::Start(); //Dirs, origins
    for(int i=0; i<warmupRounds;i++)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx,primitiveTypeHit,minDistances),
                                                 eavlOpArgs(dummy, dummyFloat, primitiveTypeHit),
                                                 RayIntersectFunctor(tri_verts_array,tri_bvh_in_array,tri_bvh_lf_array, TRIANGLE )),
                                                                                                    "intersect");
        eavlExecutor::Go();
    }

    float rayper=size/(eavlTimer::Stop(warm,"warm")/(float)warmupRounds);
    cout << "Warm up "<<rayper/1000000.f<< " Mrays/sec"<<endl;

    int test = eavlTimer::Start(); //Dirs, origins
    for(int i=0; i<testRounds;i++)
    {
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx,primitiveTypeHit,minDistances),
                                                 eavlOpArgs(dummy, dummyFloat, primitiveTypeHit),
                                                 RayIntersectFunctor(tri_verts_array,tri_bvh_in_array,tri_bvh_lf_array, TRIANGLE )),
                                                                                                    "intersect");
        eavlExecutor::Go();
    }
    rayper=size/(eavlTimer::Stop(test,"test")/(float)testRounds);
    cout << "# "<<rayper/1000000.f<<endl;


    //verify output
    eavlFloatArray *depthBuffer= new eavlFloatArray("",1,size);
    eavlFloatArray *d= new eavlFloatArray("",1,size);

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes), //dummy arg
                                             eavlOpArgs(minDistances),
                                             FloatMemsetFunctor(INFINITE)),
                                             "init");
    eavlExecutor::Go();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(hitIdx),
                                             eavlOpArgs(hitIdx),
                                             IntMemsetFunctor(-2)),
                                             "init");
    eavlExecutor::Go();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ,hitIdx,primitiveTypeHit,minDistances),
                                             eavlOpArgs(dummy, depthBuffer, primitiveTypeHit),
                                             RayIntersectFunctor(tri_verts_array,tri_bvh_in_array,tri_bvh_lf_array, TRIANGLE )),
                                                                                                    "intersect");
    eavlExecutor::Go();

    float maxDepth = 0;
    float minDepth =INFINITE;

    for(int i=0; i< size; i++)
    {
        if( depthBuffer->GetValue(i) == INFINITE) depthBuffer->SetValue(i,0);
        maxDepth= max(depthBuffer->GetValue(i), maxDepth);  
        minDepth= max(0.f,min(minDepth,depthBuffer->GetValue(i)));//??
    } 
    //for(int i=0; i< size; i++) cout<<depthBuffer->GetValue(i)<<" ";
    maxDepth=maxDepth-minDepth;
    for(int i=0; i< size; i++) depthBuffer->SetValue(i, (depthBuffer->GetValue(i)-minDepth)/maxDepth);
    
    eavlExecutor::AddOperation(new_eavlScatterOp(eavlOpArgs(depthBuffer),
                                                 eavlOpArgs(d),
                                                 eavlOpArgs(mortonIndexes)),
                                                "scatter");
    eavlExecutor::Go();

    writeBMP(height,width,d,d,d,"depth.bmp");
    delete depthBuffer;
    delete dummy;
    delete d;
    
}

void eavlRayTracerMutator::fpsTest(int warmupRounds, int testRounds)
{
#if 0
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
                                                 eavlOpArgs(hitIdx,zBuffer),
                                                 RayIntersectFunctor(tri_verts_array,tri_bvh_in_array, tri_bvh_lf_array)),
                                                 "intersect");

        eavlExecutor::Go();

        //eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,interX,interY,interZ,hitIdx),
        //                                         eavlOpArgs(interX, interY,interZ,alphas,betas),
        //                                         ReflectTriFunctorBasic(tri_verts_array,norms)),
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
                                                 eavlOpArgs(hitIdx,zBuffer),
                                                 RayIntersectFunctor(tri_verts_array,tri_bvh_in_array, tri_bvh_lf_array)),
                                                 "intersect");

        eavlExecutor::Go();

        //eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,interX,interY,interZ,hitIdx),
        //                                         eavlOpArgs(interX, interY,interZ,alphas,betas),
        //                                         ReflectTriFunctorBasic(tri_verts_array,norms)),
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
#endif
}
void eavlRayTracerMutator::freeRaw()
{
    
    deleteArrayPtr(tri_verts_raw);
    deleteArrayPtr(tri_norms_raw);
    deleteArrayPtr(tri_matIdx_raw);
    deleteArrayPtr(tri_bvh_in_raw);
    deleteArrayPtr(tri_bvh_lf_raw);
    
    deleteArrayPtr(mats_raw);

    deleteArrayPtr(sphr_scalars_raw);
    deleteArrayPtr(sphr_matIdx_raw);
    deleteArrayPtr(sphr_verts_raw);
    deleteArrayPtr(sphr_bvh_in_raw);
    deleteArrayPtr(sphr_bvh_lf_raw);
    
    deleteArrayPtr(cyl_verts_raw);
    deleteArrayPtr(cyl_matIdx_raw);
    deleteArrayPtr(cyl_bvh_in_raw);
    deleteArrayPtr(cyl_bvh_lf_raw);
    deleteArrayPtr(cyl_scalars_raw);

}

eavlFloatArray* eavlRayTracerMutator::getDepthBuffer(float proj22, float proj23, float proj32)
{ 
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(zBuffer), eavlOpArgs(zBuffer), ScreenDepthFunctor(proj22, proj23, proj32)),"convertDepth");
    eavlExecutor::Go();
    return zBuffer;
}

void eavlRayTracerMutator::freeTextures()
{
   if(verbose) cout<<"Free textures"<<endl;
   if (tri_bvh_in_array != NULL) 
    {
        tri_bvh_in_array->unbind(tri_bvh_in_tref);
        delete tri_bvh_in_array;
        tri_bvh_in_array = NULL;
    }
    if (tri_bvh_lf_array != NULL) 
    {
        tri_bvh_lf_array->unbind(tri_bvh_lf_tref);
        delete tri_bvh_lf_array;
        tri_bvh_lf_array = NULL;
    }
    if (tri_verts_array != NULL) 
    {
        tri_verts_array ->unbind(tri_verts_tref);
        delete tri_verts_array;
        tri_verts_array = NULL;

    }
    if (tri_matIdx_array != NULL) 
    {
        tri_matIdx_array->unbind(tri_matIdx_tref);
        delete tri_matIdx_array;
        tri_matIdx_array = NULL;
    }
    if (tri_norms_array != NULL) 
    {
        tri_norms_array->unbind(tri_norms_tref);
        delete tri_norms_array;
        tri_norms_array = NULL;
    }

    if (sphr_bvh_in_array != NULL) 
    {
        sphr_bvh_in_array->unbind(sphr_bvh_in_tref);
        delete sphr_bvh_in_array;
        sphr_bvh_in_array = NULL;
    }
    if (sphr_bvh_lf_array != NULL) 
    {
        sphr_bvh_lf_array->unbind(sphr_bvh_lf_tref);
        delete sphr_bvh_lf_array;
        sphr_bvh_lf_array = NULL;
    }
    if (sphr_verts_array != NULL) 
    {
        sphr_verts_array ->unbind(sphr_verts_tref);
        delete sphr_verts_array;
        sphr_verts_array = NULL;
    }
    if (sphr_scalars_array != NULL) 
    {
        sphr_scalars_array ->unbind(sphr_scalars_tref);
        delete sphr_scalars_array;
        sphr_scalars_array = NULL;
    }
    if (tri_matIdx_array != NULL) 
    {
        sphr_matIdx_array->unbind(sphr_matIdx_tref);
        delete sphr_matIdx_array;
        sphr_matIdx_array = NULL;
    }
    if (cyl_bvh_in_array != NULL) 
    {
        cyl_bvh_in_array->unbind(cyl_bvh_in_tref);
        delete cyl_bvh_in_array;
        cyl_bvh_in_array = NULL;
    }
    if (cyl_verts_array != NULL) 
    {
        cyl_verts_array->unbind(cyl_verts_tref);
        delete cyl_verts_array;
        cyl_verts_array = NULL;
    }
    if (cyl_bvh_lf_array != NULL) 
    {
        cyl_bvh_lf_array->unbind(cyl_bvh_lf_tref);
        delete cyl_bvh_lf_array;
        cyl_bvh_lf_array = NULL;
    }
    if (cyl_scalars_array != NULL) 
    {
        cyl_scalars_array->unbind(cyl_scalars_tref);
        delete cyl_scalars_array;
        cyl_scalars_array = NULL;
    }
    if (cyl_matIdx_array != NULL) 
    {
        cyl_matIdx_array->unbind(cyl_matIdx_tref);
        delete cyl_matIdx_array;
        cyl_matIdx_array = NULL;
    }

//eavlConstTexArray<float4>* cmap_array;
    if(verbose) cout<<"Done free"<<endl;
}

