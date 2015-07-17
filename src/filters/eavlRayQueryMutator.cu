// Copyright 2010-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlRayQueryMutator.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlRTUtil.h"
#include "eavlMapOp.h"
#include "eavlFilter.h"
#include "eavlTimer.h" 
#include "SplitBVH.h"

/* Triangle textures */
texture<float4> RQM_tri_bvh_in_tref;            /* BVH inner nodes */
texture<float4> RQM_tri_verts_tref;              /* vert+ scalar data */
texture<float>  RQM_tri_bvh_lf_tref;            /* BVH leaf nodes */

eavlConstTexArray<float4>* RQM_tri_bvh_in_array;
eavlConstTexArray<float4>* RQM_tri_verts_array;
eavlConstTexArray<float>*  RQM_tri_bvh_lf_array;

eavlRayQueryMutator::eavlRayQueryMutator()
{

	size = 1;
    cacheBVH = false;
    verbose = false;
    writePtr = 0;

	rayOriginX = NULL;
    rayOriginY = NULL;
    rayOriginZ = NULL;
    rayDirX = NULL;
    rayDirY = NULL;
    rayDirZ = NULL;
    indexes = NULL;
    tempFloat = NULL;
    zBuffer = NULL;
    u = NULL;
    v = NULL;

    geomDirty = true;
    sizeDirty = true;
    verbose = true;

    tri_verts_raw   = NULL;
    tri_bvh_in_raw  = NULL;     
    tri_bvh_lf_raw  = NULL;

    RQM_tri_bvh_in_array = NULL;
    RQM_tri_verts_array  = NULL;
    RQM_tri_bvh_lf_array = NULL;
    
    numTris = 0;

    if(eavlExecutor::GetExecutionMode() == eavlExecutor::ForceCPU ) cpu = true;
    else cpu = false;

}



EAVL_HOSTDEVICE int getIntersectionTri(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstTexArray<float4> *bvh,const eavlConstTexArray<float> *tri_bvh_lf_raw,
                                       const eavlConstTexArray<float4> *verts,const float &maxDistance, float &distance, float &_u, float &_v)
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

            float4 n1 = bvh->getValue(RQM_tri_bvh_in_tref, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
            float4 n2 = bvh->getValue(RQM_tri_bvh_in_tref, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3 = bvh->getValue(RQM_tri_bvh_in_tref, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
            
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
            float4 n4 = bvh->getValue(RQM_tri_bvh_in_tref, currentNode+3); //(leftChild, rightChild, pad,pad)
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
            

            currentNode = -currentNode; //swap the neg address 
            int numTri = (int)tri_bvh_lf_raw->getValue(RQM_tri_bvh_lf_tref,currentNode)+1;

            for(int i = 1; i < numTri; i++)
            {        
                    int triIndex = (int)tri_bvh_lf_raw->getValue(RQM_tri_bvh_lf_tref,currentNode+i);
                   
                    float4 a4 = verts->getValue(RQM_tri_verts_tref, triIndex*3);
                    float4 b4 = verts->getValue(RQM_tri_verts_tref, triIndex*3+1);
                    float4 c4 = verts->getValue(RQM_tri_verts_tref, triIndex*3+2);
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
                                    _v = v;
                                    _u = u;
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



struct RayIntersectFunctor{


    const eavlConstTexArray<float4> *verts;
    const eavlConstTexArray<float4> *bvh;
    const eavlConstTexArray<float>  *bvh_inner;

    RayIntersectFunctor(const eavlConstTexArray<float4> *_verts, const eavlConstTexArray<float4> *theBvh, const eavlConstTexArray<float> *_bvh_inner)
        :verts(_verts),
         bvh(theBvh),
         bvh_inner(_bvh_inner)
    {}                                                 
    EAVL_HOSTDEVICE tuple<int,float,float,float> operator()( tuple<float,float,float,float,float,float> rayTuple){
       
    int   minHit=-1; 
    float distance;
    float u = 0;
    float v = 0;
    eavlVector3 rayOrigin(get<3>(rayTuple),get<4>(rayTuple),get<5>(rayTuple)); 
    eavlVector3       ray(get<0>(rayTuple),get<1>(rayTuple),get<2>(rayTuple));

    minHit = getIntersectionTri(ray, rayOrigin, false,bvh,bvh_inner, verts,INFINITE,distance,u,v);
        
    if(minHit!=-1) return tuple<int,float,float, float>(minHit, distance, u, v);
    else           return tuple<int,float,float,float>(-1, 0.f, 0.f, 0.f);
    }
    
};



void eavlRayQueryMutator::clearOutput()
{
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes),
                                            eavlOpArgs(indexes),
                                            IntMemsetFunctor(-1.f)),
                                            "memset");
    eavlExecutor::Go();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(zBuffer),
                                            eavlOpArgs(zBuffer),
                                            FloatMemsetFunctor(0)),
                                            "memset");
    eavlExecutor::Go();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(u),
                                            eavlOpArgs(u),
                                            FloatMemsetFunctor(0)),
                                            "memset");
    eavlExecutor::Go();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(v),
                                            eavlOpArgs(v),
                                            FloatMemsetFunctor(0)),
                                            "memset");
    eavlExecutor::Go();

}

void eavlRayQueryMutator::allocateArrays()
{   
    if(verbose) cout<<"Del space\n";
	deleteClassPtr(rayDirX);
    deleteClassPtr(rayDirY);
    deleteClassPtr(rayDirZ);

    deleteClassPtr(rayOriginX);
    deleteClassPtr(rayOriginY);
    deleteClassPtr(rayOriginZ);

    deleteClassPtr(u);
    deleteClassPtr(v);

    deleteClassPtr(zBuffer);
    deleteClassPtr(indexes);
    deleteClassPtr(tempFloat);
    if(verbose) cout<<"Allocating space\n";
    indexes          = new eavlIntArray("indexes",1,size);

    rayDirX          = new eavlFloatArray("x",1,size);
    rayDirY          = new eavlFloatArray("y",1,size);
    rayDirZ          = new eavlFloatArray("z",1,size);

    rayOriginX       = new eavlFloatArray("x",1,size);
    rayOriginY       = new eavlFloatArray("y",1,size);
    rayOriginZ       = new eavlFloatArray("z",1,size);

    u                = new eavlFloatArray("r",1,size);
    v                = new eavlFloatArray("b",1,size);
    //b                = new eavlFloatArray("g",1,size);
    //a                = new eavlFloatArray("g",1,size);
    tempFloat        = new eavlFloatArray("g",1,size);

    zBuffer      = new eavlFloatArray("",1, size);
    sizeDirty = false;
}

void eavlRayQueryMutator::init()
{   
    if(verbose) cout<<"Init"<<endl;
    if(geomDirty) extractGeometry();
    clearOutput();
}

void eavlRayQueryMutator::extractGeometry()
{
    if(verbose) cerr<<"Extracting Geometry"<<endl;
    freeRaw();
    freeTextures();

    tri_verts_raw = &(scene.getTrianglePtr()->GetValue(0));
    int tri_bvh_in_size = 0;
    int tri_bvh_lf_size = 0;
    
    bool cacheExists  =false;
    bool writeCache   =true;
     
    if(cacheBVH)
    {
        cacheExists = readBVHCache(tri_bvh_in_raw, tri_bvh_in_size, tri_bvh_lf_raw, tri_bvh_lf_size, cacheName.c_str());
    }
    else 
    {
        writeCache = false;
    }

    if(!cacheExists)
    {  
        if(verbose) cout<<"Building BVH...."<<endl;
        SplitBVH *testSplit= new SplitBVH(tri_verts_raw, numTris, 0); // 0=triangle
        if(verbose) cout<<"Done building."<<endl;
        testSplit->getFlatArray(tri_bvh_in_size, tri_bvh_lf_size, tri_bvh_in_raw, tri_bvh_lf_raw);
        if( writeCache ) writeBVHCache(tri_bvh_in_raw, tri_bvh_in_size, tri_bvh_lf_raw, tri_bvh_lf_size, cacheName.c_str());
        delete testSplit;
    }

    

    RQM_tri_bvh_in_array   = new eavlConstTexArray<float4>( (float4*)tri_bvh_in_raw, tri_bvh_in_size/4, RQM_tri_bvh_in_tref, cpu);
    RQM_tri_bvh_lf_array   = new eavlConstTexArray<float>( tri_bvh_lf_raw, tri_bvh_lf_size, RQM_tri_bvh_lf_tref, cpu);
    RQM_tri_verts_array    = new eavlConstTexArray<float4>( (float4*) tri_verts_raw, numTris*3, RQM_tri_verts_tref, cpu);

    geomDirty=false;
}



void eavlRayQueryMutator::Execute()
{   
    numTris = scene.getNumTriangles();
    if(numTris < 1 ) 
    {  
        cout<<"No primitives to render. "<<endl;
        return;
    }
    
    init();
    
    int ttraverse;
    if(verbose) ttraverse = eavlTimer::Start();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayDirX,rayDirY,rayDirZ,rayOriginX,rayOriginY,rayOriginZ),
                                             eavlOpArgs(indexes, zBuffer, u,v),
                                             RayIntersectFunctor(RQM_tri_verts_array,RQM_tri_bvh_in_array,RQM_tri_bvh_lf_array)),
                                                                                                        "intersect");
    eavlExecutor::Go();
    
    if(verbose) cout<<"Traversal   RUNTIME: "<<eavlTimer::Stop(ttraverse,"traverse")<<endl;

}




void eavlRayQueryMutator::freeRaw()
{
    
    deleteArrayPtr(tri_verts_raw);
    deleteArrayPtr(tri_bvh_in_raw);
    deleteArrayPtr(tri_bvh_lf_raw);
    //cout<<"Free raw"<<endl;

}


void eavlRayQueryMutator::freeTextures()
{
    // /cout<<"Free textures"<<endl;
   if (RQM_tri_bvh_in_array != NULL) 
    {
        RQM_tri_bvh_in_array->unbind(RQM_tri_bvh_in_tref);
        delete RQM_tri_bvh_in_array;
        RQM_tri_bvh_in_array = NULL;
    }
    if (RQM_tri_bvh_lf_array != NULL) 
    {
        RQM_tri_bvh_lf_array->unbind(RQM_tri_bvh_lf_tref);
        delete RQM_tri_bvh_lf_array;
        RQM_tri_bvh_lf_array = NULL;
    }
    if (RQM_tri_verts_array != NULL) 
    {
        RQM_tri_verts_array ->unbind(RQM_tri_verts_tref);
        delete RQM_tri_verts_array;
        RQM_tri_verts_array = NULL;
    }
}

void eavlRayQueryMutator::runTest()
{
    int height = 1080;
    int width = 1920;
    setQuerySize(width*height);
    cout<<"Adding trangles"<<endl;
    /*Cornell Box*/
    addTriangle(eavlVector3(-1.01,0,0.99),      eavlVector3(1,0,0.99),          eavlVector3(1,0,-1.04));
    addTriangle(eavlVector3(-1.01,0,0.99),      eavlVector3(1,0,-1.04),         eavlVector3(-0.99,0,-1.04));
    addTriangle(eavlVector3(-1.02,1.99,0.99),   eavlVector3(-1.02,1.99,-1.04),  eavlVector3(1,1.99,-1.04));
    addTriangle(eavlVector3(-1.02,1.99,0.99),   eavlVector3(1,1.99,-1.04),      eavlVector3(1,1.99,0.99));
    addTriangle(eavlVector3(-0.99,0,-1.04),     eavlVector3(1,0,-1.04),         eavlVector3(1,1.99,-1.04));
    addTriangle(eavlVector3(-0.99,0,-1.04),     eavlVector3(1,1.99,-1.04),      eavlVector3(-1.02,1.99,-1.04));
    addTriangle(eavlVector3(1,0,-1.04),         eavlVector3(1,0,0.99),          eavlVector3(1,1.99,0.99));
    addTriangle(eavlVector3(1,0,-1.04),         eavlVector3(1,1.99,0.99),       eavlVector3(1,1.99,-1.04));
    addTriangle(eavlVector3(-1.01,0,0.99),      eavlVector3(-0.99,0,-1.04),     eavlVector3(-1.02,1.99,-1.04));
    addTriangle(eavlVector3(-1.01,0,0.99),      eavlVector3(-1.02,1.99,-1.04),  eavlVector3(-1.02,1.99,0.99));
    addTriangle(eavlVector3(0.53,0.6,0.75),     eavlVector3(0.7,0.6,0.17),      eavlVector3(0.13,0.6,0));
    addTriangle(eavlVector3(0.53,0.6,0.75),     eavlVector3(0.13,0.6,0),        eavlVector3(-0.05,0.6,0.57));
    addTriangle(eavlVector3(-0.05,0,0.57),      eavlVector3(-0.05,0.6,0.57),    eavlVector3(0.13,0.6,0));
    addTriangle(eavlVector3(-0.05,0,0.57),      eavlVector3(0.13,0.6,0),        eavlVector3(0.13,0,0));
    addTriangle(eavlVector3(0.53,0,0.75),       eavlVector3(0.53,0.6,0.75),     eavlVector3(-0.05,0.6,0.57));
    addTriangle(eavlVector3(0.53,0,0.75),       eavlVector3(-0.05,0.6,0.57),    eavlVector3(-0.05,0,0.57));
    addTriangle(eavlVector3(0.7,0,0.17),        eavlVector3(0.7,0.6,0.17),      eavlVector3(0.53,0.6,0.75));
    addTriangle(eavlVector3(0.7,0,0.17),        eavlVector3(0.53,0.6,0.75),     eavlVector3(0.53,0,0.75));
    addTriangle(eavlVector3(0.13,0,0),          eavlVector3(0.13,0.6,0),        eavlVector3(0.7,0.6,0.17));
    addTriangle(eavlVector3(0.13,0,0),          eavlVector3(0.7,0.6,0.17),      eavlVector3(0.7,0,0.17));
    addTriangle(eavlVector3(0.7,0,0.17),        eavlVector3(0.7,0.6,0.17),      eavlVector3(0.53,0.6,0.75));
    addTriangle(eavlVector3(0.7,0,0.17),        eavlVector3(0.53,0.6,0.75),     eavlVector3(0.53,0,0.75));
    addTriangle(eavlVector3(-0.53,1.2,0.09),    eavlVector3(0.04,1.2,-0.09),    eavlVector3(-0.14,1.2,-0.67));
    addTriangle(eavlVector3(-0.53,1.2,0.09),    eavlVector3(-0.14,1.2,-0.67),   eavlVector3(-0.71,1.2,-0.49));
    addTriangle(eavlVector3(-0.53,0,0.09),      eavlVector3(-0.53,1.2,0.09),    eavlVector3(-0.71,1.2,-0.49));
    addTriangle(eavlVector3(-0.53,0,0.09),      eavlVector3(-0.71,1.2,-0.49),   eavlVector3(-0.71,0,-0.49));
    addTriangle(eavlVector3(-0.71,0,-0.49),     eavlVector3(-0.71,1.2,-0.49),   eavlVector3(-0.14,1.2,-0.67));
    addTriangle(eavlVector3(-0.71,0,-0.49),     eavlVector3(-0.14,1.2,-0.67),   eavlVector3(-0.14,0,-0.67));
    addTriangle(eavlVector3(-0.14,0,-0.67),     eavlVector3(-0.14,1.2,-0.67),   eavlVector3(0.04,1.2,-0.09));
    addTriangle(eavlVector3(-0.14,0,-0.67),     eavlVector3(0.04,1.2,-0.09),    eavlVector3(0.04,0,-0.09));
    addTriangle(eavlVector3(0.04,0,-0.09),      eavlVector3(0.04,1.2,-0.09),    eavlVector3(-0.53,1.2,0.09));
    addTriangle(eavlVector3(0.04,0,-0.09),      eavlVector3(-0.53,1.2,0.09),    eavlVector3(-0.53,0,0.09));
    addTriangle(eavlVector3(0.04,0,-0.09),      eavlVector3(0.04,1.2,-0.09),    eavlVector3(-0.53,1.2,0.09));
    addTriangle(eavlVector3(0.04,0,-0.09),      eavlVector3(-0.53,1.2,0.09),    eavlVector3(-0.53,0,0.09));
    addTriangle(eavlVector3(-0.24,1.98,0.16),   eavlVector3(-0.24,1.98,-0.22),  eavlVector3(0.23,1.98,-0.22));
    addTriangle(eavlVector3(-0.24,1.98,0.16),   eavlVector3(0.23,1.98,-0.22),   eavlVector3(0.23,1.98,0.16));

    eavlVector3 look(0.f,1.f,0.f);
    eavlVector3 eye(.8f,.8f,3);
    eavlVector3 up(0.f,1.f,0.f);
    look = look - eye;
    cout<<"Set indexes"<<endl;
    eavlFloatArray * ids = new eavlFloatArray("",1,size);
    for(int i = 0; i < size; i++) ids->SetValue(i,i);
    cout<<"memset"<<endl;
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayOriginX),
                                            eavlOpArgs(rayOriginX),
                                            FloatMemsetFunctor(eye.x)),
                                            "memset");
    eavlExecutor::Go();

     eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayOriginY),
                                            eavlOpArgs(rayOriginY),
                                            FloatMemsetFunctor(eye.y)),
                                            "memset");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rayOriginZ),
                                            eavlOpArgs(rayOriginZ),
                                            FloatMemsetFunctor(eye.z)),
                                            "memset");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(ids),
                                             eavlOpArgs(rayDirX ,rayDirY, rayDirZ),
                                             RayGenFunctor(width, height, 45, 30, look, up, 1)),
                                             "ray gen");
    eavlExecutor::Go();


    Execute();
    //cout<<"After execute "<<size<<endl;

    float maxDepth = 0;
    float minDepth = INFINITE;

    for(int i = 0; i< size; i++)
    {
        //if( zBuffer->GetValue(i) == INFINITE) zBuffer->SetValue(i,0);
        maxDepth = max(zBuffer->GetValue(i), maxDepth);  
        minDepth = max(0.f,min(minDepth,zBuffer->GetValue(i)));\
    } 
    
    
    maxDepth = maxDepth - minDepth;
    
    for(int i=0; i< size; i++) zBuffer->SetValue(i, (zBuffer->GetValue(i) - minDepth) / maxDepth);

    writeBMP(height,width,zBuffer,zBuffer,zBuffer,"rayQueryTest.bmp");
}
