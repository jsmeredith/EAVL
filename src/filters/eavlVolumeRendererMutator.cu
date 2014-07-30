#include "eavlVolumeRendererMutator.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "RT/eavlRTUtil.h"
#include "eavlMapOp.h"
#include "eavlFilter.h"
#include "eavlTimer.h" 

#define USE_TEXTURE_MEM
#define END_FLAG    -1000000000
#define INFINITE    1000000
#define EPSILON     0.001f

/* Triangle textures */
texture<float4> tri_bvh_in_tref;            /* BVH inner nodes */
texture<float4> tri_verts_tref;             /* vert+ scalar data */
texture<float>  tri_bvh_lf_tref;            /* BVH leaf nodes */

#ifndef HAVE_CUDA
template<class T> class texture {};
struct float4
{
    float x,y,z,w;
};
#endif

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
    ~eavlConstArrayV2()
    {
        

#ifdef HAVE_CUDA
        //cudaFree(device);
        //CUDA_CHECK_ERROR();
#endif


    }
#ifdef __CUDA_ARCH__
#ifdef USE_TEXTURE_MEM
    EAVL_DEVICEONLY  const T getValue(texture<T> g_textureRef, int index) const
    {
        return tex1Dfetch(g_textureRef, index);
    }
    EAVL_HOSTONLY  void unbind(texture<T> g_textureRef)
    {
        cudaUnbindTexture(g_textureRef);
        CUDA_CHECK_ERROR();
    }
#else
    EAVL_DEVICEONLY const T &getValue(texture<T> g_textureRef, int index) const
    {
        return device[index];
    }
    EAVL_HOSTONLY  void unbind(texture<T> g_textureRef)
    {
        //do nothing
    }
#endif
#else
    EAVL_HOSTONLY const T &getValue(texture<T> g_textureRef, int index) const
    {
        return host[index];
    }
    EAVL_HOSTONLY  void unbind(texture<T> g_textureRef)
    {
        //do nothing
    }
#endif

};


eavlConstArrayV2<float4>* tri_bvh_in_array;
eavlConstArrayV2<float4>* tri_verts_array;
eavlConstArrayV2<float>*  tri_bvh_lf_array;

eavlVolumeRendererMutator::eavlVolumeRendererMutator()
{
	height = 1080;
	width  = 1920;
	size = height * width;
	camera.position.x = 10;
	camera.position.y = 0;
	camera.position.z = 0;
	camera.lookat.x = 0;
	camera.lookat.y = 0;
	camera.lookat.z = 0;
	camera.up.x = 0;
	camera.up.y = 1;
	camera.up.z = 0;
	camera.fovx = 45;
	camera.fovy = 30;
	camera.zoom = 1;

	rayOriginX = NULL;
    rayOriginY = NULL;
    rayOriginZ = NULL;
    rayDirX = NULL;
    rayDirY = NULL;
    rayDirZ = NULL;
    indexes = NULL;
    mortonIndexes = NULL;

    geomDirty = true;
    sizeDirty = true;
    verbose = true;

    numTets = 0;
}


EAVL_HOSTDEVICE int getIntersectionTri(const eavlVector3 rayDir, const eavlVector3 rayOrigin, bool occlusion, const eavlConstArrayV2<float4> &bvh,const eavlConstArrayV2<float> &tri_bvh_lf_raw,eavlConstArrayV2<float4> &verts,const float &maxDistance, float &distance)
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

            float4 n1=bvh.getValue(tri_bvh_in_tref, currentNode  ); //(txmin0, tymin0, tzmin0, txmax0)
            float4 n2=bvh.getValue(tri_bvh_in_tref, currentNode+1); //(tymax0, tzmax0, txmin1, tymin1)
            float4 n3=bvh.getValue(tri_bvh_in_tref, currentNode+2); //(tzmin1, txmax1, tymax1, tzmax1)
            
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
            float4 n4=bvh.getValue(tri_bvh_in_tref, currentNode+3); //(leftChild, rightChild, pad,pad)
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
            int numTri=(int)tri_bvh_lf_raw.getValue(tri_bvh_lf_tref,currentNode)+1;

            for(int i=1;i<numTri;i++)
            {        
                    int triIndex=(int)tri_bvh_lf_raw.getValue(tri_bvh_lf_tref,currentNode+i);
                   
                    float4 a4=verts.getValue(tri_verts_tref, triIndex*3);
                    float4 b4=verts.getValue(tri_verts_tref, triIndex*3+1);
                    float4 c4=verts.getValue(tri_verts_tref, triIndex*3+2);
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
                        if(u>= (0.f- EPSILON) && u<=(1.f+EPSILON))
                        {
                            eavlVector3 q=t%e1;
                            float v=(dirx*q.x+diry*q.y+dirz*q.z)*dot;
                            if(v>= (0.f- EPSILON) && v<=(1.f+EPSILON))
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
 distance=minDistance;
 return minIndex;
}

void eavlVolumeRendererMutator::allocateArrays()
{
	deleteClassPtr(rayDirX);
    deleteClassPtr(rayDirY);
    deleteClassPtr(rayDirZ);

    deleteClassPtr(rayOriginX);
    deleteClassPtr(rayOriginY);
    deleteClassPtr(rayOriginZ);

    deleteClassPtr(indexes);
    deleteClassPtr(mortonIndexes);

    indexes          = new eavlIntArray("indexes",1,size);
    mortonIndexes    = new eavlIntArray("mortonIdxs",1,size);

    rayDirX          = new eavlFloatArray("x",1,size);
    rayDirY          = new eavlFloatArray("y",1,size);
    rayDirZ          = new eavlFloatArray("z",1,size);

    rayOriginX       = new eavlFloatArray("x",1,size);
    rayOriginY       = new eavlFloatArray("y",1,size);
    rayOriginZ       = new eavlFloatArray("z",1,size);

    sizeDirty = false;
}

void eavlVolumeRendererMutator::init()
{
	size = height*width;
	if(sizeDirty) 
    {
        allocateArrays();
        createRays(); //creates the morton ray indexes
    }

    /* Set ray origins to the eye */
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(indexes), //dummy arg
                                             eavlOpArgs(rayOriginX,rayOriginY,rayOriginZ),
                                             FloatMemsetFunctor3to3(camera.position.x,camera.position.y,camera.position.z)),
                                             "init");
    eavlExecutor::Go();

    /* Copy morton indexes into idxs*/ 
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(mortonIndexes),
                                             eavlOpArgs(indexes),
                                             FloatMemcpyFunctor1to1()),
                                             "cpy");
    eavlExecutor::Go();

    if(geomDirty) extractGeometry();
}

void eavlVolumeRendererMutator::extractGeometry()
{

}



void eavlVolumeRendererMutator::Execute()
{

}

void eavlVolumeRendererMutator::createRays()
{
    float fwidth=(float)width;
    float fheight=(float)height;
    float  w,h;

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