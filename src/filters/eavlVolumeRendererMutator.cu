#include "eavlException.h"
#include "eavlExecutor.h"

#include "eavlMapOp.h"
#include "eavlFilter.h"
#include "eavlTimer.h" 


#ifndef HAVE_CUDA
template<class T> class texture {};
struct float4
{
    float x,y,z,w;
};
#endif

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


void eavlVolumeRendererMutator()
{
	height = 1080;
	width = 1920;

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



}


void eavlVolumeRendererMutator::Execute()
{
	
}

