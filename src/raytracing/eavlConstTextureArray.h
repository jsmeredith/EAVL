#ifndef EAVL_CONST_TEX_ARRAY_H
#define EAVL_CONST_TEX_ARRAY_H

#define USE_TEXTURE_MEM

#ifndef __CUDACC__
template<class T> class texture {};

#ifndef HAVE_CUDA
struct float4
{
    float x,y,z,w;
};
#endif
#endif

template<class T>
class eavlConstTexArray
{
  private: 
    bool cpu;
  public:
    T *host;
    T *device;
    
  public:
    eavlConstTexArray(T *from, int N, texture<T> &g_textureRef, bool CPU)
    {
        host = from;
        cpu = CPU;
        if(!CPU)
        {
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
    }
    ~eavlConstTexArray()
    {
        

#ifdef HAVE_CUDA
        cudaFree(device);
        CUDA_CHECK_ERROR();
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
        if(!cpu)
        {
            cudaUnbindTexture(g_textureRef);
            CUDA_CHECK_ERROR();
        }
        
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

#endif