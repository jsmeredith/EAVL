#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlSimpleVRMutator.h"
#include "eavlMapOp.h"

//declare the texture reference even if we are not using texture memory
#ifndef HAVE_CUDA
template<class T> class texture {};
struct float4
{
    float x,y,z,w;
};
#endif
texture<float4> tets_verts_tref; 
/*color map texture */
texture<float4> color_map_tref;

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

eavlConstArrayV2<float4>* tets_verts_array;
eavlConstArrayV2<float4>* color_map_array;

eavlSimpleVRMutator::eavlSimpleVRMutator()
{
	height = 500;
	width  = 500;

	samples = NULL;
	framebuffer = NULL;

	tets_raw = NULL;
	colormap_raw = NULL;

	scene = new eavlVRScene();

    geomDirty = true;
    sizeDirty = true;

    numTets = 0;
    nSamples = 200;

}

void eavlSimpleVRMutator::setColorMap3f(float* cmap,int size)
{
    colormapSize = size;
    if(color_map_array != NULL)
    {
        color_map_array->unbind(color_map_tref);
        delete color_map_array;
    }
    if(colormap_raw!=NULL)
    {
        delete colormap_raw;
    }
    color_map_raw= new float[size*4];
    
    for(int i=0;i<size;i++)
    {
        colormap_raw[i*4  ] = cmap[i*3  ];
        colormap_raw[i*4+1] = cmap[i*3+1];
        colormap_raw[i*4+2] = cmap[i*3+2];
        colormap_raw[i*4+3] = .05;          //test Alpha
        //cout<<cmap[i*3]<<" "<<cmap[i*3+1]<<" "<<cmap[i*3+2]<<endl;
    }
    color_map_array = new eavlConstArrayV2<float4>((float4*)color_map_raw, colormapSize, color_map_tref);
}

void eavlSimpleVRMutator::setDefaultColorMap()
{   cout<<"setting defaul color map"<<endl;
    if(color_map_array!=NULL)
    {
        color_map_array->unbind(color_map_tref);
        delete color_map_array;
    }
    if(color_map_raw!=NULL)
    {
        delete[] colormap_raw;
    }
    //two values all 1s
    colormapSize=2;
    colormap_raw= new float[8];
    for(int i=0;i<8;i++) color_map_raw[i]=1.f;
    color_map_array = new eavlConstArrayV2<float4>((float4*)colormap_raw, colormapSize, color_map_tref);
    cout<<"Done setting defaul color map"<<endl;

}

eavlSimpleVRMutator::~eavlSimpleVRMutator()
{

	deleteClassPtr(samples);
	deleteClassPtr(framebuffer);
	deleteClassPtr(scene);

}

void eavlSimpleVRMutator::init()
{
    if(sizeDirty)
    {
        deleteClassPtr(samples);
        deleteClassPtr(framebuffer);

        samples = new eavlFloatArray("",1,height*width*nSamples);
    }
    

}

void  eavlSimpleVRMutator::Execute()
{
    init();
	

}

void  eavlSimpleVRMutator::freeTextures()
{

    if (tets_verts_array != NULL) 
    {
        tets_verts_array->unbind(tets_verts_tref);
        delete tets_verts_array;
        tets_verts_array = NULL;
    }
    if (color_map_array != NULL) 
    {
        color_map_array->unbind(color_map_tref);
        delete color_map_array;
        color_map_array = NULL;
    }
   

}
void  eavlSimpleVRMutator::freeRaw()
{

    deleteArrayPtr(tets_raw);
    deleteArrayPtr(colormap_raw);
   

}