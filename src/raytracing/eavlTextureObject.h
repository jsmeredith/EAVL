#ifndef EAVL_TEXTURE_OBJECT_H
#define EAVL_TEXTURE_OBJECT_H
#include <typeinfo>   // operator typeid
#include "eavlRayExecutionMode.h"
#include "eavlRayDefines.h"
/*******************************************************

/
*******************************************************/


template<class T>
class eavlTextureObject
{
  private: 
    bool cpu;                   //running on the cpu or gpu
    bool isResponsible;         //Is this object resposible for these arrays.
                                //if so, delete on destroy
    bool eavlArrayProvided;     //Was the eavlArray provided? if so they have allocated cuda mem
    bool allocatedCuda;         //Texture object create cuda array

    eavlArray *array;
  public:
    T *host;
    T *device;
    unsigned long long textureObjectId;
    
  public:
    eavlTextureObject(int N, T *from, bool resposible)
    {
        cpu = eavlRayExecutionMode::isCPUOnly();

        isResponsible     = resposible;
        eavlArrayProvided = false;
        allocatedCuda     = false;
        //Check to see if this is a supported type TODO: add other types
        bool singleChannel = true;
        bool floatType     = true;
        bool unsignedType  = false;
        //Find out how to allocate CUDA resource
        if( typeid(int) == typeid(T)   || 
            typeid(float) == typeid(T) || 
            typeid(unsigned int) == typeid(T)) singleChannel = true;
        else if( typeid(int4) == typeid(T) ||
                 typeid(float4) == typeid(T)) singleChannel = false;
        else THROW(eavlException,"Unsupported type for eavlTextureObject");

        if( typeid(int4) == typeid(T) || 
            typeid(int) == typeid(T)  ||
            typeid(unsigned int) == typeid(T)) 
        {
            floatType = false;
            if( typeid(unsigned int) == typeid(T) ) unsignedType = true;
        }

        host = from;

        if(!cpu)
        {

#ifdef HAVE_CUDA
            try
            {
                allocatedCuda = true;
                cudaResourceDesc resDesc;
                memset(&resDesc, 0, sizeof(resDesc));
                int nbytes = N * sizeof(T);
                cudaMalloc((void**)&device, nbytes);
                CUDA_CHECK_ERROR();

                cudaMemcpy(device, &(host[0]), nbytes, cudaMemcpyHostToDevice);
                CUDA_CHECK_ERROR();

                //create the resource descriptor
                resDesc.resType = cudaResourceTypeLinear;   
                resDesc.res.linear.devPtr = device;
                resDesc.res.linear.sizeInBytes = nbytes;

                if(floatType) resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
                else 
                {
                    if(!unsignedType) resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
                    else resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
                }
                resDesc.res.linear.desc.x = 32; // bits per channel
                if(!singleChannel)
                {
                    resDesc.res.linear.desc.y = 32;
                    resDesc.res.linear.desc.z = 32;
                    resDesc.res.linear.desc.w = 32;
                }

                //create the texture descriptor
                cudaTextureDesc texDesc;
                memset(&texDesc, 0, sizeof(texDesc));
                texDesc.readMode = cudaReadModeElementType;
     
                //texture id 
                cudaTextureObject_t tex = 0;
                cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
                CUDA_CHECK_ERROR();

                textureObjectId = tex; //this is just a unsigned long long
            }
            catch(eavlException &e)
            {
               cerr<<"Warning: GPU Failed on eavlTextureObject. Using CPU"<<endl;
               eavlExecutor::SetExecutionMode(eavlExecutor::ForceCPU);
            }
#endif
        }
    }

    eavlTextureObject(int N, eavlArray * arrayPtr, bool resposible)
    {
        cpu = eavlRayExecutionMode::isCPUOnly();

        isResponsible     = resposible;
        allocatedCuda     = false;
        eavlArrayProvided = true;
        
        array  = arrayPtr;
        
        
        bool singleChannel  = true;
        bool floatType      = true;
        bool unsignedType   = false;
        //Find out how to allocate CUDA resource
        if( typeid(int) == typeid(T)   || 
            typeid(float) == typeid(T) || 
            typeid(unsigned int) == typeid(T)) singleChannel = true;
        else if( typeid(int4) == typeid(T) ||
                 typeid(float4) == typeid(T)) singleChannel = false;
        else THROW(eavlException,"Unsupported type for eavlTextureObject");

        if( typeid(int4) == typeid(T) || 
            typeid(int) == typeid(T)  ||
            typeid(unsigned int) == typeid(T)) 
        {
            floatType = false;
            if( typeid(unsigned int) == typeid(T) ) unsignedType = true;
        }
  
        if(!cpu)
        {
#ifdef HAVE_CUDA
	
            device = (T*) arrayPtr->GetCUDAArray();
            int nbytes = N * sizeof(T);
            
            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
           
            //create the resource descriptor
            resDesc.resType = cudaResourceTypeLinear;   
            resDesc.res.linear.devPtr = device;
            resDesc.res.linear.sizeInBytes = nbytes;

            if(floatType) resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
            else 
            {
                if(!unsignedType) resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
                else resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
            }
            resDesc.res.linear.desc.x = 32; // bits per channel

            if(!singleChannel)
            {

                resDesc.res.linear.desc.y = 32;
                resDesc.res.linear.desc.z = 32;
                resDesc.res.linear.desc.w = 32;
            }

            //create the texture descriptor
            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.readMode = cudaReadModeElementType;
            
            //texture id 
            cudaTextureObject_t tex = 0;
            cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
            CUDA_CHECK_ERROR();
        
            textureObjectId = tex; //this is just a unsigned long long
#endif
        }
        else host   = (T*) arrayPtr->GetHostArray();
    }


    ~eavlTextureObject()
    {
#ifdef HAVE_CUDA
        
        if(!cpu && isResponsible)
        {

            cudaDestroyTextureObject(textureObjectId);
            CUDA_CHECK_ERROR();    
        }
        else if(!cpu && allocatedCuda)
        {
            cudaDestroyTextureObject(textureObjectId);
            CUDA_CHECK_ERROR(); 
            cudaFree(device);
            CUDA_CHECK_ERROR();   
        }
#endif
        if(isResponsible && eavlArrayProvided) 
        {
            delete array; 
        }
        else if(isResponsible && !eavlArrayProvided)
        {
            delete[] host;
        }
    }

    //This is just a local copy only meant to access the texture
    eavlTextureObject(const eavlTextureObject &other)
    {
        host = other.host;
        device = NULL;
        array = NULL;
        textureObjectId = other.textureObjectId;
        isResponsible = false;
        eavlArrayProvided = false;
        allocatedCuda = false;
        cpu = other.cpu;
    }
   
#ifdef __CUDA_ARCH__
    EAVL_DEVICEONLY  const T getValue(int index) const
    {
        return tex1Dfetch<T>(textureObjectId, index);
    }
#else
    EAVL_HOSTONLY const T &getValue( int index) const
    {
        //cout<<"Tex getting value from index "<<index<<" ptr = "<<host<<endl;
        return host[index];
    }
#endif

};


#endif
