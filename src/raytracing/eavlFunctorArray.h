#ifndef EAVL_FUNCTOR_ARRAY
#define EAVL_FUNCTOR_ARRAY
#include "eavlArray.h"
#include "eavlRayExecutionMode.h"
/*
	eavlFunctorArray is a array wrapper that can be passed
	into a functor constructor and can be accessed 
	on either the host or device. Normally, if this
	was done, the pointer could be invalid.

	Allows arbitray reads and writes while still 
	still supporting CPU fallback.

*/

template<class T>
class eavlFunctorArray
{
 	protected:
 		T *device;
 		T *host;
 		bool cpu;

 	public:
 		eavlFunctorArray(eavlArray * arrayPtr)
 		{
 			
 			cpu = eavlRayExecutionMode::isCPUOnly();

 			if(!cpu)
 			{
 #ifdef HAVE_CUDA
 				device = (T*) arrayPtr->GetCUDAArray();
 				host = NULL;
 #endif
 			}
 			else
 			{
 				host = (T*) arrayPtr->GetHostArray();
 				device = NULL;	
 			} 
 		}
 		eavlFunctorArray(const eavlFunctorArray &other)
 		{
 			device = other.device;
 			host = other.host;
 			cpu = other.cpu;
 		}
 		~eavlFunctorArray()
 		{
 			//do nothing. We are just a wrapper.
 		}

 		EAVL_HOSTDEVICE const T& operator[](int i) const 
 		{
 #ifdef __CUDA_ARCH__
 			return device[i];
 #else  
 			return host[i];
 #endif
 		}

 		EAVL_HOSTDEVICE T& operator[](int i) 
 		{
 #ifdef __CUDA_ARCH__
 			return device[i];
 #else  
 			return host[i];
 #endif
 		}
};
#endif