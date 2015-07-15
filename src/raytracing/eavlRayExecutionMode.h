#ifndef EAVL_RAY_EXECUTION_MODE_H
#define EAVL_RAY_EXECUTION_MODE_H
#include "eavlArray.h"
#include "eavlExecutor.h"
/*
	We grab raw host and device pointers to write
	to abritrary locations and we need a way to determine
	if we can even ask for the device pointers. This has been
	an issue with computers that have CUDA installed, but have 
	no CUDA capable GPU. Since the user can set the execution 
	mode, this is safer.
*/
class eavlRayExecutionMode{

	public:
		static bool isCPUOnly()
		{
			bool onCPU = false;
			eavlIntArray theTester("Test", 1, 1);
#ifdef HAVE_CUDA
      		if(eavlExecutor::GetExecutionMode() == eavlExecutor::PreferGPU)
      		{
		        //test if cuda works: cuda may be installed but
		        //no GPU is present
		        try
		        {
		            theTester.GetCUDAArray();
		        }
		        catch(eavlException &e)
		        {
		            onCPU = true;
		        }
      		}
	        else if(eavlExecutor::GetExecutionMode() == eavlExecutor::ForceGPU) onCPU = false; //if this is set, so be it.
	        else onCPU = true;
#else 
      	    onCPU = true;
#endif
      	    return onCPU;
		}
};
#endif