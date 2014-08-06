#ifndef EAVL_SIMPLE_VR_H
#define EAVL_SIMPLE_VR_H

#include "eavlFilter.h"
#include "RT/eavlVRScene.h"
#include "RT/eavlRTUtil.h"

class eavlSimpleVRMutator : public eavlMutator
{
  public:
    eavlSimpleVRMutator();
    ~eavlSimpleVRMutator();
    void SetField(const string &name)
    {
        fieldname = name;
    }

    void setNumSamples(int nsamples)
    {
    	if(nsamples > 0)
    	{
    		if(nsamples != nSamples) sizeDirty = true;
    		nSamples = nsamples;
    	}
    	else THROW(eavlException,"Cannot have a number of samples less than 1.");
    }

    void setView(eavlMatrix4x4 vp)
    {
    	//if(vp != NULL)
    	{
    		//if(height != v.h || width != view.w) sizeDirty = true;
    		//view = v;
    	}
    	//else THROW(eavlException,"Cannot set null view.");
    }


    virtual void Execute();
    void setColorMap3f(float*,int);
    void setDefaultColorMap();

    eavlFloatArray* getFrameBuffer() { return framebuffer; }
    //eavlFloatArray* getDepthBuffer() { return zBuffer; }

  protected:
    string fieldname;
    int 	height;
    int 	width;
    int 	nSamples;
    int 	numTets;
    int 	colormapSize;
    bool 	geomDirty;
    bool	sizeDirty;


    eavlFloatArray*		samples;
    eavlFloatArray*     framebuffer;
    eavlVRScene*        scene;

    float* 		colormap_raw;
    float*      tets_raw;

    void 		init();
    void 		freeRaw();
    void 		freeTextures();

};
#endif