#ifndef EAVL_SIMPLE_VR_H
#define EAVL_SIMPLE_VR_H
#include "eavlView.h"
#include "eavlFilter.h"
#include "eavlVRScene.h"
#include "eavlRTUtil.h"
#include "eavlConstTextureArray.h"


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

    void setView(eavlView v)
    {
        int newh = v.h;
        int neww = v.w;
    	if(height != newh || width != neww) sizeDirty = true;
    	view = v;
        height = newh;
        width = neww;
    }   

    void clear()
    {
        scene->clear();
        numTets = 0;
    }

    void setVerbose(bool on)
    {
        verbose = on;
    }

    virtual void Execute();
    void setColorMap3f(float*,int);
    void setDefaultColorMap();

    eavlByteArray*  getFrameBuffer() { return framebuffer; }
    eavlFloatArray* getDepthBuffer(float, float, float);
    eavlVRScene*        scene;
  protected:
    string fieldname;
    int 	height;
    int 	width;
    int 	nSamples;
    int 	numTets;
    int 	colormapSize;
    bool 	geomDirty;
    bool	sizeDirty;
    bool    cpu;
    bool    verbose;

    eavlView view;

    eavlFloatArray*		samples;
    eavlByteArray*      framebuffer;
    

    eavlFloatArray*     ssa;
    eavlFloatArray*     ssb;
    eavlFloatArray*     ssc;
    eavlFloatArray*     ssd;
    eavlFloatArray*     zBuffer;
    eavlFloatArray*     dummy;
    eavlIntArray*       clippingFlags;
    eavlIntArray*       iterator;
    eavlIntArray*       screenIterator;
    eavlArrayIndexer*   i1;
    eavlArrayIndexer*   i2;
    eavlArrayIndexer*   i3;
    eavlArrayIndexer*   ir;
    eavlArrayIndexer*   ig;
    eavlArrayIndexer*   ib;
    eavlArrayIndexer*   ia;
    eavlArrayIndexer*   idummy;
    float* 		colormap_raw;
    float*      tets_raw;
    float*      scalars_raw;

    void 		init();
    void 		freeRaw();
    void 		freeTextures();

};
#endif