#ifndef EAVL_SIMPLE_VR_H
#define EAVL_SIMPLE_VR_H
#include "eavlView.h"
#include "eavlFilter.h"
#include "eavlVRScene.h"
#include "eavlRTUtil.h"
#include "eavlConstTextureArray.h"
#include "eavlColor.h"

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
        //setNumPasses(numPasses); //force update
    }   
    void setNumPasses(int n)
    {
        if(n > 0 ) 
        { 
            //if( n != numPasses) 
            {
                int stride = nSamples / n;
                if((nSamples % n) != 0) stride++;
                pixelsPerPass = height*width*stride;
                passNumDirty = true;
                passZStride = stride;
            }
            numPasses = n;
        }
        else cerr<<"Cannot set number of passes to negative number : "<<n<<endl;
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
    void setDataName(string name)
    {
        dataname = name;
    }

    void setTransferFunctionFile(string fname)
    {
        tfFilename = fname;
    }

    void setOpacityFactor(float factor)
    {
        opacityFactor = factor;
    }

    virtual void Execute();
    void setColorMap3f(float*,int);
    void setColorMap4f(float*,int);
    void setDefaultColorMap();
    void setBGColor(eavlColor c) { bgColor = c;}
    void setTransparentBG(bool on){ isTransparentBG = on; }
    
    eavlByteArray*  getFrameBuffer();
    eavlFloatArray* getDepthBuffer(float, float, float);
    eavlVRScene*        scene;
  protected:
    string fieldname;
    string  tfFilename;
    int     height;
    int     width;
    int     nSamples;
    int     numTets;
    int     colormapSize;
    int     numPasses;  
    int     passSize;
    int     passCountEstimate;
    int     pixelsPerPass;
    int     passZStride;
    bool    geomDirty;
    bool    sizeDirty;
    bool    passNumDirty;
    bool    cpu;
    bool    verbose;
    bool 	isTransparentBG;
    float   opacityFactor;

    double  sampleTime;
    double  compositeTime;
    double  passSelectionTime;
    double  allocateTime ;
    double  screenSpaceTime;
    double  tempTime;
    double  passFilterTime;
    double  renderTime;
    eavlColor bgColor;
    string  dataname;
    eavlView view;

    eavlFloatArray*     samples;
    eavlFloatArray*     framebuffer;
    eavlByteArray*      rgba;
    

    eavlFloatArray*     ssa; 
    eavlFloatArray*     ssb;
    eavlFloatArray*     ssc;
    eavlFloatArray*     ssd;
    eavlFloatArray*     zBuffer;
    eavlIntArray*		minSample; 			// Keeps the pixel depth of the first sample so we can  get correct zbuff
    eavlFloatArray*     dummy;
    eavlIntArray*       clippingFlags;
    eavlIntArray*       iterator;
    eavlIntArray*       screenIterator;
    eavlByteArray*      minPasses;
    eavlByteArray*      maxPasses;
    eavlIntArray*       currentPassMembers; //indexes of tets in the current pass
    eavlIntArray*       indexScan;          //container for prefix sum
    eavlIntArray*       reverseIndex;       //array to hold gather addresses 
    eavlIntArray*       passCount;          //number of tets in current pass
    eavlIntArray*       mask;               //tets in the current pass 1 = in 0 = out
    eavlArrayIndexer*   i1;
    eavlArrayIndexer*   i2;
    eavlArrayIndexer*   i3;
    eavlArrayIndexer*   ir;
    eavlArrayIndexer*   ig;
    eavlArrayIndexer*   ib;
    eavlArrayIndexer*   ia;
    eavlArrayIndexer*   idummy;
    eavlFloatArray**     tetSOA;
    float*      colormap_raw;
    float*      tets_raw;
    float*      scalars_raw;
    bool        firstPass;
    int         maxPassSize;
    int         passEstimate;
    long int    memRequirements;
    int         minimumPassesForMemory;

    void        init();
    void        freeRaw();
    void        freeTextures();
    void        findCurrentPassMembers(int);
    void        reviseSpaceEstimate(float);
    void        getBBoxPixelExtent(eavlPoint3 &smins, eavlPoint3 &smaxs);
    void        dataWriter();
    void        readTransferFunction(string);
    void        calcMemoryRequirements();
    void        clearSamplesArray();

};
#endif
