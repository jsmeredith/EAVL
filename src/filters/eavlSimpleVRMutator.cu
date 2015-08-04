#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlSimpleVRMutator.h"
#include "eavlMapOp.h"
#include "eavlColor.h"
#include "eavlPrefixSumOp_1.h"
#include "eavlReduceOp_1.h"
#include "eavlGatherOp.h"
#include "eavlSimpleReverseIndexOp.h"
#include "eavlRayExecutionMode.h"
#include "eavlRTUtil.h"
#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#define COLOR_MAP_SIZE 1024

long int scounter = 0;
long int skipped = 0;

texture<float4> scalars_tref;
texture<float4> cmap_tref;

eavlConstTexArray<float4>* color_map_array;
eavlConstTexArray<float4>* scalars_array;

#define PASS_ESTIMATE_FACTOR  5.f
eavlSimpleVRMutator::eavlSimpleVRMutator()
{   
    cpu = eavlRayExecutionMode::isCPUOnly();


    opacityFactor = 1.f;
    height = 500;
    width  = 500;    
    setNumPasses(1); //default number of passes
    samples                = NULL;
    framebuffer            = NULL;
    zBuffer                = NULL;
    minSample			   = NULL;
    iterator               = NULL;
    screenIterator         = NULL;
    colormap_raw           = NULL;
    minPasses              = NULL;
    maxPasses              = NULL;
    currentPassMembers     = NULL;
    passNumDirty           = true;
    indexScan              = NULL;
    reverseIndex           = NULL;
    scalars_array          = NULL; 

    ir = new eavlArrayIndexer(4,0);
    ig = new eavlArrayIndexer(4,1);
    ib = new eavlArrayIndexer(4,2);
    ia = new eavlArrayIndexer(4,3);
    ssa    = NULL;
    ssb    = NULL;
    ssc    = NULL;
    ssd    = NULL;
    tetSOA = NULL;
    mask   = NULL;
    rgba   = NULL;
    scene = new eavlVRScene();

    geomDirty = true;
    sizeDirty = true;

    numTets = 0;
    nSamples = 300;
    passCount = new eavlIntArray("",1,1); 
    i1 = new eavlArrayIndexer(3,0);
    i2 = new eavlArrayIndexer(3,1);
    i3 = new eavlArrayIndexer(3,2);
    idummy = new eavlArrayIndexer();
    idummy->mod = 1 ;
    dummy = new eavlFloatArray("",1,2);

    verbose = false;

    setDefaultColorMap(); 
    isTransparentBG = false;
}

eavlSimpleVRMutator::~eavlSimpleVRMutator()
{
    if(verbose) cout<<"Destructor"<<endl;
    deleteClassPtr(samples);
    deleteClassPtr(framebuffer);
    deleteClassPtr(zBuffer);
    deleteClassPtr(minSample);
    deleteClassPtr(rgba);
    deleteClassPtr(scene);
    deleteClassPtr(ssa);
    deleteClassPtr(ssb);
    deleteClassPtr(ssc);
    deleteClassPtr(ssd);
    deleteClassPtr(iterator);
    deleteClassPtr(i1);
    deleteClassPtr(i2);
    deleteClassPtr(i3);
    deleteClassPtr(ir);
    deleteClassPtr(ig);
    deleteClassPtr(ib);
    deleteClassPtr(ia);
    deleteClassPtr(idummy);
    deleteClassPtr(minPasses);
    deleteClassPtr(maxPasses);
    deleteClassPtr(indexScan);
    deleteClassPtr(mask);
    deleteClassPtr(dummy);
    if(numPasses != 1) deleteClassPtr(currentPassMembers);
    deleteClassPtr(reverseIndex);
    deleteClassPtr(screenIterator);

    freeTextures();
    freeRaw();

}


void eavlSimpleVRMutator::getBBoxPixelExtent(eavlPoint3 &smins, eavlPoint3 &smaxs)
{
    float xmin = FLT_MAX;
    float xmax = -FLT_MAX;
    float ymin = FLT_MAX;
    float ymax = -FLT_MAX;
    float zmin = FLT_MAX;
    float zmax = -FLT_MAX;

    eavlPoint3 bbox[2];
    bbox[0] = smins;
    bbox[1] = smaxs;
    for(int x = 0; x < 2 ; x++)
    {
        for(int y = 0; y < 2 ; y++)
        {
            for(int z = 0; z < 2 ; z++)
            {
                eavlPoint3 temp(bbox[x].x, bbox[y].y, bbox[z].z);

                eavlPoint3 t = view.P * view.V * temp;
                t.x = (t.x*.5+.5)  * view.w;
                t.y = (t.y*.5+.5)  * view.h;
                t.z = (t.z*.5+.5)  * (float) nSamples;
                zmin = min(zmin,t.z);
                ymin = min(ymin,t.y);
                xmin = min(xmin,t.x);
                zmax = max(zmax,t.z);
                ymax = max(ymax,t.y);
                xmax = max(xmax,t.x);
            }
        }
    }

    smins.x = xmin;
    smins.y = ymin;
    smins.z = zmin;

    smaxs.x = xmax;
    smaxs.y = ymax;
    smaxs.z = zmax;
}

//When this is called, all tets passed in are in screen space
struct ScreenSpaceFunctor
{   
    float4 *xverts;
    float4 *yverts; 
    float4 *zverts;
    eavlView         view;
    int              nSamples;
    ScreenSpaceFunctor(float4 *_xverts, float4 *_yverts,float4 *_zverts, eavlView _view, int _nSamples)
    : view(_view), xverts(_xverts),yverts(_yverts),zverts(_zverts), nSamples(_nSamples)
    {}

    EAVL_FUNCTOR tuple<float,float,float,float,float,float,float,float,float,float,float,float> operator()(tuple<int> iterator)
    {
        int tet = get<0>(iterator);
        eavlPoint3 mine(FLT_MAX,FLT_MAX,FLT_MAX);
        eavlPoint3 maxe(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        float* v[3];
        v[0] = (float*)&xverts[tet]; //x
        v[1] = (float*)&yverts[tet]; //y
        v[2] = (float*)&zverts[tet]; //z

        eavlPoint3 p[4];
        //int clipped = 0;
        for( int i=0; i< 4; i++)
        {   
            p[i].x = v[0][i];
            p[i].y = v[1][i]; 
            p[i].z = v[2][i];

            eavlPoint3 t = view.P * view.V * p[i];
            //cout<<"Before"<<t<<endl;
            // if(t.x > 1 || t.x < -1) clipped = 1;
            // if(t.y > 1 || t.y < -1) clipped = 1;
            // if(t.z > 1 || t.z < -1) clipped = 1;
            p[i].x = (t.x*.5+.5)  * view.w;
            p[i].y = (t.y*.5+.5)  * view.h;
            p[i].z = (t.z*.5+.5)  * (float) nSamples;
            //cout<<"After "<<p[i]<<endl;
        }
        

        return tuple<float,float,float,float,float,float,float,float,float,float,float,float>(p[0].x, p[0].y, p[0].z,
                                                                                                  p[1].x, p[1].y, p[1].z,
                                                                                                  p[2].x, p[2].y, p[2].z,
                                                                                                  p[3].x, p[3].y, p[3].z);
    }

   

};

struct PassRange
{   
    float4 *xverts;
    float4 *yverts;
    float4 *zverts;
    eavlView         view;
    int              nSamples;
    float            mindepth;
    float            maxdepth;
    int              numPasses;
    int              passStride;
    PassRange(float4 *_xverts, float4 *_yverts,float4 *_zverts, eavlView _view, int _nSamples, int _numPasses)
    : view(_view), xverts(_xverts),yverts(_yverts),zverts(_zverts), nSamples(_nSamples), numPasses(_numPasses)
    {
       
        passStride = nSamples / numPasses;
        //if it is not evenly divided add one pixel row so we cover all pixels
        if(((int)nSamples % numPasses) != 0) passStride++;
        
    }

    EAVL_FUNCTOR tuple<byte,byte> operator()(tuple<int> iterator)
    {
        int tet = get<0>(iterator);
        eavlPoint3 mine(FLT_MAX,FLT_MAX,FLT_MAX);
        eavlPoint3 maxe(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        float* v[3];
        v[0] = (float*)&xverts[tet]; //x
        v[1] = (float*)&yverts[tet]; //y
        v[2] = (float*)&zverts[tet]; //z

        int clipped = 0;
        eavlPoint3 p[4];

        for( int i=0; i < 4; i++)
        {   
            p[i].x = v[0][i];
            p[i].y = v[1][i]; 
            p[i].z = v[2][i];

            eavlPoint3 t = view.P * view.V * p[i];
            if(t.x > 1 || t.x < -1) clipped = 1;
            if(t.y > 1 || t.y < -1) clipped = 1;
            if(t.z > 1 || t.z < -1) clipped = 1;
            p[i].x = (t.x*.5+.5)  * view.w;
            p[i].y = (t.y*.5+.5)  * view.h;
            p[i].z = (t.z*.5+.5)  * (float) nSamples;

        }
        for(int i=0; i<4; i++)
        {    
            for (int d=0; d<3; ++d)
            {
                    mine[d] = min(p[i][d], mine[d] );
                    maxe[d] = max(p[i][d], maxe[d] );
            }
        }
        //if the tet stradles the edge, dump it TODO: extra check to make sure it is all the way outside
        float mn = min(mine[2],min(mine[1],min(mine[0], float(1e9) )));
        if(mn < 0) clipped = 1;
        
        if(clipped == 1) return tuple<byte,byte>(255,255); //not part of any pass
        int minPass = 0;
        int maxPass = 0;
        
        minPass = mine[2] / passStride; //min z coord
        maxPass = maxe[2] / passStride; //max z coord
    
        return tuple<byte,byte>(minPass, maxPass);
    }

   

};


float EAVL_HOSTDEVICE ffmin(const float &a, const float &b)
{
    #if __CUDA_ARCH__
        return fmin(a,b);
    #else
        return (a > b) ? b : a;
    #endif
}

float EAVL_HOSTDEVICE ffmax(const float &a, const float &b)
{
     #if __CUDA_ARCH__
        return fmax(a,b);
    #else
        return (a > b) ? a : b;
    #endif
    
}

struct SampleFunctor3
{   
    const eavlConstTexArray<float4> *scalars;
    eavlView         view;
    int              nSamples;
    float*           samples;
    float*           fb;
    int              passMinZPixel;
    int              passMaxZPixel;
    int              zSize;
    SampleFunctor3(const eavlConstTexArray<float4> *_scalars, eavlView _view, int _nSamples, float* _samples, int _passMinZPixel, int _passMaxZPixel,int numZperPass, float* _fb)
    : view(_view), scalars(_scalars), nSamples(_nSamples), samples(_samples)
    {
        
        passMaxZPixel  = min(int(nSamples-1), _passMaxZPixel);
        passMinZPixel  = max(0, _passMinZPixel);
        zSize = numZperPass;
        fb = _fb;
    }

    EAVL_FUNCTOR tuple<float> operator()(tuple<int,float,float,float,float,float,float,float,float,float,float,float,float> inputs )
    {
        int tet = get<0>(inputs);
        
        eavlVector3 p[4]; //TODO vectorize
        p[0].x = get<1>(inputs);
        p[0].y = get<2>(inputs);
        p[0].z = get<3>(inputs);

        p[1].x = get<4>(inputs);
        p[1].y = get<5>(inputs);
        p[1].z = get<6>(inputs);

        p[2].x = get<7>(inputs);
        p[2].y = get<8>(inputs);
        p[2].z = get<9>(inputs);

        p[3].x = get<10>(inputs);
        p[3].y = get<11>(inputs);
        p[3].z = get<12>(inputs);

        eavlVector3 v[3];
        for(int i = 1; i < 4; i++)
        {
            v[i-1] = p[i] - p[0];
        }

        //                  a         b            c       d
        //float d1 = D22(mat[1][1], mat[1][2], mat[2][1], mat[2][2]);
        float d1 = v[1].y * v[2].z - v[2].y * v[1].z;
        //float d2 = D22(mat[1][0], mat[1][2], mat[2][0], mat[2][2]);
        float d2 = v[0].y * v[2].z - v[2].y *  v[0].z;
        //float d3 = D22(mat[1][0], mat[1][1], mat[2][0], mat[2][1]);
        float d3 = v[0].y * v[1].z - v[1].y * v[0].z;

        float det = v[0].x * d1 - v[1].x * d2 + v[2].x * d3;

        if(det == 0) return tuple<float>(0.f); // dirty degenerate tetrahedron
        det  = 1.f  / det;

        //D22(mat[0][1], mat[0][2], mat[2][1], mat[2][2]);
        float d4 = v[1].x * v[2].z - v[2].x * v[1].z;
        //D22(mat[0][1], mat[0][2], mat[1][1], mat[1][2])
        float d5 = v[1].x * v[2].y - v[2].x * v[1].y;
        //D22(mat[0][0], mat[0][2], mat[2][0], mat[2][2]) 
        float d6 = v[0].x * v[2].z- v[2].x * v[0].z; 
        //D22(mat[0][0], mat[0][2], mat[1][0], mat[1][2])
        float d7 = v[0].x * v[2].y - v[2].x * v[0].y;
        //D22(mat[0][0], mat[0][1], mat[2][0], mat[2][1])
        float d8 = v[0].x * v[1].z - v[1].x * v[0].z;
        //D22(mat[0][0], mat[0][1], mat[1][0], mat[1][1])
        float d9 = v[0].x * v[1].y - v[1].x * v[0].y;
        /* need the extents again, just recalc */
        eavlPoint3 mine(FLT_MAX,FLT_MAX,FLT_MAX);
        eavlPoint3 maxe(-FLT_MAX,-FLT_MAX,-FLT_MAX);
       
        for(int i=0; i<4; i++)  //these two loops cost 2 registers
        {    
            for (int d=0; d<3; ++d) 
            {
                    mine[d] = min(p[i][d], mine[d] );
                    maxe[d] = max(p[i][d], maxe[d] );
            }
        } 

        // for(int i = 0; i < 3; i++) mine[i] = max(mine[i],0.f);
        // /*clamp*/
        maxe[0] = min(float(view.w-1), maxe[0]); //??  //these lines cost 14 registers
        maxe[1] = min(float(view.h - 1.f), maxe[1]);
        maxe[2] = min(float(passMaxZPixel), maxe[2]);
        mine[2] = max(float(passMinZPixel), mine[2]);
        //cout<<p[0]<<p[1]<<p[2]<<p[3]<<endl;
        int xmin = ceil(mine[0]);
        int xmax = floor(maxe[0]);
        int ymin = ceil(mine[1]);
        int ymax = floor(maxe[1]);
        int zmin = ceil(mine[2]);
        int zmax = floor(maxe[2]);

        float4 s = scalars->getValue(scalars_tref, tet);

        for(int z = zmin; z <= zmax; ++z)
        {
            for(int y = ymin; y <= ymax; ++y)
            { 
                //int pixel = ( y * view.w + x);
                //if(fb[pixel * 4 + 3] >= 1) {continue;}
                
                int startindex = view.w*(y + view.h*(z -passMinZPixel));
                #pragma ivdep
                for(int x=xmin; x<=xmax; ++x)
                {

                    float w1 = x - p[0].x; 
                    float w2 = y - p[0].y; 
                    float w3 = z - p[0].z; 

                    float xx =   w1 * d1 - w2 * d4 + w3 * d5;
                    xx *= det; 

                    float yy = - w1 * d2 + w2 * d6 - w3 * d7; 
                    yy *= det;

                    float zz =   w1 * d3 - w2 * d8 + w3 * d9;
                    zz *= det;
                    w1 = xx; 
                    w2 = yy; 
                    w3 = zz; 

                    float w0 = 1.f - w1 - w2 - w3;

                    int index3d = (x + startindex);//;startindex + z;
                    float lerped = w0*s.x + w1*s.y + w2*s.z + w3*s.w;
                    float a = ffmin(w0,ffmin(w1,ffmin(w2,w3)));
                    float b = ffmax(w0,ffmax(w1,ffmax(w2,w3)));
                    if((a >= 0.f && b <= 1.f)) 
                    {
                        samples[index3d] = lerped;
                        //if(lerped < 0 || lerped >1) printf("Bad lerp %f ",lerped);
                    }

                }//z
            }//y
        }//x

        return tuple<float>(0.f);
    }
};


struct CompositeFunctorFB
{   
    const eavlConstTexArray<float4> *colorMap;
    eavlView         view;
    int              nSamples;
    float*           samples;
    int              h;
    int              w;
    int              ncolors;
    float            mindepth;
    float            maxdepth;
    eavlPoint3       minComposite;
    eavlPoint3       maxComposite;
    int              zOffest;
    bool             finalPass;
    int              maxSIndx;
    int 			 minZPixel;
    float4           bgColor;
    CompositeFunctorFB( eavlView _view, int _nSamples, float* _samples, const eavlConstTexArray<float4> *_colorMap, int _ncolors, eavlPoint3 _minComposite, eavlPoint3 _maxComposite, int _zOffset, bool _finalPass, int _maxSIndx, int _minZPixel, eavlColor _bgColor)
    : view(_view), nSamples(_nSamples), samples(_samples), colorMap(_colorMap), ncolors(_ncolors), minComposite(_minComposite), maxComposite(_maxComposite), finalPass(_finalPass), maxSIndx(_maxSIndx)
    {
        w = view.w;
        h = view.h;
        zOffest = _zOffset;
        minZPixel = _minZPixel;
        bgColor.x = _bgColor.c[0];
        bgColor.y = _bgColor.c[1];
        bgColor.z = _bgColor.c[2];
        bgColor.w = _bgColor.c[3];
    }
 
    EAVL_FUNCTOR tuple<float,float,float,float,int> operator()(tuple<int, float, float, float, float, int> inputs )
    {
        int idx = get<0>(inputs);
        int x = idx%w;
        int y = idx/w;
        int minZsample = get<5>(inputs);
        //get the incoming color and return if the opacity is already 100%
        float4 color= {get<1>(inputs),get<2>(inputs),get<3>(inputs),get<4>(inputs)};
        if(color.w >= 1) return tuple<float,float,float,float,int>(color.x, color.y, color.z,color.w, minZsample);

        //pixel outside the AABB of the data set
        if((x < minComposite.x || x > maxComposite.x) ||( y < minComposite.y || y > maxComposite.y ))
        {
            return tuple<float,float,float,float,int>(bgColor.x, bgColor.y, bgColor.z, bgColor.w, minZsample);
        }
        
        for(int z = 0 ; z < zOffest; z++)
        {
                //(x + view.w*(y + zSize*z));
            int index3d = (x + w*(y + h*(z))) ;//(y*w + x)*zOffest + z;
            
            
            float value =  samples[index3d];//tsamples->getValue(samples_tref, index3d);// samples[index3d];
            
            if (value <= 0.f || value > 1.f)
                continue;
        
            int colorindex = float(ncolors-1) * value;
            float4 c = colorMap->getValue(cmap_tref, colorindex);
            c.w *= (1.f - color.w); 
            color.x = color.x  + c.x * c.w;
            color.y = color.y  + c.y * c.w;
            color.z = color.z  + c.z * c.w;
            color.w = c.w + color.w;
			minZsample = min(minZsample, minZPixel + z); //we need the closest sample to get depth buffer 
            if(color.w >=1 ) break;

        }
    
        return tuple<float,float,float,float,int>(min(1.f, color.x),  min(1.f, color.y),min(1.f, color.z),min(1.f,color.w), minZsample);
        
    }
   

};

//compisite the bakground color into the framebuffer
struct CompositeBG
{   
    float4 cc;
    CompositeBG(eavlColor &_bgColor)
    {
    	cc.x = _bgColor.c[0];
    	cc.y = _bgColor.c[1];
    	cc.z = _bgColor.c[2];
    	cc.w = _bgColor.c[3]; 
    	
    	
    }

    EAVL_FUNCTOR tuple<float,float,float,float> operator()(tuple<float, float, float, float> inputs )
    {

        float4 color= {get<0>(inputs),get<1>(inputs),get<2>(inputs),get<3>(inputs)};
        if(color.w >= 1) return tuple<float,float,float,float>(color.x, color.y, color.z,color.w);

        float4 c = cc; 
		
        c.w *= (1.f - color.w); 
        color.x = color.x  + c.x * c.w;
        color.y = color.y  + c.y * c.w;
        color.z = color.z  + c.z * c.w;
        color.w = c.w + color.w;

        return tuple<float,float,float,float>(min(1.f, color.x),  min(1.f, color.y),min(1.f, color.z),min(1.f,color.w) );
    }
};

eavlFloatArray* eavlSimpleVRMutator::getDepthBuffer(float proj22, float proj23, float proj32)
{ 

        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(minSample), eavlOpArgs(zBuffer), convertDepthFunctor(view,nSamples)),"convertDepth");
        eavlExecutor::Go();
        return zBuffer;
}

void eavlSimpleVRMutator::setColorMap3f(float* cmap,int size)
{
    if(verbose) cout<<"Setting new color map 3f"<<endl;
    colormapSize = size;
    if(color_map_array != NULL)
    {
        color_map_array->unbind(cmap_tref);
        
        delete color_map_array;
    
        color_map_array = NULL;
    }
    if(colormap_raw != NULL)
    {
        delete[] colormap_raw;
        colormap_raw = NULL;
    }
    colormap_raw= new float[size*4];
    
    for(int i=0;i<size;i++)
    {
        colormap_raw[i*4  ] = cmap[i*3  ];
        colormap_raw[i*4+1] = cmap[i*3+1];
        colormap_raw[i*4+2] = cmap[i*3+2];
        colormap_raw[i*4+3] = .01f;          //test Alpha
    }
    color_map_array = new eavlConstTexArray<float4>((float4*)colormap_raw, colormapSize, cmap_tref, cpu);
}

void eavlSimpleVRMutator::setColorMap4f(float* cmap,int size)
{
    if(verbose) cout<<"Setting new color map"<<endl;
    colormapSize = size;
    if(color_map_array != NULL)
    {
        color_map_array->unbind(cmap_tref);
        
        delete color_map_array;
    
        color_map_array = NULL;
    }
    if(colormap_raw != NULL)
    {
        delete[] colormap_raw;
        colormap_raw = NULL;
    }
    colormap_raw= new float[size*4];
    
    for(int i=0;i<size;i++)
    {
        colormap_raw[i*4  ] = cmap[i*4  ];
        colormap_raw[i*4+1] = cmap[i*4+1];
        colormap_raw[i*4+2] = cmap[i*4+2];
        colormap_raw[i*4+3] = cmap[i*4+3];          
    }
    color_map_array = new eavlConstTexArray<float4>((float4*)colormap_raw, colormapSize, cmap_tref, cpu);
}

void eavlSimpleVRMutator::setDefaultColorMap()
{   if(verbose) cout<<"setting defaul color map"<<endl;
    if(color_map_array!=NULL)
    {
        color_map_array->unbind(cmap_tref);
        delete color_map_array;
        color_map_array = NULL;
    }
    if(colormap_raw!=NULL)
    {
        delete[] colormap_raw;
        colormap_raw = NULL;
    }
    //two values all 1s
    colormapSize=2;
    colormap_raw= new float[8];
    for(int i=0;i<8;i++) colormap_raw[i]=1.f;
    color_map_array = new eavlConstTexArray<float4>((float4*)colormap_raw, colormapSize, cmap_tref, cpu);
    if(verbose) cout<<"Done setting defaul color map"<<endl;

}


void eavlSimpleVRMutator::calcMemoryRequirements()
{

    unsigned long int mem = 0; //mem in bytes

    mem += pixelsPerPass * sizeof(float);       //samples
    mem += numTets * 12 * sizeof(float);
    mem += height * width * 4 * sizeof(float);  //framebuffer
    mem += height * width * sizeof(float);      //zbuffer
    mem += numTets * 4 * sizeof(float);         //scalars
    mem += numTets * 2;                         //min and max passes (BYTEs)
    mem += numTets * sizeof(int);               //interator
    mem += height * width * sizeof(int);        //screen iterator;
    mem += passCountEstimate * 12 * sizeof(float);//screen space coords
    //find pass members arrays
    mem += numTets * 4 * sizeof(int);           //indexscan, mask, currentPassMembers
    mem += passCountEstimate * sizeof(int);     //reverse index
    double memd = (double) mem / (1024.0 * 1024.0);
    if(verbose) printf("Memory needed %10.3f MB. Do you have enough?\n", memd);
    
    if(!cpu)
    {

#ifdef HAVE_CUDA
        size_t free_byte;
        size_t total_byte;
        cudaMemGetInfo( &free_byte, &total_byte );
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        if(verbose) printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
        if(mem > free_byte)
        {
            cout<<"Warning : this will exceed memory usage by "<< (mem - free_byte) << "bytes.\n";
        }
#endif

    }

    


}

void printGPUMemUsage()
{
    #ifdef HAVE_CUDA
        size_t free_byte;
        size_t total_byte;
        cudaMemGetInfo( &free_byte, &total_byte );
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
#endif
}

void eavlSimpleVRMutator::clearSamplesArray()
{
    int clearValue = 0xbf800000; //-1 float
    size_t bytes = pixelsPerPass * sizeof(float);
    if(!cpu)
    {
#ifdef HAVE_CUDA
       cudaMemset(samples->GetCUDAArray(), clearValue,bytes);
       CUDA_CHECK_ERROR();
#endif
    }
    else
    {
       memset(samples->GetHostArray(), clearValue, bytes);   
    }

}


void eavlSimpleVRMutator::init()
{
    
    if(sizeDirty)
    {   
        setNumPasses(numPasses);
        if(verbose) cout<<"Size Dirty"<<endl;
        deleteClassPtr(samples);
        deleteClassPtr(framebuffer);
        
        deleteClassPtr(zBuffer);
        
        samples         = new eavlFloatArray("",1,pixelsPerPass);
        framebuffer     = new eavlFloatArray("",1,height*width*4);
        rgba            = new eavlByteArray("",1,height*width*4);
        zBuffer         = new eavlFloatArray("",1,height*width);
        minSample		= new eavlIntArray("",1,height*width);
        clearSamplesArray();
        if(verbose) cout<<"Samples array size "<<pixelsPerPass<<" Current CPU val "<<cpu<< endl;
        if(verbose) cout<<"Current framebuffer size "<<(height*width*4)<<endl;
        sizeDirty = false;
         
    }

    if(geomDirty && numTets > 0)
    {   
        if(verbose) cout<<"Geometry Dirty"<<endl;
        firstPass = true;
        passNumDirty = true;
        freeTextures();
        freeRaw();

        deleteClassPtr(minPasses);
        deleteClassPtr(maxPasses);
        deleteClassPtr(iterator);
        deleteClassPtr(dummy);
        deleteClassPtr(indexScan);
        deleteClassPtr(mask);

        tetSOA = scene->getEavlTetPtrs();
        
        scalars_array       = new eavlConstTexArray<float4>( (float4*) scene->getScalarPtr()->GetHostArray(), 
                                                             numTets, 
                                                             scalars_tref, 
                                                             cpu);
        minPasses = new eavlByteArray("",1, numTets);
        maxPasses = new eavlByteArray("",1, numTets);
        indexScan = new eavlIntArray("",1, numTets);
        mask = new eavlIntArray("",1, numTets);

        iterator      = new eavlIntArray("",1, numTets);
        dummy = new eavlFloatArray("",1,1); //wtf
        for(int i=0; i < numTets; i++) iterator->SetValue(i,i);
        //readTransferFunction(tfFilename);
        geomDirty = false;
    }

    //we are trying to keep the mem usage down. We will conservativily estimate the number of
    //indexes to keep in here. Edge case would we super zoomed in a particlar region which
    //would maximize the wasted space.
    
    if(!firstPass)
    {
        float ratio = maxPassSize / (float) passCountEstimate;
        if(ratio < .9 || ratio > 1.f) 
        {
            passCountEstimate = maxPassSize + (int)(maxPassSize * .1); //add a little padding here.
            passNumDirty = true;
            cout<<"Ajdusting Pass size"<<endl;
        }
    }

    if(passNumDirty)
    {
        if(verbose) cout<<"Pass Dirty"<<endl;
        if(firstPass) 
        {
       
            passCountEstimate = (int)((numTets / numPasses) * PASS_ESTIMATE_FACTOR); //TODO: see how close we can cut this
            if(numPasses == 1) passCountEstimate = numTets;
            maxPassSize =-1;
            firstPass = false;
        }
        deleteClassPtr(currentPassMembers);
        deleteClassPtr(reverseIndex);
        deleteClassPtr(ssa);
        deleteClassPtr(ssb);
        deleteClassPtr(ssc);
        deleteClassPtr(ssd);
        deleteClassPtr(screenIterator);
        if(false && numPasses == 1)
        {
            currentPassMembers = iterator;
        }
        else
        {   //we don't need to allocate this if we are only doing one pass
            currentPassMembers = new eavlIntArray("",1, passCountEstimate);
            reverseIndex = new eavlIntArray("",1, passCountEstimate); 
        }
        int size = width * height;
        screenIterator  = new eavlIntArray("",1,size);
        for(int i=0; i < size; i++) screenIterator->SetValue(i,i);
        int space  = passCountEstimate*3;
        if(space < 0) cout<<"ERROR int overflow"<<endl;
        if(verbose) cout<<"allocating pce "<<passCountEstimate<<endl;
        ssa = new eavlFloatArray("",1, passCountEstimate*3); 
        ssb = new eavlFloatArray("",1, passCountEstimate*3);
        ssc = new eavlFloatArray("",1, passCountEstimate*3);
        ssd = new eavlFloatArray("",1, passCountEstimate*3);
        passNumDirty = false;
    }
    
    calcMemoryRequirements();
}

struct PassThreshFunctor
{
    int passId;
    PassThreshFunctor(int _passId) : passId(_passId)
    {}

    EAVL_FUNCTOR tuple<int> operator()(tuple<int,int> input){
        int minp = get<0>(input);
        int maxp = get<1>(input);
        if((minp <= passId) && (maxp >= passId)) return tuple<int>(1);
        else return tuple<int>(0);
    }
};

void eavlSimpleVRMutator::findCurrentPassMembers(int pass)
{
    int passtime;
    if(verbose)  passtime = eavlTimer::Start();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(minPasses,maxPasses),
                                         eavlOpArgs(mask),
                                         PassThreshFunctor(pass)),
                                         "find pass members");

    eavlExecutor::Go();
    eavlExecutor::AddOperation(new eavlPrefixSumOp_1(mask,indexScan,false), //inclusive==true exclusive ==false
                                                     "create indexes");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new eavlReduceOp_1<eavlAddFunctor<int> >
                              (mask,
                               passCount,
                               eavlAddFunctor<int>()),
                               "count output");
    eavlExecutor::Go();

    passSize = passCount->GetValue(0);
    maxPassSize = max(maxPassSize, passSize);

    if(passSize > passCountEstimate)
    {
      cout<<"WARNING Exceeded max passSize:  maxPassSize "<<maxPassSize<<" estimate "<<passCountEstimate<<endl;  
      passNumDirty = true;
      THROW(eavlException, "exceeded max pass size.");
    } 

    if(passSize == 0)
    {
        return;
    }
    
    eavlExecutor::AddOperation(new eavlSimpleReverseIndexOp(mask,
                                                            indexScan,
                                                            reverseIndex),
                                                            "generate reverse lookup");
    eavlExecutor::Go();
    
    eavlExecutor::AddOperation(new_eavlGatherOp(eavlOpArgs(iterator),
                                                eavlOpArgs(currentPassMembers),
                                                eavlOpArgs(reverseIndex),
                                                passSize),
                                                "pull in the tets for this pass");
    eavlExecutor::Go();
    

    if(verbose) passSelectionTime += eavlTimer::Stop(passtime,"pass");
}

void  eavlSimpleVRMutator::Execute()
{
	//
	// If we are doing parallel compositing, we just want the partial
	// composites without the background color
	//
	if(isTransparentBG) 
	{
		bgColor.c[0] =0.f; 
		bgColor.c[1] =0.f; 
		bgColor.c[2] =0.f; 
		bgColor.c[3] =0.f;
	}

    //timing accumulators
    double clearTime = 0;
    passFilterTime = 0;
    compositeTime = 0;
    passSelectionTime = 0;
    sampleTime = 0;
    allocateTime = 0;
    screenSpaceTime = 0;
    renderTime = 0;
   
    int tets = scene->getNumTets();
    
   
    if(tets != numTets)
    {
        geomDirty = true;
        numTets = tets;
    }
    if(verbose) cout<<"Num Tets = "<<numTets<<endl;

    int tinit;
    if(verbose) tinit = eavlTimer::Start();
    init();
    
    if(tets < 1)
    {
    	//There is nothing to render. Set depth and framebuffer
    	eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(minSample),
                                             eavlOpArgs(minSample),
                                             IntMemsetFunctor(nSamples+1000)), //what should this be?
                                             "clear first sample");
    	eavlExecutor::Go();
    	
    	eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(framebuffer),
                                             eavlOpArgs(framebuffer),
                                             FloatMemsetFunctor(0)),
                                             "clear Frame Buffer");
    	eavlExecutor::Go();
		eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(framebuffer,*ir),
				                                             eavlIndexable<eavlFloatArray>(framebuffer,*ig),
				                                             eavlIndexable<eavlFloatArray>(framebuffer,*ib),
				                                             eavlIndexable<eavlFloatArray>(framebuffer,*ia)),
				                                  eavlOpArgs(eavlIndexable<eavlFloatArray>(framebuffer,*ir),
				                                             eavlIndexable<eavlFloatArray>(framebuffer,*ig),
				                                             eavlIndexable<eavlFloatArray>(framebuffer,*ib),
				                                             eavlIndexable<eavlFloatArray>(framebuffer,*ia)),
				                                 CompositeBG(bgColor), height*width),
				                                 "Composite");
		eavlExecutor::Go();
		return;
    }
    
    float4* xtet;
    float4* ytet;
    float4* ztet;
    if(!cpu)
    {
        //cout<<"Getting cuda array for tets."<<endl;
        xtet = (float4*) tetSOA[0]->GetCUDAArray();
        ytet = (float4*) tetSOA[1]->GetCUDAArray();
        ztet = (float4*) tetSOA[2]->GetCUDAArray();
    }
    else 
    {
        xtet = (float4*) tetSOA[0]->GetHostArray();
        ytet = (float4*) tetSOA[1]->GetHostArray();
        ztet = (float4*) tetSOA[2]->GetHostArray();
    }
    float* samplePtr;
    if(!cpu)
    {
        samplePtr = (float*) samples->GetCUDAArray();
    }
    else 
    {
        samplePtr = (float*) samples->GetHostArray();
    }

    float* alphaPtr;
    if(!cpu)
    {
        alphaPtr = (float*) framebuffer->GetCUDAArray();
    }
    else 
    {
        alphaPtr = (float*) framebuffer->GetHostArray();
    }
    if(verbose) cout<<"Init        RUNTIME: "<<eavlTimer::Stop(tinit,"init")<<endl;


    // Pixels extents are used to skip empty space in compositing.
    eavlPoint3 mins(scene->getSceneBBox().min.x,scene->getSceneBBox().min.y,scene->getSceneBBox().min.z);
    eavlPoint3 maxs(scene->getSceneBBox().max.x,scene->getSceneBBox().max.y,scene->getSceneBBox().max.z);
    getBBoxPixelExtent(mins,maxs);

    int ttot;
    if(verbose) ttot = eavlTimer::Start();

    if(verbose)
    {
        cout<<"BBox Screen Space "<<mins<<maxs<<endl; 
    }
    int tclear;
    if(verbose) tclear = eavlTimer::Start();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(framebuffer),
                                             eavlOpArgs(framebuffer),
                                             FloatMemsetFunctor(0)),
                                             "clear Frame Buffer");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(zBuffer),
                                             eavlOpArgs(zBuffer),
                                             FloatMemsetFunctor(1.f)),
                                             "clear Frame Buffer");
    eavlExecutor::Go();
    
     eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(minSample),
                                             eavlOpArgs(minSample),
                                             IntMemsetFunctor(nSamples+1000)), //TODO:Maybe this should be higher
                                             "clear first sample");
    eavlExecutor::Go();
   
    
    
     if(verbose) cout<<"ClearBuffs  RUNTIME: "<<eavlTimer::Stop(tclear,"")<<endl;

    int ttrans;
    if(verbose) ttrans = eavlTimer::Start();
    if(false && numPasses == 1)
    {
        //just set all tets to the first pass
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(minPasses),
                                             eavlOpArgs(minPasses),
                                             IntMemsetFunctor(0)),
                                             "set");
        eavlExecutor::Go();
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(maxPasses),
                                             eavlOpArgs(maxPasses),
                                             IntMemsetFunctor(0)),
                                             "set");
        eavlExecutor::Go();
        //passSize = numTets;
    }
    else
    {
        //find the min and max passes the tets belong to
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(iterator),
                                             eavlOpArgs(minPasses, maxPasses),
                                             PassRange(xtet,ytet,ztet, view, nSamples, numPasses)),
                                             "PassFilter");
        eavlExecutor::Go(); 
    }
    

    if(verbose) passFilterTime =  eavlTimer::Stop(ttrans,"ttrans");
        
    
    //cout<<"Pass Z stride "<<passZStride<<endl;
    for(int i = 0; i < numPasses; i++)
    {
        int pixelZMin = passZStride * i;
        int pixelZMax = passZStride * (i + 1) - 1;
      
        try
        {
            //if(numPasses > 1) 
                findCurrentPassMembers(i);
        }
        catch(eavlException &e)
        {
            return;
        }
        

        
        if(passSize > 0)
        {
            int tclearS;
            if(verbose) tclearS = eavlTimer::Start();
            if (i != 0) clearSamplesArray();  //this is a win on CPU for sure, gpu seems to be the same
            // eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(samples),
            //                                          eavlOpArgs(samples),
            //                                          FloatMemsetFunctor(-1.f)),
            //                                          "clear Frame Buffer");
            // eavlExecutor::Go();
            if(verbose) clearTime += eavlTimer::Stop(tclearS,"");
                
            int tsspace;
            if(verbose) tsspace = eavlTimer::Start();
           
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(currentPassMembers),
                                                     eavlOpArgs(eavlIndexable<eavlFloatArray>(ssa,*i1),
                                                                eavlIndexable<eavlFloatArray>(ssa,*i2),
                                                                eavlIndexable<eavlFloatArray>(ssa,*i3),
                                                                eavlIndexable<eavlFloatArray>(ssb,*i1),
                                                                eavlIndexable<eavlFloatArray>(ssb,*i2),
                                                                eavlIndexable<eavlFloatArray>(ssb,*i3),
                                                                eavlIndexable<eavlFloatArray>(ssc,*i1),
                                                                eavlIndexable<eavlFloatArray>(ssc,*i2),
                                                                eavlIndexable<eavlFloatArray>(ssc,*i3),
                                                                eavlIndexable<eavlFloatArray>(ssd,*i1),
                                                                eavlIndexable<eavlFloatArray>(ssd,*i2),
                                                                eavlIndexable<eavlFloatArray>(ssd,*i3)),
                                                    ScreenSpaceFunctor(xtet,ytet,ztet,view, nSamples),passSize),
                                                    "Screen Space transform");
            eavlExecutor::Go();
    
            if(verbose) screenSpaceTime += eavlTimer::Stop(tsspace,"sample");
            int tsample;
            if(verbose) tsample = eavlTimer::Start();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlIntArray>(currentPassMembers),
                                                        eavlIndexable<eavlFloatArray>(ssa,*i1),
                                                        eavlIndexable<eavlFloatArray>(ssa,*i2),
                                                        eavlIndexable<eavlFloatArray>(ssa,*i3),
                                                        eavlIndexable<eavlFloatArray>(ssb,*i1),
                                                        eavlIndexable<eavlFloatArray>(ssb,*i2),
                                                        eavlIndexable<eavlFloatArray>(ssb,*i3),
                                                        eavlIndexable<eavlFloatArray>(ssc,*i1),
                                                        eavlIndexable<eavlFloatArray>(ssc,*i2),
                                                        eavlIndexable<eavlFloatArray>(ssc,*i3),
                                                        eavlIndexable<eavlFloatArray>(ssd,*i1),
                                                        eavlIndexable<eavlFloatArray>(ssd,*i2),
                                                        eavlIndexable<eavlFloatArray>(ssd,*i3)),
                                                        eavlOpArgs(eavlIndexable<eavlFloatArray>(dummy,*idummy)), 
                                                     SampleFunctor3(scalars_array, view, nSamples, samplePtr, pixelZMin, pixelZMax, passZStride, alphaPtr),passSize),
                                                     "Sampler");
            eavlExecutor::Go();
            
            if(verbose) sampleTime += eavlTimer::Stop(tsample,"sample");
            int talloc;
            if(verbose) talloc = eavlTimer::Start();

            if(verbose) allocateTime += eavlTimer::Stop(talloc,"sample");
            //eavlArrayIndexer * ifb = new eavlArrayIndexer(1, offset);
            //cout<<"screenIterator last value "<<screenIterator->GetS
            bool finalPass = (i == numPasses - 1) ? true : false;
            int tcomp;
            if(verbose) tcomp = eavlTimer::Start();
             eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlIntArray>(screenIterator),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ir),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ig),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ib),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ia),
                                                                 eavlIndexable<eavlIntArray>(minSample)),
                                                      eavlOpArgs(eavlIndexable<eavlFloatArray>(framebuffer,*ir),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ig),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ib),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ia),
                                                                 eavlIndexable<eavlIntArray>(minSample)),
                                                     CompositeFunctorFB( view, nSamples, samplePtr, color_map_array, colormapSize, mins, maxs, passZStride, finalPass, pixelsPerPass,pixelZMin, bgColor), width*height),
                                                     "Composite");
            eavlExecutor::Go();
            if(verbose) compositeTime += eavlTimer::Stop(tcomp,"tcomp");
        }
    }//for each pass
    if(verbose) renderTime  = eavlTimer::Stop(ttot,"total render");
    if(verbose) cout<<"PassFilter  RUNTIME: "<<passFilterTime<<endl;
    cout<<"Clear Sample  RUNTIME: "<<clearTime<<endl;
    if(verbose) cout<<"PassSel     RUNTIME: "<<passSelectionTime<<" Pass AVE: "<<passSelectionTime / (float)numPasses<<endl;
    if(verbose) cout<<"ScreenSpace RUNTIME: "<<screenSpaceTime<<" Pass AVE: "<<screenSpaceTime / (float)numPasses<<endl;
    if(verbose) cout<<"Sample      RUNTIME: "<<sampleTime<<" Pass AVE: "<<sampleTime / (float)numPasses<<endl;
    if(verbose) cout<<"Composite   RUNTIME: "<<compositeTime<<" Pass AVE: "<<compositeTime / (float)numPasses<<endl;
    if(verbose) cout<<"Alloc       RUNTIME: "<<allocateTime<<" Pass AVE: "<<allocateTime / (float)numPasses<<endl;
    if(verbose) cout<<"Total       RUNTIME: "<<renderTime<<endl;
    //dataWriter();
 eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ir),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ig),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ib),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ia)),
                                                      eavlOpArgs(eavlIndexable<eavlFloatArray>(framebuffer,*ir),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ig),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ib),
                                                                 eavlIndexable<eavlFloatArray>(framebuffer,*ia)),
                                                     CompositeBG(bgColor), height*width),
                                                     "Composite");
    eavlExecutor::Go();
}


inline bool exists (const std::string& name) {
    ifstream f(name.c_str());
    if (f.good()) {
        f.close();
        return true;
    } else {
        f.close();
        return false;
    }   
}

void  eavlSimpleVRMutator::dataWriter()
{
  string sCPU = "_CPU_";
  string sGPU = "_GPU_";
  string dfile;
  if(cpu) dfile = "datafile_" + sCPU + dataname + ".dat";
  else dfile = "datafile_" + sGPU + dataname + ".dat";  
   
  if(!exists(dfile))
  {
    ofstream boilerplate;
    boilerplate.open (dfile.c_str());
    boilerplate << "Step\n";
    boilerplate << "Pass Filter\n";
    boilerplate << "Pass Selection\n";
    boilerplate << "Screen Space\n";
    boilerplate << "Sampling\n";
    boilerplate << "Compostiting\n";
    boilerplate << "Render\n";
    boilerplate.close();
  }
  string separator = ",";
  string line[7];
  double times[6];
  times[0] = passFilterTime;
  times[1] = passSelectionTime;
  times[2] = screenSpaceTime;
  times[3] = sampleTime;
  times[4] = compositeTime;
  times[5] = renderTime;

  ifstream dataIn (dfile.c_str());
  if (dataIn.is_open())
  {
    for(int i = 0; i < 7; i++)
    {
        getline (dataIn,line[i]);
        //cout << line[i] << '\n';
    }
    dataIn.close();
  }
  else
  {
    cout << "Unable to open file"<<endl;
    return; 
  }
  ofstream dataOut (dfile.c_str());
  if (dataOut.is_open())
  {
    for(int i = 0; i < 7; i++)
    {
         if(i ==  0) dataOut << line[i] << separator <<numPasses<<endl;
         else dataOut << line[i] << separator <<times[i-1]<<endl;
    }
    
    dataOut.close();
  }
  else dataOut << "Unable to open file";
    string space = " ";

}

void  eavlSimpleVRMutator::freeTextures()
{
    if (scalars_array != NULL) 
    {
        scalars_array->unbind(scalars_tref);
        delete scalars_array;
        scalars_array = NULL;
    }

}
void  eavlSimpleVRMutator::freeRaw()
{
}

void eavlSimpleVRMutator::readTransferFunction(string filename)
{

    std::fstream file(filename.c_str(), std::ios_base::in);
    if(file != NULL)
    {
        //file format number of peg points, then peg points 
        //peg point 0 0 255 255 0.0241845 //RGBA postion(float)
        int numPegs;
        file>>numPegs;
        if(numPegs >= COLOR_MAP_SIZE || numPegs < 1) 
        {
            cerr<<"Invalid number of peg points, valid range [1,1024]: "<<numPegs<<endl;
            exit(1);
        } 

        float *rgb = new float[numPegs*3];
        float *positions = new float[numPegs];
        int trash;
        for(int i = 0; i < numPegs; i++)
        {
            file>>rgb[i*3 + 0];
            file>>rgb[i*3 + 1];
            file>>rgb[i*3 + 2];
            rgb[i*3 + 0] = rgb[i*3 + 0] / 255.f; //normalize
            rgb[i*3 + 1] = rgb[i*3 + 1] / 255.f; //normalize
            rgb[i*3 + 2] = rgb[i*3 + 2] / 255.f; //normalize
            file>>trash;
            file>>positions[i];

        }
        //next we read in the free form opacity
        int numOpacity;
        file>>numOpacity;
        if(numOpacity >= COLOR_MAP_SIZE || numOpacity < 1) 
        {
            cerr<<"Invalid number of opacity points, valid range [1,1024]: "<<numOpacity<<endl;
            exit(1);
        } 
        float *opacityPoints = new float[numOpacity];
        float *opacityPositions = new float[numOpacity];
        for(int i = 0; i < numOpacity; i++)
        {
            file>>opacityPoints[i];
            opacityPoints[i] = (opacityPoints[i] / 255.f ) * opacityFactor; //normalize
            opacityPositions[i] = i / (float) numOpacity;
        }
        cout<<endl;
        //build the color map

        int rgbPeg1 = 0;
        int rgbPeg2 = 1;

        int opacityPeg1 = 0;
        int opacityPeg2 = 1;
        
        float currentPosition = 0.f;
        float *colorMap = new float[COLOR_MAP_SIZE * 4];

        //fill in rgb values
        float startPosition;
        float endPosition;
        float4 startColor = {0,0,0,0};
        float4 endColor = {0,0,0,0};
        //init color and positions
        if(positions[rgbPeg1] == 0.f)
        {
            startPosition = positions[rgbPeg1];
            startColor.x = rgb[rgbPeg1*3 + 0];
            startColor.y = rgb[rgbPeg1*3 + 1];
            startColor.z = rgb[rgbPeg1*3 + 2];
            endPosition = positions[rgbPeg2];
            endColor.x = rgb[rgbPeg2*3 + 0];
            endColor.y = rgb[rgbPeg2*3 + 1];
            endColor.z = rgb[rgbPeg2*3 + 2];
        }
        else
        {
            //cout<<"init 0 start"<<endl;
            startPosition = 0;
            //color already 0
            endPosition = positions[rgbPeg1];
            endColor.x = rgb[rgbPeg1*3 + 0];
            endColor.y = rgb[rgbPeg1*3 + 1];
            endColor.z = rgb[rgbPeg1*3 + 2];
        }

        for(int i = 0; i < COLOR_MAP_SIZE; i++)
        {
            
            currentPosition = i / (float)COLOR_MAP_SIZE;

            float t = (currentPosition - startPosition) / (endPosition - startPosition);
            colorMap[i*4 + 0] = lerp(startColor.x, endColor.x, t);
            colorMap[i*4 + 1] = lerp(startColor.y, endColor.y, t);
            colorMap[i*4 + 2] = lerp(startColor.z, endColor.z, t);

            if( (currentPosition > endPosition) )
            {
                //advance peg points

                rgbPeg1++;
                rgbPeg2++;  
                //reached the last Peg point 
                if(rgbPeg2 >= numPegs) 
                {
                    startPosition = positions[rgbPeg1];
                    startColor.x = rgb[rgbPeg1*3 + 0];
                    startColor.y = rgb[rgbPeg1*3 + 1];
                    startColor.z = rgb[rgbPeg1*3 + 2];
                    //just keep the same color, we could change this to 0
                    endPosition = 1.f;
                    endColor.x = rgb[rgbPeg1*3 + 0];
                    endColor.y = rgb[rgbPeg1*3 + 1];
                    endColor.z = rgb[rgbPeg1*3 + 2];

                }
                else
                {
                    startPosition = positions[rgbPeg1];
                    startColor.x = rgb[rgbPeg1*3 + 0];
                    startColor.y = rgb[rgbPeg1*3 + 1];
                    startColor.z = rgb[rgbPeg1*3 + 2];
                    endPosition = positions[rgbPeg2];
                    endColor.x = rgb[rgbPeg2*3 + 0];
                    endColor.y = rgb[rgbPeg2*3 + 1];
                    endColor.z = rgb[rgbPeg2*3 + 2];
                }

            }
        }

        float startAlpha = 0.f;
        float endAlpha = 1.f;
        if(positions[opacityPeg1] == 0.f)
        {
            startPosition = opacityPositions[opacityPeg1];
            startAlpha = opacityPoints[opacityPeg1];
            endPosition = opacityPositions[opacityPeg2];
            endAlpha = opacityPoints[opacityPeg2];
        }
        else
        {
            startPosition = 0.f;
            startAlpha = 0.f;
            endPosition = opacityPoints[opacityPeg1];
            endAlpha = opacityPoints[opacityPeg1];
        }
        // fill in alphas
        for(int i = 0; i < COLOR_MAP_SIZE; i++)
        {
           
            currentPosition = i / (float)COLOR_MAP_SIZE;

            float t = (currentPosition - startPosition) / (endPosition - startPosition);
            colorMap[i*4 + 3] = lerp(startAlpha, endAlpha, t);

            //cout<<colorMap[i*4+0]<<" "<<colorMap[i*4+1]<<" "<<colorMap[i*4+2]<<" "<<colorMap[i*4+3]<<" pos "<<currentPosition<<endl;
            if(currentPosition > endPosition)
            {
                //advance peg points

                opacityPeg1++;
                opacityPeg2++;  
                //reached the last Peg point
                if(opacityPeg2 >= numOpacity) 
                {
                    startPosition = opacityPositions[opacityPeg1];
                    startAlpha = opacityPoints[opacityPeg1];
                   
                    //just keep the same color, we could change this to 0
                    endPosition = 1.f;
                    endAlpha = opacityPoints[opacityPeg1];
                    

                }
                else
                {
                    startPosition = opacityPositions[opacityPeg1];
                    startAlpha = opacityPoints[opacityPeg1];
                   
                    endPosition = opacityPositions[opacityPeg2];
                    endAlpha = opacityPoints[opacityPeg2];
                   
                }
            }
        }

        setColorMap4f(colorMap, COLOR_MAP_SIZE);
        delete[] rgb;
        delete[] positions;
        delete[] opacityPoints;
        delete[] opacityPositions;
    }
    else 
    {
        cerr<<"Could not open tranfer function file : "<<filename.c_str()<<endl;
    }
}

eavlByteArray * eavlSimpleVRMutator::getFrameBuffer()
{
    
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(framebuffer),
                                             eavlOpArgs(rgba),
                                             CastToUnsignedCharFunctor()),
                                             "set");
    eavlExecutor::Go();
    return rgba;
}
