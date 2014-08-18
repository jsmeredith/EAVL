#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlSimpleVRMutator.h"
#include "eavlMapOp.h"
#include "eavlColor.h"

//declare the texture reference even if we are not using texture memory
#ifndef HAVE_CUDA
template<class T> class texture {};
struct float4
{
    float x,y,z,w;
};
#endif
texture<float4> tets_verts_tref;
texture<float4> scalars_tref;
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
eavlConstArrayV2<float4>* scalars_array;

eavlSimpleVRMutator::eavlSimpleVRMutator()
{
	height = 500;
	width  = 500;

	samples = NULL;
	framebuffer = NULL;
    zBuffer = NULL;
    ssa = NULL;
    ssb = NULL;
    ssc = NULL;
    ssd = NULL;
    clippingFlags = NULL;
    iterator = NULL;
    screenIterator = NULL;

	tets_raw = NULL;
	colormap_raw = NULL;
    scalars_raw = NULL;

	scene = new eavlVRScene();

    geomDirty = true;
    sizeDirty = true;

    numTets = 0;
    nSamples = 200;

    i1 = new eavlArrayIndexer(3,0);
    i2 = new eavlArrayIndexer(3,1);
    i3 = new eavlArrayIndexer(3,2);

    ir = new eavlArrayIndexer(4,0);
    ig = new eavlArrayIndexer(4,1);
    ib = new eavlArrayIndexer(4,2);
    ia = new eavlArrayIndexer(4,3);

    idummy = new eavlArrayIndexer();
    idummy->mod = 1 ;
    dummy = new eavlFloatArray("",1,2);

    verbose = false;
    setDefaultColorMap();
}

eavlSimpleVRMutator::~eavlSimpleVRMutator()
{
    if(verbose) cout<<"Destructor"<<endl;
    deleteClassPtr(samples);
    deleteClassPtr(framebuffer);
    deleteClassPtr(zBuffer);
    deleteClassPtr(scene);
    deleteClassPtr(ssa);
    deleteClassPtr(ssb);
    deleteClassPtr(ssc);
    deleteClassPtr(ssd);
    deleteClassPtr(clippingFlags);
    deleteClassPtr(iterator);
    deleteClassPtr(i1);
    deleteClassPtr(i2);
    deleteClassPtr(i3);
    deleteClassPtr(ir);
    deleteClassPtr(ig);
    deleteClassPtr(ib);
    deleteClassPtr(ia);
    deleteClassPtr(idummy);

    freeTextures();
    freeRaw();

}

struct ScreenSpaceFunctor
{   
    eavlConstArrayV2<float4> verts;
    eavlView         view;
    int              nSamples;
    float            mindepth;
    float            maxdepth;
    ScreenSpaceFunctor(eavlConstArrayV2<float4>* _verts, eavlView _view, int _nSamples)
    : view(_view), verts(*_verts), nSamples(_nSamples)
    {
        float dist = (view.view3d.from - view.view3d.at).norm();
        eavlPoint3 closest(0,0,-dist+view.size*.5);
        eavlPoint3 farthest(0,0,-dist-view.size*.5);
        mindepth = (view.P * closest).z;
        maxdepth = (view.P * farthest).z;

    }

    EAVL_FUNCTOR tuple<float,float,float,float,float,float,float,float,float,float,float,float,int> operator()(tuple<int> iterator)
    {
        int tet = get<0>(iterator);
        eavlPoint3 mine(FLT_MAX,FLT_MAX,FLT_MAX);
        eavlPoint3 maxe(-FLT_MAX,-FLT_MAX,-FLT_MAX);
        float4 v[4];
        v[0] = verts.getValue(tets_verts_tref, tet*4   );
        v[1] = verts.getValue(tets_verts_tref, tet*4+1 );
        v[2] = verts.getValue(tets_verts_tref, tet*4+2 );
        v[3] = verts.getValue(tets_verts_tref, tet*4+3 );
        
        eavlPoint3 p[4];

        for( int i=0; i< 4; i++)
        {   
            p[i].x = v[i].x;
            p[i].y = v[i].y; 
            p[i].z = v[i].z;

            eavlPoint3 t = view.P * view.V * p[i];
            p[i].x = (t.x*.5+.5)  * view.w;
            p[i].y = (t.y*.5+.5)  * view.h;
            p[i].z = float(nSamples) * (t.z-mindepth)/(maxdepth-mindepth);

        }
        for(int i=0; i<4; i++)
        {    
            for (int d=0; d<3; ++d)
            {
                    mine[d] = min(p[i][d], mine[d] );
                    maxe[d] = max(p[i][d], maxe[d] );
            }
        }
        int clipped = 0;

        float mn = min(maxe[2],min(maxe[1],min(maxe[0], 1e9)));
        if(mn < 0) clipped = 1;

        //if (maxe[0] < 0)
        //    return;
        //if (maxe[1] < 0)
        //    return;
        //if (maxe[2] < 0)
        //    return;
        if (mine[0] >= view.w)
            clipped = 1;
        if (mine[1] >= view.h)
            clipped = 1;
        if (mine[2] >= nSamples)
            clipped = 1;
        //cout<<"TET "<<tet<<" ";
        
        return tuple<float,float,float,float,float,float,float,float,float,float,float,float,int>(p[0].x, p[0].y, p[0].z,
                                                                                                  p[1].x, p[1].y, p[1].z,
                                                                                                  p[2].x, p[2].y, p[2].z,
                                                                                                  p[3].x, p[3].y, p[3].z, clipped);
    }

   

};

EAVL_HOSTDEVICE bool TetBarycentricCoords(eavlPoint3 p0,
                                          eavlPoint3 p1,
                                          eavlPoint3 p2,
                                          eavlPoint3 p3,
                                          eavlPoint3 p,
                                          float &b0, float &b1, float &b2, float &b3)
{

    bool inside = true;
    eavlMatrix4x4 Mn(p0.x,p0.y,p0.z, 1,
                     p1.x,p1.y,p1.z, 1,
                     p2.x,p2.y,p2.z, 1,
                     p3.x,p3.y,p3.z, 1);

    eavlMatrix4x4 M0(p.x ,p.y ,p.z , 1,
                     p1.x,p1.y,p1.z, 1,
                     p2.x,p2.y,p2.z, 1,
                     p3.x,p3.y,p3.z, 1);

    eavlMatrix4x4 M1(p0.x,p0.y,p0.z, 1,
                     p.x ,p.y ,p.z , 1,
                     p2.x,p2.y,p2.z, 1,
                     p3.x,p3.y,p3.z, 1);

    eavlMatrix4x4 M2(p0.x,p0.y,p0.z, 1,
                     p1.x,p1.y,p1.z, 1,
                     p.x ,p.y ,p.z , 1,
                     p3.x,p3.y,p3.z, 1);

    eavlMatrix4x4 M3(p0.x,p0.y,p0.z, 1,
                     p1.x,p1.y,p1.z, 1,
                     p2.x,p2.y,p2.z, 1,
                     p.x ,p.y ,p.z , 1);


    float Dn = Mn.Determinant();
    float D0 = M0.Determinant();
    float D1 = M1.Determinant();
    float D2 = M2.Determinant();
    float D3 = M3.Determinant();

    float mx = max(D3,max(D2,max(D1, max(D0,0.f))));   //if any are greater than 0, mx > 0
    float mn = min(D3,min(D2,min(D1, min(D0,1e9)))); //if any are less than 0,    mn < 0
    if(Dn == 0) inside = false; 
    if (Dn<0)
    {
        if (D0>0 || D1>0 || D2>0 || D3>0)
        inside =false;
    }
    else //if(mn < 0)
    {
        if (D0<0 || D1<0 || D2<0 || D3<0)
        inside =false;
    }

    if(inside)
    {
        b0 = D0/Dn;
        b1 = D1/Dn;
        b2 = D2/Dn;
        b3 = D3/Dn;
    }

    return inside;

}

struct SampleFunctor
{   
    eavlConstArrayV2<float4> scalars;
    eavlView         view;
    int              nSamples;
    float*           samples;
    SampleFunctor(eavlConstArrayV2<float4>* _scalars, eavlView _view, int _nSamples, float* _samples)
    : view(_view), scalars(*_scalars), nSamples(_nSamples), samples(_samples)
    {

    }

    EAVL_FUNCTOR tuple<float> operator()(tuple<int, int,float,float,float,float,float,float,float,float,float,float,float,float> inputs )
    {
        int tet = get<0>(inputs);
        int clipped = get<1>(inputs);
        if( clipped == 1) return tuple<float>(0.f);

        eavlPoint3 p[4];
        p[0].x = get<2>(inputs);
        p[0].y = get<3>(inputs);
        p[0].z = get<4>(inputs);

        p[1].x = get<5>(inputs);
        p[1].y = get<6>(inputs);
        p[1].z = get<7>(inputs);

        p[2].x = get<8>(inputs);
        p[2].y = get<9>(inputs);
        p[2].z = get<10>(inputs);

        p[3].x = get<11>(inputs);
        p[3].y = get<12>(inputs);
        p[3].z = get<13>(inputs);
        /* need the extents again, just recalc */
        eavlPoint3 mine(FLT_MAX,FLT_MAX,FLT_MAX);
        eavlPoint3 maxe(-FLT_MAX,-FLT_MAX,-FLT_MAX);

        for(int i=0; i<4; i++)
        {    
            for (int d=0; d<3; ++d)
            {
                    mine[d] = min(p[i][d], mine[d] );
                    maxe[d] = max(p[i][d], maxe[d] );
            }
        }

        for(int i = 0; i < 3; i++) mine[i] = max(mine[i],0.f);
        /*clamp*/
        maxe[0] = min(float(view.w-1), maxe[0]); //??
        maxe[1] = min(float(view.h-1), maxe[1]);
        maxe[2] = min(float(nSamples-1), maxe[2]);
        /*if(mine[0]<0) printf("min x below 0 : %f at tet %d", mine[0], tet);
        if (mine[0] < 0)
            mine[0] = 0;
        if (mine[1] < 0)
            mine[1] = 0;
        if (mine[2] < 0)
            mine[2] = 0;
        if (maxe[0] >= view.w)
            maxe[0] = view.w-1;
        if (maxe[1] >= view.h)
            maxe[1] = view.h-1;
        if (maxe[2] >= nSamples)
            maxe[2] = nSamples-1;*/
        
        int xmin = ceil(mine[0]);
        int xmax = floor(maxe[0]);
        int ymin = ceil(mine[1]);
        int ymax = floor(maxe[1]);
        int zmin = ceil(mine[2]);
        int zmax = floor(maxe[2]);

        float value;
        if (xmin > xmax || ymin > ymax || zmin > zmax) return tuple<float>(0.f);
        float4 s = scalars.getValue(scalars_tref, tet);
        //if(tet> 200000) printf("Xmin %d , Ymin %d Zmin %d xmax %d ymax %d zmax %d \n", xmin, ymin, zmin, xmax, ymax, zmax);
        for(int x=xmin; x<=xmax; ++x)
        {
            for(int y=ymin; y<=ymax; ++y)
            {
                int startindex = (y*view.w + x)*nSamples;

                for(int z=zmin; z<=zmax; ++z)
                {

                    float b0,b1,b2,b3;
                    bool isInside = TetBarycentricCoords(p[0],p[1],p[2],p[3],
                                                         eavlPoint3(x,y,z),b0,b1,b2,b3);
                    if (!isInside)
                                continue;
                    value = b0*s.x + b1*s.y + b2*s.z + b3*s.w;
                    int index3d = startindex + z;

                    samples[index3d] = value;
                    
                }//z
            }//y
        }//x
        return tuple<float>(0.f);
    }

   

};

struct CompositeFunctor
{   
    eavlConstArrayV2<float4> colorMap;
    eavlView         view;
    int              nSamples;
    float*           samples;
    int              h;
    int              w;
    int              ncolors;
    float            mindepth;
    float            maxdepth;
    CompositeFunctor( eavlView _view, int _nSamples, float* _samples,eavlConstArrayV2<float4>* _colorMap, int _ncolors)
    : view(_view), nSamples(_nSamples), samples(_samples), colorMap(*_colorMap), ncolors(_ncolors)
    {

        w = view.w;
        h = view.h;
        float dist = (view.view3d.from - view.view3d.at).norm();
        eavlPoint3 closest(0,0,-dist+view.size*.5);
        eavlPoint3 farthest(0,0,-dist-view.size*.5);
        mindepth = (view.P * closest).z;
        maxdepth = (view.P * farthest).z;

    }

    EAVL_FUNCTOR tuple<byte,byte,byte,byte,float> operator()(tuple<int> inputs )
    {
        int idx = get<0>(inputs);
        int minz = nSamples;
        int x = idx%w;
        int y = idx/w;
        eavlColor color(0,0,0,0);
        for(int z = nSamples ; z>=0; --z)
        {
            int index3d = (y*w + x)*nSamples + z;
            float value = samples[index3d];
            if (value<0 || value>1)
                continue;

            int colorindex = float(ncolors-1) * value;
            float4 clr = colorMap.getValue(color_map_tref, colorindex);
            eavlColor c(clr.x,
                        clr.y,
                        clr.z,
                        1.0);
            // use a gaussian density function as the opactiy
            float center = 0.5;
            float sigma = 0.13;
            float attenuation = 0.02; 
            float alpha = exp(-(value-center)*(value-center)/(2*sigma*sigma));
            //float alpha = value;
            alpha *= attenuation;
            color.c[0] = color.c[0] * (1.-alpha) + c.c[0] * alpha;
            color.c[1] = color.c[1] * (1.-alpha) + c.c[1] * alpha;
            color.c[2] = color.c[2] * (1.-alpha) + c.c[2] * alpha;
            color.c[3] = color.c[3] * (1.-alpha) + c.c[3] * alpha;
            minz = z;
 
        }
        
        float depth;
        if (minz < nSamples)
        {
            float projdepth = float(minz)*(maxdepth-mindepth)/float(nSamples) + mindepth;
            depth = .5 * projdepth + .5;
        }

        return tuple<byte,byte,byte,byte,float>(color.c[0]*255., color.c[1]*255., color.c[2]*255.,color.c[3]*255.,depth);
        
    }
   

};

struct CompositeFunctorFB
{   
    eavlConstArrayV2<float4> colorMap;
    eavlView         view;
    int              nSamples;
    float*           samples;
    int              h;
    int              w;
    int              ncolors;
    float            mindepth;
    float            maxdepth;
    CompositeFunctorFB( eavlView _view, int _nSamples, float* _samples,eavlConstArrayV2<float4>* _colorMap, int _ncolors)
    : view(_view), nSamples(_nSamples), samples(_samples), colorMap(*_colorMap), ncolors(_ncolors)
    {

        w = view.w;
        h = view.h;
        float dist = (view.view3d.from - view.view3d.at).norm();
        eavlPoint3 closest(0,0,-dist+view.size*.5);
        eavlPoint3 farthest(0,0,-dist-view.size*.5);
        mindepth = (view.P * closest).z;
        maxdepth = (view.P * farthest).z;

    }

    EAVL_FUNCTOR tuple<byte,byte,byte,byte,float> operator()(tuple<int> inputs )
    {
        int idx = get<0>(inputs);
        int minz = nSamples;
        int x = idx%w;
        int y = idx/w;
        eavlColor color(0,0,0,0);
        for(int z = 0 ; z < nSamples; z++)
        {
            int index3d = (y*w + x)*nSamples + z;
            float value = samples[index3d];
            if (value<0 || value>1)
                continue;

            int colorindex = float(ncolors-1) * value;
            float4 clr = colorMap.getValue(color_map_tref, colorindex);
            eavlColor c(clr.x,
                        clr.y,
                        clr.z,
                        1.0);
            // use a gaussian density function as the opactiy
            float center = 0.5;
            float sigma = 0.13;
            float attenuation = 0.02;
            float alpha = exp(-(value-center)*(value-center)/(2*sigma*sigma));
            //float alpha = value;
            alpha *= attenuation;
            color.c[0] = color.c[0]  + c.c[0] * (1.-color.c[3])*alpha;
            color.c[1] = color.c[1]  + c.c[1] * (1.-color.c[3])*alpha;
            color.c[2] = color.c[2]  + c.c[2] * (1.-color.c[3])*alpha;
            color.c[3] = color.c[3]  + c.c[3] * (1.-color.c[3])*alpha;
            minz = z;
            if(color.c[3] >=1 ) break;

        }
        
        float depth;
        if (minz < nSamples)
        {
            float projdepth = float(minz)*(maxdepth-mindepth)/float(nSamples) + mindepth;
            depth = .5 * projdepth + .5;
        }

        return tuple<byte,byte,byte,byte,float>(color.c[0]*255., color.c[1]*255., color.c[2]*255.,color.c[3]*255.,depth);
        
    }
   

};



void eavlSimpleVRMutator::setColorMap3f(float* cmap,int size)
{
    if(verbose) cout<<"Setting new color map"<<endl;
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
    colormap_raw= new float[size*4];
    
    for(int i=0;i<size;i++)
    {
        colormap_raw[i*4  ] = cmap[i*3  ];
        colormap_raw[i*4+1] = cmap[i*3+1];
        colormap_raw[i*4+2] = cmap[i*3+2];
        colormap_raw[i*4+3] = .05;          //test Alpha
    }
    color_map_array = new eavlConstArrayV2<float4>((float4*)colormap_raw, colormapSize, color_map_tref);
}

void eavlSimpleVRMutator::setDefaultColorMap()
{   if(verbose) cout<<"setting defaul color map"<<endl;
    if(color_map_array!=NULL)
    {
        color_map_array->unbind(color_map_tref);
        delete color_map_array;
    }
    if(colormap_raw!=NULL)
    {
        delete[] colormap_raw;
    }
    //two values all 1s
    colormapSize=2;
    colormap_raw= new float[8];
    for(int i=0;i<8;i++) colormap_raw[i]=1.f;
    color_map_array = new eavlConstArrayV2<float4>((float4*)colormap_raw, colormapSize, color_map_tref);
    if(verbose) cout<<"Done setting defaul color map"<<endl;

}



void eavlSimpleVRMutator::init()
{

    if(sizeDirty)
    {   
        if(verbose) cout<<"Size Dirty"<<endl;
        deleteClassPtr(samples);
        deleteClassPtr(framebuffer);
        deleteClassPtr(screenIterator);
        deleteClassPtr(zBuffer);
        
        samples         = new eavlFloatArray("",1,height*width*nSamples);
        framebuffer     = new eavlByteArray("",1,height*width*4);
        screenIterator  = new eavlIntArray("",1,height*width);
        zBuffer         = new eavlFloatArray("",1,height*width);
        
        int size = height* width;
        for(int i=0; i < size; i++) screenIterator->SetValue(i,i);

        if(verbose) cout<<"Samples array size "<<height*width*nSamples<<endl;

        sizeDirty = false;
    }

    if(geomDirty)
    {   
        if(verbose) cout<<"Geometry Dirty"<<endl;

        freeTextures();
        freeRaw();

        deleteClassPtr(ssa);
        deleteClassPtr(ssb);
        deleteClassPtr(ssc);
        deleteClassPtr(ssd);
        deleteClassPtr(clippingFlags);
        deleteClassPtr(iterator);
        deleteClassPtr(dummy);

        tets_raw = scene->getTetPtr();
        scalars_raw = scene->getScalarPtr();
        for(int i=0 ; i<100; i++) cout<<scalars_raw[i]<<" ";
        cout<<endl;
        tets_verts_array    = new eavlConstArrayV2<float4>( (float4*) tets_raw, numTets*4, tets_verts_tref); // maybe reorganize the array to push all scalars to the end, better mem accessses
        scalars_array       = new eavlConstArrayV2<float4>( (float4*) scalars_raw, numTets, scalars_tref);
        for(int i=0 ; i<100; i++) cout<<scalars_array->getValue(scalars_tref,i).x<<" "<<scalars_array->getValue(scalars_tref,i).y<<" ";
        cout<<endl;
        ssa = new eavlFloatArray("",1, numTets*3);
        ssb = new eavlFloatArray("",1, numTets*3);
        ssc = new eavlFloatArray("",1, numTets*3);
        ssd = new eavlFloatArray("",1, numTets*3);

        clippingFlags = new eavlIntArray("",1, numTets);
        iterator      = new eavlIntArray("",1, numTets);
        dummy = new eavlFloatArray("",1,numTets);
        for(int i=0; i < numTets; i++) iterator->SetValue(i,i);

        geomDirty = false;
    }
    

}

void  eavlSimpleVRMutator::Execute()
{
    if(isOnGPU) {cout<<"GPU\n"; eavlExecutor::SetExecutionMode(eavlExecutor::ForceGPU);}
    else        {cout<<"CPU\n"; eavlExecutor::SetExecutionMode(eavlExecutor::ForceCPU);}

    int tets = scene->getNumTets();
    if(tets != numTets)
    {
        geomDirty = true;
        numTets = tets;
    }
    
    if(numTets < 1)
    {
        cout<<"Nothing to render."<<endl;
        return;
    }
    cout<<"Rendering "<<numTets<<" tets Color map Size : "<<colormapSize<<endl;
    int tinit;
    if(verbose) tinit = eavlTimer::Start();
    init();
    if(verbose) cout<<"Init        RUNTIME: "<<eavlTimer::Stop(tinit,"init")<<endl;
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(framebuffer),
                                             eavlOpArgs(framebuffer),
                                             IntMemsetFunctor(0.f)),
                                             "clear Frame Buffer");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(samples),
                                             eavlOpArgs(samples),
                                             FloatMemsetFunctor(0.f)),
                                             "clear Frame Buffer");
    eavlExecutor::Go();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(zBuffer),
                                             eavlOpArgs(zBuffer),
                                             FloatMemsetFunctor(1.f)),
                                             "clear Frame Buffer");
    eavlExecutor::Go();

    int ttrans;
    if(verbose) ttrans = eavlTimer::Start();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(iterator),
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
                                                        eavlIndexable<eavlFloatArray>(ssd,*i3),
                                                        eavlIndexable<eavlIntArray>(clippingFlags)),
                                             ScreenSpaceFunctor(tets_verts_array, view, nSamples)),
                                             "Screen Space transform");
    eavlExecutor::Go();

    if(verbose) cout<<"Transform   RUNTIME: "<<eavlTimer::Stop(ttrans,"ttrans")<<endl;

    float* samplePtr;
   
    if(isOnGPU)
    {
        samplePtr = (float*) samples->GetCUDAArray();
    }
    else 
    {
        samplePtr = (float*) samples->GetHostArray();
    }

    int tsample;
    if(verbose) tsample = eavlTimer::Start();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlIntArray>(iterator),
                                                        eavlIndexable<eavlIntArray>(clippingFlags),
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
                                             SampleFunctor(scalars_array, view, nSamples, samplePtr)),
                                             "Sampler");
    eavlExecutor::Go();
    if(verbose) cout<<"Sample      RUNTIME: "<<eavlTimer::Stop(tsample,"sample")<<endl;

    int tcomp;
    if(verbose) tcomp = eavlTimer::Start();
     eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(screenIterator),
                                              eavlOpArgs(eavlIndexable<eavlByteArray>(framebuffer,*ir),
                                                         eavlIndexable<eavlByteArray>(framebuffer,*ig),
                                                         eavlIndexable<eavlByteArray>(framebuffer,*ib),
                                                         eavlIndexable<eavlByteArray>(framebuffer,*ia),
                                                         eavlIndexable<eavlFloatArray>(zBuffer)),
                                             CompositeFunctor( view, nSamples, samplePtr, color_map_array, colormapSize), height*width),
                                             "Composite");
    eavlExecutor::Go();
    if(verbose) cout<<"Composite   RUNTIME: "<<eavlTimer::Stop(tcomp,"tcomp")<<endl;

}

void  eavlSimpleVRMutator::freeTextures()
{
    if(verbose) cout<<"Free textures"<<endl;

    if (tets_verts_array != NULL) 
    {
        tets_verts_array->unbind(tets_verts_tref);
        delete tets_verts_array;
        tets_verts_array = NULL;
    }
    if (scalars_array != NULL) 
    {
        scalars_array->unbind(scalars_tref);
        delete color_map_array;
        scalars_array = NULL;
    }
   

}
void  eavlSimpleVRMutator::freeRaw()
{
    if(verbose) cout<<"Free raw"<<endl;
    deleteArrayPtr(tets_raw);
    deleteArrayPtr(scalars_raw);
   

}