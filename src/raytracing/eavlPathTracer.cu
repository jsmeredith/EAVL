#include <eavlPathTracer.h>
#include <eavl1toNScatterOp.h>
#include <eavlNto1GatherOp.h>
#include <eavlSampler.h>
#include <eavlRandOp.h>
#include <eavlMapOp.h>

eavlPathTracer::eavlPathTracer()
{

	//
	// Initialize
	//
	triGeometry = new eavlRayTriangleGeometry();
	camera = new eavlRayCamera();
	camera->setMortonSorting(false);
	rays = new eavlFullRay(camera->getWidth() * camera->getHeight());
	intersector = new eavlRayTriangleIntersector();
	scene = new eavlRTScene();
	geometryDirty = true;
	currentFrameSize = camera->getWidth() * camera->getHeight();
	frameBuffer = new eavlFloatArray("", 1, currentFrameSize * 4);
	rgbaPixels = new eavlByteArray("", 1, currentFrameSize * 4);
	depthBuffer = new eavlFloatArray("", 1, currentFrameSize);
	inShadow = new eavlIntArray("", 1, currentFrameSize);
	ambientPct = new eavlFloatArray("", 1, currentFrameSize);
    shadowX = new eavlFloatArray("",1,currentFrameSize);
    shadowY = new eavlFloatArray("",1,currentFrameSize);
    shadowZ = new eavlFloatArray("",1,currentFrameSize);
    reflectX = new eavlFloatArray("",1,currentFrameSize);
    reflectY = new eavlFloatArray("",1,currentFrameSize);
    reflectZ = new eavlFloatArray("",1,currentFrameSize);
    rSurface = new eavlFloatArray("",1,currentFrameSize);                  
    gSurface = new eavlFloatArray("",1,currentFrameSize);
    bSurface = new eavlFloatArray("",1,currentFrameSize);
    rCurrent = new eavlFloatArray("",1,currentFrameSize);                  
    gCurrent = new eavlFloatArray("",1,currentFrameSize);
    bCurrent = new eavlFloatArray("",1,currentFrameSize);
    lred = new eavlFloatArray("",1,currentFrameSize);               
    lgreen = new eavlFloatArray("",1,currentFrameSize);
    lblue = new eavlFloatArray("",1,currentFrameSize);
	bgColor.x = .5f;
	bgColor.y = .5f;
	bgColor.z = .5f;
	//
	// Create default color map
	//
	numColors = 2;
	float *defaultColorMap = new float[numColors * 3];
	for (int i = 0; i < numColors * 3; ++i)
	{
		defaultColorMap[i] = 1.f;
	}
	colorMap = new eavlTextureObject<float>(numColors * 3, defaultColorMap, true);

	materials = NULL;
	eavlMaterials = NULL;


	redIndexer   = new eavlArrayIndexer(4,0);
    greenIndexer = new eavlArrayIndexer(4,1);
    blueIndexer  = new eavlArrayIndexer(4,2);
    alphaIndexer = new eavlArrayIndexer(4,3);
    indexer   = new eavlArrayIndexer();

}

eavlPathTracer::~eavlPathTracer()
{
	delete colorMap;
	delete triGeometry;
	delete camera;
	delete intersector;
	delete scene;
	delete frameBuffer;
	delete rgbaPixels;
	delete depthBuffer;
	delete inShadow; 
    delete redIndexer;
    delete greenIndexer;
    delete blueIndexer;
    delete alphaIndexer;
    delete ambientPct;
    delete shadowX;
    delete shadowY;
    delete shadowZ;
    delete reflectX;
    delete reflectY;
    delete reflectZ;
    delete indexer;
    delete rSurface;
    delete gSurface;
    delete bSurface;
    delete rCurrent;
    delete gCurrent;
    delete bCurrent;
    delete lred;
    delete lgreen;
    delete lblue;

}


void eavlPathTracer::startScene()
{
	scene->clear();
	geometryDirty = true;
}
void eavlPathTracer::setColorMap3f(float* cmap, const int &nColors)
{
	// Colors are fed in as RGB, no alpha
	if(nColors < 1)
	{
		THROW(eavlException, "Cannot set color map size of less than 1");
	}
    delete colorMap;
    colorMap = new eavlTextureObject<float>(nColors * 3, cmap, false);
    numColors = nColors;
}

struct NormalFunctor{

    eavlTextureObject<float>  scalars;
    eavlTextureObject<float>  norms;

    eavlTextureObject<float>  colorMap;
    int colorMapSize;
    int sampleNum;

    NormalFunctor(eavlTextureObject<float>  *_scalars,
    			  eavlTextureObject<float>  *_norms,
                  eavlTextureObject<float>  *_colorMap,
                  int _colorMapSize)
        :scalars(*_scalars),
         norms(*_norms),
         colorMap(*_colorMap),
         colorMapSize(_colorMapSize)

    {
        sampleNum = 1;
    }                                                    
    EAVL_FUNCTOR tuple<float,
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                        float,
                        float> operator()( tuple<float,  // rayOrigin x
										   						    float,  // rayOrigin y
										   						    float,  // rayOrigin z
										   						    float,  // rayDir x
										   						    float,  // rayDir y
										   						    float,  // rayDir z
										   						    float,  // hit distance
										   						    float,  // alpha
										   						    float,  // beta
										   						    int     // Hit index
										   						    > input, int seed)
    {
       
       	eavlVector3 rayOrigin(get<0>(input), get<1>(input), get<2>(input));
        eavlVector3 rayDir(get<3>(input), get<4>(input), get<5>(input));
        float hitDistance = get<6>(input);
        rayDir.normalize();
        eavlVector3 intersect = rayOrigin + hitDistance * rayDir  - EPSILON * rayDir; 

        float alpha = get<7>(input);
        float beta  = get<8>(input);
        float gamma = 1.f - alpha - beta;
        int hitIndex=get<9>(input);
        if(hitIndex == -1) return tuple<float,
                                        float,
                                        float,
                                        float,
                                        float,
                                        float,
                                        float,
                                        float,
                                        float,
                                        float,
                                        float,
                                        float>(0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f);
     
        eavlVector3 aNorm, bNorm, cNorm;
        aNorm.x = norms.getValue(hitIndex * 9 + 0);
        aNorm.y = norms.getValue(hitIndex * 9 + 1);
        aNorm.z = norms.getValue(hitIndex * 9 + 2);
        bNorm.x = norms.getValue(hitIndex * 9 + 3);
        bNorm.y = norms.getValue(hitIndex * 9 + 4);
        bNorm.z = norms.getValue(hitIndex * 9 + 5);
        aNorm.x = norms.getValue(hitIndex * 9 + 6);
        aNorm.y = norms.getValue(hitIndex * 9 + 7);
        aNorm.z = norms.getValue(hitIndex * 9 + 8);

        eavlVector3 normal;
        normal = aNorm*alpha + bNorm*beta + cNorm*gamma;
        float lerpedScalar = scalars.getValue(hitIndex * 3 + 0) * alpha +
        					 scalars.getValue(hitIndex * 3 + 1) * beta  + 
        					 scalars.getValue(hitIndex * 3 + 2) * gamma;
        //reflect the ray
        normal.normalize();
        if ((normal * rayDir) > 0.0f) normal = -normal; //flip the normal if we hit the back side
        eavlVector3 reflection = rayDir - normal*2.f*(normal*rayDir);

        int   colorIdx = max(min(colorMapSize-1, (int)floor(lerpedScalar*colorMapSize)), 0);
        float shine = (colorIdx > colorIdx / 4) ? float(colorIdx) : 0.f; //someones got to be shiny
        float weight = 1.f;
        float4 color;
        color.x = colorMap.getValue(colorIdx * 3 + 0); 
        color.y = colorMap.getValue(colorIdx * 3 + 1); 
        color.z = colorMap.getValue(colorIdx * 3 + 2); 
        if(shine > 0)
        {
            bool diffuse = (seed % 2 == 0);
            if(shine == 1.f) diffuse = true;
            
            if(diffuse)
            {
                reflection = normal;
                shine = 1.f;
            }
           
            //cout<<"Before "<<reflection<<" After ";
             reflection = eavlSampler::importanceSampleHemi<eavlSampler::HALTON>(sampleNum,reflection,shine, weight, seed);
        }


        return tuple<float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float,
                     float>(normal.x, 
                            normal.y, 
                            normal.z, 
                            intersect.x, 
                            intersect.y, 
                            intersect.z, 
                            color.x, 
                            color.y, 
                            color.z, 
                            reflection.x, 
                            reflection.y, 
                            reflection.z);
    }
};

struct OccRayGenFunctor
{   
    int sampleNum;
    OccRayGenFunctor(int _sampleNum)
    {
        sampleNum = _sampleNum;
    }

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<float,float,float,int>input, int seed){
        int hitIdx = get<3>(input);
        if(hitIdx == -1) tuple<float,float,float>(0.f,0.f,0.f);
        eavlVector3 normal(get<0>(input),get<1>(input),get<2>(input));
        eavlVector3 dir = eavlSampler::hemisphere<eavlSampler::HALTON>(sampleNum, seed, normal);
        return tuple<float,float,float>(dir.x,dir.y,dir.z);
    }
};


struct WorldLightingFunctor
{   
    eavlVector3 skyColor;
    WorldLightingFunctor(eavlVector3 _skyColor)
    {
        skyColor = _skyColor;
    }

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<int,float,float,float,float,float,float>input){
        int hit = get<0>(input);
        if(hit == 1) 
        {
            eavlVector3 normal(get<1>(input), get<2>(input), get<3>(input));
            eavlVector3 dir(get<4>(input), get<5>(input), get<6>(input));
            normal.normalize();
            dir.normalize();
            float cosTheta = normal*dir; //for diffuse
            //cout<<"V "<<normal<<dir<<cosTheta<<"\n";
            cosTheta = min(max(cosTheta,0.f),1.f); //clamp this to [0,1]
            return tuple<float,float,float>(skyColor.x * cosTheta,
                                            skyColor.y * cosTheta,
                                            skyColor.z * cosTheta);
        }
        else return tuple<float,float,float>(0.f,0.f,0.f);
    }
};

struct LightingFunctor
{   
    eavlVector3 lightColor;
    LightingFunctor(eavlVector3 _lightColor)
    {
        lightColor = _lightColor;
    }

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<int,float,float,float,float,float,float>input){
        int hit = get<0>(input);
        if(hit == 1) 
        {
            eavlVector3 normal(get<1>(input), get<2>(input), get<3>(input));
            eavlVector3 dir(get<4>(input), get<5>(input), get<6>(input));
            normal.normalize();
            dir.normalize();
            float cosTheta = normal*dir; //for diffuse
            cosTheta = min(max(cosTheta,0.f),1.f); //clamp this to [0,1]
            return tuple<float,float,float>(lightColor.x * cosTheta,
                                            lightColor.y * cosTheta,
                                            lightColor.z * cosTheta);
        }
        else return tuple<float,float,float>(0.f,0.f,0.f);
    }
};



void eavlPathTracer::init()
{
	

	int numRays = camera->getWidth() * camera->getHeight();
	
	if(numRays != currentFrameSize)
	{
		delete frameBuffer;
		delete rgbaPixels;
		delete depthBuffer;
		delete inShadow;

		frameBuffer = new eavlFloatArray("", 1, numRays * 4); //rgba
		rgbaPixels  = new eavlByteArray("", 1, numRays * 4); //rgba
		depthBuffer = new eavlFloatArray("", 1, numRays);
		inShadow    = new eavlIntArray("", 1, numRays);
        ambientPct = new eavlFloatArray("",1,numRays);
        shadowX = new eavlFloatArray("",1,numRays);
        shadowY = new eavlFloatArray("",1,numRays);
        shadowZ = new eavlFloatArray("",1,numRays);
        reflectX = new eavlFloatArray("",1,numRays);
        reflectY = new eavlFloatArray("",1,numRays);
        reflectZ = new eavlFloatArray("",1,numRays);
        rSurface = new eavlFloatArray("",1,numRays);                  
        gSurface = new eavlFloatArray("",1,numRays);
        bSurface = new eavlFloatArray("",1,numRays);
        rCurrent = new eavlFloatArray("",1,numRays);                  
        gCurrent = new eavlFloatArray("",1,numRays);
        bCurrent = new eavlFloatArray("",1,numRays);
        lred = new eavlFloatArray("",1,numRays);               
        lgreen = new eavlFloatArray("",1,numRays);
        lblue = new eavlFloatArray("",1,numRays);

        
        currentFrameSize = numRays;
	}

	if(geometryDirty)
	{
		numTriangles = scene->getNumTriangles();
		if(numTriangles > 0)
		{
			triGeometry->setVertices(scene->getTrianglePtr(), numTriangles);
			triGeometry->setScalars(scene->getTriangleScalarsPtr(), numTriangles);
			triGeometry->setNormals(scene->getTriangleNormPtr(), numTriangles);
			triGeometry->setMaterialIds(scene->getTriMatIdxsPtr(), numTriangles);
			int numMaterials = scene->getNumMaterials();
			eavlMaterials = scene->getMatsPtr();
		}
		geometryDirty = false;
	}
	
	camera->createRays(rays); //this call resets hitIndexes as well

}
void eavlPathTracer::render()
{   
	camera->printSummary();

    int numSamples = 1;
    int rayDepth = 1;
    for (int p = 0; p < numSamples; ++p)
    {
        float progress = (float) p / (float) numSamples;
        cout<<"Progress "<<progress * 100.f<<"%";
        cout<<"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rSurface), //dummy arg
                                        eavlOpArgs(rSurface,gSurface,bSurface),
                                        FloatMemsetFunctor3to3(1.f, 1.f, 1.f)), //this was 1.f
                                        "init");
        eavlExecutor::Go();
        init(); //Create camera rays
        if(numTriangles < 1) 
        {
            //may be set the framebuffer and depthbuffer to background and infinite
            cerr<<"No trianles to render"<<endl;
            return;
        }
        for (int i = 0; i < rayDepth; ++i)
        {
            

        	
        	//intersector->testIntersections(rays, INFINITE, triGeometry,1,1,camera);

        	intersector->intersectionDepth(rays, INFINITE, triGeometry);
        	
        	eavlFunctorArray<float> mats(eavlMaterials);
        	eavlExecutor::AddOperation(new_eavlRandOp(eavlOpArgs(rays->rayOriginX,
        														rays->rayOriginY,
        														rays->rayOriginZ,
        														rays->rayDirX,
        														rays->rayDirY,
        														rays->rayDirZ,
        														rays->distance,
        														rays->alpha,
        														rays->beta,
        														rays->hitIdx),
                                                     eavlOpArgs(rays->normalX,
                                                     			rays->normalY,
                                                                rays->normalZ,
                                                                rays->intersectionX,
                                                                rays->intersectionY,
                                                                rays->intersectionZ,
                                                                rCurrent,
                                                                gCurrent,
                                                                bCurrent,
                                                                reflectX,
                                                                reflectY,
                                                                reflectZ),
                                                     NormalFunctor(triGeometry->scalars,
                             									   triGeometry->normals,
                                                                   colorMap,
                                                                   numColors)),
                                                     "Normal functor");
            eavlExecutor::Go();

            eavlExecutor::AddOperation(new_eavlRandOp(eavlOpArgs(rays->normalX,
                                                                 rays->normalY,
                                                                 rays->normalZ,
                                                                 rays->hitIdx),
                                                        eavlOpArgs(shadowX,shadowY,shadowZ),OccRayGenFunctor(p)),
                                                        "World Lighing sample");
            eavlExecutor::Go();

            intersector->intersectionOcclusion(rays, shadowX, shadowY, shadowZ, inShadow, indexer, INFINITE, triGeometry);

            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(inShadow, rays->normalX, rays->normalY, rays->normalZ,
                                                                shadowX, shadowY, shadowZ),
                                                     eavlOpArgs(lred, lgreen, lblue),
                                                     WorldLightingFunctor(bgColor)),
                                                     "wlighting");
            eavlExecutor::Go();	
            for (int i = 0; i < currentFrameSize; ++i)
            {
                //
                cout<<lred->GetValue(i)<<" ";
            }
            //
            // mult the current world lighting color with the current surface color, then the total surface color
            // and add its contribution to the framebuffer 
            // There are better ways to do this, but there is a limit to the number of functor arguments.
            //

        	eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rCurrent,gCurrent,bCurrent,lred,lblue,lgreen),
                                                             eavlOpArgs(lred,lblue,lgreen),
                                                             MultFunctor3to3color()),
                                                             "add");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rSurface,gSurface,bSurface,lred,lblue,lgreen),
                                                             eavlOpArgs(lred,lblue,lgreen),
                                                             MultFunctor3to3color()),
                                                             "add");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer),
                                                                eavlIndexable<eavlFloatArray>(lred),
                                                                eavlIndexable<eavlFloatArray>(lblue),
                                                                eavlIndexable<eavlFloatArray>(lgreen)),
                                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer)),
                                                             AccFunctor3to3()),
                                                             "add");
            eavlExecutor::Go();

            intersector->intersectionShadow(rays, inShadow, lightPosition, triGeometry);
            
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(inShadow, rays->normalX, rays->normalY, rays->normalZ,
                                                                shadowX, shadowY, shadowZ),
                                                     eavlOpArgs(lred, lgreen, lblue),
                                                     LightingFunctor(eavlVector3(.6f,.6f,.6f))),
                                                     "wlighting");
            eavlExecutor::Go();
            //do the direct lightiing
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rCurrent,gCurrent,bCurrent,lred,lblue,lgreen),
                                                             eavlOpArgs(lred,lblue,lgreen),
                                                             MultFunctor3to3color()),
                                                             "add");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rSurface,gSurface,bSurface,lred,lblue,lgreen),
                                                             eavlOpArgs(lred,lblue,lgreen),
                                                             MultFunctor3to3color()),
                                                             "add");
            eavlExecutor::Go();


            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer),
                                                                eavlIndexable<eavlFloatArray>(lred),
                                                                eavlIndexable<eavlFloatArray>(lblue),
                                                                eavlIndexable<eavlFloatArray>(lgreen)),
                                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer)),
                                                             AccFunctor3to3()),
                                                             "add");
            eavlExecutor::Go();

            
            //swap ray directions
            eavlFloatArray *tmp;
            tmp = rays->rayOriginX;
            rays->rayOriginX = rays->intersectionX;
            rays->intersectionX = tmp;
            tmp = rays->rayOriginY;
            rays->rayOriginY = rays->intersectionY;
            rays->intersectionY = tmp;
            tmp = rays->rayOriginZ;
            rays->rayOriginZ = rays->intersectionZ;
            rays->intersectionZ = tmp;

            tmp = rays->rayDirX;
            rays->rayDirX = reflectX;
            reflectX = tmp;
            tmp = rays->rayDirY;
            rays->rayDirY = reflectY;
            reflectY = tmp;
            tmp = rays->rayDirZ;
            rays->rayDirZ = reflectZ;
            reflectZ = tmp;

        }//ray depth
    }//path samples
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer)),
                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer)),
                                             AveFunctor(numSamples)),
                                             "add");
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer)),
                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer)),
                                             AveFunctor(numSamples)),
                                             "add");
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer)),
                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer)),
                                             AveFunctor(numSamples)),
                                             "add");
}

eavlFloatArray* eavlPathTracer::getDepthBuffer(float proj22, float proj23, float proj32)
{ 
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->distance), eavlOpArgs(depthBuffer), ScreenDepthFunctor(proj22, proj23, proj32)),"convertDepth");
    eavlExecutor::Go();
    return depthBuffer;
}

eavlByteArray* eavlPathTracer::getFrameBuffer() { return rgbaPixels; }

void eavlPathTracer::setDefaultMaterial(const float &ka,const float &kd, const float &ks)
{
	
      float old_a=scene->getDefaultMaterial().ka.x;
      float old_s=scene->getDefaultMaterial().ka.x;
      float old_d=scene->getDefaultMaterial().ka.x;
      if(old_a == ka && old_d == kd && old_s == ks) return;     //no change, do nothing
      scene->setDefaultMaterial(RTMaterial(eavlVector3(ka,ka,ka),
                                           eavlVector3(kd,kd,kd),
                                           eavlVector3(ks,ks,ks), 10.f,1));
}

void eavlPathTracer::setBackgroundColor(float r, float g, float b)
{
	float mn = min(r, min(g,b));
	float mx = max(r, max(g,b));
	if(mn < 0.f || mx > 1.f)
	{
		cerr<<"Invalid background color value: "<<r<<","<<g<<","<<b<<endl;
		return;
	}
	
	bgColor.x = r;
	bgColor.y = g;
	bgColor.z = b;
}




