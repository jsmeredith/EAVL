#include <eavlPathTracer.h>
#include <eavl1toNScatterOp.h>
#include <eavlNto1GatherOp.h>
#include <eavlSampler.h>
#include <eavlRandOp.h>
#include <eavlMapOp.h>

eavlPathTracer::eavlPathTracer()
{


    numSamples = 50;
    rayDepth = 4;
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
	seeds = new eavlIntArray("", 1, currentFrameSize);
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
	bgColor.x = 0.7f;
	bgColor.y = 0.7f;
	bgColor.z = 0.7f;
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
    delete seeds;
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
    delete rays;

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
    float albedo;

    NormalFunctor(eavlTextureObject<float>  *_scalars,
    			  eavlTextureObject<float>  *_norms,
                  eavlTextureObject<float>  *_colorMap,
                  int _colorMapSize,
                  int _sampleNum)
        :scalars(*_scalars),
         norms(*_norms),
         colorMap(*_colorMap),
         colorMapSize(_colorMapSize)

    {
        sampleNum = _sampleNum;
        albedo = .7f;
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
										   						    int,     // Hit index
                                                                    int
										   						    > input)
    {
        int hitIndex=get<9>(input);
        if(hitIndex < 0) return tuple<float,
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
       	eavlVector3 rayOrigin(get<0>(input), get<1>(input), get<2>(input));
        eavlVector3 rayDir(get<3>(input), get<4>(input), get<5>(input));
        float hitDistance = get<6>(input);
        rayDir.normalize();
        eavlVector3 intersect = rayOrigin + hitDistance * rayDir; 

        float alpha = get<7>(input);
        float beta  = get<8>(input);
        float gamma = 1.f - alpha - beta;
        
        int seed = get<10>(input) + sampleNum;

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
        //shine = 80;
        if(shine > 1)
        {
            bool diffuse = (seed % 2 == 0);
            if(shine == 1.f) diffuse = true;
            
            if(diffuse)
            {
                reflection = normal;
                shine = 1.f;        
            }
        } 

        color.x *= albedo;
        color.y *= albedo;
        color.z *= albedo;
        reflection = eavlSampler::importanceSampleHemi<eavlSampler::HALTON>(seed ,reflection,shine, weight, seed + int(reflection.y*100));
        reflection.normalize();
        intersect = intersect+(-rayDir * .0001);


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

    EAVL_FUNCTOR tuple<float,float,float> operator()(tuple<float,float,float,int, int>input, int random){
        int hitIdx = get<3>(input);
        if(hitIdx < 0) tuple<float,float,float>(0.f,0.f,0.f);
        int seed = get<4>(input) + sampleNum;
        eavlVector3 normal(get<0>(input),get<1>(input),get<2>(input));
        float w = 0;
        eavlVector3 dir = eavlSampler::importanceSampleHemi<eavlSampler::HALTON>(seed, normal, 1.f, w, random );
        return tuple<float,float,float>(dir.x,dir.y,dir.z);
    }
};

struct MissFunctor
{   
    eavlVector3 bgColor;
    MissFunctor(eavlVector3 _bgColor)
    {
        bgColor = _bgColor;
    }

    EAVL_FUNCTOR tuple<float,float,float,int> operator()(tuple<int>input){
        int hitIdx = get<0>(input);
        if(hitIdx != -1) return tuple<float,float,float,int>(0.f,0.f,0.f, hitIdx);
        return tuple<float,float,float,int>(bgColor.x, bgColor.y, bgColor.z, -2);
    }
};

struct PreethanFunctor
{   
    eavlVector3 sunDir;
    eavlVector3 betaR, betaM;
    float mieG;
    float sunIntensity;
    float sunAngularCos;
    EAVL_HOSTDEVICE float tRayleigh(float waveLength)
    {
        float ad = 0.035f;   //air depolariztion ratio
        float N = 2.545E25; // air molecular density
        float n = 1.0003;   //refractive index of air

        return (8 * pow(PI, 3) * pow(pow(n, 2) - 1, 2) * (6 + 3 * ad)) / (3 * N * pow(waveLength, 4) * (6 - 7 * ad));
    }
    //
    //  k = scattering
    //  t = turbidity (0-20)
    //
    EAVL_HOSTDEVICE float tMie(float waveLength, float k, float t)
    {

        float c = (0.2f * t ) * 10E-18;
        return 0.434 * c * PI * pow((2.f * PI) / waveLength, 2.f) * k;
    }

    PreethanFunctor(eavlVector3 _sunDir)
    {
        sunDir = _sunDir;
        sunIntensity =  1000.f* max(0.f, 1.f - exp(-((PI/2.f - acos(sunDir*eavlVector3(0,1,0)))/.5f)));
        //Earth scattering coeffs, all these can vary
        float reileigh = 1.f; 
        float mie = 0.0553f;
        mieG = .575f;     
        betaR.x = tRayleigh(680E-9) * reileigh; //3 different wavelengths in meters
        betaR.y = tRayleigh(550E-9) * reileigh;
        betaR.z = tRayleigh(450E-9) * reileigh;
        betaM.x = tMie(680E-9, 0.686f, 10.f) * mie;
        betaM.y = tMie(550E-9, 0.678f, 10.f) * mie;
        betaM.z = tMie(450E-9, 0.666f, 10.f) * mie;
        sunAngularCos =0.99995667694644844f;// cos(0.5f);
    }

    EAVL_FUNCTOR tuple<float,float,float,int> operator()(tuple<int, float,float,float>input){
        int hitIdx = get<0>(input);
        if(hitIdx != -1) return tuple<float,float,float,int>(0.f,0.f,0.f, hitIdx); 
        eavlVector3 rayDir(get<1>(input),get<2>(input),get<3>(input));
        float skyAngle = acos(max(0.f,rayDir * eavlVector3(0.f,1.f,0.f))); //this could be the camera up
        float sR =  8.4E3 / (cos(skyAngle) + 0.15f * pow(93.885f - ((skyAngle * 180.0f) / PI), -1.253f));
        float sM = 1.25E3 / (cos(skyAngle) + 0.15f * pow(93.885f - ((skyAngle * 180.0f) / PI), -1.253f));
        //cout<<"skyeAngle "<<skyAngle<<" "<<sR<<" "<<sM<<endl;
        eavlVector3 fex;
        fex.x = exp(-(betaR.x * sR + betaM.x * sM));
        fex.y = exp(-(betaR.y * sR + betaM.y * sM));
        fex.z = exp(-(betaR.z * sR + betaM.z * sM));
        
        float cosTheta = rayDir * sunDir;
        float rPhase = (3.0f / 4.0f) * (1.0f + pow(cosTheta, 2.f));
        eavlVector3 angleBetaR = betaR * rPhase; 
        float mPhase = (1.0f / (4.0f*PI)) * ((1.0f - pow(mieG, 2.f)) / pow(1.0f - 2.0f*mieG*cosTheta + pow(mieG, 2.f), 1.5f));
        eavlVector3 angleBetaM = betaM * mPhase; 
        eavlVector3 lIn;
        float t = abs(sunDir.y);
        fex.x = lerp(fex.x, 1.f - fex.x, t);
        fex.y = lerp(fex.y, 1.f - fex.y, t);
        fex.z = lerp(fex.z, 1.f - fex.z, t);

        lIn.x = sunIntensity * ((angleBetaR.x + angleBetaM.x) / (betaR.x + betaM.x)) * fex.x;
        lIn.y = sunIntensity * ((angleBetaR.y + angleBetaM.y) / (betaR.y + betaM.y)) * fex.y;
        lIn.z = sunIntensity * ((angleBetaR.z + angleBetaM.z) / (betaR.z + betaM.z)) * fex.z;
        
        eavlVector3 color = fex;
        if (cosTheta > sunAngularCos) color += sunIntensity * fex;
        color += lIn;
        color*= .01f;
        return tuple<float,float,float,int>(color.x,color.y,color.z, -2);
    }
};

struct SeedFunctor
{   
    SeedFunctor()
    {
    }

    EAVL_FUNCTOR tuple<int> operator()(tuple<int>input, int seed){
        
        int num = abs(seed) % 20000;
        return tuple<int>(num);
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



void eavlPathTracer::init(int sampleNum)
{
	

	int numRays = camera->getWidth() * camera->getHeight();
	
	if(numRays != currentFrameSize)
	{  
        cout<<"Resizing"<<endl;
		delete frameBuffer;
		delete rgbaPixels;
		delete depthBuffer;
		delete inShadow;
        delete seeds;
        delete shadowX;
        delete shadowY;
        delete shadowZ;
        delete reflectX;
        delete reflectY;
        delete reflectZ;
        delete rSurface;
        delete gSurface;
        delete bSurface;
        delete rCurrent;
        delete gCurrent;
        delete bCurrent;
        delete lred;
        delete lgreen;
        delete lblue;

		frameBuffer = new eavlFloatArray("", 1, numRays * 4); //rgba
		rgbaPixels  = new eavlByteArray("", 1, numRays * 4); //rgba
		depthBuffer = new eavlFloatArray("", 1, numRays);
		inShadow    = new eavlIntArray("", 1, numRays);
        seeds = new eavlIntArray("",1,numRays);
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
	//camera->createDOFRays(rays,seeds, sampleNum, 0.5f); //this call resets hitIndexes as well
	camera->createJitterRays(rays,seeds, sampleNum); //this call resets hitIndexes as well
    //camera->createRays(rays); //this call resets hitIndexes as well

}

void eavlPathTracer::addColor()
{
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rCurrent,gCurrent,bCurrent,lred,lgreen,lblue),
                                                             eavlOpArgs(lred,lgreen,lblue),
                                                             MultFunctor3to3color()),
                                                             "add");
    eavlExecutor::Go();
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rSurface,gSurface,bSurface,lred,lgreen,lblue),
                                                     eavlOpArgs(lred,lgreen,lblue),
                                                     MultFunctor3to3color()),
                                                     "add");
    eavlExecutor::Go();

    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer),
                                                        eavlIndexable<eavlFloatArray>(lred),
                                                        eavlIndexable<eavlFloatArray>(lgreen),
                                                        eavlIndexable<eavlFloatArray>(lblue)),
                                                     eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer)),
                                                     AccFunctor3to3()),
                                                     "add");
    eavlExecutor::Go();
}
void eavlPathTracer::render()
{   

	camera->printSummary();

    init(0); //Create camera rays
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(frameBuffer), //dummy arg
                                    eavlOpArgs(frameBuffer),
                                    FloatMemsetFunctor(0.f)), //this was 1.f
                                    "init");
    eavlExecutor::Go();
    //create a semi unique seed for the sampling sequence
    eavlExecutor::AddOperation(new_eavlRandOp(eavlOpArgs(seeds), //dummy arg
                                              eavlOpArgs(seeds),
                                              SeedFunctor()), 
                                              "Seeds");
    eavlExecutor::Go();
    

    for (int p = 0; p < numSamples; ++p)
    {
        int currentSample = p;// * rayDepth +i;
        float progress = (float) p / (float) numSamples;
        //
        cout<<"Progress "<<progress * 100.f<<"%";
        if(p != 0) cout<<"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
        eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rCurrent), //dummy arg
                                        eavlOpArgs(rCurrent,gCurrent,bCurrent), 
                                        FloatMemsetFunctor3to3(1.f, 1.f, 1.f)), //this was 1.f
                                        "init");
        eavlExecutor::Go();
        init(currentSample); //Create camera rays
        if(numTriangles < 1) 
        {
            //may be set the framebuffer and depthbuffer to background and infinite
            cerr<<"No trianles to render"<<endl;
            return;
        }
        //TODO: hitindex filtering
        for (int i = 0; i < rayDepth; ++i)
        {
           
        	intersector->intersectionDepth(rays, INFINITE, triGeometry);
            //Get background color and add it
            /*
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx), 
                                        eavlOpArgs(lred, lgreen, lblue, rays->hitIdx),
                                        MissFunctor(bgColor)), 
                                        "init");
            eavlExecutor::Go();
            */
            eavlVector3 sunDir(0,-0.1f,-1.f); 
            sunDir.normalize();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->hitIdx, 
                                                                rays->rayDirX,
                                                                rays->rayDirY,
                                                                rays->rayDirZ), 
                                        eavlOpArgs(lred, lgreen, lblue, rays->hitIdx),
                                        PreethanFunctor(sunDir)), 
                                        "init");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rCurrent,gCurrent,bCurrent,lred,lgreen,lblue),
                                                             eavlOpArgs(lred,lgreen,lblue),
                                                             MultFunctor3to3color()),
                                                             "add");
            eavlExecutor::Go();


            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer),
                                                                eavlIndexable<eavlFloatArray>(lred),
                                                                eavlIndexable<eavlFloatArray>(lgreen),
                                                                eavlIndexable<eavlFloatArray>(lblue)),
                                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer)),
                                                             AccFunctor3to3()),
                                                             "add");
            eavlExecutor::Go();
            
        	eavlFunctorArray<float> mats(eavlMaterials);
        	eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rays->rayOriginX,
        														rays->rayOriginY,
        														rays->rayOriginZ,
        														rays->rayDirX,
        														rays->rayDirY,
        														rays->rayDirZ,
        														rays->distance,
        														rays->alpha,
        														rays->beta,
        														rays->hitIdx,
                                                                seeds),
                                                     eavlOpArgs(rays->normalX,
                                                     			rays->normalY,
                                                                rays->normalZ,
                                                                rays->intersectionX,
                                                                rays->intersectionY,
                                                                rays->intersectionZ,
                                                                rSurface,
                                                                gSurface,
                                                                bSurface,
                                                                reflectX,
                                                                reflectY,
                                                                reflectZ),
                                                     NormalFunctor(triGeometry->scalars,
                             									   triGeometry->normals,
                                                                   colorMap,
                                                                   numColors,
                                                                   currentSample)),
                                                     "Normal functor");
            eavlExecutor::Go();
            
            eavlExecutor::AddOperation(new_eavlRandOp(eavlOpArgs(rays->normalX,
                                                                 rays->normalY,
                                                                 rays->normalZ,
                                                                 rays->hitIdx,
                                                                 seeds),
                                                        eavlOpArgs(shadowX,shadowY,shadowZ),OccRayGenFunctor(currentSample)),
                                                        "World Lighing sample");
            eavlExecutor::Go();
/*
            intersector->intersectionOcclusion(rays, shadowX, shadowY, shadowZ, inShadow, indexer, INFINITE, triGeometry);
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(inShadow, rays->normalX, rays->normalY, rays->normalZ,
                                                                shadowX, shadowY, shadowZ),
                                                     eavlOpArgs(lred, lgreen, lblue),
                                                     WorldLightingFunctor(eavlVector3(.55, .8156,.9921))),
                                                     "wlighting");
            eavlExecutor::Go();	
           
            //
            // mult the current world lighting color with the current surface color, then the total surface color
            // and add its contribution to the framebuffer 
            // There are better ways to do this, but there is a limit to the number of functor arguments.
            //

        	eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rCurrent,gCurrent,bCurrent,lred,lgreen,lblue),
                                                             eavlOpArgs(lred,lgreen,lblue),
                                                             MultFunctor3to3color()),
                                                             "add");
            eavlExecutor::Go();
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rSurface,gSurface,bSurface,lred,lgreen,lblue),
                                                             eavlOpArgs(lred,lgreen,lblue),
                                                             MultFunctor3to3color()),
                                                             "add");
            eavlExecutor::Go();

            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer),
                                                                eavlIndexable<eavlFloatArray>(lred),
                                                                eavlIndexable<eavlFloatArray>(lgreen),
                                                                eavlIndexable<eavlFloatArray>(lblue)),
                                                             eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer)),
                                                             AccFunctor3to3()),
                                                             "add");
            eavlExecutor::Go();
*/
            intersector->intersectionShadow(rays, inShadow, lightPosition, triGeometry);
            
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(inShadow, rays->normalX, rays->normalY, rays->normalZ,
                                                                shadowX, shadowY, shadowZ),
                                                     eavlOpArgs(lred, lgreen, lblue),
                                                     LightingFunctor(eavlVector3(.5f,.5f,.5f))),
                                                     "wlighting");
            eavlExecutor::Go();
            //do the direct lightiing
            addColor();
            
            eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(rSurface,gSurface,bSurface,rCurrent,gCurrent,bCurrent),
                                                             eavlOpArgs(rCurrent,gCurrent,bCurrent),
                                                             MultFunctor3to3color()),
                                                             "add");
            eavlExecutor::Go();
            // if(i>0)for (int z = 0; z < currentFrameSize; ++z)
            // {
            //      //
            //      if(rays->hitIdx->GetValue(z) > -1) cout<<bCurrent->GetValue(z)<<" ("<<z<<") ";
            //  }

            
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

eavlByteArray* eavlPathTracer::getFrameBuffer() 
{ 
    eavlExecutor::AddOperation(new_eavlMapOp(eavlOpArgs(eavlIndexable<eavlFloatArray>(frameBuffer,*redIndexer),
                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*greenIndexer),
                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*blueIndexer),
                                                        eavlIndexable<eavlFloatArray>(frameBuffer,*alphaIndexer)),
                                                 eavlOpArgs(eavlIndexable<eavlByteArray>(rgbaPixels,*redIndexer),
                                                            eavlIndexable<eavlByteArray>(rgbaPixels,*greenIndexer),
                                                            eavlIndexable<eavlByteArray>(rgbaPixels,*blueIndexer),
                                                            eavlIndexable<eavlByteArray>(rgbaPixels,*alphaIndexer)),
                                                 CopyFrameBuffer()),
                                                 "memcopy");
    eavlExecutor::Go();

    return rgbaPixels; 
}

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

void eavlPathTracer::setNumberOfSamples(int nSamples)
{
    numSamples = nSamples;
}

void eavlPathTracer::setRayDepth(int depth)
{
    rayDepth = depth;
}



