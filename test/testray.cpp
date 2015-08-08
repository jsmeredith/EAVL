#include "eavlCUDA.h"
#include "eavlImporterFactory.h"
#include "eavlIsosurfaceFilter.h"
#include "eavlExecutor.h"
#include "eavlRayTriangleIntersector.h"
#include "eavlRayCamera.h"
#include "eavlRayTriangleGeometry.h"
#include "eavlRayTracer.h"
#include "eavlPathTracer.h"
#include "objloader.h"
#include <string.h>
#include <sys/time.h>
#include <ctime>
#include <sstream>
#include "eavlRay.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif


eavlDataSet *ReadWholeFile(const string &);
void printUsage()
{
    cerr<<"You didn't use me correctly or I was not programmed correctly."<<endl;
    cerr<<"     -f    Wavefront OBJ filename for mesh. Example : -f conference.obj"<<endl;
    cerr<<"     -cp   Camera Position ( x y z ).       Example : -cp 1.2 0 -10.2"<<endl;
    cerr<<"     -cu   Camera Up Vector.                Example : -cu 0 1 0"<<endl;
    cerr<<"     -cla  Camera Look At Pos               Example : -cla 0 0 0"<<endl;
    cerr<<"     -fovx Field of view X                  Example : -fovx 45.0"<<endl;
    cerr<<"     -fovy Field of view Y                  Example : -fovy 45.0"<<endl;
    cerr<<"     -res  Resolution height width          Example : -res 1080 1920"<<endl;
    cerr<<"     -ao   Ambient Occculion samples/pixel  Example : -ao 16"<<endl;
    cerr<<"     -lp   Light Params.  pos+intensity      "<<endl;
    cerr<<"           cosnt linear  exponential coeffs Example : -lp 5 3.2 -20 3 1 .3 .7"<<endl;
    cerr<<"     -aa   Anti-Aliasing on                 Example : -aa"<<endl;
    cerr<<"     -cpu  Force CPU execution(GPU Default) "<<endl;
    cerr<<"     -test test tarversal only              Example : -test 50 100 (warm up and test rounds)"<<endl;
    exit(1);
}
void openmpInfo()
{
#ifdef HAVE_OPENMP
    int maxThreads=32;
    int * threads= new int[maxThreads];
    int * threadIds= new int[maxThreads];
    #pragma omp parallel for 
    for(int i=0; i<maxThreads;i++ )
    {
        threads[i]  =omp_get_num_threads();
        threadIds[i]=omp_get_thread_num();
    }

    int maxT=0;
    int maxId=0;
    for(int i=0; i<maxThreads;i++ )
    {
        maxT=std::max(threads[i],maxT);  
        maxId=std::max(threadIds[i],maxId);
    }
    cout<<"OpenMP Max threads : "<<maxT<<" Highest ThreadID : "<< maxId<<endl;
    delete[] threads;
    delete[] threadIds;
#endif
}
int main(int argc, char *argv[])
{
    try

    {  
        for(int k=0;k<1;k++){
        char *filename;
        char *outFilename = NULL;
        bool forceCPU=false;
        bool isTest=false;
        int warmups=1;
        int tests=1;
        eavlPathTracer *tracer = new eavlPathTracer();
        //eavlRayTracer *tracer = new eavlRayTracer();
        if(argc<2)
        {
            cerr<<"Must specify a file to load."<<endl;
            printUsage();
        }

        for(int i=1; i<argc;i++)
        {
            if(strcmp (argv[i],"-f")==0)
            {   
                if(argc<=i) 
                {
                    cerr<<"No file name specified."<<endl;
                    printUsage();
                }
                filename=argv[++i];

            }
            else if(strcmp (argv[i],"-cp")==0)
            {
                if(argc<=i+3) 
                {
                    cerr<<"Not enough input for camera coordinates."<<endl;
                    printUsage();
                }
                float x=0; x=atof(argv[++i]);
                float y=0; y=atof(argv[++i]);
                float z=0; z=atof(argv[++i]);
                //cerr<<x<<" "<<y<<" "<<z<<endl;
                //tracer->setCameraPos(x,y,z);
                tracer->camera->setCameraPosition(x,y,z);
            }
            else if(strcmp (argv[i],"-cu")==0)
            {
                if(argc<=i+3) 
                {
                    cerr<<"Not enough input for up vector."<<endl;
                    printUsage();
                }
                float x=0; x=atof(argv[++i]);
                float y=0; y=atof(argv[++i]);
                float z=0; z=atof(argv[++i]);
                //cerr<<x<<" "<<y<<" "<<z<<endl;
                //tracer->setUp(x,y,z);
                tracer->camera->setCameraUp(x,y,z);
            }
            else if(strcmp (argv[i],"-cla")==0)
            {
                if(argc<=i+3) 
                {
                    cerr<<"Not enough input for look at position."<<endl;
                    printUsage();
                }
                float x=0; x=atof(argv[++i]);
                float y=0; y=atof(argv[++i]);
                float z=0; z=atof(argv[++i]);
                //cerr<<x<<" "<<y<<" "<<z<<endl;
                tracer->camera->lookAtPosition(x,y,z);
            }
            else if(strcmp (argv[i],"-fovx")==0)
            {
                if(argc<=i) 
                {
                    cerr<<"Not enough input for FOV x."<<endl;
                    printUsage();
                }
                float x=0; x=atof(argv[++i]);
                if(x==0)
                {
                    cerr<<"Invalid FOV value "<<argv[i]<<endl;
                    printUsage();
                }
                tracer->camera->setFOVX(x);
            }
            else if(strcmp (argv[i],"-fovy")==0)
            {
                if(argc<=i) 
                {
                    cerr<<"Not enough input for FOV y."<<endl;
                    printUsage();
                }
                float y=0; y=atof(argv[++i]);
                if(y==0)
                {
                    cerr<<"Invalid FOV value "<<argv[i]<<endl;
                    printUsage();
                }
                tracer->camera->setFOVY(y);
            }
            else if(strcmp (argv[i],"-res")==0)
            {
                if(argc<=i+2) 
                {
                    cerr<<"Not enough input for resolution."<<endl;
                    printUsage();
                }
                float y=0; y=atoi(argv[++i]);
                float x=0; x=atoi(argv[++i]);
                if(x<1 || y<1)
                {
                    cerr<<"Invalid resolution values. Must be non-zero integers."<<endl;
                    printUsage();
                }
            
                tracer->camera->setWidth(x);
                tracer->camera->setHeight(y);
                
            }
            else if(strcmp (argv[i],"-lp")==0)
            {
                if(argc<=i+3) 
                {
                    cerr<<"Not enough input for light parameters."<<endl;
                    printUsage();
                }
                float x=0; x=atof(argv[++i]);
                float y=0; y=atof(argv[++i]);
                float z=0; z=atof(argv[++i]);
                tracer->lightPosition.x = x;
                tracer->lightPosition.y = y;
                tracer->lightPosition.z = z;
                
            }
            else if(strcmp (argv[i],"-ao")==0)
            {   cout<<"AO"<<endl;
                if(argc<=i+2) 
                {
                    cerr<<"Not enough input for ambient occlusion."<<endl;
                    printUsage();
                }
                float y=0; y=atof(argv[++i]);
                float z=0; z=atof(argv[++i]);
                if(y<1)
                {
                    cerr<<"Invalid AO value "<<argv[i]<<endl;
                    printUsage();
                 }
                //tracer->setOcclusionDistance(z);
                //tracer->setOcclusionSamples(y);
                //tracer->setOcclusionOn(true);
                
            }
            else if(strcmp (argv[i],"-test")==0)
            {
                if(argc<=i+2) 
                {
                    cerr<<"Needs more values."<<endl;
                    printUsage();
                }
                float y=0; y=atoi(argv[++i]);
                float x=0; x=atoi(argv[++i]);
                if(x<0 || y<0)
                {
                    cerr<<"Invalid test values. Must be non-zero integers."<<endl;
                    printUsage();
                }
                isTest=true;
                warmups=y;
                tests=x;
                
            }
            else if(strcmp (argv[i],"-cpu")==0)
            {
                forceCPU=true;
            }
            else if(strcmp (argv[i],"-o")==0)
            {
                outFilename=argv[++i];
                //if(outFilename!=NULL) tracer->setOutputFilename(outFilename);
            }
            else if(strcmp (argv[i],"-v")==0)
            {
                //tracer->setVerbose(true);
            }
            else
            {
                cerr<<"Unknown option : "<<argv[i]<<endl;
                printUsage();
            }
        }


        if(forceCPU)
        {
            eavlExecutor::SetExecutionMode(eavlExecutor::ForceCPU);
            //tracer->cpu=true;
        }
        else 
        {   //if this fails, it will fall back to the cpu
            eavlExecutor::SetExecutionMode(eavlExecutor::ForceGPU);
            eavlInitializeGPU();
            //tracer->cpu=false;
        }
        openmpInfo();
        //load model data from a file
        ObjReader *objreader= new ObjReader(filename);
        float *v;   //verts 
        float *n = NULL;   //norms
        int *mIdx;
        int numTris=objreader->totalTriangles;
        float *mats;
        int matCount;
        objreader->getRawData(v,n,mats,mIdx,matCount);
        
        bool hasNormals = false;
        if(n != NULL) hasNormals = true;
        for (int i = 0; i < numTris; ++i)
        {
           
           eavlVector3 a;
           eavlVector3 b;
           eavlVector3 c;

           a.x = v[i*9+0];
           a.y = v[i*9+1];
           a.z = v[i*9+2];
           b.x = v[i*9+3];
           b.y = v[i*9+4];
           b.z = v[i*9+5];
           c.x = v[i*9+6];
           c.y = v[i*9+7];
           c.z = v[i*9+8];
           if(hasNormals)
           {
                eavlVector3 an;
                eavlVector3 bn;
                eavlVector3 cn;
                an.x = n[i*9+0];
                an.y = n[i*9+1];
                an.z = n[i*9+2];
                bn.x = n[i*9+3];
                bn.y = n[i*9+4];
                bn.z = n[i*9+5];
                cn.x = n[i*9+6];
                cn.y = n[i*9+7];
                cn.z = n[i*9+8];
                tracer->scene->addTriangle(a,b,c,an,bn,cn,0,0,0);
           }
           else tracer->scene->addTriangle(a,b,c);
        }

        //eavlRayTriangleGeometry geometry;
        //geometry.setBVHCacheName(filename);
        //geometry.setVertices(verts, numTris);
        cout<<"Number of triangles "<<numTris<<endl;
        //eavlRay rays(camera->getWidth() * camera->getHeight());
        //eavlRayTriangleIntersector intersector;
        //camera->createRays(&rays);


        //tracer->scene->loadObjFile(filename);
        //tracer->setBVHCacheName(filename);
        
        //tracer->setCompactOp(false);
        //*************************************Testing****************************

        //tracer->setBVHCache(true);
        //tracer->setVerbose(true);    
        //tracer->setDepth(1);

        if(!isTest)
        {
            cout<<"Rendering to Framebuffer\n";
            tracer->render();
            if(outFilename != NULL) writeFrameBufferBMP(tracer->camera->getWidth(),
                                                        tracer->camera->getHeight(),
                                                        tracer->getFrameBuffer(),
                                                        outFilename);
        }
        else 
        {
            cout<<"Starting test.\n";
            //tracer->visualRayTrace(1,"something");
            //tracer->traversalTestISPC(warmups,tests);
            //tracer->traversalTest(warmups,tests);

            //tracer->fpsTest(warmups,tests);

            //intersector.testIntersections(&rays, 10000000, &geometry, warmups,tests, camera);
        }

        
        delete [] v;
        if(n) delete [] n;
        delete [] mIdx;
        delete [] mats;
        delete tracer;
        delete objreader;
    }
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        return 1;
    }

    return 0;
}

