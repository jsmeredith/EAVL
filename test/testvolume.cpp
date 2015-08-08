#include "eavlCUDA.h"
#include "eavlImporterFactory.h"
#include "eavlIsosurfaceFilter.h"
#include "eavlExecutor.h"
#include "eavlSimpleVRMutator.h"
#include "eavlView.h"
#include <string.h>
#include <sys/time.h>
#include <ctime>
#include <sstream>
#include <eavlRTUtil.h>
#include "eavlImporterFactory.h"
#include "eavlCellSetExplicit.h"
#include "eavlCell.h"
#ifdef HAVE_OPENMP
#include <omp.h>
#endif


eavlDataSet *ReadWholeFile(const string &);
void printUsage()
{
    cerr<<"You didn't use me correctly or I was not programmed correctly."<<endl;
    cerr<<"     -f    mesh filename                    Example : -f enzo.vtk"<<endl;
    cerr<<"     -tf    transfer function filename       Example : -tf enzoTransferFunction"<<endl;
    cerr<<"     -pass number of numPasses              Example : -pass 8"<<endl;
    cerr<<"     -mesh Index of mesh in file.           Example : -mesh 1"<<endl;
    cerr<<"     -cell Index of cellSet in file.        Example : -cell 0"<<endl;
    cerr<<"     -cell Index of field in file.          Example : -fld 0"<<endl;
    cerr<<"     -o    Image output filename .          Example : -o outfile"<<endl;
    cerr<<"     -of   Opacity factor                   Example : -of .5"<<endl;
    //cerr<<"     -cla  Camera Look At Pos               Example : -cla 0 0 0"<<endl;
    //cerr<<"     -fovx Field of view X                  Example : -fovx 45.0"<<endl;
    //cerr<<"     -fovy Field of view Y                  Example : -fovy 45.0"<<endl;
    cerr<<"     -res  Resolution height width          Example : -res width height"<<endl;
    cerr<<"     -spp  Samples per pixel                Example : -ssp 1000  "<<endl;
    cerr<<"     -cpu  Force CPU execution(GPU Default) "<<endl;
    cerr<<"     -test render timing                    Example : -test 50 100 (warm up and test rounds)"<<endl;
    cerr<<"     -closeup                               Example : -closeup"<<endl;
    exit(1);
}
void openmpInfo()
{
#ifdef HAVE_OPENMP
    int maxThreads=64;
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
    eavlView view;

    try

    {  
        char *filename;
        char *tfFilename;
        string outFilename;
        bool forceCPU = false;
        bool isTest = false;
        int warmups = 1;
        int tests = 1;
        int height = 500;
        int width = 500;
        int samples = 1000;
        int numPasses = 8;
        int meshIdx = 0;
        bool meshSpecified = false;
        int  cellSetIdx = 0;
        bool cellSetSpecified = false;
        int  fieldIdx = 0;
        bool fieldSpecified = false;
        bool verbose = false; 
        bool closeup = false;
        float opactiyFactor;

       
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
                filename = argv[++i];

            }
            else if(strcmp (argv[i],"-tf")==0)
            {
                if(argc<=i) 
                {
                    cerr<<"Not enough input for transfer function."<<endl;
                    printUsage();
                }
                tfFilename=argv[++i];
            }
            else if(strcmp (argv[i],"-res")==0)
            {
                if(argc<=i+2) 
                {
                    cerr<<"Not enough input for resolution."<<endl;
                    printUsage();
                }
                
                float x = 0; x = atoi(argv[++i]);
                float y = 0; y = atoi(argv[++i]);
                if(x<1 || y<1)
                {
                    cerr<<"Invalid resolution values. Must be non-zero integers."<<endl;
                    printUsage();
                }
                height = y;
                width  = x;
                
            }
            else if(strcmp (argv[i],"-ssp")==0)
            {
                if(argc <= i + 1) 
                {
                    cerr<<"Not enough input for samples per pixel."<<endl;
                    printUsage();
                }
                float s = 0; s = atoi(argv[++i]);
                
                if(s<1 )
                {
                    cerr<<"Invalid sample value. Must be non-zero integers."<<endl;
                    printUsage();
                }
                samples = s;
                
            }

            else if(strcmp (argv[i],"-pass")==0)
            {
                if(argc <= i + 1) 
                {
                    cerr<<"Not enough input for number of passes."<<endl;
                    printUsage();
                }
                float s = 0; s = atoi(argv[++i]);
                
                if(s<1 )
                {
                    cerr<<"Invalid sample value. Must be non-zero integers."<<endl;
                    printUsage();
                }
                numPasses = s;
                
            }

            else if(strcmp (argv[i],"-of")==0)
            {
                if(argc <= i + 1) 
                {
                    cerr<<"Not enough input for opacity factor"<<endl;
                    printUsage();
                }
                float s = 0; s = atof(argv[++i]);
                
                if(s<0 )
                {
                    cerr<<"Invalid opactiyFactor. Must be non-zero integers."<<endl;
                    printUsage();
                }
                opactiyFactor = s;
                
            }
            else if(strcmp (argv[i],"-mesh")==0)
            {
                if(argc <= i + 1) 
                {
                    cerr<<"Not enough input for mesh idx."<<endl;
                    printUsage();
                }
                float s = 0; s = atoi(argv[++i]);
                
                if(s<0 )
                {
                    cerr<<"Invalid mesh index. Must equal o of greater than 0."<<endl;
                    printUsage();
                }
                meshIdx = s;
                meshSpecified = true;
                
            }
            else if(strcmp (argv[i],"-fld")==0)
            {
                if(argc <= i + 1) 
                {
                    cerr<<"Not enough input for field idx."<<endl;
                    printUsage();
                }
                float s = 0; s = atoi(argv[++i]);
                
                if(s<0 )
                {
                    cerr<<"Invalid field index. Must equal o of greater than 0."<<endl;
                    printUsage();
                }
                fieldIdx = s;
                fieldSpecified = true;
                
            }
            else if(strcmp (argv[i],"-cell")==0)
            {
                if(argc <= i + 1) 
                {
                    cerr<<"Not enough input for number of passes."<<endl;
                    printUsage();
                }
                float s = 0; s = atoi(argv[++i]);
                
                if(s<0 )
                {
                    cerr<<"Invalid mesh index. Must equal o of greater than 0."<<endl;
                    printUsage();
                }
                cellSetIdx = s;
                cellSetSpecified = true;
                
            }
            else if(strcmp (argv[i],"-test")==0)
            {
                if(argc<=i+2) 
                {
                    cerr<<"Needs more values."<<endl;
                    printUsage();
                }
                float y = 0; y = atoi(argv[++i]);
                float x = 0; x = atoi(argv[++i]);
                if(x<1 || y<1)
                {
                    cerr<<"Invalid test values. Must be non-zero integers."<<endl;
                    printUsage();
                }
                isTest = true;
                warmups = y;
                tests = x;
                
            }
            else if(strcmp (argv[i],"-cpu")==0)
            {
                forceCPU = true;
            }
            else if(strcmp (argv[i],"-o")==0)
            {
                outFilename = argv[++i];
            }
            else if(strcmp (argv[i],"-v")==0)
            {
                verbose = true;
            }
            else if(strcmp (argv[i],"-closeup")==0)
            {
                closeup = true;
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
        }
        else 
        {   //if this fails, it will fall back to the cpu
            eavlExecutor::SetExecutionMode(eavlExecutor::ForceGPU);
            eavlInitializeGPU();
        }

        eavlSimpleVRMutator *vrenderer= new eavlSimpleVRMutator();
        vrenderer->setVerbose(verbose);
        vrenderer->scene->normalizedScalars(true);
        vrenderer->setNumPasses(numPasses);
        vrenderer->setNumSamples(samples);
        if(closeup)
        {
            string temp = outFilename;
            temp += "_cu_";
            outFilename = temp.c_str();
        } 

        // string temp = outFilename;
        // temp += tfFilename;
        // outFilename = temp.c_str();

        vrenderer->setDataName(outFilename);
        //vrenderer->setTransferFunctionFile(tfFilename);
        vrenderer->setOpacityFactor(opactiyFactor);
        openmpInfo();
        
        cout<<"------------------------Current config-------------------"<<endl;
        cout<<"Mesh Filename    : "<<filename<<endl;
        cout<<"Mesh Index       : "<<meshIdx<<endl;
        cout<<"TF Filename      : "<<tfFilename<<endl;
        cout<<"Output filename  : "<<outFilename<<endl;
        cout<<"Force CPU        : "<<forceCPU<<endl;
        cout<<"isTest           : "<<isTest<<" Warmups "<<warmups<<" tests "<<tests<<endl;
        cout<<"Screen(W x H)    : "<<width<<" x "<<height<<endl;
        cout<<"Samples Per Pix  : "<<samples<<endl;
        cout<<"Number of Passes : "<<numPasses<<endl; 
        

        eavlImporter *importer = eavlImporterFactory::GetImporterForFile(filename);
        

        //--------------------------Getting mesh info-----------------------------------------------
        vector<string> meshlist = importer->GetMeshList();
        int msize = meshlist.size();
        int domainindex = 0;

        if( (!meshSpecified &&  msize > 1) || (  meshSpecified || meshIdx > msize) )
        {
            cout<<endl<<"More than one mesh in the file. Please specify index of the mesh."<<endl;
            for(int i = 0; i < msize; i++)
            {
                cout<<"Index "<<i<<" "<<meshlist.at(i)<<endl;
            }

            exit(0);
        }
        cout<<"Using mesh       : "<<meshlist.at(meshIdx)<<endl;
        eavlDataSet *data = importer->GetMesh(meshlist.at(meshIdx), domainindex);
        
        
        
        int numCellSets = data->GetNumCellSets();
        if( (!cellSetSpecified && numCellSets > 1) || (cellSetSpecified && cellSetIdx > numCellSets))
        {
            cout<<endl<<"More than one cellset in the file. Please specify index of the cellset."<<endl;
            for(int i = 0; i < numCellSets; i++)
            {
                eavlCellSetExplicit* cs = (eavlCellSetExplicit*) data->GetCellSet(i);
                cout<<"Index "<<i<<" "<<cs->GetName()<<endl;
            }
            exit(0);
        }

        eavlCellSetExplicit* cellSet = (eavlCellSetExplicit*) data->GetCellSet(cellSetIdx);
        cout<<"Using cell set   : "<<cellSet->GetName()<<endl;

        vector<string> fieldList = importer->GetFieldList(meshlist.at(meshIdx));
        int numfields = fieldList.size();

        if( (!fieldSpecified && numfields > 1) || (fieldSpecified && fieldIdx > numfields))
        {
            cout<<endl<<"More than one field in the file. Please specify index of the field."<<endl;
            for(int i = 0; i < numfields; i++)
            {
                cout<<"Index "<<i<<" "<<fieldList.at(i)<<endl;
            }
            exit(0);
        }
        cout<<"Using field      : "<<fieldList.at(fieldIdx)<<endl;
        data->AddField(importer->GetField(fieldList.at(fieldIdx), meshlist.at(meshIdx), domainindex));
        cout<<"---------------------------------------------------------"<<endl;
        //------------------------Walk the mesh and get the data--------------------------------
        int vertexIds[4];
        int numCells = cellSet->GetNumCells();
        cout<<"Processing  "<<numCells<<" cells"<<endl;
        int nIds = 0;
        int numTets = 0;
        for(int i = 0; i < numCells; i++)
        {
            if(cellSet->GetCellNodes(i).type != EAVL_TET) continue;
            numTets++;
            cellSet->GetConnectivity(EAVL_NODES_OF_CELLS).GetElementComponents(i,nIds,vertexIds);
            
            eavlVector3 v[4];
            float scalars[4];

            for(int j = 0; j < 4; j++)
            {
                v[j].x = ((eavlFloatArray*)(data->GetField("xcoord")->GetArray()))->GetValue(vertexIds[j]);
                v[j].y = ((eavlFloatArray*)(data->GetField("ycoord")->GetArray()))->GetValue(vertexIds[j]);
                v[j].z = ((eavlFloatArray*)(data->GetField("zcoord")->GetArray()))->GetValue(vertexIds[j]);
                scalars[j] = ((eavlFloatArray*)(data->GetField(fieldList.at(fieldIdx))->GetArray()))->GetValue(vertexIds[j]);
                //cout<<scalars[j]<<" ";
            }

            vrenderer->scene->addTet(v[0], v[1], v[2], v[3], scalars[0], scalars[1], scalars[2], scalars[3]);
        }
        
        //data->PrintSummary(cout);
        

        //setup the view
        BBox bbox = vrenderer->scene->getSceneBBox();

        eavlPoint3 center = eavlPoint3((bbox.max.x + bbox.min.x) / 2,
                                       (bbox.max.y + bbox.min.y) / 2,
                                       (bbox.max.z + bbox.min.z) / 2);

        float ds_size = vrenderer->scene->getSceneMagnitude();

        view.viewtype = eavlView::EAVL_VIEW_3D;
        view.h = height;
        view.w = width;
        view.size = ds_size;
        view.view3d.perspective = true;
        view.view3d.at   = center;
        float fromDist  = (closeup) ? ds_size : ds_size*2;
        view.view3d.from = view.view3d.at + eavlVector3(fromDist,0,0);
        view.view3d.up   = eavlVector3(0,0,1);
        view.view3d.fov  = 0.5;
        view.view3d.xpan = 0;
        view.view3d.ypan = 0;
        view.view3d.zoom = 1.0;

        //set some defaults, this will change later
        view.view3d.nearplane = 0;  
        view.view3d.farplane =  1;

        view.SetupMatrices();
       
        //extract bounding box and project
        eavlPoint3 mins(bbox.min.x,bbox.min.y,bbox.min.z);
        eavlPoint3 maxs(bbox.max.x,bbox.max.y,bbox.max.z);

        mins = view.V * mins;
        maxs = view.V * maxs;

         //squeeze near and far plane to extract max samples
        view.view3d.nearplane = -maxs.z - 5; 
        view.view3d.farplane =  -mins.z + 2; 
        view.SetupMatrices();
        vrenderer->setView(view);

        cout<<"-------------Camera Params---------------"<<endl;
        cout<<"At       : "<<view.view3d.at<<endl;
        cout<<"From     : "<<view.view3d.from<<endl;
        cout<<"Look     : "<<view.view3d.at<<endl;
        cout<<"Fov      : "<<view.view3d.fov<<endl;
        cout<<"Near     : "<<view.view3d.nearplane<<endl;
        cout<<"Far      : "<<view.view3d.farplane<<endl;
        cout<<"Width    : "<<view.w<<endl;
        cout<<"Height   : "<<view.h<<endl;
        cout<<"Size     : "<<view.size<<endl;
        cout<<"vr       : "<<view.vr<<endl;
        cout<<"vl       : "<<view.vl<<endl;
        cout<<"vt       : "<<view.vt<<endl;
        cout<<"vb       : "<<view.vb<<endl;
        cout<<"-----------------------------------------"<<endl;
        
        if(!isTest)
        {
            cout<<"Rendering to Framebuffer\n";
            for(int i=0; i<1;i++)
            {
                vrenderer->Execute();
            }

            writeFrameBufferBMP(height, width, vrenderer->getFrameBuffer(), outFilename.c_str());
        }
        else 
        {
            cout<<"Starting test. Not implemented\n";
        }

        delete vrenderer;
        delete data;
        delete importer;
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        return 1;
    }

    return 0;
}
