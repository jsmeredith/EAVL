// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlFilter.h"
#include "eavlDataSet.h"
#include "eavlTimer.h"
#include "eavlException.h"

#include "eavlImporterFactory.h"

//#include "eavlRenderSurfaceParallelOSMesa.h"
#include "eavlRenderSurfaceOSMesa.h"
#include "eavlView.h"
#include "eavlSceneRenderer.h"
#include "eavlSceneRendererGL.h"
#include "eavlWorldAnnotatorGL.h"
#include "eavl3DWindow.h"
#include "eavlScene.h"

eavlDataSet *ReadMeshFromFile(const string &filename, int meshindex)
{
    eavlImporter *importer = eavlImporterFactory::GetImporterForFile(filename);

    if (!importer)
        THROW(eavlException,"Didn't determine proper file reader to use");

    vector<string> allmeshes = importer->GetMeshList();
    if (meshindex >= (int)allmeshes.size())
        return NULL;

    string meshname = allmeshes[meshindex];
    // always read the first domain for now
    eavlDataSet *out = importer->GetMesh(meshname, 0);
    vector<string> allvars = importer->GetFieldList(meshname);
    for (size_t i=0; i<allvars.size(); i++)
        out->AddField(importer->GetField(allvars[i], meshname, 0));

    return out;
}

int main(int argc, char *argv[])
{
    try
    {   
        if (argc != 4)
            THROW(eavlException,"Incorrect number of arguments");

        eavlDataSet *ds = ReadMeshFromFile(argv[1], 0);
	string field = argv[2];
	string imgFile = argv[3];

	eavlColor bg(0.15, 0.05, 0.1, 1.0);
	eavlRenderSurface *surface = new eavlRenderSurfaceOSMesa();
	eavlScene *scene = new eavl3DScene();
	eavlSceneRendererGL *sceneRenderer = new eavlSceneRendererGL();
	eavlWorldAnnotator *annotator = new eavlWorldAnnotatorGL;
	eavl3DWindow *window = new eavl3DWindow(bg, surface, scene, sceneRenderer, annotator);
	window->Initialize();
	window->Resize(512,512);
	window->Initialize();

	eavlPlot *plot = new eavlPlot(ds);
	
	plot->SetField(field);
	plot->SetColorTableByName("dense");
	scene->plots.push_back(plot);
	scene->ResetView(window);
	
	window->Paint();
	surface->SaveAs(imgFile, eavlRenderSurface::PNM);
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <infile> fieldName imgFile\n";
        return 1;
    }


    return 0;
}
