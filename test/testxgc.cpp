// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlFilter.h"
#include "eavlDataSet.h"
#include "eavlTimer.h"
#include "eavlException.h"

#include "eavlImporterFactory.h"
#include "eavlXGCImporter.h"
#include "eavlCellSetExplicit.h"

//#include "eavlRayQueryMutator.h"

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

ADIOS_SELECTION *
MakeSelection(ADIOS_VARINFO *avi, uint64_t *s, uint64_t *c)
{
    for (int i = 0; i < 3; i++)
	s[i] = c[i] = 0;
    for (int i = 0; i < avi->ndim; i++)
	c[i] = avi->dims[i];
    
    return adios_selection_boundingbox(avi->ndim, s, c);
}

eavlDataSet *ReadPsiMesh(const string &filename)
{
    MPI_Comm comm_dummy = 0;
    ADIOS_FILE *fp = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, comm_dummy);

    int nNodes = 0, nElems = 0, ptID = -1, elemID = -1;
    ADIOS_VARINFO *points = NULL, *cells = NULL, *psi;
    for (int i = 0; i < fp->nvars; i++)
    {
	ADIOS_VARINFO *avi = adios_inq_var_byid(fp, i);
	string varNm(&fp->var_namelist[i][1]);
	if (varNm == "n_t")
	{
	    nElems = (int)(*(int *)avi->value);
	    adios_free_varinfo(avi);
	}
	else if (varNm == "n_n")
	{
	    nNodes = (int)(*(int *)avi->value);
	    adios_free_varinfo(avi);
	}
	else if (varNm == "coordinates/values")
	    points = avi;
	else if (varNm == "cell_set[0]/node_connect_list")
	    cells = avi;
	else if (varNm == "psi")
	    psi = avi;
	else
	    adios_free_varinfo(avi);
    }
    cout<<"nNodes= "<<nNodes<<" nTri= "<<nElems<<endl;

    eavlDataSet *out = new eavlDataSet;
    out->SetNumPoints(nNodes);
    eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(NULL,
								    eavlCoordinatesCartesian::X,
								    eavlCoordinatesCartesian::Y);
    out->AddCoordinateSystem(coords);
    coords->SetAxis(0, new eavlCoordinateAxisField("xcoords", 0));
    coords->SetAxis(1, new eavlCoordinateAxisField("ycoords", 0));
	
    eavlArray *axisValues[2] = {new eavlFloatArray("xcoords", 1),
				new eavlFloatArray("ycoords", 1)};
    axisValues[0]->SetNumberOfTuples(nNodes);
    axisValues[1]->SetNumberOfTuples(nNodes);
    //read points.
    double *buff = new double[2*nNodes];
    uint64_t s[3], c[3];
    ADIOS_SELECTION *sel = MakeSelection(points, s, c);
    adios_schedule_read_byid(fp, sel, points->varid, 0, 1, buff);
    int retval = adios_perform_reads(fp, 1);
    adios_selection_delete(sel);
    adios_free_varinfo(points);

    for (int i = 0; i < nNodes; i++)
    {
	axisValues[0]->SetComponentFromDouble(i, 0, buff[i*2 +0]);
	axisValues[1]->SetComponentFromDouble(i, 0, buff[i*2 +1]);
    }
    out->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
    out->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
    delete [] buff;

    eavlCellSetExplicit *cellSet = new eavlCellSetExplicit("2D_cells", 2);
    eavlExplicitConnectivity conn;

    //read cells
    int *nodeList = new int[nElems*3];
    sel = MakeSelection(cells, s, c);
    adios_schedule_read_byid(fp, sel, cells->varid, 0, 1, nodeList);
    retval = adios_perform_reads(fp, 1);
    adios_selection_delete(sel);

    int nodes[3];
    for (int i = 0; i < nElems; i++)
    {
	nodes[0] = nodeList[i*3+0];
	nodes[1] = nodeList[i*3+1];
	nodes[2] = nodeList[i*3+2];
	conn.AddElement(EAVL_TRI, 3, nodes);
    }
    delete [] nodeList;
    cellSet->SetCellNodeConnectivity(conn);
    out->AddCellSet(cellSet);

    //read psi.
    buff = new double[nNodes];
    sel = MakeSelection(psi, s, c);
    adios_schedule_read_byid(fp, sel, psi->varid, 0, 1, buff);
    retval = adios_perform_reads(fp, 1);
    adios_selection_delete(sel);
    adios_free_varinfo(psi);

    eavlArray *psiValues = new eavlFloatArray("psi", 1);
    psiValues->SetNumberOfTuples(nNodes);
    for (int i = 0; i < nNodes; i++)
	psiValues->SetComponentFromDouble(i, 0, buff[i]);
    out->AddField(new eavlField(1, psiValues, eavlField::ASSOC_POINTS));
    delete [] buff;
	
    return out;
}

static inline float DOT(float *x, float *y) {return x[0]*y[0]+x[1]*y[1];}

double GetPsi(float R, float Z, eavlDataSet *psiMesh)
{
    double psi = -1.0;

    eavlCellSet *cells = psiMesh->GetCellSet(0);
    int nCells = cells->GetNumCells();
    
    //psiMesh->PrintSummary(cout);
    eavlField *Rf = psiMesh->GetField("xcoords");
    eavlField *Zf = psiMesh->GetField("ycoords");
    eavlField *psiF = psiMesh->GetField("psi");

    int cellIdx = -1;
    for (int i = 0; i < nCells; i++)
    {
	eavlCell c = cells->GetCellNodes(i);
	float r0 = Rf->GetArray()->GetComponentAsDouble(c.indices[0], 0);
	float r1 = Rf->GetArray()->GetComponentAsDouble(c.indices[1], 0);
	float r2 = Rf->GetArray()->GetComponentAsDouble(c.indices[2], 0);
	
	float z0 = Zf->GetArray()->GetComponentAsDouble(c.indices[0], 0);
	float z1 = Zf->GetArray()->GetComponentAsDouble(c.indices[1], 0);
	float z2 = Zf->GetArray()->GetComponentAsDouble(c.indices[2], 0);

	float v0[2] = {r1-r0, z1-z0}, v1[2] = {r2-r0, z2-z0}, v2[2] = {R-r0, Z-z0};
	float d00 = DOT(v0,v0);
	float d01 = DOT(v0, v1);
	float d11 = DOT(v1, v1);
	float d20 = DOT(v2, v0);
	float d21 = DOT(v2, v1);
	float denom = d00 * d11 - d01 * d01;
	
	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
	float u = 1.0f - v - w;

	//We have a triangle hit...
	if (u >= 0.0f && u <= 1.0f &&
	    v >= 0.0f && v <= 1.0f &&
	    w >= 0.0f && w <= 1.0f)
	{

	    double psi0 = psiF->GetArray()->GetComponentAsDouble(c.indices[0], 0);
	    double psi1 = psiF->GetArray()->GetComponentAsDouble(c.indices[1], 0);
	    double psi2 = psiF->GetArray()->GetComponentAsDouble(c.indices[2], 0);

	    psi = psi0*u + psi1*v + psi2*w;
	    //cout<<psi<<"= "<<psi0<<"*"<<u<<" + "<<psi1<<"*"<<v<<" + "<<psi2<<"*"<<w<<endl;
	    //cout<<"Found a cell index: "<< cellIdx<<endl;
	    cellIdx = i;
	    break;
	}
    }

    return psi;
}

static bool
IsRegion1(double R, double Z, double psi)
{
    static const double eq_x_psi = 0.266196;
    static const double eq_x_r = 1.55755;
    static const double eq_x_z = -1.17707;
    static const double eq_axis_r = 1.72485;
    static const double eq_axis_z = 0.020562;
    static const double eq_x_slope = -(eq_x_r - eq_axis_r)/(eq_x_z - eq_axis_z);
    
    if (psi <= (eq_x_psi-1e-5) && -(R-eq_x_r) * eq_x_slope + (Z-eq_x_z) > 0.0)
	return true;
    
    return false;
}

/*
static void
AddTriangles(eavlDataSet *psiMesh, eavlRayQueryMutator *rqm)
{
    //Get cells, vertices.
    eavlCellSet *cells = psiMesh->GetCellSet(0);
    eavlField *Rf = psiMesh->GetField("xcoords");
    eavlField *Zf = psiMesh->GetField("ycoords");
    eavlField *psiF = psiMesh->GetField("psi");

    eavlVector3 p0, p1, p2;
    int nCells = cells->GetNumCells();
    for (int i = 0; i < nCells; i++)
    {
	eavlCell c = cells->GetCellNodes(i);
	p0[0] = Rf->GetArray()->GetComponentAsDouble(c.indices[0], 0);
	p0[1] = Zf->GetArray()->GetComponentAsDouble(c.indices[0], 0);
	p0[2] = 0.0f;

	p1[0] = Rf->GetArray()->GetComponentAsDouble(c.indices[1], 0);
	p1[1] = Zf->GetArray()->GetComponentAsDouble(c.indices[1], 0);
	p1[2] = 0.0f;

	p2[0] = Rf->GetArray()->GetComponentAsDouble(c.indices[2], 0);
	p2[1] = Zf->GetArray()->GetComponentAsDouble(c.indices[2], 0);
	p2[2] = 0.0f;
	
	rqm->addTriangle(p0, p1, p2);
    }
}
*/

int main(int argc, char *argv[])
{
    try
    {   
        if (argc != 3)
            THROW(eavlException,"Incorrect number of arguments");

	string pFile = argv[1];
	string mFile = argv[2];
	//string pFile = "/apps/eavl/EAVL/data/xgc.restart.bp";
	//string mFile = "/apps/eavl/EAVL/data/xgc.mesh.bp";

        eavlDataSet *particles = ReadMeshFromFile(pFile, 0);
	eavlDataSet *psiMesh = ReadPsiMesh(mFile);
	//particles->PrintSummary(cout);
	//psiMesh->PrintSummary(cout);

	/*
	eavlRayQueryMutator *rqm = new eavlRayQueryMutator;

	AddTriangles(psiMesh, rqm);
	*/

	int nP = particles->GetNumPoints();
	eavlField *Rf = particles->GetField("R");
	eavlField *Zf = particles->GetField("Z");
	double eq_x_psi = 0.266196; //From xgc.equil.bp
	
	cout<<"nParticles= "<<nP<<endl;
	for (int i = 0; i < nP; i++)
	{
	    double R = Rf->GetArray()->GetComponentAsDouble(i,0);
	    double Z = Zf->GetArray()->GetComponentAsDouble(i,0);
	    double psi = GetPsi(R,Z, psiMesh);

	    if (!IsRegion1(R,Z,psi))
		cout<<i<<": open field"<<endl;
	}

    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" particle_file mesh_file\n";
        return 1;
    }


    return 0;
}
