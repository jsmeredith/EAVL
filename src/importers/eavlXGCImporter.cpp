// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.

#include "eavlXGCImporter.h"
#include "eavlCellSetExplicit.h"

eavlXGCImporter::eavlXGCImporter(const string &filename)
{
    nNodes = 0;
    nPlanes = 0;
    nElems = 0;
    fp = NULL;
    mesh_fp = NULL;
    points = NULL;
    cells = NULL;
    
    string::size_type i0 = filename.rfind("xgc.");
    string::size_type i1 = filename.rfind(".bp");
    string meshname = filename.substr(0,i0+4) + "mesh.bp";
    
    MPI_Comm comm_dummy = 0;
    fp = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, comm_dummy);
    mesh_fp = adios_read_open_file(meshname.c_str(), ADIOS_READ_METHOD_BP, comm_dummy);
    
    if (fp == NULL)
	THROW(eavlException, "XGC variable file not found.");
    if (mesh_fp == NULL)
	THROW(eavlException, "XGC mesh file not found.");

    Initialize();
}

eavlXGCImporter::~eavlXGCImporter()
{
    if (fp)
        adios_read_close(fp);
    if (mesh_fp)
        adios_read_close(mesh_fp);

    fp = NULL;
    mesh_fp = NULL;

    map<string, ADIOS_VARINFO*>::const_iterator it;
    for (it = variables.begin(); it != variables.end(); it++)
	adios_free_varinfo(it->second);
    variables.clear();
    if (points)
	adios_free_varinfo(points);    
    if (cells)
	adios_free_varinfo(cells);
    points = NULL;
    cells = NULL;
}

void
eavlXGCImporter::Initialize()
{
    variables.clear();
    for (int i = 0; i < fp->nvars; i++)
    {
	ADIOS_VARINFO *avi = adios_inq_var_byid(fp, i);
	string varNm(&fp->var_namelist[i][1]);
	if (Supported(avi))
	    variables[varNm] = avi;
	else
	{
	    //check for scalars....
	    if (avi->ndim == 0)
	    {
		if (varNm == "nnode")
		    nNodes = (int)(*(int *)avi->value);
		if (varNm == "nphi")
		    nPlanes = (int)(*(int *)avi->value);
	    }
	    adios_free_varinfo(avi);
	}
    }

    //Read mesh info.
    for (int i = 0; i < mesh_fp->nvars; i++)
    {
	ADIOS_VARINFO *avi = adios_inq_var_byid(mesh_fp, i);
	string varNm(&mesh_fp->var_namelist[i][1]);

	bool freeAVI = true;
	if (avi->ndim == 0)
	{
	    if (varNm == "n_t")
		nElems = (int)(*(int *)avi->value);
	    else if (varNm == "n_n")
	    {
		int n = (int)(*(int *)avi->value);
		if (n != nNodes)
		    THROW(eavlException, "Node count mismatch between mesh and variable file.");
	    }
	}
	else if (avi->ndim == 1 && varNm == "nextnode")
	{
	    nextNode = avi;
	    freeAVI = false;
	}
	else if (avi->ndim == 2 && varNm == "coordinates/values")
	{
	    points = avi;
	    freeAVI = false;
	}
	else if (avi->ndim == 2 && varNm == "cell_set[0]/node_connect_list")
	{
	    cells = avi;
	    freeAVI = false;
	}
	
	if (freeAVI)
	    adios_free_varinfo(avi);
    }
}

vector<string>
eavlXGCImporter::GetMeshList()
{
    vector<string> m;
    m.push_back("mesh2D");
    m.push_back("mesh3D");
    return m;
}

vector<string>
eavlXGCImporter::GetCellSetList(const std::string &mesh)
{
    vector<string> m;
    if (mesh == "mesh2D")
	m.push_back("mesh2D_cells");
    else if (mesh == "mesh3D")
	m.push_back("mesh3D_cells");

    return m;
}

vector<string>
eavlXGCImporter::GetFieldList(const std::string &mesh)
{
    vector<string> fields;
    map<string, ADIOS_VARINFO*>::const_iterator it;
    
    bool mesh2D = (mesh == "mesh2D");
    for (it = variables.begin(); it != variables.end(); it++)
    {
	if (mesh2D && it->second->ndim == 1)
	    fields.push_back(it->first);
	else if (!mesh2D && it->second->ndim == 2)
	    fields.push_back(it->first);
    }
    
    return fields;
}

eavlDataSet *
eavlXGCImporter::GetMesh(const string &name, int chunk)
{
    eavlDataSet *ds = new eavlDataSet;
    if (name == "mesh2D")
    {
	ds->SetNumPoints(nNodes);
	eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(NULL,
									eavlCoordinatesCartesian::X,
									eavlCoordinatesCartesian::Y);
	ds->AddCoordinateSystem(coords);
	coords->SetAxis(0, new eavlCoordinateAxisField("xcoords", 0));
	coords->SetAxis(1, new eavlCoordinateAxisField("ycoords", 0));
	
	eavlArray *axisValues[2] = {new eavlFloatArray("xcoords", 1),
				    new eavlFloatArray("ycoords", 1)};
	axisValues[0]->SetNumberOfTuples(nNodes);
	axisValues[1]->SetNumberOfTuples(nNodes);

	//read points
	double *buff = new double[2*nNodes];
	uint64_t s[3], c[3];
	ADIOS_SELECTION *sel = MakeSelection(points, s, c);
	adios_schedule_read_byid(mesh_fp, sel, points->varid, 0, 1, buff);
	int retval = adios_perform_reads(mesh_fp, 1);
	adios_selection_delete(sel);

        for (int i = 0; i < nNodes; i++)
	{
	    axisValues[0]->SetComponentFromDouble(i, 0, buff[i*2 +0]);
	    axisValues[1]->SetComponentFromDouble(i, 0, buff[i*2 +1]);
	}
	ds->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
	ds->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
	delete [] buff;

	eavlCellSetExplicit *cellSet = new eavlCellSetExplicit(name + "_Cells", 2);
	eavlExplicitConnectivity conn;

	//read cells
	int *nodeList = new int[nElems*3];
	sel = MakeSelection(cells, s, c);
	adios_schedule_read_byid(mesh_fp, sel, cells->varid, 0, 1, nodeList);
	retval = adios_perform_reads(mesh_fp, 1);
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
	ds->AddCellSet(cellSet);
    }
    else if (name == "mesh3D")
    {
	int nPts = nNodes*nPlanes;
	ds->SetNumPoints(nPts);
	eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(NULL,
									eavlCoordinatesCartesian::X,
									eavlCoordinatesCartesian::Y,
									eavlCoordinatesCartesian::Z);
	ds->AddCoordinateSystem(coords);
	coords->SetAxis(0, new eavlCoordinateAxisField("xcoords", 0));
	coords->SetAxis(1, new eavlCoordinateAxisField("ycoords", 0));
	coords->SetAxis(2, new eavlCoordinateAxisField("zcoords", 0));
	
	eavlArray *axisValues[3] = {new eavlFloatArray("xcoords", 1),
				    new eavlFloatArray("ycoords", 1),
				    new eavlFloatArray("zcoords", 1)};
	axisValues[0]->SetNumberOfTuples(nPts);
	axisValues[1]->SetNumberOfTuples(nPts);
	axisValues[2]->SetNumberOfTuples(nPts);

	//read points
	double *buff = new double[2*nNodes];
	uint64_t s[3], c[3];
	ADIOS_SELECTION *sel = MakeSelection(points, s, c);
	adios_schedule_read_byid(mesh_fp, sel, points->varid, 0, 1, buff);
	int retval = adios_perform_reads(mesh_fp, 1);
	adios_selection_delete(sel);

	int idx = 0;
	double dPhi = 2.0*M_PI/(double)nPlanes;
	//double dPhi = 2.0*M_PI/(double)32;
	for (int i = 0; i < nPlanes; i++)
	{
	    double phi = (double)i * dPhi;
	    for (int j = 0; j < nNodes; j++)
	    {
		double R = buff[j*2 +0];
		double Z = buff[j*2 +1];
		axisValues[0]->SetComponentFromDouble(idx, 0, R*cos(phi));
		axisValues[1]->SetComponentFromDouble(idx, 0, R*sin(phi));
		axisValues[2]->SetComponentFromDouble(idx, 0, Z);
		idx++;
	    }
	}
	    
	ds->AddField(new eavlField(1, axisValues[0], eavlField::ASSOC_POINTS));
	ds->AddField(new eavlField(1, axisValues[1], eavlField::ASSOC_POINTS));
	ds->AddField(new eavlField(1, axisValues[2], eavlField::ASSOC_POINTS));
	delete [] buff;

	eavlCellSetExplicit *cellSet = new eavlCellSetExplicit(name + "_Cells", 3);
	eavlExplicitConnectivity conn;
	
	//read cells
	int *nodeList = new int[nElems*3];
	sel = MakeSelection(cells, s, c);
	adios_schedule_read_byid(mesh_fp, sel, cells->varid, 0, 1, nodeList);
	retval = adios_perform_reads(mesh_fp, 1);
	adios_selection_delete(sel);


	int nodes[6];
	for (int i = 0; i < nPlanes; i++)
	{
	    int cellCnt = 0;
	    for (int j = 0; j < nElems*3; j+=3)
	    {
		int off = i*nNodes;
		nodes[0] = nodeList[j+0] + off;
		nodes[1] = nodeList[j+1] + off;
		nodes[2] = nodeList[j+2] + off;
		
		off = ((i==nPlanes-1) ? 0 : (i+1)*nNodes);
		nodes[3] = nodeList[j+0] + off;
		nodes[4] = nodeList[j+1] + off;
		nodes[5] = nodeList[j+2] + off;
		conn.AddElement(EAVL_WEDGE, 6, nodes);
	    }
	}
	delete [] nodeList;
	
	cellSet->SetCellNodeConnectivity(conn);
	ds->AddCellSet(cellSet);
    }

    //ds->PrintSummary(cout);
    return ds;
}

eavlField *
eavlXGCImporter::GetField(const string &name, const string &mesh, int chunk)
{
    map<string, ADIOS_VARINFO*>::const_iterator it = variables.find(name);
    if (it == variables.end())
	THROW(eavlException, string("Variable not found: ")+name);

    uint64_t s[3], c[3];
    ADIOS_SELECTION *sel = MakeSelection(it->second, s, c);
    int nt = 1;
    for (int i = 0; i < it->second->ndim; i++)
	nt *= c[i];
    double *buff = new double[nt];
    adios_schedule_read_byid(fp, sel, it->second->varid, 0, 1, buff);
    int retval = adios_perform_reads(fp, 1);
    adios_selection_delete(sel);

    eavlFloatArray *arr = new eavlFloatArray(name, 1);
    arr->SetNumberOfTuples(nt);
    for (int i = 0; i < nt; i++)
	arr->SetComponentFromDouble(i, 0, buff[i]);
    delete [] buff;

    eavlField *field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
    //field->PrintSummary(cout);
    return field;
}

bool
eavlXGCImporter::Supported(ADIOS_VARINFO *avi)
{
    return ((avi->ndim == 2 || avi->ndim == 1) &&
	    avi->type == adios_real ||
	    avi->type == adios_double);
}

ADIOS_SELECTION *
eavlXGCImporter::MakeSelection(ADIOS_VARINFO *avi, uint64_t *s, uint64_t *c)
{
    for (int i = 0; i < 3; i++)
	s[i] = c[i] = 0;
    for (int i = 0; i < avi->ndim; i++)
	c[i] = avi->dims[i];
    
    return adios_selection_boundingbox(avi->ndim, s, c);
}
