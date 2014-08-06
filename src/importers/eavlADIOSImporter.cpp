// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlADIOSImporter.h"
#include "eavlCoordinates.h"
#include "eavlCellSetAllStructured.h"

template<class T> static void
CopyValues(T *buff, eavlFloatArray *arr, int nTups);

eavlADIOSImporter::eavlADIOSImporter(const string &filename)
{
    fp = NULL;

    MPI_Comm comm_dummy = 0;
    fp = adios_read_open_file(filename.c_str(), ADIOS_READ_METHOD_BP, comm_dummy);

    Initialize();
}

eavlADIOSImporter::~eavlADIOSImporter()
{
    if (fp)
        adios_read_close(fp);
    fp = NULL;
}

void
eavlADIOSImporter::Initialize()
{
    metadata.clear();
    
    for (int i = 0; i < fp->nvars; i++)
    {
	ADIOS_VARINFO *avi = adios_inq_var_byid(fp, i);
	if (Supported(avi))
	{
	    string nm = MeshName(avi);
	    map<string,meshData>::iterator mi = metadata.find(nm);
	    if (mi == metadata.end())
	    {
		meshData md;
		md.nm = nm;
		md.dim = avi->ndim;
		md.dims[0] = md.dims[1] = md.dims[2] = 0;
		for (int j = 0; j < avi->ndim; j++)
		    md.dims[j] = avi->dims[j];
		md.vars.push_back(fp->var_namelist[i]);
		metadata[nm] = md;
	    }
	    else
	    {
		mi->second.vars.push_back(fp->var_namelist[i]);
	    }
	}
	adios_free_varinfo(avi);
    }
}

eavlDataSet *
eavlADIOSImporter::GetMesh(const string &mesh, int)
{
    map<string,meshData>::iterator it = metadata.find(mesh);

    if (it == metadata.end())
	return NULL;

    eavlDataSet *data = new eavlDataSet;
    vector<vector<double> > coords;
    vector<string> coordNames;

    coords.resize(it->second.dim);
    for (int i = 0; i < it->second.dim; i++)
    {
	coords[i].resize(it->second.dims[i]);
	for (int j = 0; j < it->second.dims[i]; j++)
		 coords[i][j] = (double)j;
    }
    coordNames.resize(it->second.dim);
    if (it->second.dim > 0) coordNames[0] = "X";
    if (it->second.dim > 1) coordNames[1] = "Y";
    if (it->second.dim > 2) coordNames[2] = "Z";

    int meshIdx = AddRectilinearMesh(data, coords, coordNames, true, "RectilinearGridCells");
    //data->PrintSummary(cout);
    return data;
}

eavlField *
eavlADIOSImporter::GetField(const string &name, const string &mesh, int)
{
    ADIOS_VARINFO *avi = NULL;
    for (int i = 0; i < fp->nvars; i++)
    {
	if (name == string(fp->var_namelist[i]))
	{
	    avi = adios_inq_var_byid(fp, i);
	    break;
	}
    }
    if (avi == NULL)
	return NULL;

    uint64_t s[3]={0,0,0}, c[3] = {0,0,0};
    for (int i = 0; i < avi->ndim; i++)
	c[i] = avi->dims[i];
    ADIOS_SELECTION *sel = adios_selection_boundingbox(avi->ndim, s, c);
    //cout<<"SELECTION: s: "<<s[0]<<" "<<s[1]<<" c: "<<c[0]<<" "<<c[1]<<endl;

    int nt = NumTuples(avi);
    eavlFloatArray *arr = new eavlFloatArray(name, 1);
    arr->SetNumberOfTuples(nt);

    int sz = NumBytes(avi);
    void *buff = new void*[sz];
    
    adios_schedule_read_byid(fp, sel, avi->varid, 0, 1, buff);
    int retval = adios_perform_reads(fp, 1);
    adios_selection_delete(sel);

    if (avi->type == adios_real)
	CopyValues((float*)buff, arr, nt);
    else if (avi->type == adios_double)
	CopyValues((double*)buff, arr, nt);
    
    adios_free_varinfo(avi);
    //arr->PrintSummary(cout);

    return new eavlField(1, arr, eavlField::ASSOC_POINTS);
}

vector<string>
eavlADIOSImporter::GetMeshList()
{
    vector<string> meshes;
    
    map<string,meshData>::iterator mi;
    for ( mi = metadata.begin(); mi != metadata.end(); mi++)
	meshes.push_back(mi->first);
    
    return meshes;
}

vector<string>
eavlADIOSImporter::GetFieldList(const string &mesh)
{
    vector<string> fields;
    map<string,meshData>::iterator mi = metadata.find(mesh);
    if (mi != metadata.end())
	for (int i = 0; i < mi->second.vars.size(); i++)
	    fields.push_back(mi->second.vars[i]);

    return fields;
}

bool
eavlADIOSImporter::Supported(ADIOS_VARINFO *avi)
{
    return ((avi->ndim == 2 || avi->ndim == 3) &&
	    avi->type == adios_real ||
	    avi->type == adios_double);
}

string
eavlADIOSImporter::MeshName(ADIOS_VARINFO *avi)
{ 
    std::vector<int64_t> dimT, dims;
    std::string meshname = "mesh_";

    for (int i=0; i<avi->ndim; i++)
        dims.insert(dims.begin(), avi->dims[i]);

    for (int i=0; i <dims.size(); i++)
    {
        std::stringstream ss;
        ss<<dims[i];
        meshname += ss.str();
        if (i<dims.size()-1)
            meshname += "x";
    }
    return meshname;
}

 int
 eavlADIOSImporter::NumTuples(ADIOS_VARINFO *avi)
 {
     int n = 1;
     for (int i = 0; i < avi->ndim; i++)
	 n *= avi->dims[i];
     return n;
 }

 int
 eavlADIOSImporter::NumBytes(ADIOS_VARINFO *avi)
 {
     int n = NumTuples(avi);
     if (avi->type == adios_real)
	 n *= sizeof(float);
     else if (avi->type == adios_double)
	 n *= sizeof(double);
     
     return n;
 }

template<class T> static void
CopyValues(T *buff, eavlFloatArray *arr, int nTups)
{
    for (int i = 0; i < nTups; i++)
    {
	arr->SetComponentFromDouble(i, 0, (double)(buff[i]));
    }
}


/*
eavlDataSet *
eavlADIOSImporter::CreateRectilinearGrid(const ADIOSVar &v)
{
    eavlDataSet *data = new eavlDataSet;
    vector<vector<double> > coords;
    vector<string> coordNames;

    coords.resize(v.dim);
    if (v.dim >= 1)
    {
        coordNames.push_back("X");
        coords[0].resize(v.count[0]);
        for (int i = 0; i < v.count[0]; i++)
            coords[0][i] = i;
    }
    if (v.dim >= 2)
    {
        coordNames.push_back("Y");
        coords[1].resize(v.count[1]);
        for (int i = 0; i < v.count[1]; i++)
            coords[1][i] = i;
    }
    if (v.dim >= 3)
    {
        coordNames.push_back("Z");
        coords[2].resize(v.count[2]);
        for (int i = 0; i < v.count[2]; i++)
            coords[2][i] = i;
    }

    int meshIdx = AddRectilinearMesh(data, coords, coordNames, true, "RectilinearGridCells");
    return data;
}
*/
