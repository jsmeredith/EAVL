// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlSiloImporter.h"
#include "eavlCoordinates.h"
#include "eavlCellSetAllStructured.h"
#include "eavlCellSetExplicit.h"
#include "eavlException.h"
#include <string.h>

#if 1 // FORCING SINGLE-PRECISION HERE
template<class T> static eavlFloatArray *
ReadValues(string nm, T **vals, int nvals, int nels)
{
    eavlFloatArray *arr = new eavlFloatArray(nm, nvals);
    arr->SetNumberOfTuples(nels);
    for (int i = 0; i < nvals; i++)
    {
        for (int j = 0; j < nels; j++)
            arr->SetComponentFromDouble(j, i, (double)(vals[i][j]));
    }

    return arr;
}

static eavlFloatArray *
ReadValues(int dataType, string nm, void **vals, int nvals, int nels)
{
    if (dataType == DB_DOUBLE)
        return ReadValues(nm, (double **)vals, nvals, nels);
    else
        return ReadValues(nm, (float **)vals, nvals, nels);
}

#else
template<class T> static eavlDoubleArray *
ReadValues(string nm, T **vals, int nvals, int nels)
{
    eavlDoubleArray *arr = new eavlDoubleArray(nm, nvals);
    arr->SetNumberOfTuples(nels);
    for (int i = 0; i < nvals; i++)
    {
        for (int j = 0; j < nels; j++)
            arr->SetComponentFromDouble(j, i, (double)(vals[i][j]));
    }

    return arr;
}

static eavlDoubleArray *
ReadValues(int dataType, string nm, void **vals, int nvals, int nels)
{
    if (dataType == DB_DOUBLE)
        return ReadValues(nm, (double **)vals, nvals, nels);
    else
        return ReadValues(nm, (float **)vals, nvals, nels);
}
#endif

/*
static bool
IsContained(string str, vector<string> &v)
{
    for(vector<string>::const_iterator it = v.begin(); it!= v.end(); ++it)
        if (str == *it)
            return true;
    
    return false;
}
*/

static string
RemoveSlash(string str)
{
    if (str[0] == '/')
        str = str.substr(1, str.size());
    return str;
}

static void MapZoneShape(int n, int *mapindices, int *inputnodes, int *outputnodes)
{
    for (int i=0; i<n; i++)
        outputnodes[i] = inputnodes[mapindices[i]];
}


eavlSiloImporter::eavlSiloImporter(const string &filename)
{
    file = DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ);
    if (file == NULL)
        THROW(eavlException,"Couldn't open Silo file!\n");

    ghosts_for_latest_mesh = new eavlByteArray(".ghost",1);

    Import();
}

eavlSiloImporter::~eavlSiloImporter()
{
    if (file)
    {
        DBClose(file);
        file = NULL;
    }
}

int
eavlSiloImporter::GetNumChunks(const string &meshname)
{
    if (multiMeshes.count(meshname) > 0)
        return multiMeshes[meshname].size();
        
    return 1;
}

static string formName(string nm, string dir)
{
    string name;
    if (dir.empty())
        name = RemoveSlash(nm);
    else
        name = RemoveSlash(dir) + "/" + RemoveSlash(nm);
    
    return name;
}

void
eavlSiloImporter::ReadQuadMeshes(DBfile *file, DBtoc *toc, string &dir)
{
    for (int i = 0; i < toc->nqmesh; i++)
    {
        DBquadmesh *m = DBGetQuadmesh(file, toc->qmesh_names[i]);
        string meshname = formName(toc->qmesh_names[i], dir);
        quadMeshes.push_back(meshname);
        DBFreeQuadmesh(m);
    }
}

void
eavlSiloImporter::ReadQuadVars(DBfile *file, DBtoc *toc, string &dir)
{
    //Read vars.
    for (int i=0; i<toc->nqvar; i++)
    {
        DBquadvar *v = DBGetQuadvar(file, toc->qvar_names[i]);
        string meshname = formName(v->meshname, dir);
        quadVars.push_back(formName(v->name,dir));
        meshForVar[formName(v->name,dir)] = meshname;
        DBFreeQuadvar(v);
    }
}

void
eavlSiloImporter::ReadUCDMeshes(DBfile *file, DBtoc *toc, string &dir)
{
    for (int i = 0; i < toc->nucdmesh; i++)
    {
        DBmultimesh *m = DBGetMultimesh(file, toc->ucdmesh_names[i]);
        string meshname = formName(toc->ucdmesh_names[i], dir);
        ucdMeshes.push_back(meshname);
        DBFreeMultimesh(m);
    }
}

void
eavlSiloImporter::ReadUCDVars(DBfile *file, DBtoc *toc, string &dir)
{
    //Read vars.
    for (int i=0; i<toc->nucdvar; i++)
    {
        DBucdvar *v = DBGetUcdvar(file, toc->ucdvar_names[i]);
        string meshname = formName(v->meshname, dir);
        ucdVars.push_back(formName(v->name,dir));
        meshForVar[formName(v->name,dir)] = meshname;
        DBFreeUcdvar(v);
    }
}

void
eavlSiloImporter::ReadMultiMeshes(DBfile *file, DBtoc *toc, string &dir)
{
    for (int i = 0; i < toc->nmultimesh; i++)
    {
        DBmultimesh *m = DBGetMultimesh(file, toc->multimesh_names[i]);
        string meshNm;
        if (dir.empty())
            meshNm = toc->multimesh_names[i];
        else
            meshNm = dir + "/" + toc->multimesh_names[i];
        
        bool foundEmpty = false;
        for (int j = 0; !foundEmpty && j < m->nblocks; j++)
            foundEmpty = (string(m->meshnames[j]) == "EMPTY");
        
        ///\todo: looks like we (mostly) support EMPTY correctly now?
        if (true) //(!foundEmpty)
        {
            vector<string> meshes;
            for (int j=0; j<m->nblocks; j++)
            {
                meshes.push_back(m->meshnames[j]);
                meshesToHide.insert(RemoveSlash(m->meshnames[j]));
            }
            multiMeshes[meshNm] = meshes;
        }
        DBFreeMultimesh(m);
    }
}

void
eavlSiloImporter::ReadMultiVars(DBfile *file, DBtoc *toc, string &dir)
{
    //Read vars.
    for (int i=0; i<toc->nmultivar; i++)
    {
        DBmultivar *v = DBGetMultivar(file, toc->multivar_names[i]);
        bool foundEmpty = false;
        for (int j = 0; !foundEmpty && j < v->nvars; j++)
            foundEmpty = (string(v->varnames[j]) == "EMPTY");
        
        ///\todo: looks like we (mostly) support EMPTY correctly now?
        if (true) //(!foundEmpty)
        {
            vector<string> vars;
            for (int j = 0; j < v->nvars; j++)
                vars.push_back(v->varnames[j]);
            multiVars[toc->multivar_names[i]] = vars;
        }
        DBFreeMultivar(v);
    }
}


void
eavlSiloImporter::ReadPointMeshes(DBfile *file, DBtoc *toc, string &dir)
{
    for (int i = 0; i < toc->nptmesh; i++)
    {
        DBpointmesh *m = DBGetPointmesh(file, toc->ptmesh_names[i]);
        string meshname = formName(toc->ptmesh_names[i], dir);
        ptMeshes.push_back(meshname);
        DBFreePointmesh(m);
    }
}

void
eavlSiloImporter::ReadPointVars(DBfile *file, DBtoc *toc, string &dir)
{
    //Read vars.
    for (int i = 0; i < toc->nptvar; i++)
    {
        DBmeshvar *v = DBGetPointvar(file, toc->ptvar_names[i]);
        string meshname = formName(v->meshname, dir);
        ptVars.push_back(formName(v->name,dir));
        meshForVar[formName(v->name,dir)] = meshname;
        DBFreeMeshvar(v);
    }
}

void
eavlSiloImporter::ReadFile(DBfile *file, string dir)
{
    DBtoc *toc = DBGetToc(file);
    if (toc == NULL)
        return;

    ReadMultiMeshes(file, toc, dir);
    ReadQuadMeshes(file, toc, dir);
    ReadUCDMeshes(file, toc, dir);
    ReadPointMeshes(file, toc, dir);
    
    ReadMultiVars(file, toc, dir);
    ReadQuadVars(file, toc, dir);
    ReadUCDVars(file, toc, dir);
    ReadPointVars(file, toc, dir);

    vector<string> dirs;
    for (int i = 0; i < toc->ndir; i++)
        dirs.push_back(toc->dir_names[i]);
    
    for (size_t i = 0; i < dirs.size(); i++)
    {
        DBSetDir(file, dirs[i].c_str());
        char currDir[128];
        DBGetDir(file, currDir);
        ReadFile(file, currDir);
        DBSetDir(file, "..");
    }
}

void
eavlSiloImporter::Import()
{
    ReadFile(file, "");
    
    //Print();
}

eavlDataSet *
eavlSiloImporter::GetMultiMesh(string nm, int chunk)
{
    string meshNm = multiMeshes[nm][chunk];
    if (meshNm == "EMPTY")
        return NULL;
    DBmultimesh *m = DBGetMultimesh(file, nm.c_str());
    if (m->meshtypes[chunk] == DB_QUADMESH)
        return GetQuadMesh(meshNm);
    else if (m->meshtypes[chunk] == DB_UCDMESH)
        return GetUCDMesh(meshNm);
    else if (m->meshtypes[chunk] == DB_POINTMESH)
        return GetPtMesh(meshNm);
    return NULL;
}

eavlDataSet *
eavlSiloImporter::GetQuadMesh(string nm)
{
    DBquadmesh *m = DBGetQuadmesh(file, nm.c_str());
    if (!m)
        return NULL;
    vector<vector<double> > coords;
    vector<string> coordNames;
        
    coords.resize(m->ndims);
    bool rectilinear = m->coordtype == DB_COLLINEAR;
    int  nzones = 1;
    for (int i = 0; i < m->ndims; i++)
    {
        coordNames.push_back(m->labels[i]);
        int dimsize = m->dims[i];
        if (dimsize > 1)
            nzones *= (dimsize-1);
        int n = (rectilinear ? dimsize : m->nnodes);
        coords[i].resize(n);
        if (m->datatype == DB_DOUBLE)
        {
            for (int j = 0; j < n; j++)
                coords[i][j] = ((double**)m->coords)[i][j];
        }
        else
        {
            for (int j = 0; j < n; j++)
                coords[i][j] = (double) ((float**)m->coords)[i][j];
        }
        
    }

    eavlDataSet *data = new eavlDataSet;
    if (rectilinear)
    {
        AddRectilinearMesh(data, coords,
                           coordNames, true,
                           m->name+string("_Cells"));
    }
    else
    {
        AddCurvilinearMesh(data, m->dims, coords,
                           coordNames, true,
                           m->name+string("_Cells"));
        /*
        // For various reasons, we might want to use separated coords instead.
        meshIdx = AddCurvilinearMesh_SepCoords(data, m->dims, coords,
                                     coordNames, true,
                                     m->name+string("_Cells"));
        */
    }

    ghosts_for_latest_mesh->SetNumberOfTuples(nzones);
    for (int i=0; i<nzones; i++)
    {
        ///\todo: fill this in from min_index[3] and max_index[3] in DBquadmesh
        ghosts_for_latest_mesh->SetComponentFromDouble(i,0,  false);
    }
    return data;
}

eavlDataSet *
eavlSiloImporter::GetUCDMesh(string nm)
{
    DBucdmesh *m = DBGetUcdmesh(file, nm.c_str());
    if (!m)
        return NULL;
    eavlDataSet *data = new eavlDataSet;
    data->SetNumPoints(m->nnodes);
    
    //Read the points.
    eavlCoordinatesCartesian *coords;
    if (m->ndims == 1)
    {
        coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X);
    }
    else if (m->ndims == 2)
    {
        coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y);
    }
    else if (m->ndims == 3)
    {
        coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);
    }
    else
    {
        THROW(eavlException,"unxpected number of dimensions");
    }

    for (int i = 0; i < m->ndims; i++)
    {
        coords->SetAxis(i, new eavlCoordinateAxisField("coords", i));
    }
    data->AddCoordinateSystem(coords);
    
    eavlArray *axisValues = new eavlFloatArray("coords", m->ndims);
    axisValues->SetNumberOfTuples(m->nnodes);

    for (int d = 0; d < m->ndims; d++)
        for (int i = 0; i < m->nnodes; i++)
        {
            if (m->datatype == DB_DOUBLE)
                axisValues->SetComponentFromDouble(i, d, ((double **)m->coords)[d][i]);
            else
                axisValues->SetComponentFromDouble(i, d, ((float **)m->coords)[d][i]);
        }
    
    eavlField *field = new eavlField(1,axisValues,eavlField::ASSOC_POINTS);
    data->AddField(field);

    eavlCellSetExplicit *cells = new eavlCellSetExplicit(nm + "_Cells",
                                                         m->zones->ndims);
    
    int nl_index = 0;
    eavlCellShape st;
    eavlExplicitConnectivity conn;
    int tmp_indices[8];
    for (int i = 0; i < m->zones->nshapes; i++)
    {
        for (int j = 0; j < m->zones->shapecnt[i]; j++)
        {
            int nNodes = -1;
            int *zone_nodes = &(m->zones->nodelist[nl_index]);
            if (m->zones->shapetype[i] == DB_ZONETYPE_TRIANGLE)
            {
                st = EAVL_TRI;
                nNodes = 3;
            }
            else if (m->zones->shapetype[i] == DB_ZONETYPE_QUAD)
            {
                st = EAVL_QUAD;
                nNodes = 4;
            }
            else if (m->zones->shapetype[i] == DB_ZONETYPE_TET)
            {
                st = EAVL_TET;
                nNodes = 4;

                int tetIndices[] = {1,0,2,3};
                MapZoneShape(4, tetIndices, zone_nodes, tmp_indices);
                zone_nodes = tmp_indices;
            }
            else if (m->zones->shapetype[i] == DB_ZONETYPE_PYRAMID)
            {
                st = EAVL_PYRAMID;
                nNodes = 5;

                int pyrIndices[] = {0,3,2,1,4};
                MapZoneShape(5, pyrIndices, zone_nodes, tmp_indices);
                zone_nodes = tmp_indices;
            }
            else if (m->zones->shapetype[i] == DB_ZONETYPE_PRISM)
            {
                st = EAVL_WEDGE;
                nNodes = 6;

                int wedgeIndices[] = {2,1,5,3,0,4};
                MapZoneShape(6, wedgeIndices, zone_nodes, tmp_indices);
                zone_nodes = tmp_indices;
            }
            else if (m->zones->shapetype[i] == DB_ZONETYPE_HEX)
            {
                st = EAVL_HEX;
                nNodes = 8;
            }
            else
                THROW(eavlException,"Unsupported silo zone type!");
            
            conn.AddElement(st, nNodes, zone_nodes);
            nl_index += nNodes;
        }
    }
    cells->SetCellNodeConnectivity(conn);
    data->AddCellSet(cells);

    ghosts_for_latest_mesh->SetNumberOfTuples(m->zones->nzones);
    for (int i=0; i<m->zones->nzones; i++)
    {
        bool g = (i < m->zones->min_index || i > m->zones->max_index);
        ghosts_for_latest_mesh->SetComponentFromDouble(i,0,  g);
    }

    DBFreeUcdmesh(m);
    return data;
}

eavlDataSet *
eavlSiloImporter::GetPtMesh(string nm)
{
    DBpointmesh *m = DBGetPointmesh(file, nm.c_str());
    if (!m)
        return NULL;
    eavlDataSet *data = new eavlDataSet;
    data->SetNumPoints(m->nels);
    
    //Read the points.
    eavlCoordinatesCartesian *coords;
    if (m->ndims == 1)
    {
        coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X);
    }
    else if (m->ndims == 2)
    {
        coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y);
    }
    else if (m->ndims == 3)
    {
        coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);
    }
    else
    {
        THROW(eavlException,"unxpected number of dimensions");
    }

    for (int i = 0; i < m->ndims; i++)
    {
        coords->SetAxis(i, new eavlCoordinateAxisField("coords", i));
    }
    data->AddCoordinateSystem(coords);
    
    eavlArray *axisValues = new eavlFloatArray("coords", m->ndims);
    axisValues->SetNumberOfTuples(m->nels);

    for (int d = 0; d < m->ndims; d++)
        for (int i = 0; i < m->nels; i++)
        {
            if (m->datatype == DB_DOUBLE)
                axisValues->SetComponentFromDouble(i, d, ((double **)m->coords)[d][i]);
            else
                axisValues->SetComponentFromDouble(i, d, ((float **)m->coords)[d][i]);
        }
    
    eavlField *field = new eavlField(1,axisValues,eavlField::ASSOC_POINTS);
    data->AddField(field);

    DBFreePointmesh(m);
    return data;
}

eavlDataSet *
eavlSiloImporter::GetMesh(const string &meshname, int chunk)
{
    ///\todo: use mesh name
    if (multiMeshes.count(meshname) > 0)
        return GetMultiMesh(meshname, chunk);

    for (size_t i=0; i<quadMeshes.size(); i++)
        if (quadMeshes[i] == meshname)
            return GetQuadMesh(meshname);

    for (size_t i=0; i<ucdMeshes.size(); i++)
        if (ucdMeshes[i] == meshname)
            return GetUCDMesh(meshname);

    for (size_t i=0; i<ptMeshes.size(); i++)
        if (ptMeshes[i] == meshname)
            return GetPtMesh(meshname);

    return NULL;
}

eavlField *
eavlSiloImporter::GetField(const string &name, const string &meshname, int chunk)
{
    string varPath = name;
    
    if (!multiMeshes.empty())
    {
        for (map<string, vector<string> >::iterator it = multiVars.begin(); it != multiVars.end(); it++)
        {
            if (name == it->first)
            {
                varPath = it->second[chunk];
                break;
                
            }
        }
    }
    if (varPath == "EMPTY")
        return NULL;

    eavlField *field = NULL;
    if (name == ".ghost")
    {
        eavlByteArray *ghost = new eavlByteArray(".ghost",1);
        ghost->SetNumberOfTuples(ghosts_for_latest_mesh->GetNumberOfTuples());
        for (int i=0; i<ghosts_for_latest_mesh->GetNumberOfTuples(); i++)
            ghost->SetComponentFromDouble(i,0,
                           ghosts_for_latest_mesh->GetComponentAsDouble(i,0));
        field = new eavlField(0, ghost, eavlField::ASSOC_CELL_SET, meshname+"_Cells");
        return field;
    }

    DBObjectType varType = DBInqVarType(file, varPath.c_str());
    if (varType == DB_QUADVAR)
    { 
        DBquadvar *v = DBGetQuadvar(file, varPath.c_str());
        eavlArray *arr = ReadValues(v->datatype, name, v->vals, v->nvals, v->nels);
        if (v->align[0] == 0.)
            field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
        else
            field = new eavlField(0, arr, eavlField::ASSOC_CELL_SET, meshname+"_Cells");
        DBFreeQuadvar(v);
    }
    else if (varType == DB_UCDVAR)
    {
        DBucdvar *v = DBGetUcdvar(file, varPath.c_str());
        eavlArray *arr = ReadValues(v->datatype, name, v->vals, v->nvals, v->nels);
        if (v->centering == DB_NODECENT)
            field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
        else
            field = new eavlField(0, arr, eavlField::ASSOC_CELL_SET, meshname+"_Cells");
        DBFreeUcdvar(v);
    }
    else if (varType == DB_POINTVAR)
    {
        DBmeshvar *v = DBGetPointvar(file, varPath.c_str());
        eavlArray *arr = ReadValues(v->datatype, name, v->vals, v->nvals, v->nels);
        field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
        DBFreeMeshvar(v);
    }
    else
        THROW(eavlException,"UNSUPPORTED VARIABLE TYPE.");

    return field;
}

vector<string>
eavlSiloImporter::GetMeshList()
{
    vector<string> meshes;

    for (map<string, vector<string> >::iterator it = multiMeshes.begin(); it != multiMeshes.end(); it++)
    {
        meshes.push_back(it->first);
    }

    for (size_t i = 0; i < quadMeshes.size(); i++)
    {
        if (meshesToHide.count(quadMeshes[i]) <= 0)
            meshes.push_back(quadMeshes[i]);
    }

    for (size_t i = 0; i < ucdMeshes.size(); i++)
    {
        if (meshesToHide.count(ucdMeshes[i]) <= 0)
            meshes.push_back(ucdMeshes[i]);
    }

    for (size_t i = 0; i < ptMeshes.size(); i++)
    {
        if (meshesToHide.count(ptMeshes[i]) <= 0)
            meshes.push_back(ptMeshes[i]);
    }

    return meshes;
}

vector<string>
eavlSiloImporter::GetFieldList(const string &meshname)
{
    bool haveGhost = false;

    vector<string> fields;
    for (map<string, vector<string> >::iterator it = multiVars.begin(); it != multiVars.end(); it++)
    {
        //cerr << "meshname="<<meshname<<"\n";
        //cerr << "it->first="<<it->first<<"\n";
        //cerr << "it->second="<<it->second<<"\n";
        //cerr << "meshForVar[RemoveSlash(it->second[0])]="<<meshForVar[RemoveSlash(it->second[0])]<<endl;
        //cerr << "multiMeshes[meshname]="<<multiMeshes[meshname]<<endl;

        if (multiMeshes.count(meshname) <= 0)
            continue; // we didn't read a multimesh with this name

        if (meshForVar[RemoveSlash(it->second[0])] ==
            RemoveSlash(multiMeshes[meshname][0]))
        {
            fields.push_back(it->first);
            ///\todo: we don't know if we need this without checking...
            haveGhost = true;
        }
    }

    for (size_t i = 0; i < quadVars.size(); i++)
    {
        if (meshForVar[quadVars[i]] == meshname)
        {
            fields.push_back(quadVars[i]);
            haveGhost = true;
        }
    }

    for (size_t i = 0; i < ucdVars.size(); i++)
    {
        if (meshForVar[ucdVars[i]] == meshname)
        {
            fields.push_back(ucdVars[i]);
            haveGhost = true;
        }
    }

    for (size_t i = 0; i < ptVars.size(); i++)
    {
        if (meshForVar[ptVars[i]] == meshname)
        {
            fields.push_back(ptVars[i]);
        }
    }

    if (haveGhost)
        fields.push_back(".ghost");


    return fields;
}
void
eavlSiloImporter::Print()
{
    if (!multiMeshes.empty())
    {
        cout<<"MultiMeshes -----"<<endl;
        for (map<string, vector<string> >::iterator it = multiMeshes.begin(); it != multiMeshes.end(); it++)
        {
            cout<<"  "<<it->first<<" [";
            for (size_t i = 0; i < it->second.size(); i++)
                cout<<it->second[i]<<" ";
            cout<<"]"<<endl;
        }
    }
    
    if (!quadMeshes.empty())
    {
        cout<<"QuadMeshes -----"<<endl;
        for (size_t i = 0; i < quadMeshes.size(); i++)
            cout<<" "<<quadMeshes[i]<<endl;
    }
    if (!ucdMeshes.empty())
    {
        cout<<"UCDMeshes -----"<<endl;
        for (size_t i = 0; i < ucdMeshes.size(); i++)
            cout<<" "<<ucdMeshes[i]<<endl;
    }

    if (!multiVars.empty())
    {
        cout<<"MultiVars -----"<<endl;
        for (map<string, vector<string> >::iterator it = multiVars.begin(); it != multiVars.end(); it++)
        {
            cout<<"  "<<it->first<<" [";
            for (size_t i = 0; i < it->second.size(); i++)
                cout<<it->second[i]<<" ";
            cout<<"]"<<endl;
        }
    }
    
    if (!quadVars.empty())
    {
        cout<<"QuadVars -----"<<endl;
        for (size_t i = 0; i < quadVars.size(); i++)
            cout<<" "<<quadVars[i]<<endl;
    }
    if (!ucdVars.empty())
    {
        cout<<"UCDVars -----"<<endl;
        for (size_t i = 0; i < ucdVars.size(); i++)
            cout<<" "<<ucdVars[i]<<endl;
    }
}

vector<string>
eavlSiloImporter::GetCellSetList(const std::string &mesh)
{
    return vector<string>(1, mesh + "_Cells");
}
