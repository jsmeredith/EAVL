// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlChimeraImporter.h"
#include "eavlCoordinates.h"
#include "eavlCellSetAllStructured.h"
#include "eavlException.h"

/*
static bool
IsContained(string str, vector<string> &v)
{
    for(vector<string>::const_iterator it = v.begin(); it!= v.end(); ++it)
        if (str == *it)
            return true;
    
    return false;
}

static string
RemoveSlash(string str)
{
    if (str[0] == '/')
        str = str.substr(1, str.size());
    return str;
}
*/

eavlChimeraImporter::eavlChimeraImporter(const string &filename)
{
    file = DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ);
    if (file == NULL)
        THROW(eavlException,"Couldn't open Chimera file!\n");
    
    Import();
}

eavlChimeraImporter::~eavlChimeraImporter()
{
    if (file)
    {
        DBClose(file);
        file = NULL;
    }
}

static void
AddUnique(const string &str, vector<string> &v)
{
    for (size_t i = 0; i < v.size(); i++)
        if (str == v[i])
            return;
    v.push_back(str);
}

static void
PrintStrVec(const vector<string> &v)
{
    cout<<"[";
    if (v.size() > 0)
    {
        for (size_t i = 0; i < v.size()-1; i++)
            cout<<v[i]<<", ";
        cout<<v[v.size()-1];
    }
    cout<<"]"<<endl;
}

void
eavlChimeraImporter::Import()
{
    data = new eavlDataSet;

    DBtoc *toc = DBGetToc(file);
    if (toc->nqmesh < 1)
        THROW(eavlException,"No quad mesh in chimera file.");

    //Figure out what kind of mesh we have.
    vector<pair<string,int> > meshDims;
    
    //Fill in the spatial
    DBquadmesh *m = DBGetQuadmesh(file, toc->qmesh_names[0]);
    int numSpatial = m->ndims;
    for (int i = 0; i < numSpatial; i++)
        meshDims.push_back(pair<string,int>(m->labels[i], m->dims[i]));
    DBFreeQuadmesh(m);
    
    //Read the neutrino data.
    DBSetDir(file, "neutrino");
    toc = DBGetToc(file);
    
    //See what we have.
    vector<string> neutFlav, energyGrp, vars;
    for (int vi = 0; vi < toc->nqvar; vi++)
    {
        //psi0_c_F1_E04
        string varNm = toc->qvar_names[vi];
        string v = varNm.substr(0, 6); //"psi?_?"
        string F = varNm.substr(8,1);   //"1"
        string E = varNm.substr(11,2);   //"04"
        
        AddUnique(v, vars);
        AddUnique(F, neutFlav);
        AddUnique(E, energyGrp);
    }
    //cout<<"Flav: "; PrintStrVec(neutFlav);
    //cout<<"EG  : "; PrintStrVec(energyGrp);
    //cout<<"VARS: "; PrintStrVec(vars);

    if (! neutFlav.empty())
        meshDims.push_back(pair<string,int> ("neutrinoFlavor", neutFlav.size()));
    if (! energyGrp.empty())
        meshDims.push_back(pair<string,int> ("energyGroup", energyGrp.size()));

    vector<vector<double> > coords;
    vector<string> coordNames;


    //Create the high D mesh.
    coords.resize(meshDims.size());
    coordNames.resize(meshDims.size());
    for (size_t i = 0; i < meshDims.size(); i++)
    {
        coordNames[i] = meshDims[i].first;
        coords[i].resize(meshDims[i].second);
    }
    PrintStrVec(coordNames);

    //Add the spatial coordinates.
    DBSetDir(file, "/");
    toc = DBGetToc(file);
    m = DBGetQuadmesh(file, toc->qmesh_names[0]);
    int i;
    for (i = 0; i < numSpatial; i++)
    {
        if (m->datatype == DB_DOUBLE)
        {
            for (int j = 0; j < m->dims[i]; j++)
                coords[i][j] = ((double**)m->coords)[i][j];
        }
        else
        {
            for (int j = 0; j < m->dims[i]; j++)
                coords[i][j] = (double) ((float**)m->coords)[i][j];
        }
    }
    DBFreeQuadmesh(m);

    //Add the higher D coordinates. (***note  i not set to 0).
    for (; i < (int)meshDims.size(); i++)
    {
        for (int j = 0; j < meshDims[i].second; j++)
            coords[i][j] = (double)j;
    }

    AddRectilinearMesh(data, coords, coordNames, true, "chimera_Cells");


    //Add the variables.
    DBSetDir(file, "neutrino");
    toc = DBGetToc(file);

    int numCells = 1;
    for (size_t i = 0; i < meshDims.size(); i++)
        numCells *= meshDims[i].second;
    
    for (size_t i = 0; i < vars.size(); i++)
    {
        eavlFloatArray *arr = new eavlFloatArray(vars[i], 1);
        arr->SetNumberOfTuples(numCells);
        
        const char *varPat = "%s_F%s_E%s";
        char varFileName[64];
        
        int idx = 0;
        for (size_t e = 0; e < energyGrp.size(); e++)
        {
            for (size_t f = 0; f < neutFlav.size(); f++)
            {
                sprintf(varFileName, varPat, vars[i].c_str(), neutFlav[f].c_str(), energyGrp[e].c_str());
                DBquadvar *v = DBGetQuadvar(file, varFileName);
                //cout<<"    "<<varFileName<<" "<<v->nels<<endl;
                int idx2 = 0;
                for (int z = 0; z < meshDims[2].second; z++)
                    for (int y = 0; y < meshDims[1].second; y++)                
                        for (int x = 0; x < meshDims[0].second; x++)
                        {
                            arr->SetComponentFromDouble(idx,0, (double)((float **)v->vals)[0][idx2]);
                            idx++;
                            idx2++;
                        }
                DBFreeQuadvar(v);
            }
        }
        
        DBquadvar *v0 = DBGetQuadvar(file, toc->qvar_names[0]);
        eavlField *field = NULL;
        if (v0->align[0] == 0.)
            field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
        else
            field = new eavlField(0, arr, eavlField::ASSOC_CELL_SET, "chimera_Cells");
        data->AddField(field);

        DBFreeQuadvar(v0);
    }
}

eavlDataSet *
eavlChimeraImporter::GetMesh(const string &mesh, int)
{
    return data;
}

eavlField *
eavlChimeraImporter::GetField(const string &name, const string &mesh, int)
{
    return data->GetField(name);
}

vector<string>
eavlChimeraImporter::GetFieldList(const string &mesh)
{
    vector<string> fields;
    for (int i = 0; i < data->GetNumFields(); i++)
        fields.push_back(data->GetField(i)->GetArray()->GetName());

    return fields;
}
