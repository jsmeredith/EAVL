// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlADIOSImporter.h"
#include "eavlCoordinates.h"
#include "eavlCellSetAllStructured.h"

eavlADIOSImporter::eavlADIOSImporter(const string &filename)
{
    file = new ADIOSFileObject(filename);
    Import();
}

eavlADIOSImporter::~eavlADIOSImporter()
{
    if (file)
    {
	file->Close();
	delete file;
	file = NULL;
    }
}

void
eavlADIOSImporter::Import()
{
    file->Open();
}

eavlDataSet *
eavlADIOSImporter::GetMesh(const string &mesh, int)
{
    if (file->variables.empty())
        return NULL;
    
    map<string, ADIOSVar>::iterator it = file->variables.begin();
    
    ADIOSVar v = it->second;

    eavlDataSet *data = new eavlDataSet;
    vector<vector<double> > coords;
    vector<string> coordNames;
    
    if (v.dim > 3 || v.dim < 1)
        return NULL;
    

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

eavlField *
eavlADIOSImporter::GetField(const string &name, const string &mesh, int)
{
    if (file->variables.empty())
	return NULL;
    
    map<string, ADIOSVar>::iterator it = file->variables.find(name);
    if (it == file->variables.end())
	return NULL;
    
    eavlFloatArray *arr = NULL;
    file->ReadVariable(name, 0, &arr);
    eavlField *field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
    return field;
}

vector<string>
eavlADIOSImporter::GetFieldList(const string &mesh)
{
    vector<string> fields;
    map<string, ADIOSVar>::iterator it;
    for (it = file->variables.begin(); it != file->variables.end(); it++)
    {
        fields.push_back(it->second.name);
	//cout<<"var: "<<it->second.name<<endl;
    }
    return fields;
}

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
