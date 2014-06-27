// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlPixieImporter.h"

eavlPixieImporter::eavlPixieImporter(const string &filename) :
    eavlADIOSImporter(filename)
{
}

eavlPixieImporter::~eavlPixieImporter()
{
}

vector<string>
eavlPixieImporter::GetFieldList()
{
    vector<string> fields;
    map<string, ADIOSVar>::iterator it;
    for (it = file->variables.begin(); it != file->variables.end(); it++)
    {
	if (it->second.dim == 3)
	{
	    fields.push_back(it->second.name);
	}
    }
    return fields;
}

eavlDataSet *
eavlPixieImporter::GetMesh(int)
{
    ADIOSVar v;
    map<string, ADIOSVar>::iterator it;
    for (it = file->variables.begin(); it != file->variables.end(); it++)
    {
	if (it->second.dim == 3)
	    return CreateRectilinearGrid(it->second);
    }
    
    return NULL;
}
