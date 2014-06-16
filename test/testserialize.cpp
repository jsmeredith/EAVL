// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlFilter.h"
#include "eavlDataSet.h"
#include "eavlTimer.h"
#include "eavlException.h"

#include "eavlImporterFactory.h"


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
        if (argc != 2)
            THROW(eavlException,"Incorrect number of arguments");

        int meshindex = 0;
        eavlDataSet *data = ReadMeshFromFile(argv[1], meshindex);
	
	ofstream f1("dump.dat", ios::binary);
	eavlStream s1(f1);
	data->serialize(s1);
	f1.close();
	//To create the baseline...
	//data->PrintSummary(cout);

	eavlDataSet *d2 = new eavlDataSet;
	ifstream f2("dump.dat", ios::binary);
	eavlStream s2(f2);
	d2->deserialize(s2);
	d2->PrintSummary(cout);
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <infile>\n";
        return 1;
    }
    return 0;
}
