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
	
	ofstream f1("dump.dat");
	eavlStream s1(f1);
	data->serialize(s1);
	f1.close();
	data->PrintSummary(cout);

	cout<<"*******************"<<endl;
	cout<<"READ BACK THE FILE:"<<endl;
	cout<<"*******************"<<endl;
#if 1
	eavlDataSet *d2 = new eavlDataSet;
	ifstream f2("dump.dat");
	eavlStream s2(f2);
	d2->deserialize(s2);
	d2->PrintSummary(cout);
#endif





#if 0
	ofstream f1("dump.dat");
	eavlStream s1(f1);
	string str;
	int n;
	bool b;
	vector<int> a;

	str = "abc";
	n = 10;
	a.push_back(2);
	a.push_back(3);
	b = false;
	s1 <<str<<n<<a<<b;
	f1.close();
		
	ifstream t("dump.dat");
	eavlStream ss(t);

	ss >> str;
	ss >> n;
	ss >> a;
	ss >> b;
	cout<<str<<" "<<n<<" "<<a.size()<<" "<<b<<endl;
#endif

#if 0

	ifstream t("dump.dat");
	eavlStream ss(t);
	
	//data->PrintSummary(cout);
	cout<<"READ BACK THE FILE:"<<endl;
	string str;
	int n, n1, n2, n3;
	bool b;
	size_t sz;
	vector<float> a;
	ss >> str;
	ss >> n;
	ss >> sz;
	ss >> sz;
	ss >> str;
	ss >> n >> n >> n;
	ss >> str;
	ss >> str;
	ss >> n;
	ss >> n;
	ss >> b;
	ss >> a;
	/*
	cout<<str<<endl; //eavlDataSet
	cout<<n<<endl;   //npoints
	cout<<sz<<endl;  // discreetcoords
	*/
	//ss >> str;  cout<<"field: "<<str<<endl;
	//ss >> n >> n1 >> n2;
#endif

	/*
        while (data)
        {
            data->PrintSummary(cout);
            ++meshindex;
            data = ReadMeshFromFile(argv[1], meshindex);
            if (data)
                cout << "\n\n";
        }
	*/
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: "<<argv[0]<<" <infile>\n";
        return 1;
    }
    return 0;
}
