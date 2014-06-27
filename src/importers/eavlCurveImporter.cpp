// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.

#include "eavlCurveImporter.h"

#include <fstream>
#include <sstream>

using namespace std;

static string GenerateName(int count)
{
    char tmp[256];
    sprintf(tmp, "curve%03d", count);
    return tmp;
}


eavlCurveImporter::eavlCurveImporter(const string &filename)
{
    ifstream in(filename.c_str(), ios::in);

    string curName = "";
    vector<double> curX;
    vector<double> curY;

    char line[4096];
    in.getline(line,4096);
    while (in.good() && !in.eof())
    {
        if (line[0] == '#')
        {
            if (curX.size() > 0)
            {
                if (curName == "")
                    curName = GenerateName(curveX.size());
                curveX[curName].insert(curveX[curName].begin(),
                                       curX.begin(), curX.end());;
                curveY[curName].insert(curveY[curName].begin(),
                                       curY.begin(), curY.end());;
            }
            curName = "";
            curX.clear();
            curY.clear();

            istringstream istr(&(line[1]));
            istr >> curName;
        }
        else
        {
            double x, y;
            istringstream istr(line);
            istr >> x >> y;
            curX.push_back(x);
            curY.push_back(y);
        }

        in.getline(line,4096);
    }

    // EOF (or other problem); add the existing curve to the list
    if (curX.size() > 0)
    {
        if (curName == "")
            curName = GenerateName(curveX.size());
        curveX[curName].insert(curveX[curName].begin(),
                               curX.begin(), curX.end());
        curveY[curName].insert(curveY[curName].begin(),
                               curY.begin(), curY.end());
    }


    in.close();
}

eavlCurveImporter::~eavlCurveImporter()
{
}

eavlDataSet*
eavlCurveImporter::GetMesh(const string &name, int chunk)
{
    eavlDataSet *data = new eavlDataSet;
    vector< vector<double> > coords;
    coords.resize(1);
    coords[0].insert(coords[0].end(), curveX[name].begin(), curveX[name].end());
    vector<string> coordNames(1, "X");
    AddRectilinearMesh(data, coords, coordNames,
                       true, // note: if this is false, empty result in GetCellSetList
                       "cells");
    return data;
}

eavlField*
eavlCurveImporter::GetField(const string &name, const string &mesh, int chunk)
{
    std::vector<double> &vals = curveY[mesh];
    int n = vals.size();
    eavlFloatArray *arr = new eavlFloatArray(name, 1, n);
    for (int i=0; i<n; ++i)
        arr->SetValue(i, vals[i]);
    return new eavlField(1, arr, eavlField::ASSOC_POINTS);
}

vector<string>
eavlCurveImporter::GetMeshList()
{
    vector<string> retval;
    for (map<string, vector<double> >::iterator it = curveX.begin();
         it != curveX.end(); ++it)
    {
        retval.push_back(it->first);
    }
    return retval;
}

vector<string>
eavlCurveImporter::GetFieldList(const std::string &mesh)
{
    vector<string> retval;
    retval.push_back("vals");
    return retval;
}

vector<string>
eavlCurveImporter::GetCellSetList(const std::string &mesh)
{
    vector<string> retval;
    retval.push_back("cells");
    return retval;
}
