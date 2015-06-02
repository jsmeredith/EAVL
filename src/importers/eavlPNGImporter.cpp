// Copyright 2010-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlPNGImporter.h"
#include "lodepng.h"

eavlPNGImporter::eavlPNGImporter(const string &filename)
{
  std::ifstream file(filename.c_str(), std::ios::in|std::ios::binary|std::ios::ate);

  std::streamsize size = 0;
  if(file.seekg(0, std::ios::end).good()) size = file.tellg();
  if(file.seekg(0, std::ios::beg).good()) size -= file.tellg();

  if (size <= 0)
      THROW(eavlException, "Error reading file");

  std::vector<unsigned char> buffer(size);
  file.read((char*)(&buffer[0]), size);
  file.close();

  int error = lodepng::decode(rgba, width, height, &buffer[0], buffer.size());
  if (error != 0)
      THROW(eavlException, "Error decoding file to PNG");
}

eavlPNGImporter::eavlPNGImporter(const unsigned char *buffer, long long size)
{
  int error = lodepng::decode(rgba, width, height, buffer, size);
  if (error != 0)
      THROW(eavlException, "Error decoding file to PNG");
}


eavlPNGImporter::~eavlPNGImporter()
{
}

vector<string> 
eavlPNGImporter::GetFieldList(const std::string &mesh)
{
    vector<string> retval;
    retval.push_back("r");
    retval.push_back("g");
    retval.push_back("b");
    retval.push_back("a");
    ///\todo: support rgb, rgba, i, ia?
    return retval;
}

vector<string>
eavlPNGImporter::GetCellSetList(const std::string &mesh)
{
    return std::vector<string>(1, "pixels");
}


eavlDataSet*
eavlPNGImporter::GetMesh(const string &name, int chunk)
{
    eavlDataSet *data = new eavlDataSet;

    vector<vector<double> > coords;
    vector<string> coordNames;

    vector<double> x(width+1);
    for (unsigned int i=0; i<=width; i++)
        x[i] = i;

    vector<double> y(height+1);
    for (unsigned int i=0; i<=height; i++)
        y[i] = i;

    coordNames.push_back("x");
    coordNames.push_back("y");
    coords.push_back(x);
    coords.push_back(y);
    
    AddRectilinearMesh(data, coords, coordNames, true, "pixels");

    return data;
}

eavlField*
eavlPNGImporter::GetField(const string &name, const string &mesh, int chunk)
{
    int component = 0;
    if (name == "r")
        component = 0;
    if (name == "g")
        component = 1;
    if (name == "b")
        component = 2;
    if (name == "a")
        component = 3;

    int n = width*height;
    eavlByteArray *arr = new eavlByteArray(name, 1, n);
    for (int i=0; i<n; ++i)
    {
        arr->SetValue(i, rgba[4*i + component]);
    }

    return new eavlField(0, arr, eavlField::ASSOC_CELL_SET, "pixels");
}

