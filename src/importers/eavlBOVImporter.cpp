// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlBOVImporter.h"
#include "eavlException.h"
#include <string.h>

#ifdef HAVE_ZLIB
#include <zlib.h>
#endif

eavlBOVImporter::eavlBOVImporter(const string &filename)
{
    for (int i = 0; i < 3; i++)
    {
        dataSize[i] = 1;
        brickSize[i] = 1;
        brickOrigin[i] = 0.0;
        brickXAxis[i] = 0.0;
        brickYAxis[i] = 0.0;
        brickZAxis[i] = 0.0;
    }
    numComponents = 1;
    filePath = "";
    dataFilePattern = "";
    dataT = FLOAT;
    variable = "";
    numChunks = 0;
    nodalCentering = true;
    swapBytes = false;
    hasBoundaries = false;

    ReadTOC(filename);
}

eavlBOVImporter::~eavlBOVImporter()
{
}

int
eavlBOVImporter::GetNumChunks(const string&)
{
    return numChunks;
}

vector<string>
eavlBOVImporter::GetFieldList(const string&)
{
    vector<string> fields;

    fields.push_back(variable);
    return fields;
}


eavlDataSet *
eavlBOVImporter::GetMesh(const string&, int chunk)
{
    int nx = dataSize[0] / brickSize[0];
    int ny = dataSize[1] / brickSize[1];
    int nz = dataSize[2] / brickSize[2];

    int z_off = chunk / (nx*ny);
    int y_off = (chunk % (nx*ny)) / nx;
    int x_off = chunk % nx;

    float x_step = brickSize[0] / (nx);
    float y_step = brickSize[1] / (ny);
    float z_step = brickSize[2] / (nz);
    float x_start = brickOrigin[0] + x_step*x_off;
    float x_stop  = brickOrigin[0] + x_step*(x_off+1);
    float y_start = brickOrigin[1] + y_step*y_off;
    float y_stop  = brickOrigin[1] + y_step*(y_off+1);
    float z_start = brickOrigin[2] + z_step*z_off;
    float z_stop  = brickOrigin[2] + z_step*(z_off+1);

    if (hasBoundaries)
        THROW(eavlException,"ADD CODE FOR THIS!");
    int dx = brickSize[0];
    int dy = brickSize[1];
    int dz = brickSize[2];
    if (!nodalCentering)
    {
        dx += 1;
        dy += 1;
        dz += 1;
    }

    vector<vector<double> > coords;
    vector<string> coordNames;

    coords.resize(3);
    coordNames.resize(3);
    coordNames[0] = "XDir";
    coordNames[1] = "YDir";
    coordNames[2] = "ZDir";

    coords[0].resize(dx);
    for (int i = 0; i < dx; i++)
        coords[0][i] = x_start + i * (x_stop-x_start) / (dx-1);
    coords[0][dx-1] = x_stop;

    coords[1].resize(dy);
    for (int i = 0; i < dy; i++)
        coords[1][i] = y_start + i * (y_stop-y_start) / (dy-1);
    coords[1][dy-1] = y_stop;

    coords[2].resize(dz);
    for (int i = 0 ; i < dz ; i++)
        coords[2][i] = z_start + i * (z_stop-z_start) / (dz-1);
    coords[2][dz-1] = z_stop;

    eavlDataSet *data = new eavlDataSet;
    AddRectilinearMesh(data, coords, coordNames, true, "E");
    return data;
}

template<class T> static eavlFloatArray *
CopyValues(string nm, T *buff, int nTups, int nComps)
{
    eavlFloatArray *arr = new eavlFloatArray(nm, nComps);
    arr->SetNumberOfTuples(nTups);
    int idx = 0;
    for (int i = 0; i < nComps; i++)
    {
        for (int j = 0; j < nTups; j++, idx++)
            arr->SetComponentFromDouble(j, i, (double)(buff[idx]));
    }

    return arr;
}

eavlField *
eavlBOVImporter::GetField(const string &var, const string &mesh, int chunk)
{
    string fileName = DataFileFromChunk(chunk);
    bool gzipped = (fileName.length() > 3 && fileName.substr(fileName.length()-3) == ".gz");

    int nTuples = brickSize[0]*brickSize[1]*brickSize[2];

    size_t typeSz = SizeOfDataType();
    size_t sz = nTuples*numComponents*typeSz;
    void *buff = new void*[sz];
    if (gzipped)
    {
#ifdef HAVE_ZLIB
        gzFile fp = gzopen(fileName.c_str(), "r");
        size_t nread = gzread(fp, buff, sz);
        gzclose(fp);
        if (nread != sz)
            THROW(eavlException,"error reading "+fileName);
#else
        THROW(eavlException,"Found .gz BOV file, but BOV was not compiled with ZLIB support");
#endif
    }
    else
    {
        FILE *fp = fopen(fileName.c_str(), "rb");
        size_t nread = fread(buff, 1, sz, fp);
        fclose(fp);
        if (nread != sz)
            THROW(eavlException,"error reading "+fileName);
    }

    if (swapBytes)
        cerr<<"SWAP BYTES NOT SUPPORTED. But it still seems to work... ????"<<endl;

    eavlArray *arr;
    if (dataT == FLOAT)
        arr = CopyValues(var, (float *)buff, nTuples, numComponents);
    else if (dataT == DOUBLE)
        arr = CopyValues(var, (double *)buff, nTuples, numComponents);
    else if (dataT == INT)
        arr = CopyValues(var, (int *)buff, nTuples, numComponents);
    else if (dataT == SHORT)
        arr = CopyValues(var, (short *)buff, nTuples, numComponents);
    else if (dataT == BYTE)
        arr = CopyValues(var, (unsigned char *)buff, nTuples, numComponents);
    else
        THROW(eavlException, "Unknown data type in BOV file");

    eavlField *field = NULL;
    if (nodalCentering)
        field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
    else
        field = new eavlField(1, arr, eavlField::ASSOC_CELL_SET, "E");


    return field;
}

string
eavlBOVImporter::DataFileFromChunk(int chunk)
{
    string df;
    if (numChunks == 1)
        df = dataFilePattern;
    else
    {
        char str[512];
        sprintf(str, dataFilePattern.c_str(), chunk);
        df = str;
    }
    return filePath+df;
}

void
eavlBOVImporter::ReadTOC(const string &fn)
{
    size_t slashPos = fn.rfind("/");
    if (slashPos == 0)
        filePath = "./";
    else
        filePath = fn.substr(0, slashPos+1);

    FILE *fp = fopen(fn.c_str(), "r");

    char buff[1024];
    while (fgets(buff, 1024, fp) != NULL)
    {
        if (buff[0] == '\0' || buff[0] == '#')
            continue;
        //remove newline
        buff[strlen(buff)-1] = '\0';

        const char *key = "DATA_FILE: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            dataFilePattern = &buff[strlen(key)];
            continue;
        }

        key = "DATA SIZE: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            sscanf(&buff[strlen(key)], "%d %d %d", &dataSize[0], &dataSize[1], &dataSize[2]);
            continue;
        }
        key = "DATA FORMAT: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            string dataFormat = &buff[strlen(key)];

            continue;
        }
        key = "DATA_COMPONENTS: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            numComponents = atoi(&buff[strlen(key)]);
            continue;
        }
        key = "VARIABLE: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            char *ptr = &buff[strlen(key)];
            //Take off trailing and leading quotes.
            ptr++;
            ptr[strlen(ptr)-1] = '\0';
            variable = ptr;
            continue;
        }
        key = "CENTERING: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            if (strcasecmp(&buff[strlen(key)], "nodal") == 0)
                nodalCentering = true;
            else
                nodalCentering = false;

            continue;
        }
        key = "DATA_ENDIAN: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            bool isLittle;
            if (strcasecmp(&buff[strlen(key)], "little"))
                isLittle = true;
            else
                isLittle = false;

#ifdef WORDS_BIGENDIAN
            swapBytes = isLittle;
#else
            swapBytes = !isLittle;
#endif
            continue;
        }

        key = "DATA_BRICKLETS: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            sscanf(&buff[strlen(key)], "%d %d %d", &brickSize[0], &brickSize[1], &brickSize[2]);
            continue;
        }

        key = "BRICK_ORIGIN: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            sscanf(&buff[strlen(key)], "%f %f %f", &brickOrigin[0], &brickOrigin[1], &brickOrigin[2]);
            continue;
        }
        key = "BRICK X_AXIS: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            sscanf(&buff[strlen(key)], "%f %f %f", &brickXAxis[0], &brickXAxis[1], &brickXAxis[2]);
            continue;
        }
        key = "BRICK Y_AXIS: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            sscanf(&buff[strlen(key)], "%f %f %f", &brickYAxis[0], &brickYAxis[1], &brickYAxis[2]);
            continue;
        }
        key = "BRICK Z_AXIS: ";
        if (strncmp(buff, key, strlen(key)) == 0)
        {
            sscanf(&buff[strlen(key)], "%f %f %f", &brickZAxis[0], &brickZAxis[1], &brickZAxis[2]);
            continue;
        }
    }
    fclose(fp);

    int nX = dataSize[0], nY = dataSize[1], nZ = dataSize[2];
    if (brickSize[0] > 1 || brickSize[1] > 1 || brickSize[2] > 1)
        numChunks = (nX/brickSize[0])*(nY/brickSize[1])*(nZ/brickSize[2]);
    else
        numChunks = 1;
}

size_t
eavlBOVImporter::SizeOfDataType()
{
    if (dataT == FLOAT)
        return sizeof(float);
    if (dataT == DOUBLE)
        return sizeof(double);
    if (dataT == BYTE)
        return sizeof(char);
    if (dataT == INT)
        return sizeof(int);
    if (dataT == SHORT)
        return sizeof(short);
    return 0;
}
