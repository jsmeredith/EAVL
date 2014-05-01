// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlVTKImporter.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlException.h"

#include <cstring>

bool debug =false;


template <int T>
inline void byte_swap_element(char *p);

template <>
inline void byte_swap_element<1>(char *)
{
}

template <>
inline void byte_swap_element<2>(char *p)
{
    char val;
    val  = p[0];
    p[0] = p[1];
    p[1] = val;
}

template <>
inline void byte_swap_element<4>(char *p)
{
    char val;
    val  = p[0];
    p[0] = p[3];
    p[3] = val;
    val  = p[1];
    p[1] = p[2];
    p[2] = val;
}

template <>
inline void byte_swap_element<8>(char *p)
{
    char val;
    val  = p[0];
    p[0] = p[7];
    p[7] = val;
    val  = p[1];
    p[1] = p[6];
    p[6] = val;
    val  = p[2];
    p[2] = p[5];
    p[5] = val;
    val  = p[3];
    p[3] = p[4];
    p[4] = val;
}

// oddly, this is already defined.  neat!
//#define LITTLE_ENDIAN

template <class T>
inline void byte_swap(vector<T> &v)
{
#ifdef LITTLE_ENDIAN
    int n = v.size();
    for (int e=0; e<n; e++)
    {
        byte_swap_element<sizeof(T)>(reinterpret_cast<char*>(&(v[e])));
    }
#endif
}

// ----------------------------------------------------------------------------
// from vtkCellType.h:

// Linear cells
#define VTK_EMPTY_CELL     0
#define VTK_VERTEX         1
#define VTK_POLY_VERTEX    2
#define VTK_LINE           3
#define VTK_POLY_LINE      4
#define VTK_TRIANGLE       5
#define VTK_TRIANGLE_STRIP 6
#define VTK_POLYGON        7
#define VTK_PIXEL          8
#define VTK_QUAD           9
#define VTK_TETRA         10
#define VTK_VOXEL         11
#define VTK_HEXAHEDRON    12
#define VTK_WEDGE         13
#define VTK_PYRAMID       14
#define VTK_PENTAGONAL_PRISM 15
#define VTK_HEXAGONAL_PRISM  16

// Quadratic, isoparametric cells
#define VTK_QUADRATIC_EDGE       21
#define VTK_QUADRATIC_TRIANGLE   22
#define VTK_QUADRATIC_QUAD       23
#define VTK_QUADRATIC_TETRA      24
#define VTK_QUADRATIC_HEXAHEDRON 25
#define VTK_QUADRATIC_WEDGE      26
#define VTK_QUADRATIC_PYRAMID    27

// Special class of cells formed by convex group of points
#define VTK_CONVEX_POINT_SET 41

// Higher order cells in parametric form
#define VTK_PARAMETRIC_CURVE        51
#define VTK_PARAMETRIC_SURFACE      52
#define VTK_PARAMETRIC_TRI_SURFACE  53
#define VTK_PARAMETRIC_QUAD_SURFACE 54
#define VTK_PARAMETRIC_TETRA_REGION 55
#define VTK_PARAMETRIC_HEX_REGION   56

// Higher order cells
#define VTK_HIGHER_ORDER_EDGE        60
#define VTK_HIGHER_ORDER_TRIANGLE    61
#define VTK_HIGHER_ORDER_QUAD        62
#define VTK_HIGHER_ORDER_POLYGON     63
#define VTK_HIGHER_ORDER_TETRAHEDRON 64
#define VTK_HIGHER_ORDER_WEDGE       65
#define VTK_HIGHER_ORDER_PYRAMID     66 
#define VTK_HIGHER_ORDER_HEXAHEDRON  67
// ----------------------------------------------------------------------------

void toupper(char *p)
{
    while (*p != '\0')
    {
        if (*p >= 'a' && *p <= 'z')
            *p += int('A') - int('a');
        p++;
    }
}

void toupper(string &s)
{
    int n = s.length();
    while (n > 0)
    {
        --n;
        if (s[n] >= 'a' && s[n] <= 'z')
            s[n] += int('A') - int('a');
    }
}

// ----------------------------------------------------------------------------
eavlVTKImporter::DataType eavlVTKImporter::DataTypeFromString(const string &s)
{
    if (s=="BIT")            return dt_bit;
    if (s=="UNSIGNED_CHAR")  return dt_unsigned_char;
    if (s=="CHAR")           return dt_char;
    if (s=="UNSIGNED_SHORT") return dt_unsigned_short;
    if (s=="SHORT")          return dt_short;
    if (s=="UNSIGNED_INT")   return dt_unsigned_int;
    if (s=="INT")            return dt_int;
    if (s=="UNSIGNED_LONG")  return dt_unsigned_long;
    if (s=="LONG")           return dt_long;
    if (s=="FLOAT")          return dt_float;
    if (s=="DOUBLE")         return dt_double;
    THROW(eavlException,string("Unexpected type: ")+s);
}

string eavlVTKImporter::StringFromDataType(eavlVTKImporter::DataType dt)
{
    if (dt == dt_bit)            return "bit";
    if (dt == dt_unsigned_char)  return "unsigned_char";
    if (dt == dt_char)           return "char";
    if (dt == dt_unsigned_short) return "unsigned_short";
    if (dt == dt_short)          return "short";
    if (dt == dt_unsigned_int)   return "unsigned_int";
    if (dt == dt_int)            return "int";
    if (dt == dt_unsigned_long)  return "unsigned_long";
    if (dt == dt_long)           return "long";
    if (dt == dt_float)          return "float";
    if (dt == dt_double)         return "double";
    throw;
}

eavlVTKImporter::DataSetType
eavlVTKImporter::DataSetTypeFromString(const string &s)
{
    if (s == "STRUCTURED_POINTS") return DS_STRUCTURED_POINTS;
    if (s == "STRUCTURED_GRID")   return DS_STRUCTURED_GRID;
    if (s == "RECTILINEAR_GRID")  return DS_RECTILINEAR_GRID;
    if (s == "POLYDATA")          return DS_POLYDATA;
    if (s == "UNSTRUCTURED_GRID") return DS_UNSTRUCTURED_GRID;
    return DS_UNKNOWN;
}

string
eavlVTKImporter::StringFromDataSetType(eavlVTKImporter::DataSetType dst)
{
    if (dst == DS_STRUCTURED_POINTS) return "STRUCTURED_POINTS";
    if (dst == DS_STRUCTURED_GRID) return "STRUCTURED_GRID";
    if (dst == DS_RECTILINEAR_GRID) return "RECTILINEAR_GRID";
    if (dst == DS_POLYDATA) return "POLYDATA";
    if (dst == DS_UNSTRUCTURED_GRID) return "UNSTRUCTURED_GRID";
    return "UNKNOWN";
}



bool eavlVTKImporter::GetNextLine()
{
    // skip blank lines
    // turn keywords into uppercase
    strcpy(buff, "");
    if (debug) cerr << "++ initialized to null\n";
    while (buff[0] == '\0')
    {
        if (debug) cerr << "++ entered pass\n";
        is->getline(buff, 4096);
        if (debug) cerr << "++ got buff; len="<<strlen(buff)<<endl;
        if (debug) cerr << "++ buff first byte = "<<int(buff[0])<<endl;
        if (!(*is) || is->eof())
            return false;
        if (debug) cerr << "++ not eof or bad\n";
    }
    if (debug) cerr << "++ breaking out of loop\n";
    strcpy(bufforig,buff);
    toupper(buff);
    return true;
}

template <class IT, class OT>
inline void BinaryReadThenCopyToVector(istream *is, int n, vector<OT> &v)
{
    vector<IT> t(n);
    is->read(reinterpret_cast<char*>(&t[0]), sizeof(IT) * n);
    byte_swap(t); // meaningless for size 1 types, of course
    for (int i=0; i<n; i++) v[i] = OT(t[i]);
}

template <class IT>
inline void BinaryReadThenCopyToArray(istream *is, int nt, int nc, eavlArray *arr)
{
    vector<IT> t(nt*nc);
    is->read(reinterpret_cast<char*>(&t[0]), sizeof(IT) * nt*nc);
    byte_swap(t);
    for(int i = 0; i < nt; i++)
        for (int j = 0; j < nc; j++)
            arr->SetComponentFromDouble(i, j, t[i]);
}

void
eavlVTKImporter::ReadIntoArray(DataType dt, eavlArray *arr)
{
    int nt = arr->GetNumberOfTuples();
    int nc = arr->GetNumberOfComponents();

    if (binary)
    {
        switch (dt)
        {
          case dt_bit:
            THROW(eavlException,"don't know how to support bits in binary files");

          case dt_unsigned_char:
            BinaryReadThenCopyToArray<unsigned char>(is, nt, nc, arr);
            break;

          case dt_char:
            BinaryReadThenCopyToArray<char>(is, nt, nc, arr);
            break;

          case dt_unsigned_short:
            BinaryReadThenCopyToArray<unsigned short>(is, nt, nc, arr);
            break;

          case dt_short:
            BinaryReadThenCopyToArray<signed short>(is, nt, nc, arr);
            break;

          case dt_unsigned_int:
            BinaryReadThenCopyToArray<unsigned int>(is, nt, nc, arr);
            break;

          case dt_int:
            BinaryReadThenCopyToArray<signed int>(is, nt, nc, arr);
            break;

          case dt_unsigned_long:
            BinaryReadThenCopyToArray<unsigned long>(is, nt, nc, arr);
            break;

          case dt_long:
            BinaryReadThenCopyToArray<signed long>(is, nt, nc, arr);
            break;

          case dt_float:
            BinaryReadThenCopyToArray<float>(is, nt, nc, arr);
            break;

          case dt_double:
            BinaryReadThenCopyToArray<double>(is, nt, nc, arr);
            break;

          default:
            THROW(eavlException,"incorrect DataType for eavlArray");

        }       
        is->getline(buff, 4096); // skip the EOL
    }
    else
    {
        double v;
        for(int i = 0; i < nt; i++)
            for(int j = 0; j < nc; j++)
            {
                (*is) >> v;
                arr->SetComponentFromDouble(i, j, v);
            }
        is->getline(buff,4096); // skip the EOL
    }
}

template <class T>
void
eavlVTKImporter::ReadIntoVector(int n,DataType dt,vector<T> &v)
{
    v.resize(n);
    if (binary)
    {
        switch (dt)
        {
          case dt_bit:
            THROW(eavlException,"don't know how to support bits in binary files");

          case dt_unsigned_char:
            BinaryReadThenCopyToVector<unsigned char>(is, n, v);
            break;

          case dt_char:
            BinaryReadThenCopyToVector<char>(is, n, v);
            break;

          case dt_unsigned_short:
            BinaryReadThenCopyToVector<unsigned short>(is, n, v);
            break;

          case dt_short:
            BinaryReadThenCopyToVector<signed short>(is, n, v);
            break;

          case dt_unsigned_int:
            BinaryReadThenCopyToVector<unsigned int>(is, n, v);
            break;

          case dt_int:
            BinaryReadThenCopyToVector<signed int>(is, n, v);
            break;

          case dt_unsigned_long:
            BinaryReadThenCopyToVector<unsigned long>(is, n, v);
            break;

          case dt_long:
            BinaryReadThenCopyToVector<signed long>(is, n, v);
            break;

          case dt_float:
            BinaryReadThenCopyToVector<float>(is, n, v);
            break;

          case dt_double:
            BinaryReadThenCopyToVector<double>(is, n, v);
            break;

        }       
        is->getline(buff, 4096); // skip the EOL
    }
    else
    {
        for (int i=0; i<n; i++)
            (*is) >> v[i];
        is->getline(buff,4096); // skip the EOL
    }
}

/*
void
eavlVTKImporter::Print(ostream &out)
{
    out << "Comment: "<<comment<<endl;
    out << "Binary: "<<(binary ? "true" : "false")<<endl;
    out << "Structure: "<<StringFromDataSetType(structure)<<endl;
    if (structure == DS_STRUCTURED_GRID || structure == DS_RECTILINEAR_GRID)
    {
        out << "Dims: " << dims[0] << "," << dims[1] << "," << dims[2] << endl;
    }
    out << "NPoints:" <<npoints<<endl;
    out << "Points:"<<endl;
    out << "    " << x << endl;
    out << "    " << y << endl;
    out << "    " << z << endl;
    out << "NCells:" <<ncells<<endl;
    if (structure == DS_UNSTRUCTURED_GRID || structure == DS_POLYDATA)
    {
        out << "Cells:"<<endl;
        out << "    Connectivity["<<cell_connectivity.size()<<"] = ";
        PrintVectorSummary(out,cell_connectivity);
        out << endl;
        out << "    CellTypes["<<cell_types.size()<<"] = ";
        PrintVectorSummary(out,cell_types);
        out << endl;
    }
    out << "DataSet Arrays: "<< arrays_dataset.size()<<endl;
    for (unsigned int i=0; i<arrays_dataset.size(); i++)
        out << "    " << arrays_dataset[i] << endl;
    out << "Cell Arrays: "<< arrays_cells.size()<<endl;
    for (unsigned int i=0; i<arrays_cells.size(); i++)
        out << "    " << arrays_cells[i] << endl;
    out << "Point Arrays: "<< arrays_points.size()<<endl;
    for (unsigned int i=0; i<arrays_points.size(); i++)
        out << "    " << arrays_points[i] << endl;
}
*/

eavlVTKImporter::eavlVTKImporter(const string &filename)
{
    is = new ifstream(filename.c_str(), ios::in);
    if (is->fail())
        THROW(eavlException, string("Could not open file ")+filename);

    Import();
}

eavlVTKImporter::eavlVTKImporter(const char *data, size_t len)
{
    string str(data, len);
    is = new istringstream(str);
    Import();
}

eavlVTKImporter::~eavlVTKImporter()
{
    if (is)
        delete is;
    is = NULL;
}

void
eavlVTKImporter::Import()
{
    data = new eavlDataSet;

    ParseVersion();
    ParseHeader();
    ParseFormat();
    ParseStructure();
    ParseAttributes();
}

vector<string>
eavlVTKImporter::GetFieldList(const string &mesh)
{
    vector<string> retval;
    for (map<string,eavlField*>::iterator it = vars.begin();
         it != vars.end(); it++)
    {
        retval.push_back(it->first);
    }
    return retval;
}

vector<string>
eavlVTKImporter::GetCellSetList(const std::string &mesh)
{
    vector<string> retval;
    for (int i=0; i<data->GetNumCellSets(); i++)
    {
        retval.push_back(data->GetCellSet(i)->GetName());
    }
    return retval;
}


eavlDataSet *
eavlVTKImporter::GetMesh(const string &mesh, int chunk)
{
    return data;
}

eavlField *
eavlVTKImporter::GetField(const string &name, const string &mesh, int chunk)
{
    return vars[name];
}

// --------------------
void
eavlVTKImporter::ParseVersion()
{
    is->getline(buff, 4096);
    toupper(buff);
    string line(buff);
    if (line != "# VTK DATAFILE VERSION 1.0" &&
        line != "# VTK DATAFILE VERSION 2.0" &&
        line != "# VTK DATAFILE VERSION 3.0")
    {
        THROW(eavlException,
              string("Expected version 1.0, 2.0, or 3.0 in version.") +
              "  Got: '"+line+"'.");
    }
}

// --------------------
void
eavlVTKImporter::ParseHeader()
{
    is->getline(buff, 4096);
    comment = buff;
}

// --------------------
void
eavlVTKImporter::ParseFormat()
{
    is->getline(buff, 4096);
    toupper(buff);
    string format(buff);
    if (format == "ASCII")
        binary = false;
    else if (format == "BINARY")
        binary = true;
    else
        THROW(eavlException,"Didn't get ASCII or BINARY for format line.");
}
// --------------------

void
eavlVTKImporter::AddArray(eavlFloatArray *arr, eavlVTKImporter::Location loc)
{
    string name = arr->GetName();
    if (name == "xcoord" || name == "ycoord" || name == "zcoord" || name == "coords")
    {
        name = "file." + name;
        arr->SetName(name);
    }   

    // if we split cells by dimensionality, split the fields too
    if (loc == LOC_CELLS &&
        cell_to_cell_splitmap.size() != 0)
    {
        int counts[4] = {0,0,0,0};
        int n = cell_to_cell_splitmap.size();
        for (int i=0; i<n; i++)
        {
            counts[cell_to_cell_splitmap[i]]++;
        }

        int realCellIndex = -1;
        for (int f=0; f<4; f++)
        {
            if (counts[f] == 0)
                continue;

            realCellIndex++;
            int nc = arr->GetNumberOfComponents();
            eavlFloatArray *a = new eavlFloatArray(name,nc);
            a->SetNumberOfTuples(counts[f]);
            int ctr = 0;
            for (int i=0; i<n; i++)
            {
                if (cell_to_cell_splitmap[i] == f)
                {
                    for (int c=0; c<nc; c++)
                    {
                        a->SetComponentFromDouble(ctr, c, 
                                               arr->GetComponentAsDouble(i,c));
                    }
                    ctr++;
                }
            }

            ///\todo: note: this assumes you add arrays after creating cell sets
            eavlField *field = new eavlField(0, a, eavlField::ASSOC_CELL_SET,
                                             data->GetCellSet(realCellIndex)->GetName());
            vars[name] = field;
        }
    }
    else
    {
        eavlField *field = NULL;
        switch (loc)
        {
          case LOC_DATASET:
            field = new eavlField(0, arr, eavlField::ASSOC_WHOLEMESH);
            break;
          case LOC_POINTS:
            field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
            break;
          case LOC_CELLS:
            field = new eavlField(0, arr, eavlField::ASSOC_CELL_SET,
                                  data->GetCellSet(0)->GetName()); // if we're here, we only had one cell set
            break;
        }
        vars[name] = field;
    }
}

void
eavlVTKImporter::ParseFieldData(eavlVTKImporter::Location loc)
{
    // assume already filled BUFF
    istringstream sin(buff);
    string s;
    sin >> s;
    if (s != "FIELD")
        THROW(eavlException,string("Internal error; expected "
                                   "FIELD in already parsed line, got")+s);
    sin >> s; // get the name
    // ignore the name; it's not useful
    sin >> s;
    int narrays;
    narrays = atoi(s.c_str());
    if (narrays < 1)
        THROW(eavlException,string("Expected some number of arrays; got: ")+s);
    for (int i=0; i<narrays; i++)
    {
        string an;
        int    ac;
        int    at;
        string ad;

        *is >> an;
        *is >> ac;
        *is >> at;
        *is >> ad;
        toupper(ad);

        eavlFloatArray *arr = new eavlFloatArray(an,ac);
        arr->SetNumberOfTuples(at);
        //Changed to ReadIntoArray
        //int n = arr->GetNumberOfComponents() * arr->GetNumberOfTuples();
        //ReadIntoVector(n, DataTypeFromString(ad), arr->values);
        is->getline(buff, 4096); // skip the EOL
        
        ReadIntoArray(DataTypeFromString(ad), arr);

        AddArray(arr, loc);
    }
}

void
eavlVTKImporter::ParseScalars(eavlVTKImporter::Location loc)
{
    // assume already filled BUFF
    istringstream sin(bufforig);
    string s;
    sin >> s;
    toupper(s);
    if (s != "SCALARS")
        THROW(eavlException,string("Internal error; expected "
                                   "SCALARS in already parsed line, got")+s);

    string an;
    int    ac;
    string ad;

    sin >> an;
    sin >> ad;
    toupper(ad);
    ac = 0;
    sin >> ac;
    if (ac < 1)
        ac = 1;

    GetNextLine(); // read and ignore the lookup table

    eavlFloatArray *a = new eavlFloatArray(an,ac);

    int ntotalcells = 0;
    for (int i=0; i<data->GetNumCellSets(); i++)
        ntotalcells += data->GetCellSet(i)->GetNumCells();

    if (loc == LOC_CELLS)
        a->SetNumberOfTuples(ntotalcells);
    else if (loc == LOC_POINTS)
        a->SetNumberOfTuples(data->GetNumPoints());
    else
        THROW(eavlException,"internal error in ParseScalars; loc must be points or cells");

    //ReadIntoVector(a->GetNumberOfComponents()*a->GetNumberOfTuples(), DataTypeFromString(ad), a->values);
    ReadIntoArray(DataTypeFromString(ad), a);

    AddArray(a, loc);
}

void
eavlVTKImporter::ParseVectors(eavlVTKImporter::Location loc)
{
    // assume already filled BUFF
    istringstream sin(bufforig);
    string s;
    sin >> s;
    toupper(s);
    if (s != "VECTORS")
        THROW(eavlException,string("Internal error; expected "
                                   "VECTORS in already parsed line, got")+s);

    string an;
    int    ac;
    string ad;

    sin >> an;
    sin >> ad;
    toupper(ad);
    ac = 3;

    eavlFloatArray *a = new eavlFloatArray(an,ac);

    int ntotalcells = 0;
    for (int i=0; i<data->GetNumCellSets(); i++)
        ntotalcells += data->GetCellSet(i)->GetNumCells();

    if (loc == LOC_CELLS)
        a->SetNumberOfTuples(ntotalcells);
    else if (loc == LOC_POINTS)
        a->SetNumberOfTuples(data->GetNumPoints());
    else
        THROW(eavlException,"internal error in ParseVectors; loc must be points or cells");

    //ReadIntoVector(a->GetNumberOfComponents()*a->GetNumberOfTuples(), DataTypeFromString(ad), a->values);
    ReadIntoArray(DataTypeFromString(ad), a);

    AddArray(a, loc);
}

void
eavlVTKImporter::ParseNormals(eavlVTKImporter::Location loc)
{
    // assume already filled BUFF
    istringstream sin(bufforig);
    string s;
    sin >> s;
    toupper(s);
    if (s != "NORMALS")
        THROW(eavlException,string("Internal error; expected "
                                   "NORMALS in already parsed line, got")+s);

    string an;
    int    ac;
    string ad;

    sin >> an;
    sin >> ad;
    toupper(ad);
    ac = 3;

    eavlFloatArray *a = new eavlFloatArray(an,ac);

    int ntotalcells = 0;
    for (int i=0; i<data->GetNumCellSets(); i++)
        ntotalcells += data->GetCellSet(i)->GetNumCells();

    if (loc == LOC_CELLS)
        a->SetNumberOfTuples(ntotalcells);
    else if (loc == LOC_POINTS)
        a->SetNumberOfTuples(data->GetNumPoints());
    else
        THROW(eavlException,"internal error in ParseNormals; loc must be points or cells");

    //ReadIntoVector(a->GetNumberOfComponents()*a->GetNumberOfTuples(), DataTypeFromString(ad), a->values);
    ReadIntoArray(DataTypeFromString(ad), a);

    AddArray(a, loc);
}

// --------------------
void
eavlVTKImporter::ParseStructure()
{
    // header line
    GetNextLine();
    istringstream sin(buff);
    string s;
    sin >> s;
    if (s != "DATASET")
        THROW(eavlException,string("Expected DATASET at beginning of structure line; got ")+s);
    sin >> s;
    structure = DataSetTypeFromString(s);
    if (structure == DS_UNKNOWN)
        THROW(eavlException,string("Got unknown data set structure: ") + s);

    // get first line of actual structure *now*, since it may be field data
    GetNextLine();
    if (strncmp(buff, "FIELD", 5) == 0)
    {
        ParseFieldData(LOC_DATASET);
        GetNextLine();
    }

    // actual structure
    switch (structure)
    {
      case DS_UNKNOWN:
        throw;
      case DS_STRUCTURED_POINTS:
        Parse_Structured_Points();
        break;
      case DS_STRUCTURED_GRID:
        Parse_Structured_Grid();
        break;
      case DS_RECTILINEAR_GRID:
        Parse_Rectilinear_Grid();
        break;
      case DS_POLYDATA:
        Parse_Polydata();
        break;
      case DS_UNSTRUCTURED_GRID:
        Parse_Unstructured_Grid();
        break;
    }
}

void
eavlVTKImporter::ParsePoints(eavlLogicalStructureRegular *log)
{
    //GetNextLine(); // assume it's read already
    istringstream sin(buff);
    string s;
    sin >> s;
    if (s != "POINTS")
        THROW(eavlException,string("Expected POINTS, got ")+s);
    int npoints;
    sin >> npoints;
    sin >> s;
    data->SetNumPoints(npoints);
    DataType dt = DataTypeFromString(s);
    vector<double> vals;
    ReadIntoVector(npoints*3, dt, vals);

#if 0
    eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(log,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);

    data->AddCoordinateSystem(coords);
    coords->SetAxis(0,new eavlCoordinateAxisField("coords",0));
    coords->SetAxis(1,new eavlCoordinateAxisField("coords",1));
    coords->SetAxis(2,new eavlCoordinateAxisField("coords",2));
    /*
    coords->axisTypes.push_back(eavlCoordinateAxisType(eavlCoordinateAxisType::X,"X"));
    coords->axisTypes.push_back(eavlCoordinateAxisType(eavlCoordinateAxisType::Y,"Y"));
    coords->axisTypes.push_back(eavlCoordinateAxisType(eavlCoordinateAxisType::Z,"Z"));
    coords->fieldName = "coords";
    */

    eavlArray *axisValues = new eavlFloatArray("coords",3);
    axisValues->SetNumberOfTuples(data->GetNumPoints());
    for (int i=0; i<data->GetNumPoints(); i++)
    {
        axisValues->SetComponentFromDouble(i, 0, vals[i*3+0]);
        axisValues->SetComponentFromDouble(i, 1, vals[i*3+1]);
        axisValues->SetComponentFromDouble(i, 2, vals[i*3+2]);
    }

    eavlField *field = new eavlField(1, axisValues, eavlField::ASSOC_POINTS);
    data->AddField(field);
#else
    eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(log,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);

    data->AddCoordinateSystem(coords);
    coords->SetAxis(0,new eavlCoordinateAxisField("xcoord",0));
    coords->SetAxis(1,new eavlCoordinateAxisField("ycoord",0));
    coords->SetAxis(2,new eavlCoordinateAxisField("zcoord",0));

    eavlArray *axisValues[3] = {
        new eavlFloatArray("xcoord",1),
        new eavlFloatArray("ycoord",1),
        new eavlFloatArray("zcoord",1)
    };
    for (int d=0; d<3; d++)
    {
        
        axisValues[d]->SetNumberOfTuples(data->GetNumPoints());
        for (int i=0; i<data->GetNumPoints(); i++)
            axisValues[d]->SetComponentFromDouble(i, 0, vals[i*3+d]);

        eavlField *field = new eavlField(1, axisValues[d], eavlField::ASSOC_POINTS);
        data->AddField(field);
    }
#endif
}


void
eavlVTKImporter::Parse_Structured_Points()
{
    //GetNextLine(); // assume it's read already
    THROW(eavlException,"Structured points is unsupported\n");
}

void
eavlVTKImporter::Parse_Structured_Grid()
{
    //GetNextLine(); // assume it's read already

    istringstream sin(buff);
    string s;
    sin >> s;
    if (s != "DIMENSIONS")
        THROW(eavlException,string("Expected DIMENSIONS, got ")+s);
    int ndims[3];
    sin >> ndims[0] >> ndims[1] >> ndims[2];

    ///\todo: this "3" is redundant and actually conflicts with the next lines
    ///       (i.e. we overwrite it with the real logical dimension)
    int dim = 0;
    int dims[3];
    if (ndims[0] > 1)
        dims[dim++] = ndims[0];
    if (ndims[1] > 1)
        dims[dim++] = ndims[1];
    if (ndims[2] > 1)
        dims[dim++] = ndims[2];

    eavlRegularStructure reg; 
    reg.SetNodeDimension(dim, dims);
    eavlLogicalStructureRegular *log = new eavlLogicalStructureRegular(dim,reg);
    data->SetLogicalStructure(log);

    GetNextLine();
    ParsePoints(log);

    int zi = ndims[0]-1;
    int zj = ndims[1]-1;
    int zk = ndims[2]-1;
    if (zi<1) zi=1;
    if (zj<1) zj=1;
    if (zk<1) zk=1;

    eavlCellSetAllStructured *cells =
        new eavlCellSetAllStructured("StructuredGridCells", reg);
    data->AddCellSet(cells);

    GetNextLine();
}

void
eavlVTKImporter::Parse_Rectilinear_Grid()
{
    //GetNextLine(); // assume it's read already

    istringstream sin(buff);
    string s;
    int n;
    sin >> s;
    if (s != "DIMENSIONS")
        THROW(eavlException,string("Expected DIMENSIONS, got ")+s);
    int ndims[3];
    sin >> ndims[0] >> ndims[1] >> ndims[2];

    GetNextLine();
    istringstream xin(buff);
    xin >> s;
    if (s != "X_COORDINATES")
        THROW(eavlException,string("Expected X_COORDINATES, got ")+s);
    vector<double> x;
    xin >> n;
    x.resize(n);
    xin >> s;
    toupper(s);
    ReadIntoVector(n,DataTypeFromString(s), x);

    GetNextLine();
    istringstream yin(buff);
    yin >> s;
    if (s != "Y_COORDINATES")
        THROW(eavlException,string("Expected Y_COORDINATES, got ")+s);
    vector<double> y;
    yin >> n;
    y.resize(n);
    yin >> s;
    toupper(s);
    ReadIntoVector(n,DataTypeFromString(s), y);

    GetNextLine();
    istringstream zin(buff);
    zin >> s;
    if (s != "Z_COORDINATES")
        THROW(eavlException,string("Expected Z_COORDINATES, got ")+s);
    vector<double> z;
    zin >> n;
    z.resize(n);
    zin >> s;
    toupper(s);
    ReadIntoVector(n,DataTypeFromString(s), z);

    vector<vector<double> > coords;
    coords.push_back(x);
    coords.push_back(y);
    coords.push_back(z);
    vector<string> coordNames;
    coordNames.push_back("xcoord");
    coordNames.push_back("ycoord");
    coordNames.push_back("zcoord");

    AddRectilinearMesh(data, coords, coordNames, true, "RectilinearGridCells");

    GetNextLine();
}

void
eavlVTKImporter::Parse_Polydata()
{
    //GetNextLine(); // assume it's read already
    ParsePoints(NULL);

    eavlCellSetExplicit *cells[4] = {
        new eavlCellSetExplicit("PolyDataVertices", 0),
        new eavlCellSetExplicit("PolyDataLines", 1),
        new eavlCellSetExplicit("PolyDataPolygons", 2),
        new eavlCellSetExplicit("PolyDataTriStrips", 2)
    };
    eavlExplicitConnectivity newconn[4];

    GetNextLine();

    while (strncmp(buff, "VERTICES",        8)  == 0 ||
           strncmp(buff, "LINES",           5)  == 0 ||
           strncmp(buff, "POLYGONS",        8)  == 0 ||
           strncmp(buff, "TRIANGLE_STRIPS", 15) == 0)
    {
        istringstream sin(buff);
        string s;
        sin >> s;
        eavlCellShape st;
        int index = -1;
        if (s == "VERTICES")
        {
            st = EAVL_POINT;
            index = 0;
        }
        else if (s == "LINES")
        {
            st = EAVL_BEAM;
            index = 1;
        }
        else if (s == "POLYGONS")
        {
            st = EAVL_POLYGON;
            index = 2;
        }
        else if (s == "TRIANGLE_STRIPS")
        {
            st = EAVL_TRISTRIP;
            index = 3;
        }
        else
        {
            THROW(eavlException, "Unknown cell set in vtk polydata");
        }

        int nnew;
        int nvals;
        sin >> nnew;
        sin >> nvals;

        vector<int> cv;
        ReadIntoVector(nvals, dt_int, cv);

        ///\todo: this may not be right; we're currently enforcing
        ///       that in CELL_DATA, VERTICES is the first section,
        ///       LINES the second, etc.  It's possible VTK does
        ///       something else, though.
        int cv_index = 0;
        for (int i=0; i<nnew; i++)
        {
            int n = cv[cv_index];
            newconn[index].AddElement(st,  n,  &(cv[cv_index+1]));
            // we need to split fields by dimensionality
            cell_to_cell_splitmap.push_back(index);
            cv_index += n+1;
        }

        if (!GetNextLine())
            break;
    }
    for (int e=0; e<4; e++)
    {
        if (newconn[e].GetNumElements() > 0)
        {
            cells[e]->SetCellNodeConnectivity(newconn[e]);
            data->AddCellSet(cells[e]);
        }
        else
        {
            delete cells[e];
        }
    }
}

void
eavlVTKImporter::Parse_Unstructured_Grid()
{
    //GetNextLine(); // assume it's read already
    ParsePoints(NULL);

    eavlCellSetExplicit *cells[4];
    eavlExplicitConnectivity newconn[4];
    for (int e=0; e<4; e++)
    {
        cells[e] = new eavlCellSetExplicit(string("UnstructuredGridCells")
                                           + char('0'+e) + "D", e);
    }

    GetNextLine();
    istringstream sin1(buff);
    string s;
    sin1 >> s;
    if (s != "CELLS")
        THROW(eavlException,string("Expected CELLS; got ")+s);
    int ncells;
    sin1 >> ncells;
    int nvals1;
    sin1 >> nvals1;
    vector<int> orig_connectivity;
    ReadIntoVector(nvals1, dt_int, orig_connectivity);

    GetNextLine();
    istringstream sin2(buff);
    sin2 >> s;
    if (s != "CELL_TYPES")
        THROW(eavlException,string("Expected CELL_TYPES; got ")+s);
    int nvals2;
    sin2 >> nvals2;
    if (nvals2 != ncells)
        THROW(eavlException,"Mismatch in num cells between CELLS line and CELL_TYPES line.");
    vector<int> cell_types;
    ReadIntoVector(nvals2, dt_int, cell_types);
    int conn_index = 0;
    for (int i=0; i<nvals2; i++)
    {
        eavlCellShape st = EAVL_OTHER;
        int d = -1;
        switch (cell_types[i])
        {
          //case VTK_EMPTY_CELL:   d=0; st = EAVL_OTHER??;  break;
          case VTK_VERTEX:         d=0; st = EAVL_POINT;    break;
          //case VTK_POLY_VERTEX   d=0; st = EAVL_OTHER??;  break;
          case VTK_LINE:           d=1; st = EAVL_BEAM;     break;
          //case VTK_POLY_LINE:    d=1; st = EAVL_OTHER??;  break;
          case VTK_TRIANGLE:       d=2; st = EAVL_TRI;      break;
          case VTK_TRIANGLE_STRIP: d=2; st = EAVL_TRISTRIP; break;
          case VTK_POLYGON:        d=2; st = EAVL_POLYGON;  break;
          case VTK_PIXEL:          d=2; st = EAVL_PIXEL;    break;
          case VTK_QUAD:           d=2; st = EAVL_QUAD;     break;
          case VTK_TETRA:          d=3; st = EAVL_TET;      break;
          case VTK_VOXEL:          d=3; st = EAVL_VOXEL;    break;
          case VTK_HEXAHEDRON:     d=3; st = EAVL_HEX;      break;
          case VTK_WEDGE:          d=3; st = EAVL_WEDGE;    break;
          case VTK_PYRAMID:        d=3; st = EAVL_PYRAMID;  break;
        }
        if (d < 0)
        {
            int npts = orig_connectivity[conn_index];
            conn_index += npts+1;
            continue;
        }


        int npts = orig_connectivity[conn_index];
        newconn[d].AddElement(st,  npts,  &(orig_connectivity[conn_index+1]));
        // we need to split fields by dimensionality
        cell_to_cell_splitmap.push_back(d);
        conn_index += npts+1;
    }
    for (int e=0; e<4; e++)
    {
        if (newconn[e].GetNumElements() > 0)
        {
            cells[e]->SetCellNodeConnectivity(newconn[e]);
            data->AddCellSet(cells[e]);
        }
        else
        {
            delete cells[e];
        }
    }

    GetNextLine();
}

// --------------------
void
eavlVTKImporter::ParseAttributes()
{
    Location loc = LOC_DATASET;
    while ((*is) && !(is->eof()))
    {
        istringstream sin(buff);
        string s;
        sin >> s;

        if (s == "CELL_DATA")
        {
            loc = LOC_CELLS;
            int n;
            sin >> n;
        }
        else if (s == "POINT_DATA")
        {
            loc = LOC_POINTS;
            int n;
            sin >> n;
            if (n != data->GetNumPoints())
                THROW(eavlException,"Mismatch between original num pts and POINT_DATA value");
        }
        else if (loc == LOC_DATASET)
        {
            THROW(eavlException,string("Got something other than CELL_DATA or POINT_DATA "
                                       "after data set structure: ")+bufforig);
        }
        else
        {
            if (s == "FIELD")
            {
                ParseFieldData(loc);
            }
            else if (s == "SCALARS")
            {
                ParseScalars(loc);
            }
            else if (s == "VECTORS")
            {
                ParseVectors(loc);
            }
            else if (s == "NORMALS")
            {
                ParseNormals(loc);
            }
            else if (s == "COLOR_SCALARS" ||
                     s == "LOOKUP_TABLE" ||
                     s == "TEXTURE_COORDINATES" ||
                     s == "TENSORS")
            {
                THROW(eavlException,string("Parsing of ")+s+" data set attributes "
                      "is unimplemented.");
            }
            else
            {
                THROW(eavlException,string("Unexpected data set attribute: ")+s);
            }
        }

        GetNextLine();
    }
}

