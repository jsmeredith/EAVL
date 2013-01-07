// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "STL.h"
#include "ADIOSFileObject.h"
#include "eavlArray.h"
#include "eavlException.h"
#include "adios.h"
#include "adios_error.h"

typedef struct
{
    ADIOS_VARINFO * v;
    uint64_t        start[10];
    uint64_t        count[10];
    uint64_t        writesize; // size of subset this process writes, 0: do not write
} VarInfo;

static bool
SupportedVariable(ADIOS_VARINFO *avi);

static void
ConvertToFloat(float *data, int &n, ADIOS_DATATYPES &t, const void *readData);
template<class T> static void
ConvertTo(T *data, int &n, ADIOS_DATATYPES &t, const void *readData);

// ****************************************************************************
//  Method: ADIOSFileObject::ADIOSFileObject
//
//  Purpose:
//      ADIOSFileObject constructor
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Feb 10 15:55:14 EST 2010
//
// ****************************************************************************

ADIOSFileObject::ADIOSFileObject(const char *fname)
{
    fileName = fname;
    fp = NULL;
}

// ****************************************************************************
//  Method: ADIOSFileObject::ADIOSFileObject
//
//  Purpose:
//      ADIOSFileObject constructor
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Feb 10 15:55:14 EST 2010
//
// ****************************************************************************

ADIOSFileObject::ADIOSFileObject(const std::string &fname)
{
    fileName = fname;
    fp = NULL;
}


// ****************************************************************************
//  Method: ADIOSFileObject::ADIOSFileObject
//
//  Purpose:
//      ADIOSFileObject dtor.
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Feb 10 15:55:14 EST 2010
//
// ****************************************************************************

ADIOSFileObject::~ADIOSFileObject()
{
    Close();
}

#if 0
// ****************************************************************************
//  Method: ADIOSFileObject::NumTimeSteps
//
//  Purpose:
//      Return number of timesteps.
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Feb 10 15:55:14 EST 2010
//
// ****************************************************************************

int
ADIOSFileObject::NumTimeSteps()
{
    Open();
    return fp->ntimesteps;
}

// ****************************************************************************
//  Method: ADIOSFileObject::GetCycles
//
//  Purpose:
//      Return cycles.
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Feb 10 15:55:14 EST 2010
//
// ****************************************************************************

void
ADIOSFileObject::GetCycles(std::vector<int> &cycles)
{
    Open();
    
    cycles.resize(0);
    for (int i = 0; i < fp->ntimesteps; i++)
        cycles.push_back(fp->tidx_start + i);
}

// ****************************************************************************
// Method:  ADIOSFileObject::GetCycles
//
// Purpose:
//   Return cycles.
//
// Programmer:  Dave Pugmire
// Creation:    January 27, 2011
//
// ****************************************************************************

void
ADIOSFileObject::GetCycles(std::string &varNm, std::vector<int> &cycles)
{
    Open();
    cycles.resize(0);

    varIter vi = variables.find(varNm);
    if (vi == variables.end())
    {
        return;
    }
    ADIOSVar v = vi->second;
    
    int tupleSz = adios_type_size(v.type, NULL);
    uint64_t start[4] = {0,0,0,0}, count[4] = {0,0,0,0};

    count[0] = v.count[0];
    int ntuples = v.count[0];
    
    void *readData = malloc(ntuples*tupleSz);
    OpenGroup(v.groupIdx);
    uint64_t retval = adios_read_var_byid(gps[v.groupIdx], v.varid, start, count, readData);
    CloseGroup(v.groupIdx);

    if (retval > 0)
    {
        cycles.resize(ntuples);
        ConvertTo(&cycles[0], ntuples, v.type, readData);
    }
    free(readData);
}


// ****************************************************************************
// Method:  ADIOSFileObject::GetTimes
//
// Purpose:
//   
// Programmer:  Dave Pugmire
// Creation:    January 26, 2011
//
// ****************************************************************************

void
ADIOSFileObject::GetTimes(std::string &varNm, std::vector<double> &times)
{
    Open();
    times.resize(0);
    
    varIter vi = variables.find(varNm);
    if (vi == variables.end())
    {
        return;
    }
    ADIOSVar v = vi->second;

    int tupleSz = adios_type_size(v.type, NULL);
    uint64_t start[4] = {0,0,0,0}, count[4] = {0,0,0,0};

    count[0] = v.count[0];
    int ntuples = v.count[0];
    
    void *readData = malloc(ntuples*tupleSz);
    OpenGroup(v.groupIdx);
    uint64_t retval = adios_read_var_byid(gps[v.groupIdx], v.varid, start, count, readData);
    CloseGroup(v.groupIdx);

    if (retval > 0)
    {
        times.resize(ntuples);
        ConvertTo(&times[0], ntuples, v.type, readData);
    }
    free(readData);
}
#endif

// ****************************************************************************
//  Method: ADIOSFileObject::Open
//
//  Purpose:
//      Open a file.
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Feb 10 15:55:14 EST 2010
//
//  Modifications:
//
//   Dave Pugmire, Tue Mar  9 12:40:15 EST 2010
//   Major overhaul. Added scalars, attributes, and reorganized the class.   
//
// ****************************************************************************

bool
ADIOSFileObject::Open()
{
    if (IsOpen())
        return true;
    
    int err;
    ADIOS_READ_METHOD read_method = ADIOS_READ_METHOD_BP;
    int timeoutSec = 0;

#ifdef PARALLEL
    err = adios_read_init_method(read_method, (MPI_Comm)VISIT_MPI_COMM, "");
    fp = adios_read_open_stream(fileName.c_str(), read_method, (MPI_Comm)VISIT_MPI_COMM, 
				ADIOS_LOCKMODE_ALL, timeoutSec);
#else
    err = adios_read_init_method(read_method, 0, "");
    /*
    fp = adios_read_open_stream(fileName.c_str(), read_method, 0,
				ADIOS_LOCKMODE_ALL, timeoutSec);
    */
    fp = adios_read_open_file(fileName.c_str(), read_method, 0);
#endif
    
    char errmsg[1024];
    if (0)//(!err || fp == NULL)
    {
        sprintf(errmsg, "Error opening bp file %s:\n%s", fileName.c_str(), adios_errmsg());
	THROW(eavlException, errmsg);
    }
    if (adios_errno == err_file_not_found || adios_errno == err_end_of_stream)
	THROW(eavlException, "ADIOS open failed.");
    
    char **groupNames;
    int64_t gh;
    VarInfo *varinfo;
    ADIOS_VARINFO *v;

    while (adios_errno != err_end_of_stream)
    {
	adios_get_grouplist(fp, &groupNames);
	adios_declare_group(&gh, groupNames[0], "", adios_flag_yes);
	
	varinfo = (VarInfo *) malloc (sizeof(VarInfo) * fp->nvars);

	for (int i=0; i<fp->nvars; i++) 
	{
	    //cout <<"Get info on variable "<<i<<" "<<fp->var_namelist[i]<<endl;
	    varinfo[i].v = adios_inq_var_byid(fp, i);
	    if (varinfo[i].v == NULL)
		THROW(eavlException, "ADIOS Importer: variable inquiry failed.");

	    if (!SupportedVariable(varinfo[i].v))
		continue;
	    
	    // add variable to map, map id = variable path without the '/' in the beginning
	    ADIOSVar v(fp->var_namelist[i], varinfo[i].v);
	    variables[v.name] = v;

	    // print variable type and dimensions
        }
	break;
    }
}

// ****************************************************************************
//  Method: ADIOSFileObject::Close
//
//  Purpose:
//      Close a file.
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Feb 10 15:55:14 EST 2010
//
//  Modifications:
//
//   Dave Pugmire, Tue Mar  9 12:40:15 EST 2010
//   Major overhaul. Added scalars, attributes, and reorganized the class.   
//
// ****************************************************************************

void
ADIOSFileObject::Close()
{
    if (fp)
	adios_read_close(fp);
    fp = NULL;
}


// ****************************************************************************
//  Method: ADIOSFileObject::GetIntScalar
//
//  Purpose:
//      Return an integer scalar
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

bool
ADIOSFileObject::GetIntScalar(const std::string &nm, int &val)
{
    Open();
    std::map<std::string, ADIOSScalar>::const_iterator s = scalars.find(nm);
    if (s == scalars.end() || !s->second.IsInt())
        return false;
    
    val = s->second.AsInt();
    return true;
}

// ****************************************************************************
//  Method: ADIOSFileObject::GetDoubleScalar
//
//  Purpose:
//      Return a double scalar
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

bool
ADIOSFileObject::GetDoubleScalar(const std::string &nm, double &val)
{
    Open();
    std::map<std::string, ADIOSScalar>::const_iterator s = scalars.find(nm);
    if (s == scalars.end() || !s->second.IsDouble())
        return false;
    
    val = s->second.AsDouble();
    return true;
}

// ****************************************************************************
//  Method: ADIOSFileObject::GetStringScalar
//
//  Purpose:
//      Return a string scalar
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Mar 24 16:32:09 EDT 2010
//
// ****************************************************************************

bool
ADIOSFileObject::GetStringScalar(const std::string &nm, std::string &val)
{
    Open();
    std::map<std::string, ADIOSScalar>::const_iterator s = scalars.find(nm);
    if (s == scalars.end() || !s->second.IsString())
        return false;
    
    val = s->second.AsString();
    return true;
}

// ****************************************************************************
//  Method: ADIOSFileObject::GetIntAttr
//
//  Purpose:
//      Return integer attribute
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

bool
ADIOSFileObject::GetIntAttr(const std::string &nm, int &val)
{
    Open();
    std::map<std::string, ADIOSAttr>::const_iterator a = attributes.find(nm);
    if (a == attributes.end() || !a->second.IsInt())
        return false;

    val = a->second.AsInt();
    return true;
}

// ****************************************************************************
//  Method: ADIOSFileObject::GetStringAttr
//
//  Purpose:
//      Return string attribute
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

bool
ADIOSFileObject::GetStringAttr(const std::string &nm, std::string &val)
{
    Open();
    std::map<std::string, ADIOSAttr>::const_iterator a = attributes.find(nm);
    if (a == attributes.end() || !a->second.IsString())
        return false;

    val = a->second.AsString();
    return true;
}
// ****************************************************************************
//  Method: ADIOSFileObject::ReadVariable
//
//  Purpose:
//      Read variable.
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Mar 17 15:29:24 EDT 2010
//
//  Modifications:
//
//
// ****************************************************************************

bool
ADIOSFileObject::ReadVariable(const std::string &nm,
                              int ts,
			      eavlFloatArray **array)
{
    Open();

    varIter vi = variables.find(nm);
    if (vi == variables.end())
    {
        //cout<<"Variable "<<nm<<" not found."<<endl;
        return false;
    }
    ADIOSVar v = vi->second;

    int tupleSz = adios_type_size(v.type, NULL);
    int ntuples = 1;
    uint64_t start[4] = {0,0,0,0}, count[4] = {0,0,0,0};
    v.GetReadArrays(ts, start, count, &ntuples);
    
    (*array) = new eavlFloatArray(nm, 1);
    (*array)->SetNumberOfTuples(ntuples);
    float *data = (float *)(*array)->GetRawPointer(eavlArray::HOST);
    void *readData = (void *)data;

    bool convertData = (v.type != adios_real);
    if (convertData)
        readData = malloc(ntuples*tupleSz);

    ADIOS_SELECTION *sel = adios_selection_boundingbox(v.dim, start, count);
    adios_schedule_read_byid(fp, sel, v.varid, 0, 1, readData);
    adios_perform_reads(fp, 1);

    if (convertData)
    {
	ConvertTo(data, ntuples, v.type, readData);
        free(readData);
    }

    return true;
}

// ****************************************************************************
//  Function: SupportedVariable
//
//  Purpose:
//      Determine if this variable is supported.
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

static bool
SupportedVariable(ADIOS_VARINFO *avi)
{
    if (/*(avi->ndim == 1 && avi->timedim >= 0) ||  // scalar with time*/
	(avi->ndim == 0) ||
	(avi->ndim == 1) ||
        (avi->ndim > 3) ||  // >3D array
        avi->type == adios_long_double ||
        avi->type == adios_complex || 
        avi->type == adios_double_complex)
    {
        return false;
    }
    
    return true;
}

// ****************************************************************************
//  Method: ADIOSVar::ADIOSVar
//
//  Purpose:
//      Constructor
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

ADIOSVar::ADIOSVar()
{
    start[0] = start[1] = start[2] = 0;
    count[0] = count[1] = count[2] = 0;
    global[0] = global[1] = global[2] = 0;
    dim = 0;
    type=adios_unknown; varid=-1;
    extents[0] = extents[1] = 0.0;
}

// ****************************************************************************
//  Method: ADIOSVar::ADIOSVar
//
//  Purpose:
//      Constructor
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

ADIOSVar::ADIOSVar(const std::string &nm, ADIOS_VARINFO *avi)
{
    name = nm;
    type = avi->type;
    double valMin = 0.0, valMax = 0.0;

    /*
    if (avi->gmin && avi->gmax)
    {
        if (type == adios_integer)
        {
            valMin = (double)(*((int*)avi->gmin));
            valMax = (double)(*((int*)avi->gmax));
        }
        else if (type == adios_real)
        {
            valMin = (double)(*((float*)avi->gmin));
            valMax = (double)(*((float*)avi->gmax));
        }
        else if (type == adios_double)
        {
            valMin = (double)(*((double*)avi->gmin));
            valMax = (double)(*((double*)avi->gmax));
        }
    }
    */

    extents[0] = valMin;
    extents[1] = valMax;
    varid = avi->varid;
    dim = avi->ndim;

    for (int i = 0; i < 3; i++)
    {
        start[i] = 0;
        count[i] = 0;
    }
    
    int idx = 0;
    //ADIOS is ZYX.
    if (dim == 3)
    {
        count[0] = avi->dims[idx+2];
        count[1] = avi->dims[idx+1];
        count[2] = avi->dims[idx+0];
    }
    else if (dim == 2)
    {
        count[0] = avi->dims[idx+1];
        count[1] = avi->dims[idx+0];
    }
    else if (dim == 1)
    {
        count[0] = avi->dims[0];
    }

    for (int i = 0; i < 3; i++)
        global[i] = count[i];
}

// ****************************************************************************
//  Method: ADIOSVar::GetReadArrays
//
//  Purpose:
//      Fill in start/count arrays for adios API.
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

void
ADIOSVar::GetReadArrays(int ts, uint64_t *s, uint64_t *c, int *ntuples)
{
    *ntuples = 1;

    s[0] = s[1] = s[2] = s[3] = 0;
    c[0] = c[1] = c[2] = c[3] = 0;

    int idx = 0;
    if (dim == 1)
    {
        s[idx] = start[0];
        c[idx] = count[0];
        idx++;
        *ntuples *= (int)count[0];
    }
    //ADIOS is ZYX.
    else if (dim == 2)
    {
        s[idx] = start[1];
        c[idx] = count[1];
        idx++;
        s[idx] = start[0];
        c[idx] = count[0];
        *ntuples *= (int)count[0];
        *ntuples *= (int)count[1];
    }
    else if (dim == 3)
    {
        s[idx] = start[2];
        c[idx] = count[2];
        idx++;
        s[idx] = start[1];
        c[idx] = count[1];
        idx++;
        s[idx] = start[0];
        c[idx] = count[0];
        *ntuples *= (int)count[0];
        *ntuples *= (int)count[1];
        *ntuples *= (int)count[2];
    }
}


// ****************************************************************************
//  Method: ConvertTo
//
//  Purpose:
//      Convert arrays to different types.
//
//  Programmer: Dave Pugmire
//  Creation:   Wed Mar 17 15:29:24 EDT 2010
//
//  Modifications:
//
//
// ****************************************************************************

template<class T0, class T1> static void
CopyArray( T0 *inData, T1 *outData, int n)
{
    for (int i = 0; i < n; i++)
        outData[i] = (T1)inData[i];
}
template<class T> static void
ConvertTo(T *data, int &n, ADIOS_DATATYPES &t, const void *readData)
{
    switch(t)
    {
      case adios_unsigned_byte:
        CopyArray((const unsigned char *)readData, data, n);
        break;
      case adios_byte:
        CopyArray((const char *)readData, data, n);
        break;
      case adios_unsigned_short:
        CopyArray((const unsigned short *)readData, data, n);
        break;
      case adios_short:
        CopyArray((const short *)readData, data, n);
        break;
      case adios_unsigned_integer:
        CopyArray((const unsigned int *)readData, data, n);
        break;
      case adios_integer:
        CopyArray((const int *)readData, data, n);
        break;
      case adios_unsigned_long:
        CopyArray((const unsigned long *)readData, data, n);
        break;
      case adios_long:
        CopyArray((const long *)readData, data, n);
        break;
      case adios_double:
        CopyArray((const double *)readData, data, n);
        break;
      default:
	THROW(eavlException,"Inavlid variable type");
        break;        
    }
}
