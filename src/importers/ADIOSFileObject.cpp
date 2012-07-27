// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "STL.h"
#include "ADIOSFileObject.h"
#include "eavlArray.h"
#include "eavlException.h"

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
    gps = NULL;
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
    gps = NULL;
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

#ifdef PARALLEL
    fp = adios_fopen(fileName.c_str(), (MPI_Comm)VISIT_MPI_COMM);
#else
    fp = adios_fopen(fileName.c_str(), 0);
#endif
    
    char errmsg[1024];
    if (fp == NULL)
    {
        sprintf(errmsg, "Error opening bp file %s:\n%s", fileName.c_str(), adios_errmsg());
	THROW(eavlException,errmsg);
    }

    
    ADIOS_VARINFO *avi;
    gps = (ADIOS_GROUP **) malloc(fp->groups_count * sizeof(ADIOS_GROUP *));
    if (gps == NULL)
	THROW(eavlException,"The file could not be opened. Not enough memory");
    
    /*
    cout << "ADIOS BP file: " << fileName << endl;
    cout << "# of groups: " << fp->groups_count << endl;
    cout << "# of variables: " << fp->vars_count << endl;
    cout << "# of attributes:" << fp->attrs_count << endl;
    cout << "time steps: " << fp->ntimesteps << " from " << fp->tidx_start << endl;
    */

    //Read in variables/scalars.
    variables.clear();
    scalars.clear();
    for (int gr=0; gr<fp->groups_count; gr++)
    {
        //cout <<  "  group " << fp->group_namelist[gr] << ":" << endl;
        gps[gr] = adios_gopen_byid(fp, gr);
        if (gps[gr] == NULL)
        {
            sprintf(errmsg, "Error opening group %s in bp file %s:\n%s", fp->group_namelist[gr], fileName.c_str(), adios_errmsg());
	    THROW(eavlException,errmsg);
        }
        
        for (int vr=0; vr<gps[gr]->vars_count; vr++)
        {
            avi = adios_inq_var_byid(gps[gr], vr);
            if (avi == NULL)
            {
                sprintf(errmsg, "Error opening inquiring variable %s in group %s of bp file %s:\n%s", 
                        gps[gr]->var_namelist[vr], fp->group_namelist[gr], fileName.c_str(), adios_errmsg());
		THROW(eavlException,errmsg);
            }

            if (SupportedVariable(avi))
            {
                //Scalar
                if (avi->ndim == 0)
                {
                    ADIOSScalar s(gps[gr]->var_namelist[vr], avi);
                    scalars[s.Name()] = s;
                    //cout<<"  added scalar "<<s<<endl;
                }
                //Variable
                else
                {
                    // add variable to map, map id = variable path without the '/' in the beginning
                    ADIOSVar v(gps[gr]->var_namelist[vr], gr, avi);
                    variables[v.name] = v;
                    //cout<<"  added variable "<< v.name<<endl;
                }
            }
            else
	    {
		/*
                cout<<"Skipping variable: "<<gps[gr]->var_namelist[vr]<<" dim= "<<avi->ndim
                      <<" timedim= "<<avi->timedim
                      <<" type= "<<adios_type_to_string(avi->type)<<endl;
		*/
	    }
            
            adios_free_varinfo(avi);
        }
        //Read in attributes.
        for (int a = 0; a < gps[gr]->attrs_count; a++)
        {
            int sz;
            void *data = NULL;
            ADIOS_DATATYPES attrType;

            if (adios_get_attr_byid(gps[gr], a, &attrType, &sz, &data) != 0)
            {
                //cout<<"Failed to get attr: "<<gps[gr]->attr_namelist[a]<<endl;
                continue;
            }
            
            ADIOSAttr attr(gps[gr]->attr_namelist[a], attrType, data);
            attributes[attr.Name()] = attr;
            free(data);
        }

        adios_gclose(gps[gr]);
        gps[gr] = NULL;
    }

    return true;
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
    if (fp && gps)
    {
        for (int gr=0; gr<fp->groups_count; gr++)
            if (gps[gr] != NULL)
                adios_gclose(gps[gr]);
    }
    
    if (gps)
        free(gps);
    if (fp)
        adios_fclose(fp);
    
    fp = NULL;
    gps = NULL;
}


// ****************************************************************************
//  Method: ADIOSFileObject::OpenGroup
//
//  Purpose:
//      Open a group.
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

void
ADIOSFileObject::OpenGroup(int grpIdx)
{
    if (!gps)
        return;
    if (gps[grpIdx] == NULL)
        gps[grpIdx] = adios_gopen_byid(fp, grpIdx);

    if (gps[grpIdx] == NULL)
    {
        std::string errmsg = "Error opening group "+std::string(fp->group_namelist[grpIdx])+" in " + fileName;
	THROW(eavlException,errmsg);
    }
}

// ****************************************************************************
//  Method: ADIOSFileObject::CloseGroup
//
//  Purpose:
//      Close a group.
//
//  Programmer: Dave Pugmire
//  Creation:   Tue Mar  9 12:40:15 EST 2010
//
// ****************************************************************************

void
ADIOSFileObject::CloseGroup(int grpIdx)
{
    if (!gps)
        return;
    if (gps[grpIdx] != NULL)
    {
        int val = adios_gclose(gps[grpIdx]);
        gps[grpIdx] = NULL;
        if (val != 0)
        {
            std::string errmsg = "Error closing group "+std::string(fp->group_namelist[grpIdx])+" in " + fileName;
	    THROW(eavlException,errmsg);
        }
    }
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

    //cout<<"ARR: adios_read_var:"<<endl<<v<<endl;
    OpenGroup(v.groupIdx);

    uint64_t retval = adios_read_var_byid(gps[v.groupIdx], v.varid, start, count, readData);
    CloseGroup(v.groupIdx);

    if (convertData)
    {
        if (retval > 0)
            ConvertTo(data, ntuples, v.type, readData);
        free(readData);
    }

    return (retval > 0);
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
        (avi->ndim > 3 && avi->timedim == -1) ||  // >3D array with no time
        (avi->ndim > 4 && avi->timedim >= 0)  ||  // >3D array with time
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
    type=adios_unknown; groupIdx=-1, varid=-1, timedim=-1;
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

ADIOSVar::ADIOSVar(const std::string &nm, int grpIdx, ADIOS_VARINFO *avi)
{
    name = nm;
    type = avi->type;
    double valMin = 0.0, valMax = 0.0;

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

    extents[0] = valMin;
    extents[1] = valMax;
    timedim = avi->timedim;
    groupIdx = grpIdx;
    varid = avi->varid;
    if (avi->timedim == -1)
        dim = avi->ndim;
    else
    {
        if (avi->ndim == 1)
            dim = avi->ndim;
        else
            dim = avi->ndim - 1;
    }

    for (int i = 0; i < 3; i++)
    {
        start[i] = 0;
        count[i] = 0;
    }
    
    int idx = (timedim == -1 ? 0 : 1);
    if (dim == 1 && timedim == 0)
        idx = 0;
    
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
    if (timedim >= 0)
    {
        s[idx] = ts;
        c[idx] = 1;
        idx++;
    }

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
