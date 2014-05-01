// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlNetCDFDecomposingImporter.h"
#include "eavlCoordinates.h"
#include "eavlCellSetAllStructured.h"
#include "eavlException.h"

static const bool debugoutput = false;


// ****************************************************************************
// The following methods are convenience methods to help database plugins
// do dynamic domain decomposion for rectilinear grids. There are four 
// methods...
//
// 1 DetermineRectilinearDecomposition: Used to compute a domain decomposition
// 2 ComputeDomainLogicalCoords: Used to compute the logical indices of a domain 
// 3 ComputeDomainBounds: Used to compute spatial bounds of a domain
// 4 RectilinearDecompCost: Used *internally* to help compute domain decomp.
//
//   DetermineRectilinearDecomposition determines the number of domains in
//   each dimension to decompose the mesh into. For example, for a mesh that
//   is 100 zones in X and 50 zones in Y and a total number of domains (e.g.
//   processor count) of 6, it would return (3,2,0) meaning 3 domains in X
//   by 2 domains in Y by zero domains in Z like so...
//   
//      +---------+---------+---------+
//   1  |    3    |    4    |    5    |    
//      +---------+---------+---------+
//   0  |    0    |    1    |    2    |    
//      +---------+---------+---------+
//           0         1         2
//
//   The numbers inside each box above are the MPI ranks associated with the
//   domains or, in other words, the domain numbers.
//
//   ComputeDomainLogicalCoords is used to map an MPI rank into a set of 
//   logical domain indices. It would be called, for example, on processor of
//   rank 5 to return the indices (2,1,0) meaning that domain '5' has logical
//   indices 2,1,0.
//
//   ComputeDomainBounds is used to compute the logical bounds of a domain
//   'slot' along a given axis. For example, for the case above, slot 0's
//   domain in the Y axis would go from 0 to 49 while slot 1 would go from
//   50 to 99.
//
//   These routines could be improved to support simple ghosting.
//
// ****************************************************************************

// ****************************************************************************
//  Function: RectilinearDecompCost 
//
//  Purpose:
///     Compute the "cost" of a decomposition in terms of the amount of
///     communication algorithms might require. Note, this is part of the
///     domain decomposition algorithm taken from Matt O'Brien's domain
///     decomposition library
//      
//  Programmer: Mark C. Miller (plagiarized from Matt O'Brien)
//  Creation:   September 20, 2004 
//
//  Modification:
//
// ****************************************************************************
double
RectilinearDecompCost(int i, int j, int k, int nx, int ny, int nz)
{
    double costtot;

    i--;
    j--;
    k--;

    costtot = 0;

    /* estimate cost based on size of communicating surfaces */

    costtot+=i*((ny*nz));
    costtot+=j*((nx*nz));
    costtot+=k*((ny*nx));
    costtot+=2*i*j*((nz));
    costtot+=2*i*k*((ny));
    costtot+=2*j*k*((nx));
    costtot+=4*i*j*k;

    return(costtot);
}

// ****************************************************************************
//  Function: DetermineRectilinearDecomposition 
//
//  Purpose:
///     Decides how to decompose a rectilinear mesh into numbers of processors
///     along each independent axis. This code was taken from Matt O'Brien's
///     domain decomposition library. 
//
//      ndims : is the number of logical dimensions in the mesh
//      n     : is the number of desired domains
//      nx    : size of global, logical mesh in x
//      ny    : size of global, logical mesh in y
//      nz    : size of global, logical mesh in z
//      imin  : (named consistent with orig. code) domain count along x axis
//      jmin  : (named consistent with orig. code) domain count along y axis
//      kmin  : (named consistent with orig. code) domain count along z axis
//
//      After calling, it should be the case that imin * jmin * kmin = n;
//      Therefore, one can decompose the mesh into an array of 'domains' that
//      is imin x jmin x kmin.
//
//      If n is a prime number, I think the result is imin = n, jmin = kmin = 1
//      
//  Programmer: Mark C. Miller (plagiarized from Matt O'Brien)
//  Creation:   September 20, 2004 
//
//  Modification:
//
// ****************************************************************************
double
ComputeRectilinearDecomposition(int ndims, int n, int nx, int ny, int nz,
   int *imin, int *jmin, int *kmin)
{
    int i,j;
    double cost, costmin = 1e80;

    if (ndims == 2)
        nz = 1;

    /* find all two or three product factors of the number of domains
       and evaluate the communication cost */

   for (i = 1; i <= n; i++)
   {
      if (n%i == 0)
      {
         if (ndims == 3)
         {
            for (j = 1; j <= i; j++)
            {
               if ((i%j) == 0)
               {
                  cost = RectilinearDecompCost(j, i/j, n/i, nx, ny, nz);
                  
                  if (cost < costmin)
                  {
                     *imin = j;
                     *jmin = i/j;
                     *kmin = n/i;
                     costmin = cost;
                  }
               }
            }
         }
         else
         {
            cost = RectilinearDecompCost(i, n/i, 1, nx, ny, nz);
            
            if (cost < costmin)
            {
               *imin = i;
               *jmin = n/i;
               *kmin = 1;
               costmin = cost;
            }
         } 
      }
   }

   return costmin;
}


// ****************************************************************************
//  Function: ComputeDomainLogicalCoords
//
//  Purpose:
///   Given the number of domains along each axis in a decomposition and
///   the rank of a processor, this routine will determine the domain logical
///   coordinates of the processor's domain.
//
//  dataDim            : number of logical dimensions in the mesh
//  domCount[]         : array of counts of domains in each of x, y, z axes
//  rank               : zero-origin rank of calling processors
//  domLogicalCoords[] : returned logical indices of the domain associated
//                       with this rank.
//
//  For example, in a 6 processor case decomposed in a 3 x 2 array of domains
//  like so...
//
//      +---------+---------+---------+
//   1  |    3    |    4    |    5    |    
//      +---------+---------+---------+
//   0  |    0    |    1    |    2    |    
//      +---------+---------+---------+
//           0         1         2
//
//   Calling this method on processor of rank 5 would return
//   domLogicalCoords = {2,1,0}
//      
//  Programmer: Mark C. Miller
//  Creation:   September 20, 2004 
//
//  Modification:
//
// ****************************************************************************
void
ComputeDomainLogicalCoords(int dataDim, int domCount[3], int rank,
    int domLogicalCoords[3])
{
    int r = rank;

    // handle Z (K logical) axis
    if (dataDim == 3)
    {
        domLogicalCoords[2] = r / (domCount[1]*domCount[0]);
        r = r % (domCount[1]*domCount[0]);
    }

    // handle Y (J logical) axis
    domLogicalCoords[1] = r / domCount[0];
    r = r % domCount[0];

    // handle X (I logical) axis
    domLogicalCoords[0] = r;
}

// ****************************************************************************
//  Function: ComputeDomainBounds 
//
//  Purpose:
///   Given the global zone count along an axis, the domain count for
///   the same axis and a domain's logical index along the same axis, compute
///   the starting global zone index along this axis and the count of zones
///   along this axis for the associated domain.
//
//  For example, in the case a 2D mesh that is globally 100 zones in X by
//  100 zones in Y, if we want to obtain the bounds in X of domains (1,0)
//  (rank 1) or (1,1) (rank 4)...
//
//      +---------+---------+---------+
//   1  |    3    |    4    |    5    |    
//      +---------+---------+---------+
//   0  |    0    |    1    |    2    |    
//      +---------+---------+---------+
//           0         1         2
//
//   We would call this method with globalZoneCount = 100, domCount = 3
//   because there are 3 domains along the X axis, domLogicalCoord = 1
//   because we are dealing with the domain whose index is 1 along the
//   X axis.
//      
//  Programmer: Mark C. Miller
//  Creation:   September 20, 2004 
//
//  Modification:
//
//    Mark C. Miller, Mon Aug 14 13:59:33 PDT 2006
//    Moved here from ViSUS plugin so it can be used by other plugins
//
// ****************************************************************************
void
ComputeDomainBounds(int globalZoneCount, int domCount, int domLogicalCoord,
    int *globalZoneStart, int *zoneCount)
{
    int domZoneCount       = globalZoneCount / domCount;
    int domsWithExtraZones = globalZoneCount % domCount;
    int domZoneCountPlus1  = domZoneCount;
    if (domsWithExtraZones > 0)
        domZoneCountPlus1 = domZoneCount + 1;

    int i;
    int stepSize = domZoneCount;
    *globalZoneStart = 0;
    for (i = 0; i < domLogicalCoord; i++)
    {
        *globalZoneStart += stepSize;
        if (i >= domCount - domsWithExtraZones)
            stepSize = domZoneCountPlus1;
    }
    if (i >= domCount - domsWithExtraZones)
        stepSize = domZoneCountPlus1;
    *zoneCount = stepSize;
}




// ****************************************************************************
//  Method:  DoDomainDecomposition
//
//  Purpose:
///   Find a domain decomposition for this problem based on num processors.
//
//  Arguments:
//    none
//
//  Programmer:  Jeremy Meredith
//  Creation:    July 15, 2008
//
// ****************************************************************************
void DoDomainDecomposition(int nprocs, int rank,
                           int zx, int zy, int zz,
                           int domainGlobalStart[3],
                           int domainGlobalCount[3],
                           int localRealStart[3],
                           int localRealCount[3],
                           bool addGhosts)
{
    int globalZoneCount[3] = {zx, zy, zz};

    // fill in domainCounts with # domains in each dimension
    int domainCounts[3];
    ComputeRectilinearDecomposition(3, // 3D
                                    nprocs, // total domains
                                    globalZoneCount[0], // zx
                                    globalZoneCount[1], // zy
                                    globalZoneCount[2], // zz
                                    &domainCounts[0], // domX
                                    &domainCounts[1], // domY
                                    &domainCounts[2]);// domZ

    if (debugoutput)
        cerr << "domain grid: "
             << domainCounts[0] << " x "
             << domainCounts[1] << " x "
             << domainCounts[2] << endl;

    // find the domain-logical indices for this rank
    int domainIndex[3];
    ComputeDomainLogicalCoords(3, domainCounts, rank, domainIndex);

    if (debugoutput)
        cerr << "rank "<<rank<<" is at domain grid index: "
             << domainIndex[0] << ","
             << domainIndex[1] << ","
             << domainIndex[2] << endl;


    // find local cell indices in the global space
    // account for ghost zones
    for (int axis = 0; axis < 3; axis++)
    {
        ComputeDomainBounds(globalZoneCount[axis],
                            domainCounts[axis],
                            domainIndex[axis],
                            &domainGlobalStart[axis],
                            &domainGlobalCount[axis]);

        if (debugoutput)
            cerr << "  without ghosts: domain local indices: "
                 << "start[" << axis << "] = " << domainGlobalStart[axis]
                 << ",  count[" << axis << "] = " << domainGlobalCount[axis]
                 << endl;

        localRealStart[axis] = 0;
        localRealCount[axis] = domainGlobalCount[axis];

        // now account for ghost cells by expanding the requested cells
        if (addGhosts)
        {
            if (domainIndex[axis] > 0)
            {
                localRealStart[axis]++;
                domainGlobalStart[axis]--;
                domainGlobalCount[axis]++;
            }
            if (domainIndex[axis] < domainCounts[axis]-1)
            {
                domainGlobalCount[axis]++;
            }

            if (debugoutput)
                cerr << "  with ghosts:    domain local indices: "
                     << "start[" << axis << "] = " << domainGlobalStart[axis]
                     << ",  count[" << axis << "] = " << domainGlobalCount[axis]
                     << endl;

            if (debugoutput)
                cerr << "  with ghosts:    local real zone indices: "
                     << "start[" << axis << "] = " << localRealStart[axis]
                     << ",  count[" << axis << "] = " << localRealCount[axis]
                     << endl;
        }
    }
}



eavlNetCDFDecomposingImporter::eavlNetCDFDecomposingImporter(int numdomains,
                                                             const string &filename)
{
    numchunks = numdomains;
    file = new NcFile(filename.c_str(), NcFile::ReadOnly);
     
    if (!file->is_valid())
    {
        THROW(eavlException,"Couldn't open file!\n");
    }

    if (debugoutput) cerr << "num_dims="<<file->num_dims()<<endl;
    if (debugoutput) cerr << "num_vars="<<file->num_vars()<<endl;
    if (debugoutput) cerr << "num_atts="<<file->num_atts()<<endl;

    for (int i=0; i<file->num_dims(); i++)
    {
        NcDim *d = file->get_dim(i);
        if (debugoutput) cerr << "  dim["<<i<<"]: name="<<d->name()<<" size="<<d->size()<<endl;
    }

    for (int i=0; i<file->num_atts(); i++)
    {
        NcAtt *a = file->get_att(i);
        if (debugoutput) cerr << "  att["<<i<<"]: name="<<a->name()<<" numvals="<<a->num_vals()<<endl;
    }

    bool found_grid = false;

    for (int i=0; i<file->num_vars(); i++)
    {
        NcVar *v = file->get_var(i);
        if (debugoutput) 
        {
            cerr << "  var["<<i<<"]: name="<<v->name();
            cerr << "  ndims="<<v->num_dims();
            cerr << "  dims = ";
            for (int j=0; j<v->num_dims(); j++)
            {
                cerr << v->get_dim(j)->name();
                if (j<v->num_dims()-1)
                    cerr << "*";
            }
            cerr << endl;
        }

        // Here's the condition for what we're going to use;
        // we only support one mesh for the moment, so we're picking one.
        // We're only reading the first time step for now.
        if (v->num_dims() == 4 && string(v->get_dim(0)->name())=="time")
        {
            if (!found_grid)
            {
                dims.push_back(v->get_dim(1));
                dims.push_back(v->get_dim(2));
                dims.push_back(v->get_dim(3));
                found_grid = true;
                vars.push_back(v);
                if (debugoutput) cerr << "     * using as first real var\n";
            }
            else
            {
                if (string(v->get_dim(1)->name()) == dims[0]->name() &&
                    string(v->get_dim(2)->name()) == dims[1]->name() &&
                    string(v->get_dim(3)->name()) == dims[2]->name())
                {
                    vars.push_back(v);
                    if (debugoutput) cerr << "     * using as another var; matches the first real one's dims\n";
                }
            }
        }

    }
}


eavlNetCDFDecomposingImporter::~eavlNetCDFDecomposingImporter()
{
    file->close();
}

eavlDataSet*
eavlNetCDFDecomposingImporter::GetMesh(const string &mesh, int chunk)
{
    // NOTE: the data ordering isn't what we expected; for the moment
    // we've swapped X, Y, and Z to some degree, but this is a good use
    // case to make sure we're doing it "right".

    int domainGlobalStart[3];
    int domainGlobalCount[3];
    int localRealStart[3];
    int localRealCount[3];
    int ngx_zones = dims[0]->size() - 1; // nodal -> zonal
    int ngy_zones = dims[1]->size() - 1; // nodal -> zonal
    int ngz_zones = dims[2]->size() - 1; // nodal -> zonal
    DoDomainDecomposition(numchunks, chunk,
                          ngx_zones, ngy_zones, ngz_zones,
                          domainGlobalStart,
                          domainGlobalCount,
                          localRealStart,
                          localRealCount,
                          true); // no ghosts for now
    int nlx_nodes = domainGlobalCount[0] + 1;  // zonal -> nodal
    int nly_nodes = domainGlobalCount[1] + 1;  // zonal -> nodal
    int nlz_nodes = domainGlobalCount[2] + 1;  // zonal -> nodal

    eavlDataSet *data = new eavlDataSet;

    vector<vector<double> > coords;
    vector<string> coordNames;

    // z
    coordNames.push_back(dims[2]->name());
    {
        vector<double> c;
        int nc = nlz_nodes;
        c.resize(nc);
        for (int i = 0; i < nc; i++)
            c[i] = (double)i + domainGlobalStart[2];
        coords.push_back(c);
    }

    // y
    coordNames.push_back(dims[1]->name());
    {
        vector<double> c;
        int nc = nly_nodes;
        c.resize(nc);
        for (int i = 0; i < nc; i++)
            c[i] = (double)i + domainGlobalStart[1];
        coords.push_back(c);
    }

    // x
    coordNames.push_back(dims[0]->name());
    {
        vector<double> c;
        int nc = nlx_nodes;
        c.resize(nc);
        for (int i = 0; i < nc; i++)
            c[i] = (double)i + domainGlobalStart[0];
        coords.push_back(c);
    }
    
    AddRectilinearMesh(data, coords, coordNames, true,
                       "RectilinearGridCells");

    return data;
}

vector<string>
eavlNetCDFDecomposingImporter::GetFieldList(const string &mesh)
{
    vector<string> retval;
    retval.push_back(".ghost");
    for (unsigned int v=0; v<vars.size(); v++)
    {
        NcVar *var = vars[v];
        retval.push_back(var->name());
        ///\TODO: REMOVE THIS DEBUG LINE HERE:
        //if (v>=0) break;
    }
    return retval;
}

eavlField*
eavlNetCDFDecomposingImporter::GetField(const string &name, const string &mesh, int chunk)
{
    ///\todo: repeat of GetMesh logic
    int domainGlobalStart[3];
    int domainGlobalCount[3];
    int localRealStart[3];
    int localRealCount[3];
    int ngx_zones = dims[0]->size() - 1; // nodal -> zonal
    int ngy_zones = dims[1]->size() - 1; // nodal -> zonal
    int ngz_zones = dims[2]->size() - 1; // nodal -> zonal
    DoDomainDecomposition(numchunks, chunk,
                          ngx_zones, ngy_zones, ngz_zones,
                          domainGlobalStart,
                          domainGlobalCount,
                          localRealStart,
                          localRealCount,
                          true); // no ghosts for now
    int nlx_nodes = domainGlobalCount[0] + 1;  // zonal -> nodal
    int nly_nodes = domainGlobalCount[1] + 1;  // zonal -> nodal
    int nlz_nodes = domainGlobalCount[2] + 1;  // zonal -> nodal
    int n = nlx_nodes*nly_nodes*nlz_nodes;

    if (name == ".ghost")
    {
        int nlx_zones = domainGlobalCount[0];  // zonal -> nodal
        int nly_zones = domainGlobalCount[1];  // zonal -> nodal
        int nlz_zones = domainGlobalCount[2];  // zonal -> nodal
        if (nlz_zones < 1)
            nlz_zones = 1;
        int nzones = nlx_zones * nly_zones * nlz_zones;
        eavlFloatArray *arr = new eavlFloatArray(name, 1);
        arr->SetNumberOfTuples(nzones);

        for (int k = 0; k < nlz_zones; k++)
        {
            for (int j = 0; j < nly_zones; j++)
            {
                for (int i = 0; i < nlx_zones; i++)
                {
                    //int index = k*nly_zones*nlx_zones + j*nlx_zones + i;
                    ///\todo: again, backwards from what I'd have thought!
                    int index = i*nlz_zones*nly_zones + j*nlz_zones + k;
                    if (i < localRealStart[0] ||
                        j < localRealStart[1] ||
                        k < localRealStart[2] ||
                        i >= localRealStart[0] + localRealCount[0] ||
                        j >= localRealStart[1] + localRealCount[1] ||
                        k >= localRealStart[2] + localRealCount[2])
                    {
                        arr->SetComponentFromDouble(index,0, true);
                    }
                    else
                    {
                        arr->SetComponentFromDouble(index,0, false);
                    }
                }
            }
        }
        eavlField *field = new eavlField(0, arr, eavlField::ASSOC_CELL_SET, "RectilinearGridCells");
        return field;
    }

    for (unsigned int v=0; v<vars.size(); v++)
    {
        NcVar *var = vars[v];
        if (name != var->name())
            continue;

        if (debugoutput) cerr << "reading var "<<v+1<<" / "<<vars.size()<<endl;
        eavlFloatArray *arr = new eavlFloatArray(var->name(), 1);
        arr->SetNumberOfTuples(n);

        //arr->SetNumberOfTuples(var->num_vals());
        //NcValues *vals = var->values();
        //int n = var->num_vals();

        switch (var->type())
        {
          case ncFloat:
          {
              float *inp = new float[n];
              var->set_cur(0,
                           domainGlobalStart[0],
                           domainGlobalStart[1],
                           domainGlobalStart[2]);
              var->get(inp, 1, nlx_nodes, nly_nodes, nlz_nodes);
              for (int i=0; i<n; i++)
              {
                  //cout << "inp["<<i<<"] = "<<inp[i]<<endl;
                  arr->SetComponentFromDouble(i,0, inp[i]);
              }
              break;
          }
          default:
              THROW(eavlException,"Unsupported netcdf type\n");
        }

        eavlField *field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
        return field;
    }

    return NULL;
}

int
eavlNetCDFDecomposingImporter::GetNumChunks(const string &mesh)
{
    return numchunks;
}
