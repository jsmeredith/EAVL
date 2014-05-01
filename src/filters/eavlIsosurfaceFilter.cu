// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlIsosurfaceFilter.h"

#include "eavlExecutor.h"
#include "eavlCellSetExplicit.h"
#include "eavlDestinationTopologyPackedMapOp.h"
#include "eavlCombinedTopologyPackedMapOp.h"
#include "eavlCoordinates.h"
#include "eavlGatherOp.h"
#include "eavlMapOp.h"
#include "eavlPrefixSumOp_1.h"
#include "eavlReduceOp_1.h"
#include "eavlReverseIndexOp.h"
#include "eavlSimpleReverseIndexOp.h"
#include "eavlSourceTopologyMapOp.h"
#include "eavlInfoTopologyMapOp.h"
#include "eavlInfoTopologyPackedMapOp.h"
#include "eavlSourceTopologyGatherMapOp.h"
#include "eavlException.h"

#include "eavlNewIsoTables.h"
#include "eavlTimer.h"

class HiLoToCaseFunctor
{
  public:
    template <class IN>
    EAVL_FUNCTOR int operator()(int shapeType, int n, int ids[],
                                const IN hilo)
    {
        int caseindex = collect(ids[n-1], hilo);
        for (int i=n-2; i>=0; --i)
            caseindex = 2*caseindex + collect(ids[i], hilo);
        return caseindex;
    }
};


class CalcAlphaFunctor
{
    float target;
  public:
    CalcAlphaFunctor(float tgt) : target(tgt) { }
    template <class IN>
    EAVL_FUNCTOR float operator()(int shapeType, int n, int ids[],
                                  const IN vals)
    {
        // we're assuming vals[0] != vals[1] here, but note
        // that we only call this routine for edges which will
        // be present in the final output, which only happens
        // if one edge node was > tgt and one was <= tgt, so 
        // they must be different in the way we call it.
        float a = collect(ids[0], vals);
        float b = collect(ids[1], vals);
        return (target - a) / (b - a);
    }
};

class LinterpFunctor1
{
  public:
    template <class IN>
    EAVL_FUNCTOR float operator()(int shapeType, int n, int ids[],
                                  const IN vals, float alpha)
    {
        float a = collect(ids[0], vals);
        float b = collect(ids[1], vals);
        return a + alpha*(b-a);
    }
};

class LinterpFunctor2
{
  public:
    template <class IN>
    EAVL_FUNCTOR tuple<float,float> operator()(int shapeType, int n, int ids[],
                                               const IN vals, float alpha)
    {
        tuple<float,float> a = collect(ids[0], vals);
        tuple<float,float> b = collect(ids[1], vals);
        return tuple<float,float>(get<0>(a) + alpha*(get<0>(b)-get<0>(a)),
                                  get<1>(a) + alpha*(get<1>(b)-get<1>(a)));
    }
};

class LinterpFunctor3
{
  public:
    ///\todo: this makes a good spot to test whether to use collecttypes for speed
    /// or if tuples are just as fast (and more convenient).
    template <class IN>
    EAVL_FUNCTOR tuple<float,float,float> operator()(int shapeType, int n, int ids[],
                                                     const IN vals, float alpha)
    {
        tuple<float,float,float> a = collect(ids[0], vals);
        tuple<float,float,float> b = collect(ids[1], vals);
        return tuple<float,float,float>(get<0>(a) + alpha*(get<0>(b)-get<0>(a)),
                                        get<1>(a) + alpha*(get<1>(b)-get<1>(a)),
                                        get<2>(a) + alpha*(get<2>(b)-get<2>(a)));
    }
};

struct ConnectivityDererenceFunctor3
{
    EAVL_FUNCTOR tuple<int,int,int> operator()(int shapeType, int n, int ids[], tuple<int,int,int> localids)
    {
        return tuple<int,int,int>(ids[get<0>(localids)],
                                  ids[get<1>(localids)],
                                  ids[get<2>(localids)]);
    }
};

struct ConnectivityDererenceFunctor2
{
    EAVL_FUNCTOR tuple<int,int> operator()(int shapeType, int n, int ids[], tuple<int,int> localids)
    {
        return tuple<int,int>(ids[get<0>(localids)],
                              ids[get<1>(localids)]);
    }
};

struct ConnectivityDererenceFunctor1
{
    EAVL_FUNCTOR int operator()(int shapeType, int n, int ids[], int localids)
    {
        return ids[localids];
    }
};


class FirstTwoItemsDifferFunctor
{
  public:
    template <class IN>
    EAVL_FUNCTOR int operator()(int shapeType, int n, int ids[],
                                const IN vals)
    {
        return collect(ids[0],vals) != collect(ids[1],vals);
    }
};


class Iso3DLookupCounts
{
    eavlConstArray<byte> tetcount;
    eavlConstArray<byte> pyrcount;
    eavlConstArray<byte> wdgcount;
    eavlConstArray<byte> hexcount;
    eavlConstArray<byte> voxcount;
  public:
    Iso3DLookupCounts(eavlConstArray<byte> *tetcount_,
                      eavlConstArray<byte> *pyrcount_,
                      eavlConstArray<byte> *wdgcount_,
                      eavlConstArray<byte> *hexcount_,
                      eavlConstArray<byte> *voxcount_)
        : tetcount(*tetcount_),
          pyrcount(*pyrcount_),
          wdgcount(*wdgcount_),
          hexcount(*hexcount_),
          voxcount(*voxcount_)
    {
    }
    EAVL_FUNCTOR int operator()(int shapeType, int caseindex)
    {
        switch (shapeType)
        {
          case EAVL_TET:     return tetcount[caseindex];
          case EAVL_PYRAMID: return pyrcount[caseindex];
          case EAVL_WEDGE:   return wdgcount[caseindex];
          case EAVL_HEX:     return hexcount[caseindex];
          case EAVL_VOXEL:   return voxcount[caseindex];
        }
        return 0;
    }
};


class Iso3DLookupTris
{
    eavlConstArray<int>  tetstart;
    eavlConstArray<byte> tetgeom;
    eavlConstArray<int>  pyrstart;
    eavlConstArray<byte> pyrgeom;
    eavlConstArray<int>  wdgstart;
    eavlConstArray<byte> wdggeom;
    eavlConstArray<int>  hexstart;
    eavlConstArray<byte> hexgeom;
    eavlConstArray<int>  voxstart;
    eavlConstArray<byte> voxgeom;
  public:
    Iso3DLookupTris(eavlConstArray<int>  *tetstart_,
                    eavlConstArray<byte> *tetgeom_,
                    eavlConstArray<int>  *pyrstart_,
                    eavlConstArray<byte> *pyrgeom_,
                    eavlConstArray<int>  *wdgstart_,
                    eavlConstArray<byte> *wdggeom_,
                    eavlConstArray<int>  *hexstart_,
                    eavlConstArray<byte> *hexgeom_,
                    eavlConstArray<int>  *voxstart_,
                    eavlConstArray<byte> *voxgeom_)
        : tetstart(*tetstart_), tetgeom(*tetgeom_),
          pyrstart(*pyrstart_), pyrgeom(*pyrgeom_),
          wdgstart(*wdgstart_), wdggeom(*wdggeom_),
          hexstart(*hexstart_), hexgeom(*hexgeom_),
          voxstart(*voxstart_), voxgeom(*voxgeom_)
    {
    }
    EAVL_FUNCTOR tuple<int,int,int> operator()(int shapeType, tuple<int,int> index)
    {
        int caseindex = get<0>(index);
        int subindex = get<1>(index);
                                                   
        int startindex;
        int localedge0, localedge1, localedge2;
        switch (shapeType)
        {
          case EAVL_TET:
            startindex = tetstart[caseindex] + 3*subindex;
            localedge0 = tetgeom[startindex+0];
            localedge1 = tetgeom[startindex+1];
            localedge2 = tetgeom[startindex+2];
            break;
          case EAVL_PYRAMID:
            startindex = pyrstart[caseindex] + 3*subindex;
            localedge0 = pyrgeom[startindex+0];
            localedge1 = pyrgeom[startindex+1];
            localedge2 = pyrgeom[startindex+2];
            break;
          case EAVL_WEDGE:
            startindex = wdgstart[caseindex] + 3*subindex;
            localedge0 = wdggeom[startindex+0];
            localedge1 = wdggeom[startindex+1];
            localedge2 = wdggeom[startindex+2];
            break;
          case EAVL_HEX:
            startindex = hexstart[caseindex] + 3*subindex;
            localedge0 = hexgeom[startindex+0];
            localedge1 = hexgeom[startindex+1];
            localedge2 = hexgeom[startindex+2];
            break;
          case EAVL_VOXEL:
            startindex = voxstart[caseindex] + 3*subindex;
            localedge0 = voxgeom[startindex+0];
            localedge1 = voxgeom[startindex+1];
            localedge2 = voxgeom[startindex+2];
            break;
          default:
            localedge0 = localedge1 = localedge2 = 0;
            break;
        }
        return tuple<int,int,int>(localedge0,localedge1,localedge2);
    }
};

class Iso2DLookupCounts
{
    eavlConstArray<byte> tricount;
    eavlConstArray<byte> quacount;
    eavlConstArray<byte> pixcount;
  public:
    Iso2DLookupCounts(eavlConstArray<byte> *tricount_,
                      eavlConstArray<byte> *quacount_,
                      eavlConstArray<byte> *pixcount_)
        : tricount(*tricount_),
          quacount(*quacount_),
          pixcount(*pixcount_)
    {
    }
    EAVL_FUNCTOR int operator()(int shapeType, int caseindex)
    {
        switch (shapeType)
        {
          case EAVL_TRI:    return tricount[caseindex];
          case EAVL_QUAD:   return quacount[caseindex];
          case EAVL_PIXEL:  return pixcount[caseindex];
        }
        return 0;
    }
};


class Iso2DLookupLines
{
    eavlConstArray<int>  tristart;
    eavlConstArray<byte> trigeom;
    eavlConstArray<int>  quastart;
    eavlConstArray<byte> quageom;
    eavlConstArray<int>  pixstart;
    eavlConstArray<byte> pixgeom;
  public:
    Iso2DLookupLines(eavlConstArray<int>  *tristart_,
                     eavlConstArray<byte> *trigeom_,
                     eavlConstArray<int>  *quastart_,
                     eavlConstArray<byte> *quageom_,
                     eavlConstArray<int>  *pixstart_,
                     eavlConstArray<byte> *pixgeom_)
        : tristart(*tristart_), trigeom(*trigeom_),
          quastart(*quastart_), quageom(*quageom_),
          pixstart(*pixstart_), pixgeom(*pixgeom_)
    {
    }
    EAVL_FUNCTOR tuple<int,int> operator()(int shapeType, tuple<int,int> index)
    {
        int caseindex = get<0>(index);
        int subindex = get<1>(index);
                                                   
        int startindex;
        int localedge0, localedge1;
        switch (shapeType)
        {
          case EAVL_TRI:
            startindex = tristart[caseindex] + 2*subindex;
            localedge0 = trigeom[startindex+0];
            localedge1 = trigeom[startindex+1];
            break;
          case EAVL_QUAD:
            startindex = quastart[caseindex] + 2*subindex;
            localedge0 = quageom[startindex+0];
            localedge1 = quageom[startindex+1];
            break;
          case EAVL_PIXEL:
            startindex = pixstart[caseindex] + 2*subindex;
            localedge0 = pixgeom[startindex+0];
            localedge1 = pixgeom[startindex+1];
            break;
          default:
            localedge0 = localedge1 = 0;
            break;
        }
        return tuple<int,int>(localedge0,localedge1);
    }
};


class Iso1DLookupCounts
{
  public:
    Iso1DLookupCounts()
    {
    }
    EAVL_FUNCTOR int operator()(int shapeType, int caseindex)
    {
        return (caseindex==1 || caseindex==2) ? 1 : 0;
    }
};


class Iso1DLookupPoints
{
  public:
    Iso1DLookupPoints()
    {
    }
    EAVL_FUNCTOR int operator()(int shapeType, tuple<int,int> index)
    {
        // Assume shapeType == EAVL_BEAM, it's always a point
        // intersecting the single edge composing the beam shape.
        // in other words, always return edge index 0;
        return 0;
    }
};


eavlIsosurfaceFilter::eavlIsosurfaceFilter()
{
    hiloArray = NULL;
    caseArray = NULL;
    numoutArray = NULL;
    outindexArray = NULL;
    totalout = NULL;
    edgeInclArray = NULL;
    outpointindexArray = NULL;
    totaloutpts = NULL;
}

eavlIsosurfaceFilter::~eavlIsosurfaceFilter()
{
    if (hiloArray)
        delete hiloArray;
    if (caseArray)
        delete caseArray;
    if (numoutArray)
        delete numoutArray;
    if (outindexArray)
        delete outindexArray;
    if (totalout)
        delete totalout;
    if (edgeInclArray)
        delete edgeInclArray;
    if (outpointindexArray)
        delete outpointindexArray;
    if (totaloutpts)
        delete totaloutpts;
}

void
eavlIsosurfaceFilter::Execute()
{
    eavlTimer::Suspend();

    int th_init = eavlTimer::Start();
    eavlInitializeIsoTables();

    int inCellSetIndex = input->GetCellSetIndex(cellsetname);
    eavlCellSet *inCells = input->GetCellSet(cellsetname);
    int dimension = inCells->GetDimensionality();

    eavlField   *inField = input->GetField(fieldname);
    if (inField->GetAssociation() != eavlField::ASSOC_POINTS)
        THROW(eavlException,"Isosurface expected point-centered field");

    int npts = input->GetNumPoints();
    int ncells = inCells->GetNumCells();

    ///\todo: assuming first coordinate system
    eavlCoordinates *coordsys = input->GetCoordinateSystem(0);
    int spatialdim = coordsys->GetDimension();

    //
    // allocate internal storage arrays
    //
    if (!hiloArray)
        hiloArray = new eavlByteArray("hilo", 1, npts);
    if (!caseArray)
        caseArray = new eavlByteArray("isocase", 1, ncells);
    if (!numoutArray)
        numoutArray = new eavlIntArray("numout", 1, ncells);
    if (!outindexArray)
        outindexArray = new eavlIntArray("outindex", 1, ncells);
    if (!totalout)
        totalout = new eavlIntArray("totalout", 1, 1);
    if (!edgeInclArray)
        edgeInclArray = new eavlIntArray("edgeIncl", 1, inCells->GetNumEdges());
    if (!outpointindexArray)
        outpointindexArray = new eavlIntArray("outpointindex", 1, inCells->GetNumEdges());
    if (!totaloutpts)
        totaloutpts = new eavlIntArray("totaloutpts", 1, 1);


    //
    // set up output mesh
    //
    eavlCellSetExplicit *outCellSet = new eavlCellSetExplicit("iso", dimension-1);
    output->AddCellSet(outCellSet);
    eavlTimer::Stop(th_init, "initialization");

    //
    // do isosurface
    //

    // map scalars to above/below (hi/lo) booleans
    eavlExecutor::AddOperation(
        new_eavlMapOp(eavlOpArgs(inField->GetArray()),
                      eavlOpArgs(hiloArray),
                      eavlLessThanConstFunctor<float>(value)),
        "generate hi/lo boolean");

    // map the cell nodes' hi/lo as a bitfield, i.e. into a case index
    eavlExecutor::AddOperation(
        new_eavlSourceTopologyMapOp(inCells,
                                    EAVL_NODES_OF_CELLS,
                                    eavlOpArgs(hiloArray),
                                    eavlOpArgs(caseArray),
                                    HiLoToCaseFunctor()),
        "generate case index per cell");

    // look up case index in the table to get output counts
    ///\todo: we need a "EAVL_CELLS" equivalent here; we don't care
    /// what "from" topo type, just that we want the mapping for cells.
    if (dimension == 3)
    {
        eavlExecutor::AddOperation(
            new_eavlInfoTopologyMapOp(inCells,
                                  EAVL_NODES_OF_CELLS,
                                  eavlOpArgs(caseArray),
                                  eavlOpArgs(numoutArray),
                                  Iso3DLookupCounts(eavlTetIsoTriCount,
                                                    eavlPyrIsoTriCount,
                                                    eavlWdgIsoTriCount,
                                                    eavlHexIsoTriCount,
                                                    eavlVoxIsoTriCount)),
        "look up output tris per cell case");
    }
    else if (dimension == 2)
    {
        eavlExecutor::AddOperation(
             new_eavlInfoTopologyMapOp(inCells,
                                  EAVL_NODES_OF_CELLS,
                                  eavlOpArgs(caseArray),
                                  eavlOpArgs(numoutArray),
                                  Iso2DLookupCounts(eavlTriIsoLineCount,
                                                    eavlQuadIsoLineCount,
                                                    eavlPixelIsoLineCount)),
        "look up output lines per cell case");
    }
    else // (dimension == 1)
    {
        eavlExecutor::AddOperation(
             new_eavlInfoTopologyMapOp(inCells,
                                  EAVL_NODES_OF_CELLS,
                                  eavlOpArgs(caseArray),
                                  eavlOpArgs(numoutArray),
                                  Iso1DLookupCounts()),
        "look up output points per cell case");
    }

    // exclusive scan output counts to get output index
    eavlExecutor::AddOperation(
        new eavlPrefixSumOp_1(numoutArray,
                              outindexArray,
                              false),
        "scan to generate starting out geom index");


    // count overall geometry
    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlAddFunctor<int> >
            (numoutArray,
             totalout,
             eavlAddFunctor<int>()),
        "sumreduce to count output geom");

    // figure out which edges we'll need in the end (not-equal hi-lo)

    ///\todo: if this int array is changed to a byte array, the prefix sum a little later fails.
    /// I would expect it to throw an error (array types don't match because we're putting
    /// the scan result into an int array), but I'm just getting a segfault?
    eavlExecutor::AddOperation(
        new_eavlSourceTopologyMapOp(inCells,
                              EAVL_NODES_OF_EDGES,
                              eavlOpArgs(hiloArray),
                              eavlOpArgs(edgeInclArray),
                              FirstTwoItemsDifferFunctor()),
        "flag edges that have differing hi/lo as they will generate pts in output");
    //for (int i=0; i<inCells->GetNumEdges(); i++) {if (edgeInclArray->GetValue(i)) cerr << "USES EDGE: "<<i<<endl;}

    // generate output-point-to-input-edge map
    // exclusive scan output edge inclusion (i.e. count) to get output index
    eavlExecutor::AddOperation(new eavlPrefixSumOp_1(edgeInclArray,
                                                     outpointindexArray,
                                                     false),
                               "scan edge flags to find starting output point index for each input edge");
    //for (int i=0; i<inCells->GetNumEdges(); i++) {if (edgeInclArray->GetValue(i)) cerr << "EDGE #: "<<i<<" is at index "<<outpointindexArray->GetValue(i)<<endl;}

    // sum reduction to count edges
    eavlExecutor::AddOperation(
        new eavlReduceOp_1<eavlAddFunctor<int> >
            (edgeInclArray,
             totaloutpts,
             eavlAddFunctor<int>()),
        "sumreduce to count output pts (from edges)");

    //
    // We can now execute the plan up to the point, and then
    // we'll know how many output cells and nodes we'll have.
    // That lets us create some final arrays and resize them,
    // then execute the final stage.
    //
    eavlExecutor::Go();
    int noutgeom = totalout->GetValue(0);
    int noutpts = totaloutpts->GetValue(0);
    //cerr << "TOTAL GEOMETRY = "<<noutgeom<<endl;
    //cerr << "TOTAL NEW OUTPUT POINTS = "<<noutpts<<endl;


    ///\todo: some of these are temporary and should be deleted eventually
    /// (though right now we put them all in the output for debugging)
    int each_outgeom_count = dimension;
    eavlIntArray *revPtEdgeIndex = new eavlIntArray("revPtEdgeIndex",1,noutpts);
    eavlIntArray *revInputIndex = new eavlIntArray("revInputIndex", 1, noutgeom);
    eavlIntArray *revInputSubindex = new eavlIntArray("revInputSubindex", 1, noutgeom);
    eavlByteArray *outcaseArray = new eavlByteArray("outcase", 1, noutgeom);
    eavlIntArray *localouttriArray = new eavlIntArray("localouttri", each_outgeom_count, noutgeom);
    eavlIntArray *outtriArray = new eavlIntArray("outtri", each_outgeom_count, noutgeom);
    eavlIntArray *outconn = new eavlIntArray("outconn", each_outgeom_count, noutgeom);
    eavlFloatArray *alpha = new eavlFloatArray("alpha", 1, noutpts);
    eavlFloatArray *newx = spatialdim < 1 ? NULL : new eavlFloatArray("newx", 1, noutpts);
    eavlFloatArray *newy = spatialdim < 2 ? NULL : new eavlFloatArray("newy", 1, noutpts);
    eavlFloatArray *newz = spatialdim < 3 ? NULL : new eavlFloatArray("newz", 1, noutpts);

    // do reverse index for outpts to inedges
    eavlExecutor::AddOperation(
        new eavlSimpleReverseIndexOp(edgeInclArray,
                                     outpointindexArray,
                                     revPtEdgeIndex),
        "generate reverse lookup: output point to input edge");


    //outpointindexArray->PrintSummary(cerr);

    // generate output(tri)-to-input(cell) map
    eavlExecutor::AddOperation(
        new eavlReverseIndexOp(numoutArray,
                               outindexArray,
                               revInputIndex,
                               revInputSubindex,
                               5), ///<\todo: is this right (or even needed)?
        "generate reverse lookup: output triangle to input cell");

    //revInputIndex->PrintSummary(cerr);
    //revInputSubindex->PrintSummary(cerr);

    // gather input cell lookup to output-length array
    eavlExecutor::AddOperation(
        new_eavlGatherOp(eavlOpArgs(caseArray),
                         eavlOpArgs(outcaseArray),
                         eavlOpArgs(revInputIndex)),
        "copy input case from cells to output array for each generated triangle");

    // look up case+subindex in the table using input cell to get output geom
    ///\todo: is this operation plus the gatherop prior to it not just a combined topology gather map?
    /// No, the problem is that the caseArray is sparsely indexed through the indices array,
    /// while the revInputSubIndex array is densely indexed (i.e. directly), but both are
    /// arrays on the output topology.  We don't have an InfoTopologyMap with two separate
    /// inputs, one sparse and one dense.  (It might make a useful addition, I suppose, but
    /// not necessarily.)
    ///\todo: need EAVL_CELLS instead of nodes-of-cells.
    if (dimension == 3)
    {
        eavlExecutor::AddOperation(
            new_eavlInfoTopologyPackedMapOp(inCells,
                                        EAVL_NODES_OF_CELLS,
                                        eavlOpArgs(outcaseArray,
                                                   revInputSubindex),
                                        eavlOpArgs(eavlIndexable<eavlIntArray>(localouttriArray,0),
                                                   eavlIndexable<eavlIntArray>(localouttriArray,1),
                                                   eavlIndexable<eavlIntArray>(localouttriArray,2)),
                                        eavlOpArgs(revInputIndex),
                                        Iso3DLookupTris(eavlTetIsoTriStart, eavlTetIsoTriGeom,
                                                        eavlPyrIsoTriStart, eavlPyrIsoTriGeom,
                                                        eavlWdgIsoTriStart, eavlWdgIsoTriGeom,
                                                        eavlHexIsoTriStart, eavlHexIsoTriGeom,
                                                        eavlVoxIsoTriStart, eavlVoxIsoTriGeom)),
            "generate cell-local output triangle edge indices");

        // map local cell edges to global ones from input mesh
        eavlExecutor::AddOperation(
            new_eavlDestinationTopologyPackedMapOp(inCells,
                                               EAVL_EDGES_OF_CELLS,
                                               eavlOpArgs(eavlIndexable<eavlIntArray>(localouttriArray, 0),
                                                          eavlIndexable<eavlIntArray>(localouttriArray, 1),
                                                          eavlIndexable<eavlIntArray>(localouttriArray, 2)),
                                               eavlOpArgs(eavlIndexable<eavlIntArray>(outtriArray, 0),
                                                          eavlIndexable<eavlIntArray>(outtriArray, 1),
                                                          eavlIndexable<eavlIntArray>(outtriArray, 2)),
                                               eavlOpArgs(revInputIndex),
                                               ConnectivityDererenceFunctor3()),
            "dereference cell-local edges to global edge ids");
    }
    else if (dimension == 2)
    {
        eavlExecutor::AddOperation(
            new_eavlInfoTopologyPackedMapOp(inCells,
                                        EAVL_NODES_OF_CELLS,
                                        eavlOpArgs(outcaseArray,
                                                   revInputSubindex),
                                        eavlOpArgs(eavlIndexable<eavlIntArray>(localouttriArray,0),
                                                   eavlIndexable<eavlIntArray>(localouttriArray,1)),
                                        eavlOpArgs(revInputIndex),
                                        Iso2DLookupLines(eavlTriIsoLineStart, eavlTriIsoLineGeom,
                                                         eavlQuadIsoLineStart, eavlQuadIsoLineGeom,
                                                         eavlPixelIsoLineStart, eavlPixelIsoLineGeom)),
           "generate cell-local output beam edge indices");

        // map local cell edges to global ones from input mesh
        eavlExecutor::AddOperation(
            new_eavlDestinationTopologyPackedMapOp(inCells,
                                               EAVL_EDGES_OF_CELLS,
                                               eavlOpArgs(eavlIndexable<eavlIntArray>(localouttriArray, 0),
                                                          eavlIndexable<eavlIntArray>(localouttriArray, 1)),
                                               eavlOpArgs(eavlIndexable<eavlIntArray>(outtriArray, 0),
                                                          eavlIndexable<eavlIntArray>(outtriArray, 1)),
                                               eavlOpArgs(revInputIndex),
                                               ConnectivityDererenceFunctor2()),
            "dereference cell-local edges to global edge ids");
    }
    else // (dimension == 1)
    {
        eavlExecutor::AddOperation(
            new_eavlInfoTopologyPackedMapOp(inCells,
                                        EAVL_NODES_OF_CELLS,
                                        eavlOpArgs(outcaseArray,
                                                   revInputSubindex),
                                        eavlOpArgs(eavlIndexable<eavlIntArray>(localouttriArray,0)),
                                        eavlOpArgs(revInputIndex),
                                        Iso1DLookupPoints()),
           "generate cell-local output point edge indices");

        // map local cell edges to global ones from input mesh
        eavlExecutor::AddOperation(
            new_eavlDestinationTopologyPackedMapOp(inCells,
                                               EAVL_EDGES_OF_CELLS,
                                               eavlOpArgs(eavlIndexable<eavlIntArray>(localouttriArray, 0)),
                                               eavlOpArgs(eavlIndexable<eavlIntArray>(outtriArray, 0)),
                                               eavlOpArgs(revInputIndex),
                                               ConnectivityDererenceFunctor1()),
            "dereference cell-local edges to global edge ids");
    }

    // map global edge indices for triangles to output point index

    // NOTE: By not creating an eavlIndexable with an array component
    // we're interested in for outconn and outtriArray, even though
    // they are three component arrays, we are treating them as if
    // they were single-component arrays here.  (If we didn't, then
    // we'd have to make three gather calls here.)
    eavlExecutor::AddOperation(new_eavlGatherOp(eavlOpArgs(outpointindexArray),
                                                eavlOpArgs(outconn),
                                                eavlOpArgs(outtriArray)),
                               "turn input edge ids (for output triangles) into output point ids");
                                                    
    // generate alphas for each output 
    eavlExecutor::AddOperation(
        new_eavlSourceTopologyGatherMapOp(inCells,
                                          EAVL_NODES_OF_EDGES,
                                          eavlOpArgs(inField->GetArray()),
                                          eavlOpArgs(alpha),
                                          eavlOpArgs(revPtEdgeIndex),
                                          CalcAlphaFunctor(value)),
        "generate alphas");


    // using the alphas, interpolate to create the new coordinate arrays
    if (spatialdim == 1)
    {
        eavlExecutor::AddOperation(
            new_eavlCombinedTopologyPackedMapOp(inCells,
                                                EAVL_NODES_OF_EDGES,
                                                eavlOpArgs(input->GetIndexableAxis(0)),
                                                eavlOpArgs(alpha),
                                                eavlOpArgs(newx),
                                                eavlOpArgs(revPtEdgeIndex),
                                                LinterpFunctor1()),
            "generate x coords");
    }
    else if (spatialdim == 2)
    {
        eavlExecutor::AddOperation(
            new_eavlCombinedTopologyPackedMapOp(inCells,
                                                EAVL_NODES_OF_EDGES,
                                                eavlOpArgs(input->GetIndexableAxis(0),
                                                           input->GetIndexableAxis(1)),
                                                eavlOpArgs(alpha),
                                                eavlOpArgs(newx,
                                                           newy),
                                                eavlOpArgs(revPtEdgeIndex),
                                                LinterpFunctor2()),
            "generate xy coords");
    }
    else if (spatialdim == 3)
    {
        eavlExecutor::AddOperation(
            new_eavlCombinedTopologyPackedMapOp(inCells,
                                                EAVL_NODES_OF_EDGES,
                                                eavlOpArgs(input->GetIndexableAxis(0),
                                                           input->GetIndexableAxis(1),
                                                           input->GetIndexableAxis(2)),
                                                eavlOpArgs(alpha),
                                                eavlOpArgs(newx,
                                                           newy,
                                                           newz),
                                                eavlOpArgs(revPtEdgeIndex),
                                                LinterpFunctor3()),
            "generate xyz coords");
    }


    /// interpolate the point vars and gather the cell vars
    for (int i=0; i<input->GetNumFields(); i++)
    {
        eavlField *f = input->GetField(i);
        eavlArray *a = f->GetArray();

        // we already did the coordinate fields
        if (coordsys->IsCoordinateAxisField(a->GetName()))
            continue;

        if (f->GetArray()->GetNumberOfComponents() != 1)
        {
            ///\todo: currently only handle point and cell scalar fields
            continue;
        }

        if (f->GetAssociation() == eavlField::ASSOC_POINTS)
        {
            eavlArray *outArr = a->Create(a->GetName(), 1, noutpts);
            eavlExecutor::AddOperation(
                new_eavlCombinedTopologyPackedMapOp(inCells,
                                                    EAVL_NODES_OF_EDGES,
                                                    eavlOpArgs(a),
                                                    eavlOpArgs(alpha),
                                                    eavlOpArgs(outArr),
                                                    eavlOpArgs(revPtEdgeIndex),
                                                    LinterpFunctor1()),
              "interpolate nodal field");
            output->AddField(new eavlField(1, outArr, eavlField::ASSOC_POINTS));
        }
        else if (f->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                 f->GetAssocCellSet() == input->GetCellSet(inCellSetIndex)->GetName())
        {
            eavlArray *outArr = a->Create(a->GetName(), 1, noutgeom);
            eavlExecutor::AddOperation(new_eavlGatherOp(eavlOpArgs(a),
                                                        eavlOpArgs(outArr),
                                                        eavlOpArgs(revInputIndex)),
                                       "gather cell field");
            output->AddField(
                new eavlField(1, outArr, eavlField::ASSOC_CELL_SET, "iso"));
        }
        else
        {
            // skip field: either wrong cell set or not nodal/zonal assoc
        }
    }

    //
    // finalize output mesh
    //
    output->SetNumPoints(noutpts);

    if (spatialdim == 1)
    {
        eavlCoordinatesCartesian *newcoordsys = 
            new eavlCoordinatesCartesian(NULL,
                                         eavlCoordinatesCartesian::X);
        newcoordsys->SetAxis(0, new eavlCoordinateAxisField("newx", 0));
        output->AddCoordinateSystem(newcoordsys);
        output->AddField(new eavlField(1, newx, eavlField::ASSOC_POINTS));
    }
    else if (spatialdim == 2)
    {
        eavlCoordinatesCartesian *newcoordsys = 
            new eavlCoordinatesCartesian(NULL,
                                         eavlCoordinatesCartesian::X,
                                         eavlCoordinatesCartesian::Y);
        newcoordsys->SetAxis(0, new eavlCoordinateAxisField("newx", 0));
        newcoordsys->SetAxis(1, new eavlCoordinateAxisField("newy", 0));
        output->AddCoordinateSystem(newcoordsys);
        output->AddField(new eavlField(1, newx, eavlField::ASSOC_POINTS));
        output->AddField(new eavlField(1, newy, eavlField::ASSOC_POINTS));
    }
    else if (spatialdim == 3)
    {
        eavlCoordinatesCartesian *newcoordsys = 
            new eavlCoordinatesCartesian(NULL,
                                         eavlCoordinatesCartesian::X,
                                         eavlCoordinatesCartesian::Y,
                                         eavlCoordinatesCartesian::Z);
        newcoordsys->SetAxis(0, new eavlCoordinateAxisField("newx", 0));
        newcoordsys->SetAxis(1, new eavlCoordinateAxisField("newy", 0));
        newcoordsys->SetAxis(2, new eavlCoordinateAxisField("newz", 0));
        output->AddCoordinateSystem(newcoordsys);
        output->AddField(new eavlField(1, newx, eavlField::ASSOC_POINTS));
        output->AddField(new eavlField(1, newy, eavlField::ASSOC_POINTS));
        output->AddField(new eavlField(1, newz, eavlField::ASSOC_POINTS));
    }

    //
    // Finish it!
    //
    eavlExecutor::Go();

    // do the cells
    int th_get_conn_to_host = eavlTimer::Start();
    outconn->GetTuple(0);
    eavlTimer::Stop(th_get_conn_to_host, "send connectivity back to host");

    eavlExplicitConnectivity conn;
    int th_create_final_cell_set = eavlTimer::Start();
    if (dimension == 3)
    {
        conn.shapetype.resize(noutgeom);
        conn.connectivity.resize(4*noutgeom);
        for (int i=0; i<noutgeom; i++)
            conn.shapetype[i] = EAVL_TRI;
        for (int i=0; i<noutgeom; i++)
        {
            const int *o = outconn->GetTuple(i);
            conn.connectivity[i*4+0] = 3;
            conn.connectivity[i*4+1] = o[0];
            conn.connectivity[i*4+2] = o[1];
            conn.connectivity[i*4+3] = o[2];
        }
    }
    else if (dimension == 2)
    {
        conn.shapetype.resize(noutgeom);
        conn.connectivity.resize(3*noutgeom);
        for (int i=0; i<noutgeom; i++)
            conn.shapetype[i] = EAVL_BEAM;
        for (int i=0; i<noutgeom; i++)
        {
            const int *o = outconn->GetTuple(i);
            conn.connectivity[i*3+0] = 2;
            conn.connectivity[i*3+1] = o[0];
            conn.connectivity[i*3+2] = o[1];
        }
    }
    else // (dimension == 1)
    {
        conn.shapetype.resize(noutgeom);
        conn.connectivity.resize(2*noutgeom);
        for (int i=0; i<noutgeom; i++)
            conn.shapetype[i] = EAVL_POINT;
        for (int i=0; i<noutgeom; i++)
        {
            const int *o = outconn->GetTuple(i);
            conn.connectivity[i*2+0] = 1;
            conn.connectivity[i*2+1] = o[0];
        }
    }
    eavlTimer::Stop(th_create_final_cell_set, "create final connectivity for cell set");
    int th_create_revindex = eavlTimer::Start();
    outCellSet->SetCellNodeConnectivity(conn);
    eavlTimer::Stop(th_create_revindex, "create reverse index for connectivity");


    // if we want to debug some of the temporary arrays, we can add them back
    // to the input and write out the input data set
    if (false)
    {
        // note: if we do this, we can't delete them in the destructor
        input->AddField(new eavlField(0, hiloArray, eavlField::ASSOC_POINTS));
        input->AddField(new eavlField(0, caseArray, eavlField::ASSOC_CELL_SET, "iso"));
        input->AddField(new eavlField(0, numoutArray, eavlField::ASSOC_CELL_SET, "iso"));
        input->AddField(new eavlField(0, outindexArray, eavlField::ASSOC_CELL_SET, "iso"));
    }

    if (false)
    {
        output->AddField(new eavlField(0, revPtEdgeIndex, eavlField::ASSOC_POINTS));
        output->AddField(new eavlField(0, revInputIndex, eavlField::ASSOC_CELL_SET, "iso"));
        output->AddField(new eavlField(0, revInputSubindex, eavlField::ASSOC_CELL_SET, "iso"));
        output->AddField(new eavlField(0, outcaseArray, eavlField::ASSOC_CELL_SET, "iso"));
        output->AddField(new eavlField(0, localouttriArray, eavlField::ASSOC_CELL_SET, "iso"));
        output->AddField(new eavlField(0, outtriArray, eavlField::ASSOC_CELL_SET, "iso"));    
        output->AddField(new eavlField(0, outconn, eavlField::ASSOC_CELL_SET, "iso"));
        output->AddField(new eavlField(1, alpha, eavlField::ASSOC_POINTS));
    }
    else
    {
        delete revPtEdgeIndex;
        delete revInputIndex;
        delete revInputSubindex;
        delete outcaseArray;
        delete localouttriArray;
        delete outtriArray;
        delete outconn;
        delete alpha;
    }

    eavlTimer::Resume();
}
