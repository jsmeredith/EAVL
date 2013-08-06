// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlIsosurfaceFilter.h"

#include "eavlExecutor.h"
#include "eavlCellSetExplicit.h"
#include "eavlDestinationTopologyPackedMapOp.h"
#include "eavlCombinedTopologyPackedMapOp.h"
#include "eavlCoordinates.h"
#include "eavlGatherOp_1.h"
#include "eavlMapOp.h"
#include "eavlPrefixSumOp_1.h"
#include "eavlReduceOp_1.h"
#include "eavlReverseIndexOp.h"
#include "eavlSimpleReverseIndexOp.h"
#include "eavlSimpleTopologyMapOp.h"
#include "eavlInfoTopologyMapOp.h"
#include "eavlInfoTopologyPackedMapOp.h"
#include "eavlSimpleTopologyGatherMapOp.h"
#include "eavlException.h"

#include "eavlNewIsoTables.h"
#include "eavlTimer.h"

class HiLoToCaseFunctor
{
  public:
    template <class IN>
    EAVL_FUNCTOR int operator()(int shapeType, int n, int ids[],
                                IN hilo)
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
                                  IN vals)
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

class LinterpFunctor
{
  public:
    template <class IN>
    EAVL_FUNCTOR float operator()(int shapeType, int n, int ids[],
                                  IN vals, float alpha)
    {
        float a = collect(ids[0], vals);
        float b = collect(ids[1], vals);
        return a + alpha*(b-a);
    }
};

class LinterpFunctor3
{
  public:
    ///\todo: this makes a good spot to test whether to use collecttypes for speed
    /// or if tuples are just as fast (and more convenient).
    template <class IN>
    EAVL_FUNCTOR tuple<float,float,float> operator()(int shapeType, int n, int ids[],
                                                     IN vals, float alpha)
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


class FirstTwoItemsDifferFunctor
{
  public:
    template <class IN>
    EAVL_FUNCTOR int operator()(int shapeType, int n, int ids[],
                                IN vals)
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

    eavlField   *inField = input->GetField(fieldname);
    if (inField->GetAssociation() != eavlField::ASSOC_POINTS)
        THROW(eavlException,"Isosurface expected point-centered field");

    int npts = input->GetNumPoints();
    int ncells = inCells->GetNumCells();

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
    eavlCellSetExplicit *outCellSet = new eavlCellSetExplicit("iso",2);
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
        new_eavlSimpleTopologyMapOp(inCells,
                              EAVL_NODES_OF_CELLS,
                              eavlOpArgs(hiloArray),
                              eavlOpArgs(caseArray),
                              HiLoToCaseFunctor()),
        "generate case index per cell");

    // look up case index in the table to get output counts
    ///\todo: we need a "EAVL_CELLS" equivalent here; we don't care
    /// what "from" topo type, just that we want the mapping for cells.
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
        new_eavlSimpleTopologyMapOp(inCells,
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
    eavlIntArray *revPtEdgeIndex = new eavlIntArray("revPtEdgeIndex",1,noutpts);
    eavlIntArray *revInputIndex = new eavlIntArray("revInputIndex", 1, noutgeom);
    eavlIntArray *revInputSubindex = new eavlIntArray("revInputSubindex", 1, noutgeom);
    eavlByteArray *outcaseArray = new eavlByteArray("outcase", 1, noutgeom);
    eavlIntArray *localouttriArray = new eavlIntArray("localouttri", 3, noutgeom);
    eavlIntArray *outtriArray = new eavlIntArray("outtri", 3, noutgeom);
    eavlIntArray *outconn = new eavlIntArray("outconn", 3, noutgeom);
    eavlFloatArray *alpha = new eavlFloatArray("alpha", 1, noutpts);
    eavlFloatArray *newx = new eavlFloatArray("newx", 1, noutpts);
    eavlFloatArray *newy = new eavlFloatArray("newy", 1, noutpts);
    eavlFloatArray *newz = new eavlFloatArray("newz", 1, noutpts);

    // do reverse index for outpts to inedges
    eavlExecutor::AddOperation(
        new eavlSimpleReverseIndexOp(edgeInclArray,
                                     outpointindexArray,
                                     revPtEdgeIndex),
        "generate reverse lookup: output point to input edge");


    // generate output(tri)-to-input(cell) map
    eavlExecutor::AddOperation(
        new eavlReverseIndexOp(numoutArray,
                               outindexArray,
                               revInputIndex,
                               revInputSubindex,
                               5),
        "generate reverse lookup: output triangle to input cell");

    // gather input cell lookup to output-length array
    eavlExecutor::AddOperation(
        new eavlGatherOp_1(caseArray,
                           outcaseArray,
                           revInputIndex),
        "copy input case from cells to output array for each generated triangle");

    // look up case+subindex in the table using input cell to get output geom
    ///\todo: is this operation plus the gatherop prior to it not just a combined topology gather map?
    /// No, the problem is that the caseArray is sparsely indexed through the indices array,
    /// while the revInputSubIndex array is densely indexed (i.e. directly), but both are
    /// arrays on the output topology.  We don't have an InfoTopologyMap with two separate
    /// inputs, one sparse and one dense.  (It might make a useful addition, I suppose, but
    /// not necessarily.)
    ///\todo: need EAVL_CELLS instead of nodes-of-cells.
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

    // map global edge indices for triangles to output point index
    ///\todo: this would be better with a gatherop_3, but even better
    /// if we had an easy way to flatten the 3-component array into a
    /// single-component array, since all components are treated identically.
    eavlExecutor::AddOperation(new eavlGatherOp_1(outpointindexArray,
                                                  eavlArrayWithLinearIndex(outconn,0),
                                                  eavlArrayWithLinearIndex(outtriArray,0)),
                               "(a) turn input edge ids (for output triangles) into output point ids");
    eavlExecutor::AddOperation(new eavlGatherOp_1(outpointindexArray,
                                                  eavlArrayWithLinearIndex(outconn,1),
                                                  eavlArrayWithLinearIndex(outtriArray,1)),
                               "(b) turn input edge ids (for output triangles) into output point ids");
    eavlExecutor::AddOperation(new eavlGatherOp_1(outpointindexArray,
                                                  eavlArrayWithLinearIndex(outconn,2),
                                                  eavlArrayWithLinearIndex(outtriArray,2)),
                               "(c) turn input edge ids (for output triangles) into output point ids");
    
                                                    
    //
    // get the original coordinate arrays
    //
    eavlCoordinates *cs = input->GetCoordinateSystem(0);
    if (cs->GetDimension() != 3)
        THROW(eavlException,"eavlIsosurfaceFilter assumes 3D coordinates");

    eavlCoordinateAxisField *axis0 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(0));
    eavlCoordinateAxisField *axis1 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(1));
    eavlCoordinateAxisField *axis2 = dynamic_cast<eavlCoordinateAxisField*>(cs->GetAxis(2));

    if (!axis0 || !axis1 || !axis2)
        THROW(eavlException,"eavlIsosurfaceFilter expects only field-based coordinate axes");

    eavlField *field0 = input->GetField(axis0->GetFieldName());
    eavlField *field1 = input->GetField(axis1->GetFieldName());
    eavlField *field2 = input->GetField(axis2->GetFieldName());
    eavlArray *arr0 = field0->GetArray();
    eavlArray *arr1 = field1->GetArray();
    eavlArray *arr2 = field2->GetArray();
    if (!arr0 || !arr1 || !arr2)
    {
        THROW(eavlException,"eavlIsosurfaceFilter couldn't get coordinate arrays");
    }

    eavlIndexable<eavlArray> ali0(arr0, axis0->GetComponent());
    eavlIndexable<eavlArray> ali1(arr1, axis1->GetComponent());
    eavlIndexable<eavlArray> ali2(arr2, axis2->GetComponent());
    
    eavlLogicalStructureRegular *logReg = dynamic_cast<eavlLogicalStructureRegular*>(input->GetLogicalStructure());
    if (logReg)
    {
        eavlRegularStructure &reg = logReg->GetRegularStructure();

        if (field0->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            ali0 = eavlIndexable<eavlArray>(arr0, axis0->GetComponent(), reg, field0->GetAssocLogicalDim());
        if (field1->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            ali1 = eavlIndexable<eavlArray>(arr1, axis1->GetComponent(), reg, field1->GetAssocLogicalDim());
        if (field2->GetAssociation() == eavlField::ASSOC_LOGICALDIM)
            ali2 = eavlIndexable<eavlArray>(arr2, axis2->GetComponent(), reg, field2->GetAssocLogicalDim());
    }

    // generate alphas for each output 
    eavlExecutor::AddOperation(
        new_eavlSimpleTopologyGatherMapOp(inCells,
                                          EAVL_NODES_OF_EDGES,
                                          eavlOpArgs(inField->GetArray()),
                                          eavlOpArgs(alpha),
                                          eavlOpArgs(revPtEdgeIndex),
                                          CalcAlphaFunctor(value)),
        "generate alphas");

    // using the alphas, interpolate to create the new coordinate arrays
    eavlExecutor::AddOperation(
        new_eavlCombinedTopologyPackedMapOp(inCells,
                                            EAVL_NODES_OF_EDGES,
                                            eavlOpArgs(ali0,ali1,ali2),
                                            eavlOpArgs(alpha),
                                            eavlOpArgs(newx,newy,newz),
                                            eavlOpArgs(revPtEdgeIndex),
                                            LinterpFunctor3()),
        "generate xyz coords");

    /// interpolate the point vars and gather the cell vars
    for (int i=0; i<input->GetNumFields(); i++)
    {
        eavlField *f = input->GetField(i);
        eavlArray *a = f->GetArray();
        if (f == field0 ||
            f == field1 ||
            f == field2)
        {
            // we already did the coord fields
            continue;
        }
        if (f->GetArray()->GetNumberOfComponents() != 1)
        {
            ///\todo: currently only handle point and cell fields
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
                                                    LinterpFunctor()),
              "interpolate nodal field");
            output->AddField(new eavlField(1, outArr, eavlField::ASSOC_POINTS));
        }
        else if (f->GetAssociation() == eavlField::ASSOC_CELL_SET &&
                 f->GetAssocCellSet() == inCellSetIndex)
        {
            eavlArray *outArr = a->Create(a->GetName(), 1, noutgeom);
            eavlExecutor::AddOperation(new eavlGatherOp_1(a,
                                                          outArr,
                                                          revInputIndex),
                                       "gather cell field");
            output->AddField(
                new eavlField(1, outArr, eavlField::ASSOC_CELL_SET, 0));
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

    eavlCoordinatesCartesian *coordsys =
        new eavlCoordinatesCartesian(NULL,
                                     eavlCoordinatesCartesian::X,
                                     eavlCoordinatesCartesian::Y,
                                     eavlCoordinatesCartesian::Z);
    ///\todo: assuming 3D coords
    coordsys->SetAxis(0, new eavlCoordinateAxisField("newx", 0));
    coordsys->SetAxis(1, new eavlCoordinateAxisField("newy", 0));
    coordsys->SetAxis(2, new eavlCoordinateAxisField("newz", 0));
    output->AddCoordinateSystem(coordsys);

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
        input->AddField(new eavlField(0, caseArray, eavlField::ASSOC_CELL_SET, 0));
        input->AddField(new eavlField(0, numoutArray, eavlField::ASSOC_CELL_SET, 0));
        input->AddField(new eavlField(0, outindexArray, eavlField::ASSOC_CELL_SET, 0));
    }

    if (false)
    {
        output->AddField(new eavlField(0, revPtEdgeIndex, eavlField::ASSOC_POINTS));
        output->AddField(new eavlField(0, revInputIndex, eavlField::ASSOC_CELL_SET, 0));
        output->AddField(new eavlField(0, revInputSubindex, eavlField::ASSOC_CELL_SET, 0));
        output->AddField(new eavlField(0, outcaseArray, eavlField::ASSOC_CELL_SET, 0));
        output->AddField(new eavlField(0, localouttriArray, eavlField::ASSOC_CELL_SET, 0));
        output->AddField(new eavlField(0, outtriArray, eavlField::ASSOC_CELL_SET, 0));    
        output->AddField(new eavlField(0, outconn, eavlField::ASSOC_CELL_SET, 0));
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

    output->AddField(new eavlField(1, newx, eavlField::ASSOC_POINTS));
    output->AddField(new eavlField(1, newy, eavlField::ASSOC_POINTS));
    output->AddField(new eavlField(1, newz, eavlField::ASSOC_POINTS));
    eavlTimer::Resume();
}
