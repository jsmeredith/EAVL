// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_GATHER_MAP_OP_2_3_H
#define EAVL_CELL_GATHER_MAP_OP_2_3_H

#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlOpDispatch_2_3_int.h"
#include "eavlException.h"

#ifndef DOXYGEN

template <class F,
          class I0, class I1, class O01, class O2>
struct cpuCellGatherMapOp_2_3_regular
{
    static void call(int nout,
                     eavlRegularStructure &reg,
                     I0  *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I1  *i1, int i1div, int i1mod, int i1mul, int i1add,
                     O01 *o0, int o0mul, int o0add,
                     O01 *o1, int o1mul, int o1add,
                     O2  *o2, int o2mul, int o2add,
                     int *idx, int idxmul, int idxadd,
                     F &functor)
    {
        const int shapeType  = (reg.dimension == 1 ? EAVL_BEAM : 
                                (reg.dimension == 2 ? EAVL_PIXEL :
                                 (reg.dimension == 3 ? EAVL_VOXEL : EAVL_OTHER)));
        for (int i=0; i<nout; i++)
        {
            int index_idx = i*idxmul + idxadd;
            int in_index = idx[index_idx];

            int index_i0 = ((in_index / i0div) % i0mod) * i0mul + i0add;
            int index_i1 = ((in_index / i1div) % i1mod) * i1mul + i1add;

            I0 val_i0 = i0[index_i0];
            I1 val_i1 = i1[index_i1];

            float out0, out1, out2;
            functor(shapeType,
                    val_i0, val_i1,
                    out0, out1, out2);

            int index_o0 = i*o0mul + o0add;
            int index_o1 = i*o1mul + o1add;
            int index_o2 = i*o2mul + o2add;
            o0[index_o0] = out0;
            o1[index_o1] = out1;
            o2[index_o2] = out2;
        }
    }
};


template <class F,
          class I0, class I1, class O01, class O2>
struct cpuCellGatherMapOp_2_3_explicit
{
    static void call(int nout,
                     eavlExplicitConnectivity &conn,
                     I0  *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I1  *i1, int i1div, int i1mod, int i1mul, int i1add,
                     O01 *o0, int o0mul, int o0add,
                     O01 *o1, int o1mul, int o1add,
                     O2  *o2, int o2mul, int o2add,
                     int *idx, int idxmul, int idxadd,
                     F &functor)
    {
        for (int i=0; i<nout; i++)
        {
            int index_idx = i*idxmul + idxadd;
            int in_index = idx[index_idx];

            int shapeType = conn.shapetype[in_index];

            int index_i0 = ((in_index / i0div) % i0mod) * i0mul + i0add;
            int index_i1 = ((in_index / i1div) % i1mod) * i1mul + i1add;

            I0 val_i0 = i0[index_i0];
            I1 val_i1 = i1[index_i1];

            float out0, out1, out2;
            functor(shapeType,
                    val_i0, val_i1,
                    out0, out1, out2);

            int index_o0 = i*o0mul + o0add;
            int index_o1 = i*o1mul + o1add;
            int index_o2 = i*o2mul + o2add;
            o0[index_o0] = out0;
            o1[index_o1] = out1;
            o2[index_o2] = out2;
        }
    }
};

#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlCellSparseMapOp_2_3
//
// Purpose:
///   This is like a eavlCellMapOp, but you can think of the output
///   arrays as being sparse cell-centered arrays (or compacted, if
///   you will). In other words, for each cell index in the lookup
///   array, it passes the input cell values plus the corresponding
///   cell shape information to a functor and stores it at the same
///   location in the output arrays.
///
///   \todo: with the same caveat that a "cell" op is possibly more efficient
///   than a "topology" op, isn't this technically the same as a 
///   eavlTopologyGatherMapOp_2_0_3?  (I think?)  Anyway, this one was
///   never used, we can probably just delete it.
///
//
// Programmer:  Jeremy Meredith
// Creation:    March 11, 2012
//
// Modifications:
// ****************************************************************************
template <class F>
class eavlCellGatherMapOp_2_3 : public eavlOperation
{
  protected:
    eavlCellSet     *cells;
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex inArray1;
    eavlArrayWithLinearIndex outArray0;
    eavlArrayWithLinearIndex outArray1;
    eavlArrayWithLinearIndex outArray2;
    eavlArrayWithLinearIndex indicesArray;
    F functor;
  public:
    eavlCellGatherMapOp_2_3(eavlCellSet *inCells,
                            eavlArrayWithLinearIndex in0,
                            eavlArrayWithLinearIndex in1,
                            eavlArrayWithLinearIndex out0,
                            eavlArrayWithLinearIndex out1,
                            eavlArrayWithLinearIndex out2,
                            eavlArrayWithLinearIndex indices,
                            F f)
        : cells(inCells),
          inArray0(in0), inArray1(in1),
          outArray0(out0), outArray1(out1), outArray2(out2),
          indicesArray(indices),
          functor(f)
    {
    }
    virtual void GoCPU()
    {
        int n = outArray0.array->GetNumberOfTuples();

        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            eavlDispatch_2_3_int<cpuCellGatherMapOp_2_3_explicit>
                (n, eavlArray::HOST,
                 elExp->cellNodeConnectivity,
                 inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                 inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                 outArray0.array, outArray0.mul, outArray0.add,
                 outArray1.array, outArray1.mul, outArray1.add,
                 outArray2.array, outArray2.mul, outArray2.add,
                 indicesArray.array, indicesArray.mul, indicesArray.add,
                 functor);
        }
        else if (elStr)
        {
            eavlDispatch_2_3_int<cpuCellGatherMapOp_2_3_regular>
                (n, eavlArray::HOST,
                 elStr->GetRegularStructure(),
                 inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                 inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                 outArray0.array, outArray0.mul, outArray0.add,
                 outArray1.array, outArray1.mul, outArray1.add,
                 outArray2.array, outArray2.mul, outArray2.add,
                 indicesArray.array, indicesArray.mul, indicesArray.add,
                 functor);
        }
        else
        {
            THROW(eavlException,"eavlCellMapOp didn't understand the mesh type.");
        }
    }
    virtual void GoGPU()
    {
        THROW(eavlException,"unimplemented");
    }
};



#endif
