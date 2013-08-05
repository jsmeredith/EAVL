// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CONNECTIVITY_DEREFERENCE_OP_3_H
#define EAVL_CONNECTIVITY_DEREFERENCE_OP_3_H

#include "eavlOperation.h"
#include "eavlArray.h"
#include "eavlException.h"
#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"

void eavlConnectivityDereferenceOp_3_CPU_regular(int n,
                            eavlRegularConnectivity reg,
                            int *i0, int i0div, int i0mod, int i0mul, int i0add,
                            int *i1, int i1div, int i1mod, int i1mul, int i1add,
                            int *i2, int i2div, int i2mod, int i2mul, int i2add,
                            int *o0, int o0mul, int o0add,
                            int *o1, int o1mul, int o1add,
                            int *o2, int o2mul, int o2add,
                            int *idx, int idxmul, int idxadd)
{
    int compIds[12]; // most likely components is 12 edges for a 3D hexahedron
    for (int i=0; i<n; i++)
    {
        int index_idx = i*idxmul + idxadd;
        int in_index = idx[index_idx];

        int ncomps;
        reg.GetElementComponents(in_index, ncomps, compIds);

        int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
        int index_i1 = ((i / i1div) % i1mod) * i1mul + i1add;
        int index_i2 = ((i / i2div) % i2mod) * i2mul + i2add;
        
        int index_o0 = i*o0mul + o0add;
        int index_o1 = i*o1mul + o1add;
        int index_o2 = i*o2mul + o2add;

        o0[index_o0] = compIds[i0[index_i0]];
        o1[index_o1] = compIds[i1[index_i1]];
        o2[index_o2] = compIds[i2[index_i2]];
    }
}

void eavlConnectivityDereferenceOp_3_CPU_explicit(int n,
                             eavlExplicitConnectivity &conn,
                            int *i0, int i0div, int i0mod, int i0mul, int i0add,
                            int *i1, int i1div, int i1mod, int i1mul, int i1add,
                            int *i2, int i2div, int i2mod, int i2mul, int i2add,
                            int *o0, int o0mul, int o0add,
                            int *o1, int o1mul, int o1add,
                            int *o2, int o2mul, int o2add,
                            int *idx, int idxmul, int idxadd)
{
    int compIds[12]; // most likely components is 12 edges for a 3D hexahedron
    for (int i=0; i<n; i++)
    {
        int index_idx = i*idxmul + idxadd;
        int in_index = idx[index_idx];

        int ncomps;
        conn.GetElementComponents(in_index, ncomps, compIds);

        int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
        int index_i1 = ((i / i1div) % i1mod) * i1mul + i1add;
        int index_i2 = ((i / i2div) % i2mod) * i2mul + i2add;
        
        int index_o0 = i*o0mul + o0add;
        int index_o1 = i*o1mul + o1add;
        int index_o2 = i*o2mul + o2add;

        o0[index_o0] = compIds[i0[index_i0]];
        o1[index_o1] = compIds[i1[index_i1]];
        o2[index_o2] = compIds[i2[index_i2]];
    }
}

#if defined __CUDACC__
#ifndef HAVE_OLD_GPU

__global__ void
eavlConnectivityDereferenceOp_3_kernel_regular(int n,
                            eavlRegularConnectivity reg,
                            int *i0, int i0div, int i0mod, int i0mul, int i0add,
                            int *i1, int i1div, int i1mod, int i1mul, int i1add,
                            int *i2, int i2div, int i2mod, int i2mul, int i2add,
                            int *o0, int o0mul, int o0add,
                            int *o1, int o1mul, int o1add,
                            int *o2, int o2mul, int o2add,
                            int *idx, int idxmul, int idxadd)
{
    int compIds[12]; // most likely components is 12 edges for a 3D hexahedron
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < n; i += numThreads)
    {
        int index_idx = i*idxmul + idxadd;
        int in_index = idx[index_idx];

        int ncomps;
        int shapeType = reg.GetElementComponents(in_index, ncomps, compIds);

        int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
        int index_i1 = ((i / i1div) % i1mod) * i1mul + i1add;
        int index_i2 = ((i / i2div) % i2mod) * i2mul + i2add;
        
        int index_o0 = i*o0mul + o0add;
        int index_o1 = i*o1mul + o1add;
        int index_o2 = i*o2mul + o2add;

        o0[index_o0] = compIds[i0[index_i0]];
        o1[index_o1] = compIds[i1[index_i1]];
        o2[index_o2] = compIds[i2[index_i2]];
    }
}

__global__ void
eavlConnectivityDereferenceOp_3_kernel_explicit(int n,
                            eavlExplicitConnectivity conn,
                            int *i0, int i0div, int i0mod, int i0mul, int i0add,
                            int *i1, int i1div, int i1mod, int i1mul, int i1add,
                            int *i2, int i2div, int i2mod, int i2mul, int i2add,
                            int *o0, int o0mul, int o0add,
                            int *o1, int o1mul, int o1add,
                            int *o2, int o2mul, int o2add,
                            int *idx, int idxmul, int idxadd)
{
    int compIds[12]; // most likely components is 12 edges for a 3D hexahedron
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < n; i += numThreads)
    {
        int index_idx = i*idxmul + idxadd;
        int in_index = idx[index_idx];

        int ncomps;
        int shapeType = conn.GetElementComponents(in_index, ncomps, compIds);

        int index_i0 = ((i / i0div) % i0mod) * i0mul + i0add;
        int index_i1 = ((i / i1div) % i1mod) * i1mul + i1add;
        int index_i2 = ((i / i2div) % i2mod) * i2mul + i2add;
        
        int index_o0 = i*o0mul + o0add;
        int index_o1 = i*o1mul + o1add;
        int index_o2 = i*o2mul + o2add;

        o0[index_o0] = compIds[i0[index_i0]];
        o1[index_o1] = compIds[i1[index_i1]];
        o2[index_o2] = compIds[i2[index_i2]];
    }
}

#endif
#endif


// ****************************************************************************
// Class:  eavlConnectivityDereferenceOp_3
//
// Purpose:
///   Turns a set of local connectivity information (e.g. an edge number
///   0 to 11 for a voxel) and cell indices, and using the appropriate
///   topology map, converts it to a global index (e.g. global edge number).
///   This version operates on three arrays simultaneously, e.g. for
///   converting a triangle defined by local edge indices into a global
///   set of edge indices.
//
// Programmer:  Jeremy Meredith
// Creation:    April 10, 2012
//
// Modifications:
// ****************************************************************************
class eavlConnectivityDereferenceOp_3 : public eavlOperation
{
  protected:
    eavlCellSet     *cells;
    eavlTopology topology;
    eavlArrayWithLinearIndex inArray0, inArray1, inArray2;
    eavlArrayWithLinearIndex outArray0, outArray1, outArray2;
    eavlArrayWithLinearIndex indexArray;
  public:
    eavlConnectivityDereferenceOp_3(eavlCellSet *inCells,
                         eavlTopology topo,
                         eavlArrayWithLinearIndex in0,
                         eavlArrayWithLinearIndex in1,
                         eavlArrayWithLinearIndex in2,
                         eavlArrayWithLinearIndex out0,
                         eavlArrayWithLinearIndex out1,
                         eavlArrayWithLinearIndex out2,
                         eavlArrayWithLinearIndex index)
        : cells(inCells),
          topology(topo),
          inArray0(in0),
          inArray1(in1),
          inArray2(in2),
          outArray0(out0),
          outArray1(out1),
          outArray2(out2),
          indexArray(index)
    {
    }

    virtual void GoCPU()
    {
        int n = outArray0.array->GetNumberOfTuples();
        ///\todo: assert all output arrays have same length?

        eavlIntArray *i0 = dynamic_cast<eavlIntArray*>(inArray0.array);
        eavlIntArray *i1 = dynamic_cast<eavlIntArray*>(inArray1.array);
        eavlIntArray *i2 = dynamic_cast<eavlIntArray*>(inArray2.array);
        eavlIntArray *o0 = dynamic_cast<eavlIntArray*>(outArray0.array);
        eavlIntArray *o1 = dynamic_cast<eavlIntArray*>(outArray1.array);
        eavlIntArray *o2 = dynamic_cast<eavlIntArray*>(outArray2.array);
        eavlIntArray *idx = dynamic_cast<eavlIntArray*>(indexArray.array);
        if (!i0 || !i1 || !i2 || !o0 || !o1 || !o2 || !idx)
            THROW(eavlException,"eavlConnectivityDereferenceOp expects all integer arrays.");

        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            eavlConnectivityDereferenceOp_3_CPU_explicit
                (n,
                 elExp->GetConnectivity(topology),
                 (int*)i0->GetHostArray(), inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                 (int*)i1->GetHostArray(), inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                 (int*)i2->GetHostArray(), inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                 (int*)o0->GetHostArray(), outArray0.mul, outArray0.add,
                 (int*)o1->GetHostArray(), outArray1.mul, outArray1.add,
                 (int*)o2->GetHostArray(), outArray2.mul, outArray2.add,
                 (int*)idx->GetHostArray(), indexArray.mul, indexArray.add);
        }
        else if (elStr)
        {
            eavlConnectivityDereferenceOp_3_CPU_regular
                (n,
                 eavlRegularConnectivity(elStr->GetRegularStructure(),topology),
                 (int*)i0->GetHostArray(), inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                 (int*)i1->GetHostArray(), inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                 (int*)i2->GetHostArray(), inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                 (int*)o0->GetHostArray(), outArray0.mul, outArray0.add,
                 (int*)o1->GetHostArray(), outArray1.mul, outArray1.add,
                 (int*)o2->GetHostArray(), outArray2.mul, outArray2.add,
                 (int*)idx->GetHostArray(), indexArray.mul, indexArray.add);
        }
        else
        {
            THROW(eavlException,"eavlConnectivityDereferenceOp_3 didn't understand the mesh type.");
        }
    }

    virtual void GoGPU()
    {
#ifdef HAVE_OLD_GPU
        THROW(eavlException,"This method not supported when old GPU support is enabled.");
#else
#if defined __CUDACC__
        int n = outArray0.array->GetNumberOfTuples();
        eavlIntArray *i0 = dynamic_cast<eavlIntArray*>(inArray0.array);
        eavlIntArray *i1 = dynamic_cast<eavlIntArray*>(inArray1.array);
        eavlIntArray *i2 = dynamic_cast<eavlIntArray*>(inArray2.array);
        eavlIntArray *o0 = dynamic_cast<eavlIntArray*>(outArray0.array);
        eavlIntArray *o1 = dynamic_cast<eavlIntArray*>(outArray1.array);
        eavlIntArray *o2 = dynamic_cast<eavlIntArray*>(outArray2.array);
        eavlIntArray *idx = dynamic_cast<eavlIntArray*>(indexArray.array);
        if (!i0 || !i1 || !i2 || !o0 || !o1 || !o2 || !idx)
            THROW(eavlException,"eavlConnectivityDereferenceOp expects all integer arrays.");

        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            int numThreads = 256;
            int numBlocks  = 32;
            eavlExplicitConnectivity &conn = elExp->GetConnectivity(topology);
            conn.shapetype.NeedOnDevice();
            conn.connectivity.NeedOnDevice();
            conn.mapCellToIndex.NeedOnDevice();

            eavlConnectivityDereferenceOp_3_kernel_explicit<<<numBlocks,numThreads>>>
                (n,
                 conn,
                 (int*)i0->GetCUDAArray(), inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                 (int*)i1->GetCUDAArray(), inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                 (int*)i2->GetCUDAArray(), inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                 (int*)o0->GetCUDAArray(), outArray0.mul, outArray0.add,
                 (int*)o1->GetCUDAArray(), outArray1.mul, outArray1.add,
                 (int*)o2->GetCUDAArray(), outArray2.mul, outArray2.add,
                 (int*)idx->GetCUDAArray(), indexArray.mul, indexArray.add);
            CUDA_CHECK_ERROR();

            conn.shapetype.NeedOnDevice();
            conn.connectivity.NeedOnDevice();
            conn.mapCellToIndex.NeedOnDevice();
        }
        else if (elStr)
        {
            int numThreads = 256;
            int numBlocks  = 32;
            eavlConnectivityDereferenceOp_3_kernel_regular<<<numBlocks,numThreads>>>
                (n,
                 eavlRegularConnectivity(elStr->GetRegularStructure(),topology),
                 (int*)i0->GetCUDAArray(), inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                 (int*)i1->GetCUDAArray(), inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                 (int*)i2->GetCUDAArray(), inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                 (int*)o0->GetCUDAArray(), outArray0.mul, outArray0.add,
                 (int*)o1->GetCUDAArray(), outArray1.mul, outArray1.add,
                 (int*)o2->GetCUDAArray(), outArray2.mul, outArray2.add,
                 (int*)idx->GetCUDAArray(), indexArray.mul, indexArray.add);
            CUDA_CHECK_ERROR();
        }
        else
        {
            THROW(eavlException,"eavlConnectivityDereferenceOp_3 didn't understand the mesh type.");
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
#endif
    }
};

#endif
