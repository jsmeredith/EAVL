// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_CELL_MAP_OP_1_1_H
#define EAVL_CELL_MAP_OP_1_1_H

#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOperation.h"
#include "eavlOpDispatch_1_1.h"
#include "eavlException.h"
#include <time.h>
#include <omp.h>

///\todo: this entire iterator is now useless, right?
/// In other words, eavlCellMapOp_1_1 is just eavlTopologyMapOp_0_1_1, right?
/// The only question: we have to pick a topology component-to-component,
/// e.g. nodes-to-cells, or edges-to-cells, when we really just want "cells".
/// Maybe that implies a different iterator, though I believe Cell here
/// is too specific and we might want to support Face and Edge as well.

#ifndef DOXYGEN

template <class F, class I0, class O0>
struct cpuCellMapOp_1_1_regular
{
    static void call(int ncells,
                     eavlRegularStructure &reg,
                     I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *o0, int o0mul, int o0add,
                     F &functor)
    {
        const int shapeType  = (reg.dimension == 1 ? EAVL_BEAM : 
                                (reg.dimension == 2 ? EAVL_PIXEL :
                                 (reg.dimension == 3 ? EAVL_VOXEL : EAVL_OTHER)));
        for (int index = 0; index < ncells; ++index)
        {
            o0[index * o0mul + o0add] = functor(shapeType,
                                                i0[((index / i0div) % i0mod) * i0mul + i0add]);
        }
    }
};

template <class F>
void
callCellMapCPUStructured_1_1(int nCells,
                              eavlRegularStructure reg,
                              eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                              eavlArray *o0, int o0mul, int o0add,
                              F &functor)
{
    eavlDispatch_1_1<cpuCellMapOp_1_1_regular>(nCells,
                                                  eavlArray::HOST,
                                                  reg,
                                                  i0, i0div, i0mod, i0mul, i0add,
                                                  o0, o0mul, o0add,
                                                  functor);
}

template <class F, class I0, class O0>
struct cpuCellMapOp_1_1_explicit
{
    static void call(int ncells,
                     eavlExplicitConnectivity &conn,
                     I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *o0, int o0mul, int o0add,
                     F &functor)
    {
        for (int index = 0; index < ncells; ++index)
        {
            int shapeType = conn.shapetype[index];
            o0[index * o0mul + o0add] = functor(shapeType,
                                                i0[((index / i0div) % i0mod) * i0mul + i0add]);
        }
    }
};


template <class F>
void
callCellMapCPUExplicit_1_1(int nCells,
                              eavlExplicitConnectivity &conn,
                              eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                              eavlArray *o0, int o0mul, int o0add,
                              F &functor)
{
    eavlDispatch_1_1<cpuCellMapOp_1_1_explicit>(nCells,
                                                  eavlArray::HOST,
                                                  conn,
                                                  i0, i0div, i0mod, i0mul, i0add,
                                                  o0, o0mul, o0add,
                                                  functor);
}


#if defined __CUDACC__

template <class F, class I0, class O0>
__global__ void
cellMapKernelRegular_1_1(int n,
                           eavlRegularStructure reg,
                           I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                           O0 *o0, int o0mul, int o0add,
                           F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    const int shapeType  = (reg.dimension == 1 ? EAVL_BEAM : 
                            (reg.dimension == 2 ? EAVL_PIXEL :
                             (reg.dimension == 3 ? EAVL_VOXEL : EAVL_OTHER)));
    for (int index = threadID; index < n; index += numThreads)
    {
        o0[index * o0mul + o0add] = functor(shapeType,
                                            i0[((index / i0div) % i0mod) * i0mul + i0add]);
    }
}

template <class F, class I0, class O0>
struct gpuCellMapOp_1_1_regular
{
    static void call(int ncells,
                     eavlRegularStructure &reg,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *d_o0, int o0mul, int o0add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        cellMapKernelRegular_1_1<<< blocks, threads >>>(ncells,
                                                           reg,
                                                           d_i0, i0div, i0mod, i0mul, i0add,
                                                           d_o0, o0mul, o0add,
                                                           functor);
        CUDA_CHECK_ERROR();
    }
};


template <class F>
void callCellMapKernelStructured_1_1(int nCells,
                          eavlRegularStructure reg,
                          eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                          eavlArray *o0, int o0mul, int o0add,
                          F &functor)
{
    int n = o0->GetNumberOfTuples();
    ///\todo: assert num for all output arrays is the same?

    i0->GetCUDAArray();
    o0->GetCUDAArray();

    // run the kernel
    eavlDispatch_1_1<gpuCellMapOp_1_1_regular>(nCells,
                                                  eavlArray::DEVICE,
                                                  reg,
                                                  i0, i0div, i0mod, i0mul, i0add,
                                                  o0, o0mul, o0add,
                                                  functor);
}

template <class F, class I0, class O0>
__global__ void
cellMapKernelExplicit_1_1(int n,
                             eavlExplicitConnectivity conn,
                             I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                             O0 *o0, int o0mul, int o0add,
                             F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int index = threadID; index < n; index += numThreads)
    {
        int shapeType = conn.shapetype[index];
        o0[index * o0mul + o0add] = functor(shapeType,
                                            i0[((index / i0div) % i0mod) * i0mul + i0add]);
    }
}


template <class F, class I0, class O0>
struct gpuCellMapOp_1_1_explicit
{
    static void call(int ncells,
                     eavlExplicitConnectivity &conn,
                     I0 *d_i0, int i0div, int i0mod, int i0mul, int i0add,
                     O0 *d_o0, int o0mul, int o0add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        cellMapKernelExplicit_1_1<<< blocks, threads >>>(ncells,
                                                            conn,
                                                            d_i0, i0div, i0mod, i0mul, i0add,
                                                            d_o0, o0mul, o0add,
                                                            functor);
        CUDA_CHECK_ERROR();
    }
};

template <class F>
void callCellMapKernelExplicit_1_1(int nCells,
                            eavlExplicitConnectivity &conn,
                            eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                            eavlArray *o0, int o0mul, int o0add,
                            F &functor)
{
    int n = o0->GetNumberOfTuples();
    ///\todo: assert num for all output arrays is the same?


    conn.shapetype.NeedOnDevice();
    conn.connectivity.NeedOnDevice();
    conn.mapCellToIndex.NeedOnDevice();

    // Send all the arrays to the device if needed; we're only doing this
    // right now to make timing the kernel itself easier later.
    i0->GetCUDAArray();
    o0->GetCUDAArray();

    // run the kernel
    eavlDispatch_1_1<gpuCellMapOp_1_1_explicit>(nCells,
                                                    eavlArray::DEVICE,
                                                    conn,
                                                    i0, i0div, i0mod, i0mul, i0add,
                                                    o0, o0mul, o0add,
                                                    functor);

    conn.shapetype.NeedOnHost();
    conn.connectivity.NeedOnHost();
    conn.mapCellToIndex.NeedOnHost();
}

#endif

#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlCellMapOp_1_1<F>
//
// Purpose:
///   This is like a simple map operation, except that the input and output
///   arrays are defined to be on the mesh cells.
//
// Programmer:  Jeremy Meredith
// Creation:    March 29, 2012
//
// Modifications:
// ****************************************************************************
template <class F>
class eavlCellMapOp_1_1 : public eavlOperation
{
  protected:
    eavlCellSet     *cells;
    eavlArrayWithLinearIndex inArray0;
    eavlArrayWithLinearIndex outArray0;
    F                functor;
  public:
    eavlCellMapOp_1_1(eavlCellSet *inCells,
                         eavlArrayWithLinearIndex in0,
                         eavlArrayWithLinearIndex out0,
                         F f)
        : cells(inCells),
          inArray0(in0),
          outArray0(out0),
          functor(f)
    {
    }
    virtual void GoCPU()
    {
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            callCellMapCPUExplicit_1_1(elExp->GetNumCells(),
                                   elExp->GetConnectivity(EAVL_NODES_OF_CELLS),
                                   inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                   outArray0.array, outArray0.mul, outArray0.add,
                                   functor);
        }
        else if (elStr)
        {
            callCellMapCPUStructured_1_1(elStr->GetNumCells(),
                                 elStr->GetRegularStructure(),
                                 inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                 outArray0.array, outArray0.mul, outArray0.add,
                                 functor);
        }
        else
        {
            THROW(eavlException,"eavlCellMapOp didn't understand the mesh type.");
        }
    }
    virtual void GoGPU()
    {
#if defined __CUDACC__
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            ///\todo: assert that div,mod are always 1,INT_MAX?
            callCellMapKernelExplicit_1_1(elExp->GetNumCells(),
                                   elExp->GetConnectivity(EAVL_NODES_OF_CELLS),
                                   inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                   outArray0.array, outArray0.mul, outArray0.add,
                                   functor);
        }
        else if (elStr)
        {
            ///\todo: ideas/suggestions here:
            /// 1) for a rectilinear mesh, can we just create a 2D (or ideally fake up a 3d,
            ///    too -- see http://forums.nvidia.com/index.php?showtopic=164979)
            ///    thread grid and use thread IDs .x and .y for logical index?
            ///    that obviously saves a lot of divs and mods
            /// 2) create a specialized versions without div/mod (i.e. 
            ///    for curvilinear grids).  note: this was only a 10% speedup
            ///    when I tried it.  obviously not the bottleneck right now on an 8800GTS...
            /// 3) create a specialized version for separated coords.
            ///    since the one without div/mod wasn't a big speedup, this
            ///    is may be pointless right now....
            ///\todo: assert the out arrays are not logical -- i.e. div,mod are 1,INT_MAX?
            callCellMapKernelStructured_1_1(elStr->GetNumCells(),
                                 elStr->GetRegularStructure(),
                                 inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                 outArray0.array, outArray0.mul, outArray0.add,
                                 functor);
        }
        else
        {
            THROW(eavlException,"eavlCellMapOp didn't understand the mesh type.");
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
}; 

#endif
