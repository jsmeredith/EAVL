// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TOPOLOGY_MAP_OP_3_0_3_H
#define EAVL_TOPOLOGY_MAP_OP_3_0_3_H

#include "eavlCellSet.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlDataSet.h"
#include "eavlArray.h"
#include "eavlOperation.h"
#include "eavlOpDispatch_3_3.h"
#include "eavlTopology.h"
#include "eavlException.h"
#include <time.h>
#include <omp.h>

#ifndef DOXYGEN
template <class F, class I01, class I2, class O01, class O2>
struct cpu_topologyMapRegular_3_0_3
{
    static void call(int nitems,
                     eavlRegularConnectivity &reg,
                     I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                     O01 *o0, int o0mul, int o0add,
                     O01 *o1, int o1mul, int o1add,
                     O2  *o2, int o2mul, int o2add,
                     F &functor)
    {
        int nodeIds[12];
        int npts;
        for (int index = 0; index < nitems; index++)
        {
            int shapeType = reg.GetElementComponents(index, npts, nodeIds);

            float x[12], y[12], z[12];
            for (int n=0; n<npts; n++)
            {
                int node = nodeIds[n];
                x[n] = i0[((node / i0div) % i0mod) * i0mul + i0add];
                y[n] = i1[((node / i1div) % i1mod) * i1mul + i1add];
                z[n] = i2[((node / i2div) % i2mod) * i2mul + i2add];
            }

            float out0, out1, out2;

            functor(shapeType, npts,
                    x, y, z,
                    out0, out1, out2);

            o0[index * o0mul + o0add] = out0;
            o1[index * o1mul + o1add] = out1;
            o2[index * o2mul + o2add] = out2;
        }
    }
};

template <class F, class I01, class I2, class O01, class O2>
struct cpu_topologyMapExplicit_3_0_3
{
    static void call(int nitems,
                     eavlExplicitConnectivity &conn,
                     I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                     O01 *o0, int o0mul, int o0add,
                     O01 *o1, int o1mul, int o1add,
                     O2  *o2, int o2mul, int o2add,
                     F &functor)
    {
        int nodeIds[12];
        for (int index = 0; index < nitems; index++)
        {
            int npts;
            int shapeType = conn.GetElementComponents(index, npts, nodeIds);

            /// \todo: we're converting explicitly to float here,
            /// forcing floats in the functor operator()
                float x[12], y[12], z[12];
                for (int n=0; n<npts; n++)
                {
                    int node = nodeIds[n];
                    x[n] = i0[((node / i0div) % i0mod) * i0mul + i0add];
                    y[n] = i1[((node / i1div) % i1mod) * i1mul + i1add];
                    z[n] = i2[((node / i2div) % i2mod) * i2mul + i2add];
                }

                float out0, out1, out2;

                functor(shapeType, npts,
                        x, y, z,
                        out0, out1, out2);

                o0[index * o0mul + o0add] = out0;
                o1[index * o1mul + o1add] = out1;
                o2[index * o2mul + o2add] = out2;
        }
    }
};

#if defined __CUDACC__

template <class F, class I01, class I2, class O01, class O2>
__global__ void
topologyMapKernelRegular_3_0_3(int nitems,
                            eavlRegularConnectivity reg,
                            I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                            I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                            I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                            O01 *o0, int o0mul, int o0add,
                            O01 *o1, int o1mul, int o1add,
                            O2  *o2, int o2mul, int o2add,
                            F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeIds[12];
    int npts;
    for (int index = threadID; index < nitems; index += numThreads)
    {
        int shapeType = reg.GetElementComponents(index, npts, nodeIds);

        ///\todo: really, we're EITHER doing div/mod or mul/add for each dim.
        ///       we should be able to optimize this quite a bit more....
        ///       No, that's not necessarily true; we could, in theory, have
        ///       a 3-vector on each value of a logical dimension, right?
        ///       Not that there aren't many ways to optimize this still....
        float x[12], y[12], z[12];
        for (int n=0; n<npts; n++)
        {
            int node = nodeIds[n];
            x[n] = i0[((node / i0div) % i0mod) * i0mul + i0add];
            y[n] = i1[((node / i1div) % i1mod) * i1mul + i1add];
            z[n] = i2[((node / i2div) % i2mod) * i2mul + i2add];
        }

        float out0, out1, out2;

        functor(shapeType, npts,
                x, y, z,
                out0, out1, out2);

        o0[index * o0mul + o0add] = out0;
        o1[index * o1mul + o1add] = out1;
        o2[index * o2mul + o2add] = out2;
    }
}

template <class F, class I01, class I2, class O01, class O2>
struct gpuTopologyMapOp_3_0_3_regular
{
    static void call(int nitems,
                     eavlRegularConnectivity &reg,
                     I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                     O01 *o0, int o0mul, int o0add,
                     O01 *o1, int o1mul, int o1add,
                     O2  *o2, int o2mul, int o2add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);
        topologyMapKernelRegular_3_0_3<<< blocks, threads >>>(nitems,
                                                           reg,
                                                           i0, i0div, i0mod, i0mul, i0add,
                                                           i1, i1div, i1mod, i1mul, i1add,
                                                           i2, i2div, i2mod, i2mul, i2add,
                                                           o0, o0mul, o0add,
                                                           o1, o1mul, o1add,
                                                           o2, o2mul, o2add,
                                                           functor);
        CUDA_CHECK_ERROR();
    }
};


template <class F>
void callTopologyMapKernelStructured_3_0_3(int nitems,
                          eavlRegularConnectivity reg,
                          eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                          eavlArray *i1, int i1div, int i1mod, int i1mul, int i1add,
                          eavlArray *i2, int i2div, int i2mod, int i2mul, int i2add,
                          eavlArray *o0, int o0mul, int o0add,
                          eavlArray *o1, int o1mul, int o1add,
                          eavlArray *o2, int o2mul, int o2add,
                          F &functor)
{
    ///\todo: assert num for all output arrays is the same?

    ///\todo: just doing this to make timing easier; can probably remove it
    i0->GetCUDAArray();
    i1->GetCUDAArray();
    i2->GetCUDAArray();
    o0->GetCUDAArray();
    o1->GetCUDAArray();
    o2->GetCUDAArray();

    // run the kernel
    eavlDispatch_3_3<gpuTopologyMapOp_3_0_3_regular>(nitems,
                                                  eavlArray::DEVICE,
                                                  reg,
                                                  i0, i0div, i0mod, i0mul, i0add,
                                                  i1, i1div, i1mod, i1mul, i1add,
                                                  i2, i2div, i2mod, i2mul, i2add,
                                                  o0, o0mul, o0add,
                                                  o1, o1mul, o1add,
                                                  o2, o2mul, o2add,
                                                  functor);
}

template <class F, class I01, class I2, class O01, class O2>
__global__ void
topologyMapKernelExplicit_3_0_3(int nitems,
                             eavlExplicitConnectivity conn,
                             I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                             I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                             I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                             O01 *o0, int o0mul, int o0add,
                             O01 *o1, int o1mul, int o1add,
                             O2  *o2, int o2mul, int o2add,
                             F functor)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID   = blockIdx.x * blockDim.x + threadIdx.x;
    int nodeIds[12];
    for (int index = threadID; index < nitems; index += numThreads)
    {
        int npts;
        int shapeType = conn.GetElementComponents(index, npts, nodeIds);

        /// \todo: we're converting explicitly to float here,
        /// forcing floats in the functor operator()
        float x[12], y[12], z[12];
        for (int n=0; n<npts; n++)
        {
            int node = nodeIds[n];
            x[n] = i0[((node / i0div) % i0mod) * i0mul + i0add];
            y[n] = i1[((node / i1div) % i1mod) * i1mul + i1add];
            z[n] = i2[((node / i2div) % i2mod) * i2mul + i2add];
        }

        float out0, out1, out2;

        functor(shapeType, npts,
                x, y, z,
                out0, out1, out2);

        o0[index * o0mul + o0add] = out0;
        o1[index * o1mul + o1add] = out1;
        o2[index * o2mul + o2add] = out2;
    }
}


template <class F, class I01, class I2, class O01, class O2>
struct gpuTopologyMapOp_3_0_3_explicit
{
    static void call(int nitems,
                     eavlExplicitConnectivity &conn,
                     I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                     I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                     I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                     O01 *o0, int o0mul, int o0add,
                     O01 *o1, int o1mul, int o1add,
                     O2  *o2, int o2mul, int o2add,
                     F &functor)
    {
        // fixing at 32 threads, 64 blocks for now, with thread coarsening
        int numThreads = 256;
        dim3 threads(numThreads,   1, 1);
        dim3 blocks (32,           1, 1);

        topologyMapKernelExplicit_3_0_3<<< blocks, threads >>>(nitems,
                                                            conn,
                                                            i0, i0div, i0mod, i0mul, i0add,
                                                            i1, i1div, i1mod, i1mul, i1add,
                                                            i2, i2div, i2mod, i2mul, i2add,
                                                            o0, o0mul, o0add,
                                                            o1, o1mul, o1add,
                                                            o2, o2mul, o2add,
                                                            functor);
        CUDA_CHECK_ERROR();
    }
};

template <class F>
void callTopologyMapKernelExplicit_3_0_3(int nitems,
                            eavlExplicitConnectivity &conn,
                            eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                            eavlArray *i1, int i1div, int i1mod, int i1mul, int i1add,
                            eavlArray *i2, int i2div, int i2mod, int i2mul, int i2add,
                            eavlArray *o0, int o0mul, int o0add,
                            eavlArray *o1, int o1mul, int o1add,
                            eavlArray *o2, int o2mul, int o2add,
                            F &functor)
{
    conn.shapetype.NeedOnDevice();
    conn.connectivity.NeedOnDevice();
    conn.mapCellToIndex.NeedOnDevice();

    // Send all the arrays to the device if needed; we're only doing this
    // right now to make timing the kernel itself easier later.
    i0->GetCUDAArray();
    i1->GetCUDAArray();
    i2->GetCUDAArray();
    o0->GetCUDAArray();
    o1->GetCUDAArray();
    o2->GetCUDAArray();

    // run the kernel
    eavlDispatch_3_3<gpuTopologyMapOp_3_0_3_explicit>(nitems,
                                                   eavlArray::DEVICE,
                                                   conn,
                                                   i0, i0div, i0mod, i0mul, i0add,
                                                   i1, i1div, i1mod, i1mul, i1add,
                                                   i2, i2div, i2mod, i2mul, i2add,
                                                   o0, o0mul, o0add,
                                                   o1, o1mul, o1add,
                                                   o2, o2mul, o2add,
                                                   functor);


    conn.shapetype.NeedOnHost();
    conn.connectivity.NeedOnHost();
    conn.mapCellToIndex.NeedOnHost();
}

#endif
#endif // DOXYGEN

// ****************************************************************************
// Class:  eavlTopologyMapOp_3_0_3<F>
//
// Purpose:
///   Map from one topological element in a mesh to another, with 3
///   input arrays on the source topology and 3 output arrays on the
///   destination topology.
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    September 6, 2011
//
// Modifications:
// ****************************************************************************
template <class F>
class eavlTopologyMapOp_3_0_3 : public eavlOperation
{
  protected:
    eavlCellSet     *cells;
    eavlArrayWithLinearIndex inArray0, inArray1, inArray2;
    eavlArrayWithLinearIndex outArray0, outArray1, outArray2;
    F                functor;
    eavlTopology topology;
  public:
    eavlTopologyMapOp_3_0_3(eavlCellSet *inCells,
                         eavlTopology topo,
                         eavlArrayWithLinearIndex in0,
                         eavlArrayWithLinearIndex in1,
                         eavlArrayWithLinearIndex in2,
                         eavlArrayWithLinearIndex out0,
                         eavlArrayWithLinearIndex out1,
                         eavlArrayWithLinearIndex out2,
                         F f)
        : cells(inCells),
          topology(topo),
          inArray0(in0),
          inArray1(in1),
          inArray2(in2),
          outArray0(out0),
          outArray1(out1),
          outArray2(out2),
          functor(f)
    {
    }
    virtual void GoCPU()
    {
        eavlCellSetExplicit *elExp = dynamic_cast<eavlCellSetExplicit*>(cells);
        eavlCellSetAllStructured *elStr = dynamic_cast<eavlCellSetAllStructured*>(cells);
        if (elExp)
        {
            eavlDispatch_3_3<cpu_topologyMapExplicit_3_0_3>(outArray0.array->GetNumberOfTuples(),
                                                            eavlArray::HOST,
                                                            elExp->GetConnectivity(topology),
                                                            inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                                            inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                                            inArray2.array, inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                                                            outArray0.array, outArray0.mul, outArray0.add,
                                                            outArray1.array, outArray1.mul, outArray1.add,
                                                            outArray2.array, outArray2.mul, outArray2.add,
                                                            functor);
        }
        else if (elStr)
        {
            eavlRegularConnectivity conn = eavlRegularConnectivity(elStr->GetRegularStructure(),topology);
            eavlDispatch_3_3<cpu_topologyMapRegular_3_0_3>(outArray0.array->GetNumberOfTuples(),
                                                           eavlArray::HOST,
                                                           conn,
                                                           inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                                           inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                                           inArray2.array, inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                                                           outArray0.array, outArray0.mul, outArray0.add,
                                                           outArray1.array, outArray1.mul, outArray1.add,
                                                           outArray2.array, outArray2.mul, outArray2.add,
                                                           functor);
        }
        else
        {
            THROW(eavlException,"eavlTopologyMapOp didn't understand the mesh type.");
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
            callTopologyMapKernelExplicit_3_0_3(outArray0.array->GetNumberOfTuples(),
                                   elExp->GetConnectivity(topology),
                                   inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                   inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                   inArray2.array, inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                                   outArray0.array, outArray0.mul, outArray0.add,
                                   outArray1.array, outArray1.mul, outArray1.add,
                                   outArray2.array, outArray2.mul, outArray2.add,
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
            callTopologyMapKernelStructured_3_0_3(outArray0.array->GetNumberOfTuples(),
                                 eavlRegularConnectivity(elStr->GetRegularStructure(),topology),
                                 inArray0.array, inArray0.div, inArray0.mod, inArray0.mul, inArray0.add,
                                 inArray1.array, inArray1.div, inArray1.mod, inArray1.mul, inArray1.add,
                                 inArray2.array, inArray2.div, inArray2.mod, inArray2.mul, inArray2.add,
                                 outArray0.array, outArray0.mul, outArray0.add,
                                 outArray1.array, outArray1.mul, outArray1.add,
                                 outArray2.array, outArray2.mul, outArray2.add,
                                 functor);
        }
        else
        {
            THROW(eavlException,"eavlTopologyMapOp didn't understand the mesh type.");
        }
#else
        THROW(eavlException,"Executing GPU code without compiling under CUDA compiler.");
#endif
    }
}; 

#endif
