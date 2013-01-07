// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_OP_DISPATCH_1_1_INT_H
#define EAVL_OP_DISPATCH_1_1_INT_H

#include "eavlException.h"
// ----------------------------------------------------------------------------

///\todo: note: this is untested!

template <template <typename KF, typename KI0, typename KO0> class K,
          class F,
          class S, class I0, class O0>
void eavlDispatch_1_1_int_final(int n, eavlArray::Location loc,
                             S &structure,
                             I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                             O0 *o0, int o0mul, int o0add,
                             int *idx, int idxmul, int idxadd,
                             F &functor)
{
    K<F,I0,O0>::call(n, structure,
                     i0, i0div, i0mod, i0mul, i0add,
                     o0, o0mul, o0add,
                     idx, idxmul, idxadd,
                     functor);
}

template <template <typename KF, typename KI0, typename KO0> class K,
          class F,
          class S, class I0>
void eavlDispatch_1_1_int_stage2(int n, eavlArray::Location loc,
                              S &structure,
                              I0 *i0, int i0div, int i0mod, int i0mul, int i0add,
                              eavlArray *o0, int o0mul, int o0add,
                              int *idx, int idxmul, int idxadd,
                              F &functor)
{
    eavlFloatArray  *o0_f = dynamic_cast<eavlFloatArray*>(o0);
    eavlByteArray   *o0_b = dynamic_cast<eavlByteArray*>(o0);
    eavlIntArray    *o0_i = dynamic_cast<eavlIntArray*>(o0);

    if (o0_f)
        eavlDispatch_1_1_int_final<K>(n, loc, structure,
                                   i0, i0div, i0mod, i0mul, i0add,
                                   (float*)o0_f->GetRawPointer(loc), o0mul, o0add,
                                   idx, idxmul, idxadd,
                                   functor);
    else if (o0_b)
        eavlDispatch_1_1_int_final<K>(n, loc, structure,
                                   i0, i0div, i0mod, i0mul, i0add,
                                   (byte*)o0_b->GetRawPointer(loc), o0mul, o0add,
                                   idx, idxmul, idxadd,
                                   functor);
    else if (o0_i)
        eavlDispatch_1_1_int_final<K>(n, loc, structure,
                                   i0, i0div, i0mod, i0mul, i0add,
                                   (int*)o0_i->GetRawPointer(loc), o0mul, o0add,
                                   idx, idxmul, idxadd,
                                   functor);
    else
        THROW(eavlException,"Unknown array type");
};

template <template <typename KF, typename KI0, typename KO0> class K,
          class F,
          class S>
void eavlDispatch_1_1_int(int n, eavlArray::Location loc,
                       S &structure,
                       eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                       eavlArray *o0, int o0mul, int o0add,
                       eavlArray *idx, int idxmul, int idxadd,
                       F &functor)
{
    eavlIntArray    *idx_i = dynamic_cast<eavlIntArray*>(idx);
    if (!idx_i)
        THROW(eavlException,"Expected integer array for indices in eavlDispatch_1_1_int.");

    eavlFloatArray  *i0_f = dynamic_cast<eavlFloatArray*>(i0);
    eavlByteArray   *i0_b = dynamic_cast<eavlByteArray*>(i0);
    eavlIntArray    *i0_i = dynamic_cast<eavlIntArray*>(i0);

    if (i0_f)
        eavlDispatch_1_1_int_stage2<K>(n, loc, structure,
                                    (float*)i0_f->GetRawPointer(loc), i0div, i0mod, i0mul, i0add,
                                    o0, o0mul, o0add,
                                    (int*)idx_i->GetRawPointer(loc), idxmul, idxadd,
                                    functor);
    else if (i0_b)
        eavlDispatch_1_1_int_stage2<K>(n, loc, structure, 
                                    (byte*)i0_b->GetRawPointer(loc), i0div, i0mod, i0mul, i0add,
                                    o0, o0mul, o0add,
                                    (int*)idx_i->GetRawPointer(loc), idxmul, idxadd,
                                    functor);
    else if (i0_i)
        eavlDispatch_1_1_int_stage2<K>(n, loc, structure,
                                    (int*)i0_i->GetRawPointer(loc), i0div, i0mod, i0mul, i0add,
                                    o0, o0mul, o0add,
                                    (int*)idx_i->GetRawPointer(loc), idxmul, idxadd,
                                    functor);
    else
        THROW(eavlException,"Unknown array type");
};

#endif
