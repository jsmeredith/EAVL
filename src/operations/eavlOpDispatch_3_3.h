// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_OP_DISPATCH_3_3_H
#define EAVL_OP_DISPATCH_3_3_H

#include "eavlException.h"

// ----------------------------------------------------------------------------

template <template <typename KF, typename KI01, typename KI2, typename KO01, typename KO2> class K,
          class F,
          class S, class I01, class I2, class O01, class O2>
void eavlDispatch_3_3_final(int n, eavlArray::Location loc,
                             S &structure,
                             I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                             I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                             I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                             O01 *o0, int o0mul, int o0add,
                             O01 *o1, int o1mul, int o1add,
                             O2  *o2, int o2mul, int o2add,
                             F &functor)
{
    K<F,I01,I2,O01,O2>::call(n, structure,
                             i0, i0div, i0mod, i0mul, i0add,
                             i1, i1div, i1mod, i1mul, i1add,
                             i2, i2div, i2mod, i2mul, i2add,
                             o0, o0mul, o0add,
                             o1, o1mul, o1add,
                             o2, o2mul, o2add,
                             functor);
}

template <template <typename KF, typename KI01, typename KI2, typename KO01, typename KO2> class K,
          class F,
          class S, class I01, class I2, class O01>
void eavlDispatch_3_3_stage4(int n, eavlArray::Location loc,
                       S &structure,
                       I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                       I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                       I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                       O01 *o0, int o0mul, int o0add,
                       O01 *o1, int o1mul, int o1add,
                       eavlArray *o2, int o2mul, int o2add,
                       F &functor)
{
    eavlFloatArray  *o2_f = dynamic_cast<eavlFloatArray*>(o2);
    eavlByteArray   *o2_b = dynamic_cast<eavlByteArray*>(o2);
    eavlIntArray    *o2_i = dynamic_cast<eavlIntArray*>(o2);

    if (o2_f)
        eavlDispatch_3_3_final<K>(n, loc, structure,
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    (float*)o2_f->GetRawPointer(loc), o2mul, o2add,
                                    functor);
    else if (o2_b)
        eavlDispatch_3_3_final<K>(n, loc, structure, 
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    (byte*)o2_b->GetRawPointer(loc), o2mul, o2add,
                                    functor);
    else if (o2_i)
        eavlDispatch_3_3_final<K>(n, loc, structure,
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    (int*)o2_i->GetRawPointer(loc), o2mul, o2add,
                                    functor);
    else
        THROW(eavlException,"Unknown array type");
}


template <template <typename KF, typename KI01, typename KI2, typename KO01, typename KO2> class K,
          class F,
          class S, class I01, class I2>
void eavlDispatch_3_3_stage3(int n, eavlArray::Location loc,
                       S &structure,
                       I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                       I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                       I2  *i2, int i2div, int i2mod, int i2mul, int i2add,
                       eavlArray *o0, int o0mul, int o0add,
                       eavlArray *o1, int o1mul, int o1add,
                       eavlArray *o2, int o2mul, int o2add,
                       F &functor)
{
    eavlFloatArray  *o0_f = dynamic_cast<eavlFloatArray*>(o0);
    eavlByteArray   *o0_b = dynamic_cast<eavlByteArray*>(o0);
    eavlIntArray    *o0_i = dynamic_cast<eavlIntArray*>(o0);

    eavlFloatArray  *o1_f = dynamic_cast<eavlFloatArray*>(o1);
    eavlByteArray   *o1_b = dynamic_cast<eavlByteArray*>(o1);
    eavlIntArray    *o1_i = dynamic_cast<eavlIntArray*>(o1);

    if ((o0_f && !o1_f) ||
        (o0_b && !o1_b) ||
        (o0_i && !o1_i))
        THROW(eavlException,"With N>=3 arguments as input or output, the first N-1 must be of the same type");
        

    if (o0_f)
        eavlDispatch_3_3_stage4<K>(n, loc, structure,
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    (float*)o0_f->GetRawPointer(loc), o0mul, o0add,
                                    (float*)o1_f->GetRawPointer(loc), o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else if (o0_b)
        eavlDispatch_3_3_stage4<K>(n, loc, structure, 
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    (byte*)o0_b->GetRawPointer(loc), o0mul, o0add,
                                    (byte*)o1_b->GetRawPointer(loc), o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else if (o0_i)
        eavlDispatch_3_3_stage4<K>(n, loc, structure,
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    (int*)o0_i->GetRawPointer(loc), o0mul, o0add,
                                    (int*)o1_i->GetRawPointer(loc), o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else
        THROW(eavlException,"Unknown array type");

}

template <template <typename KF, typename KI01, typename KI2, typename KO01, typename KO2> class K,
          class F,
          class S, class I01>
void eavlDispatch_3_3_stage2(int n, eavlArray::Location loc,
                       S &structure,
                       I01 *i0, int i0div, int i0mod, int i0mul, int i0add,
                       I01 *i1, int i1div, int i1mod, int i1mul, int i1add,
                       eavlArray *i2, int i2div, int i2mod, int i2mul, int i2add,
                       eavlArray *o0, int o0mul, int o0add,
                       eavlArray *o1, int o1mul, int o1add,
                       eavlArray *o2, int o2mul, int o2add,
                       F &functor)
{
    eavlFloatArray  *i2_f = dynamic_cast<eavlFloatArray*>(i2);
    eavlByteArray   *i2_b = dynamic_cast<eavlByteArray*>(i2);
    eavlIntArray    *i2_i = dynamic_cast<eavlIntArray*>(i2);

    if (i2_f)
        eavlDispatch_3_3_stage3<K>(n, loc, structure,
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    (float*)i2_f->GetRawPointer(loc), i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else if (i2_b)
        eavlDispatch_3_3_stage3<K>(n, loc, structure, 
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    (byte*)i2_b->GetRawPointer(loc), i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else if (i2_i)
        eavlDispatch_3_3_stage3<K>(n, loc, structure,
                                    i0, i0div, i0mod, i0mul, i0add,
                                    i1, i1div, i1mod, i1mul, i1add,
                                    (int*)i2_i->GetRawPointer(loc), i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else
        THROW(eavlException,"Unknown array type");
};


template <template <typename KF, typename KI01, typename KI2, typename KO01, typename KO2> class K,
          class F,
          class S>
void eavlDispatch_3_3(int n, eavlArray::Location loc,
                       S &structure,
                       eavlArray *i0, int i0div, int i0mod, int i0mul, int i0add,
                       eavlArray *i1, int i1div, int i1mod, int i1mul, int i1add,
                       eavlArray *i2, int i2div, int i2mod, int i2mul, int i2add,
                       eavlArray *o0, int o0mul, int o0add,
                       eavlArray *o1, int o1mul, int o1add,
                       eavlArray *o2, int o2mul, int o2add,
                       F &functor)
{
    eavlFloatArray  *i0_f = dynamic_cast<eavlFloatArray*>(i0);
    eavlByteArray   *i0_b = dynamic_cast<eavlByteArray*>(i0);
    eavlIntArray    *i0_i = dynamic_cast<eavlIntArray*>(i0);

    eavlFloatArray  *i1_f = dynamic_cast<eavlFloatArray*>(i1);
    eavlByteArray   *i1_b = dynamic_cast<eavlByteArray*>(i1);
    eavlIntArray    *i1_i = dynamic_cast<eavlIntArray*>(i1);

    if ((i0_f && !i1_f) ||
        (i0_b && !i1_b) ||
        (i0_i && !i1_i))
        THROW(eavlException,"With N>=3 arguments as input or output, the first N-1 must be of the same type");
        

    if (i0_f)
        eavlDispatch_3_3_stage2<K>(n, loc, structure,
                                    (float*)i0_f->GetRawPointer(loc), i0div, i0mod, i0mul, i0add,
                                    (float*)i1_f->GetRawPointer(loc), i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else if (i0_b)
        eavlDispatch_3_3_stage2<K>(n, loc, structure, 
                                    (byte*)i0_b->GetRawPointer(loc), i0div, i0mod, i0mul, i0add,
                                    (byte*)i1_b->GetRawPointer(loc), i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else if (i0_i)
        eavlDispatch_3_3_stage2<K>(n, loc, structure,
                                    (int*)i0_i->GetRawPointer(loc), i0div, i0mod, i0mul, i0add,
                                    (int*)i1_i->GetRawPointer(loc), i1div, i1mod, i1mul, i1add,
                                    i2, i2div, i2mod, i2mul, i2add,
                                    o0, o0mul, o0add,
                                    o1, o1mul, o1add,
                                    o2, o2mul, o2add,
                                    functor);
    else
        THROW(eavlException,"Unknown array type");
};

#endif
