// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_OPERATION_H
#define EAVL_OPERATION_H

#include <climits>
#include <cfloat>
#include "eavlException.h"
#include "eavlConfig.h"
#include "eavlArray.h"
#include "eavlRegularStructure.h"
#include "eavlExplicitConnectivity.h"
#include "eavlCUDA.h"

///\todo: finalize this and make it its own thing
struct eavlArrayWithLinearIndex
{
    eavlArray *array;
    int div;
    int mod;
    int mul;
    int add;
    eavlArrayWithLinearIndex() : array(NULL)
    {
    }
    eavlArrayWithLinearIndex(const eavlArrayWithLinearIndex &awli)
        : array(awli.array),
          div(awli.div),
          mod(awli.mod),
          mul(awli.mul),
          add(awli.add)
    {
    }
    void operator=(const eavlArrayWithLinearIndex &awli)
    {
        array = awli.array;
        div = awli.div;
        mod = awli.mod;
        mul = awli.mul;
        add = awli.add;
    }
    eavlArrayWithLinearIndex(eavlArray *arr)
        : array(arr),
          div(1),
          mod(INT_MAX),
          mul(1),
          add(0)
    {
        if (array->GetNumberOfComponents() != 1)
            THROW(eavlException,"Must specify a component for a multi-component array");
    }
    eavlArrayWithLinearIndex(eavlArray *arr, int component)
        : array(arr),
          div(1),
          mod(INT_MAX),
          mul(arr->GetNumberOfComponents()),
          add(component)
    {
    }
    ///\todo: we're assuming a logical dimension must be a NODE array, and
    ///       we're probably not just making that assumption right here, either!
    eavlArrayWithLinearIndex(eavlArray *arr,
                           eavlRegularStructure reg, int logicaldim)
        : array(arr),
          div(reg.CalculateNodeIndexDivForDimension(logicaldim)),
          mod(reg.CalculateNodeIndexModForDimension(logicaldim)),
          mul(1),
          add(0)
    {
        if (array->GetNumberOfComponents() != 1)
            THROW(eavlException,"Must specify a component for a multi-component array");
    }
    ///\todo: we're assuming a logical dimension must be a NODE array, and
    ///       we're probably not just making that assumption right here, either!
    eavlArrayWithLinearIndex(eavlArray *arr, int component,
                           eavlRegularStructure reg, int logicaldim)
        : array(arr),
          div(reg.CalculateNodeIndexDivForDimension(logicaldim)),
          mod(reg.CalculateNodeIndexModForDimension(logicaldim)),
          mul(arr->GetNumberOfComponents()),
          add(component)
    {
    }
    void Print(ostream &out)
    {
        cout << "array="<<array<<" div="<<div<<" mod="<<mod<<" mul="<<mul<<" add="<<add<<endl;
    }
};


// ****************************************************************************
// Class:  eavlOperation
//
// Purpose:
///
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    September 2, 2011
//
// ****************************************************************************

template<class T>
struct eavlAddFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return a + b; }
    T identity() { return 0; }
};


template<class T>
struct eavlMulFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return a * b; }
    T identity() { return 1; }
};

template<class T>
struct eavlMaxFunctor
{
    EAVL_FUNCTOR T operator()(T a, T b) { return (a > b) ? a : b; }
    T identity();
};


template<class T>
struct eavlMinFunctor
{
  public:
    EAVL_FUNCTOR T operator()(T a, T b) { return (a < b) ? a : b; }
    T identity();
};

template<class T>
struct eavlLessThanConstFunctor
{
  private:
    T target;
  public:
    eavlLessThanConstFunctor(const T &t) : target(t) { }
    EAVL_FUNCTOR bool operator()(float x) { return x < target; }
};

template<class T>
struct eavlNotEqualFunctor
{
    EAVL_FUNCTOR bool operator()(T a, T b) { return a != b; }
};

struct DummyFunctor
{
    void operator()(void) { }
};


class eavlOperation 
{
    friend class eavlExecutor;
  public:
  protected:
    virtual void GoCPU() = 0;
    virtual void GoGPU() = 0;
};

#endif
