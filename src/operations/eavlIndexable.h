#ifndef EAVL_INDEXABLE_H
#define EAVL_INDEXABLE_H

#include "eavlRegularStructure.h"

class eavlArrayIndexer
{
  public:
    ///\todo: order doesn't match existing EAVL order
    int div, mod;
    int mul, add;
    eavlArrayIndexer() : div(1), mod(1e9), mul(1), add(0)
    {
    }
    eavlArrayIndexer(int mul, int add) : div(1), mod(1e9), mul(mul), add(add)
    {
    }
    eavlArrayIndexer(int div, int mod, int mul, int add) : div(div), mod(mod), mul(mul), add(add)
    {
    }
    virtual void Print(ostream &) const
    {
    }
    EAVL_HOSTDEVICE int index(int i) const { return (((i/div)%mod)*mul)+add; }
};

template <class T>
class eavlIndexable
{
  public:
    typedef T type;
    typedef eavlIndexable<T> indexable_type;
    T *array;
    eavlArrayIndexer indexer;
    eavlIndexable(T *arr) : array(arr)
    {
    }
    eavlIndexable(T *arr, eavlArrayIndexer ind) : array(arr), indexer(ind)
    {
    }
    eavlIndexable(T *arr, int comp) : array(arr), indexer(arr->GetNumberOfComponents(), comp)
    {
    }
    eavlIndexable(T *arr, int comp, eavlRegularStructure &reg, int logicaldim)
        : array(arr),
          indexer(reg.CalculateNodeIndexDivForDimension(logicaldim),
                  reg.CalculateNodeIndexModForDimension(logicaldim),
                  arr->GetNumberOfComponents(),
                  comp)
    {
    }
    int length() const
    {
        // We don't need to account for div/mod here because
        // we're only using this to determine how many
        // output values we have, and output values don't have
        // div/mod.  (I'm not sure we could do it accurately
        // even if we wanted to....)
        int nvalues = array->GetNumberOfTuples() * array->GetNumberOfComponents();
        return int((nvalues - indexer.add) / indexer.mul);
    }
    virtual void Print(ostream &)
    {
    }
};

template <class T>
struct make_indexable_class
{
    static inline eavlIndexable<T> make_indexable(T *t)
    {
        return eavlIndexable<T>(t);
    }
    static inline eavlIndexable<T> make_indexable(T *t, eavlArrayIndexer i)
    {
        return eavlIndexable<T>(t,i);
    }
    static inline eavlIndexable<T> make_indexable(T *t, int comp)
    {
        return eavlIndexable<T>(t,comp);
    }
};

template <class T>
inline eavlIndexable<T> make_indexable(T *t)
{
    return make_indexable_class<T>::make_indexable(t);
}

template <class T>
inline eavlIndexable<T> make_indexable(T *t, eavlArrayIndexer i)
{
    return make_indexable_class<T>::make_indexable(t, i);
}

template <class T>
inline eavlIndexable<T> make_indexable(T *t, int comp)
{
    return make_indexable_class<T>::make_indexable(t, comp);
}

#endif
