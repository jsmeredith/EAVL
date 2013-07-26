#ifndef EAVL_INDEXABLE_H
#define EAVL_INDEXABLE_H


class eavlArrayIndexer
{
  public:
    ///\todo: order doesn't match existing EAVL order
    int add, mul, div, mod;
    eavlArrayIndexer(int a=0, int m=1, int d=1, int o=1e9) : add(a), mul(m), div(d), mod(o) { }
    virtual void Print(ostream &out)
    {
        out << "eavlArrayIndexer{"<<add<<","<<mul<<",<<"<<div<<","<<mod<<"}\n";
    }
    EAVL_HOSTDEVICE int index(int i) { return (((i/div)%mod)*mul)+add; }
};

template <class T>
class eavlIndexable
{
  public:
    typedef T type;
    typedef eavlIndexable<T> indexable_type;
    T *array;
    eavlArrayIndexer indexer;
    eavlIndexable(T *arr) : array(arr) { }
    eavlIndexable(T *arr, eavlArrayIndexer ind) : array(arr), indexer(ind) { }
    virtual void Print(ostream &out)
    {
        out << "Indexable\n";
    }
};

template <class T>
struct make_indexable_class
{
    static inline eavlIndexable<T> make_indexable(T *t, eavlArrayIndexer i)
    {
        return eavlIndexable<T>(t);
    }
};

template <class T>
inline eavlIndexable<T> make_indexable(T *t, eavlArrayIndexer i=eavlArrayIndexer())
{
    return make_indexable_class<T>::make_indexable(t, i);
}

#endif
