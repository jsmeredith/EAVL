// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_UTILITY_H
#define EAVL_UTILITY_H

#include "eavlArray.h"
#include "eavlFlatArray.h"

void CalculateTicks(double lower, double upper, bool minor,
                    vector<double> &positions,
                    vector<double> &proportions,
                    int modifyTickQuantity=0); ///< -1 for less, +1 for more

void CalculateTicksLogarithmic(double lower, double upper, bool minor,
                               vector<double> &positions,
                               vector<double> &proportions,
                               int modifyTickQuantity=0); ///< -1 for less, +1 for more

template <class T>
string VecPrint(T *const v, unsigned int n, unsigned int nmax, unsigned int group=1e9)
{
    ostringstream out;
    if (n <= nmax)
    {
        for (unsigned int i=0 ; i<n; i++)
        {
            out << v[i];
            if (i<n-1)
            {
                if (i%group == group-1)
                    out << ", ";
                out << " ";
            }
        }
        return out.str();
    }

    unsigned int num = nmax/2;
    unsigned int i=0;
    for ( ; i<num; i++)
    {
        out << v[i];
        if (i%group == group-1)
            out << ", ";
        out << " ";
    }
    out << "... ";
    for (i=n - num ; i<n; i++)
    {
        out << v[i];
        if (i%group == group-1)
            out << ", ";
        if (i<n-1)
            out << " ";
    }
    return out.str();
}


template <class T>
string VecPrint(const T &v, unsigned int nmax, unsigned int group=1e9)
{
    return VecPrint(&(v[0]), v.size(), nmax, group);
}

template <class T>
void PrintVectorSummary(ostream &out, const vector<T> &v)
{
    out << VecPrint(v, 20);
}

template <class T>
void PrintVectorSummary(ostream &out, const T *v, int n)
{
    out << VecPrint(v, n, 20);
}

template <class T>
void PrintVectorSummary(ostream &out, const eavlFlatArray<T> &v)
{
    out << VecPrint(v, 20);
}



#endif
