// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef STL_H
#define STL_H

#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <algorithm>
#include <cassert>
#include <iomanip>

using std::istream;
using std::ostream;
using std::ifstream;
using std::ofstream;
using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::ios;
using std::vector;
using std::deque;
using std::string;
using std::map;
using std::set;
using std::sort;
using std::pair;
using std::istringstream;
using std::ostringstream;
using std::filebuf;
using std::setprecision;
using std::fixed;

///\todo: This file is lazy.  Get rid of this entire STL.h file.

/*
template<class T>
inline ostream& operator<<(ostream& out,
                           const vector<T> &v)
{
    typename vector<T>::const_iterator b = v.begin();
    typename vector<T>::const_iterator e = v.end();

    out<<"[";
    while (b != e)
    {
        out<<*b;
        advance(b,1);
        if (b != e)
            out<<" ";
    }
    out<<"]";
    return out;
}
*/


#endif
