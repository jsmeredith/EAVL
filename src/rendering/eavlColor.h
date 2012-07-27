// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COLOR_H
#define EAVL_COLOR_H

#include "STL.h"

// ****************************************************************************
// Class:  eavlColor
//
// Purpose:
///   It's a color!
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    March  9, 2011
//
// ****************************************************************************
class eavlColor
{
  public:
    float c[4];
    eavlColor()
    {
        c[0] = c[1] = c[2] = 0;
        c[3] = 1;
    }
    eavlColor(float r_, float g_, float b_, float a_=1)
    {
        c[0] = r_;
        c[1] = g_;
        c[2] = b_;
        c[3] = a_;
    }
    friend ostream &operator<<(ostream &out, const eavlColor &c)
    {
        out << "["<<c.c[0]<<","<<c.c[1]<<","<<c.c[2]<<","<<c.c[3]<<"]";
        return out;
    }

    static eavlColor black;
    static eavlColor grey10;
    static eavlColor grey20;
    static eavlColor grey30;
    static eavlColor grey40;
    static eavlColor grey50;
    static eavlColor grey60;
    static eavlColor grey70;
    static eavlColor grey80;
    static eavlColor grey90;
    static eavlColor white;

    static eavlColor red;
    static eavlColor green;
    static eavlColor blue;

    static eavlColor cyan;
    static eavlColor magenta;
    static eavlColor yellow;
};



#endif
