// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
        c[0] = 0;
        c[1] = 0;
        c[2] = 0;
        c[3] = 1;
    }
    eavlColor(float r_, float g_, float b_, float a_=1)
    {
        c[0] = r_;
        c[1] = g_;
        c[2] = b_;
        c[3] = a_;
    }
    inline void SetComponentFromByte(int i, unsigned char v)
    {
        // Note that though GetComponentAsByte below
        // multiplies by 256, we're dividing by 255. here.
        // This is, believe it or not, still correct.
        // That's partly because we always round down in
        // that method.  For example, if we set the float
        // here using byte(1), /255 gives us .00392, which
        // *256 gives us 1.0035, which is then rounded back
        // down to byte(1) below.  Or, if we set the float
        // here using byte(254), /255 gives us .99608, which
        // *256 gives us 254.996, which is then rounded
        // back down to 254 below.  So it actually reverses
        // correctly, even though the mutliplier and
        // divider don't match between these two methods.
        c[i] = float(v) / 255.;
        // clamp?
        if (c[i]<0) c[i] = 0;
        if (c[i]>1) c[i] = 1;
    }
    inline unsigned char GetComponentAsByte(int i)
    {
        // We need this to match what OpenGL/Mesa do.
        // Why?  Well, we need to set glClearColor
        // using floats, but the frame buffer comes
        // back as bytes (and is internally such) in
        // most cases.  In one example -- parallel 
        // compositing -- we need the byte values
        // returned from here to match the byte values
        // returned in the frame buffer.  Though
        // a quick source code inspection of Mesa
        // led me to believe I should do *255., in 
        // fact this led to a mismatch.  *256. was
        // actually closer.  (And arguably more correct
        // if you think the byte value 255 should share
        // approximately the same range in the float [0,1]
        // space as the other byte values.)  Note in the
        // inverse method above, though, we still use 255;
        // see SetComponentFromByte for an explanation of
        // why that is correct, if non-obvious.
        int tv = c[i] * 256.;
        return (tv < 0) ? 0 : (tv > 255) ? 255 : tv;
    }
    void GetRGBA(unsigned char &r, unsigned char &g,
                 unsigned char &b, unsigned char &a)
    {
        r = GetComponentAsByte(0);
        g = GetComponentAsByte(1);
        b = GetComponentAsByte(2);
        a = GetComponentAsByte(3);
    }
    double RawBrightness()
    {
        return (c[0]+c[1]+c[2])/3.;
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
