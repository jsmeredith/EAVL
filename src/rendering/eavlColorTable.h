// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COLOR_TABLE_H
#define EAVL_COLOR_TABLE_H

#include "STL.h"
#include "eavlColor.h"

// ****************************************************************************
// Class:  eavlColorTable
//
// Purpose:
///   It's a color table!
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern
// Creation:    March  9, 2011
//
// ****************************************************************************
class eavlColorTable
{
  public:
    bool  smooth;
    class ColorControlPoint
    {
      public:
        float position;
        eavlColor color;
        ColorControlPoint(float p, const eavlColor &c) : position(p), color(c) { }
    };
  public:
    vector<ColorControlPoint> pts;
    eavlColor Map(float c)
    {
        int n = pts.size();
        if (n == 0)
            return eavlColor(.5,.5,.5);
        if (n == 1 || c <= pts[0].position)
            return pts[0].color;
        if (c >= pts[n-1].position)
            return pts[n-1].color;
        int second;
        for (second=1; second<n-1; second++)
        {
            if (c < pts[second].position)
                break;
        }
        int first = second-1;
        float seg = pts[second].position-pts[first].position;
        float alpha;
        if (seg == 0)
            alpha = .5;
        else
            alpha = (c - pts[first].position)/seg;
        if (smooth)
        {
            return eavlColor(pts[first].color.c[0] * (1.-alpha) + pts[second].color.c[0] * alpha,
                             pts[first].color.c[1] * (1.-alpha) + pts[second].color.c[1] * alpha,
                             pts[first].color.c[2] * (1.-alpha) + pts[second].color.c[2] * alpha);
        }
        else
        {
            if (alpha < .5)
                return pts[first].color;
            else
                return pts[second].color;
        }
            
    }
    eavlColorTable(const string &name = "default")
    {
        smooth = true;
        if (name == "grey" || name == "gray")
        {
            pts.push_back(ColorControlPoint(0.0, eavlColor( 0, 0, 0)));
            pts.push_back(ColorControlPoint(1.0, eavlColor( 1, 1, 1)));
        }
        else if (name == "blue")
        {
            pts.push_back(ColorControlPoint(0.00, eavlColor( 0, 0, 0)));
            pts.push_back(ColorControlPoint(0.33, eavlColor( 0, 0,.5)));
            pts.push_back(ColorControlPoint(0.66, eavlColor( 0,.5, 1)));
            pts.push_back(ColorControlPoint(1.00, eavlColor( 1, 1, 1)));
        }
        else if (name == "orange")
        {
            pts.push_back(ColorControlPoint(0.00, eavlColor( 0, 0, 0)));
            pts.push_back(ColorControlPoint(0.33, eavlColor(.5, 0, 0)));
            pts.push_back(ColorControlPoint(0.66, eavlColor( 1,.5, 0)));
            pts.push_back(ColorControlPoint(1.00, eavlColor( 1, 1, 1)));
        }
        else if (name == "temperature")
        {
            pts.push_back(ColorControlPoint(0.05, eavlColor( 0, 0, 1)));
            pts.push_back(ColorControlPoint(0.35, eavlColor( 0, 1, 1)));
            pts.push_back(ColorControlPoint(0.50, eavlColor( 1, 1, 1)));
            pts.push_back(ColorControlPoint(0.65, eavlColor( 1, 1, 0)));
            pts.push_back(ColorControlPoint(0.95, eavlColor( 1, 0, 0)));
        }
        else if (name == "rainbow")
        {
            pts.push_back(ColorControlPoint(0.00, eavlColor( 0, 0, 1)));
            pts.push_back(ColorControlPoint(0.20, eavlColor( 0, 1, 1)));
            pts.push_back(ColorControlPoint(0.45, eavlColor( 0, 1, 0)));
            pts.push_back(ColorControlPoint(0.55, eavlColor( .7, 1, 0)));
            pts.push_back(ColorControlPoint(0.6,  eavlColor( 1, 1, 0)));
            pts.push_back(ColorControlPoint(0.75, eavlColor( 1, .5, 0)));
            pts.push_back(ColorControlPoint(0.9,  eavlColor( 1, 0, 0)));
            pts.push_back(ColorControlPoint(0.98, eavlColor( 1, 0, .5)));
            pts.push_back(ColorControlPoint(1.0,  eavlColor( 1, 0, 1)));
        }
        else if (name == "levels")
        {
            pts.push_back(ColorControlPoint(0.0, eavlColor( 0, 0, 1)));
            pts.push_back(ColorControlPoint(0.2, eavlColor( 0, 0, 1)));
            pts.push_back(ColorControlPoint(0.2, eavlColor( 0, 1, 1)));
            pts.push_back(ColorControlPoint(0.4, eavlColor( 0, 1, 1)));
            pts.push_back(ColorControlPoint(0.4, eavlColor( 0, 1, 0)));
            pts.push_back(ColorControlPoint(0.6, eavlColor( 0, 1, 0)));
            pts.push_back(ColorControlPoint(0.6, eavlColor( 1, 1, 0)));
            pts.push_back(ColorControlPoint(0.8, eavlColor( 1, 1, 0)));
            pts.push_back(ColorControlPoint(0.8, eavlColor( 1, 0, 0)));
            pts.push_back(ColorControlPoint(1.0, eavlColor( 1, 0, 0)));
        }
        else if (name == "dense" || name == "sharp")
        {
            smooth = (name == "dense") ? true : false;
            pts.push_back(ColorControlPoint(0.0, eavlColor(0.28, 0.28, 0.86)));
            pts.push_back(ColorControlPoint(0.1, eavlColor(0.00, 0.00, 0.36)));
            pts.push_back(ColorControlPoint(0.2, eavlColor(0.00, 1.00, 1.00)));
            pts.push_back(ColorControlPoint(0.3, eavlColor(0.00, 0.50, 0.00)));
            pts.push_back(ColorControlPoint(0.4, eavlColor(1.00, 1.00, 0.00)));
            pts.push_back(ColorControlPoint(0.5, eavlColor(0.75, 0.57, 0.00)));
            pts.push_back(ColorControlPoint(0.6, eavlColor(1.00, 0.47, 0.00)));
            pts.push_back(ColorControlPoint(0.7, eavlColor(0.58, 0.00, 0.00)));
            pts.push_back(ColorControlPoint(0.8, eavlColor(1.00, 0.03, 0.17)));
            pts.push_back(ColorControlPoint(0.9, eavlColor(0.69, 0.14, 0.38)));
            pts.push_back(ColorControlPoint(1.0, eavlColor(1.00, 0.00, 1.00)));
        }
        else //if (name == "tmp") // or anything else
        {
            // note: this is 'dense' with some control points shifted
            // to emphasize one range; it's probably not useful long-term
            pts.push_back(ColorControlPoint(0.0, eavlColor(0.28, 0.28, 0.86)));
            pts.push_back(ColorControlPoint(0.2, eavlColor(0.00, 0.00, 0.36)));
            pts.push_back(ColorControlPoint(0.4, eavlColor(0.00, 1.00, 1.00)));
            pts.push_back(ColorControlPoint(0.6, eavlColor(0.00, 0.50, 0.00)));
            pts.push_back(ColorControlPoint(0.7, eavlColor(1.00, 1.00, 0.00)));
            pts.push_back(ColorControlPoint(0.73, eavlColor(0.75, 0.57, 0.00)));
            pts.push_back(ColorControlPoint(0.78, eavlColor(1.00, 0.47, 0.00)));
            pts.push_back(ColorControlPoint(0.82, eavlColor(0.58, 0.00, 0.00)));
            pts.push_back(ColorControlPoint(0.92, eavlColor(1.00, 0.03, 0.17)));
            pts.push_back(ColorControlPoint(0.96, eavlColor(0.69, 0.14, 0.38)));
            pts.push_back(ColorControlPoint(1.0, eavlColor(1.00, 0.00, 1.00)));
        }
    }
};

#endif

