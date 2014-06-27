// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
    eavlColor Map(float c) const
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
    eavlColorTable(string name)
    {
        if (name == "" || name == "default")
            name = "dense";

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
            pts.push_back(ColorControlPoint(0.0, eavlColor(0.26, 0.22, 0.92)));
            pts.push_back(ColorControlPoint(0.1, eavlColor(0.00, 0.00, 0.52)));
            pts.push_back(ColorControlPoint(0.2, eavlColor(0.00, 1.00, 1.00)));
            pts.push_back(ColorControlPoint(0.3, eavlColor(0.00, 0.50, 0.00)));
            pts.push_back(ColorControlPoint(0.4, eavlColor(1.00, 1.00, 0.00)));
            pts.push_back(ColorControlPoint(0.5, eavlColor(0.60, 0.47, 0.00)));
            pts.push_back(ColorControlPoint(0.6, eavlColor(1.00, 0.47, 0.00)));
            pts.push_back(ColorControlPoint(0.7, eavlColor(0.61, 0.18, 0.00)));
            pts.push_back(ColorControlPoint(0.8, eavlColor(1.00, 0.03, 0.17)));
            pts.push_back(ColorControlPoint(0.9, eavlColor(0.63, 0.12, 0.34)));
            pts.push_back(ColorControlPoint(1.0, eavlColor(1.00, 0.40, 1.00)));
        }
        else if (name == "thermal")
        {
            pts.push_back(ColorControlPoint(0.0, eavlColor(0.30, 0.00, 0.00)));
            pts.push_back(ColorControlPoint(0.25,eavlColor(1.00, 0.00, 0.00)));
            pts.push_back(ColorControlPoint(0.50,eavlColor(1.00, 1.00, 0.00)));
            pts.push_back(ColorControlPoint(0.55,eavlColor(0.80, 0.55, 0.20)));
            pts.push_back(ColorControlPoint(0.60,eavlColor(0.60, 0.37, 0.40)));
            pts.push_back(ColorControlPoint(0.65,eavlColor(0.40, 0.22, 0.60)));
            pts.push_back(ColorControlPoint(0.75,eavlColor(0.00, 0.00, 1.00)));
            pts.push_back(ColorControlPoint(1.00,eavlColor(1.00, 1.00, 1.00)));
        }
        // The following five tables are perceeptually linearized colortables
        // (4 rainbow, one heatmap) from BSD-licensed code by Matteo Niccoli.
        // See: http://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
        else if (name == "IsoL")
        {
            double n = 5;
            pts.push_back(ColorControlPoint(0./n,  eavlColor(0.9102, 0.2236, 0.8997)));
            pts.push_back(ColorControlPoint(1./n,  eavlColor(0.4027, 0.3711, 1.0000)));
            pts.push_back(ColorControlPoint(2./n,  eavlColor(0.0422, 0.5904, 0.5899)));
            pts.push_back(ColorControlPoint(3./n,  eavlColor(0.0386, 0.6206, 0.0201)));
            pts.push_back(ColorControlPoint(4./n,  eavlColor(0.5441, 0.5428, 0.0110)));
            pts.push_back(ColorControlPoint(5./n,  eavlColor(1.0000, 0.2288, 0.1631)));
        }
        else if (name == "CubicL")
        {
            double n = 15;
            pts.push_back(ColorControlPoint(0./n,  eavlColor(0.4706, 0.0000, 0.5216)));
            pts.push_back(ColorControlPoint(1./n,  eavlColor(0.5137, 0.0527, 0.7096)));
            pts.push_back(ColorControlPoint(2./n,  eavlColor(0.4942, 0.2507, 0.8781)));
            pts.push_back(ColorControlPoint(3./n,  eavlColor(0.4296, 0.3858, 0.9922)));
            pts.push_back(ColorControlPoint(4./n,  eavlColor(0.3691, 0.5172, 0.9495)));
            pts.push_back(ColorControlPoint(5./n,  eavlColor(0.2963, 0.6191, 0.8515)));
            pts.push_back(ColorControlPoint(6./n,  eavlColor(0.2199, 0.7134, 0.7225)));
            pts.push_back(ColorControlPoint(7./n,  eavlColor(0.2643, 0.7836, 0.5756)));
            pts.push_back(ColorControlPoint(8./n,  eavlColor(0.3094, 0.8388, 0.4248)));
            pts.push_back(ColorControlPoint(9./n,  eavlColor(0.3623, 0.8917, 0.2858)));
            pts.push_back(ColorControlPoint(10./n, eavlColor(0.5200, 0.9210, 0.3137)));
            pts.push_back(ColorControlPoint(11./n, eavlColor(0.6800, 0.9255, 0.3386)));
            pts.push_back(ColorControlPoint(12./n, eavlColor(0.8000, 0.9255, 0.3529)));
            pts.push_back(ColorControlPoint(13./n, eavlColor(0.8706, 0.8549, 0.3608)));
            pts.push_back(ColorControlPoint(14./n, eavlColor(0.9514, 0.7466, 0.3686)));
            pts.push_back(ColorControlPoint(15./n, eavlColor(0.9765, 0.5887, 0.3569)));
        }
        else if (name == "CubicYF")
        {
            double n = 15;
            pts.push_back(ColorControlPoint(0./n,  eavlColor(0.5151, 0.0482, 0.6697)));
            pts.push_back(ColorControlPoint(1./n,  eavlColor(0.5199, 0.1762, 0.8083)));
            pts.push_back(ColorControlPoint(2./n,  eavlColor(0.4884, 0.2912, 0.9234)));
            pts.push_back(ColorControlPoint(3./n,  eavlColor(0.4297, 0.3855, 0.9921)));
            pts.push_back(ColorControlPoint(4./n,  eavlColor(0.3893, 0.4792, 0.9775)));
            pts.push_back(ColorControlPoint(5./n,  eavlColor(0.3337, 0.5650, 0.9056)));
            pts.push_back(ColorControlPoint(6./n,  eavlColor(0.2795, 0.6419, 0.8287)));
            pts.push_back(ColorControlPoint(7./n,  eavlColor(0.2210, 0.7123, 0.7258)));
            pts.push_back(ColorControlPoint(8./n,  eavlColor(0.2468, 0.7612, 0.6248)));
            pts.push_back(ColorControlPoint(9./n,  eavlColor(0.2833, 0.8125, 0.5069)));
            pts.push_back(ColorControlPoint(10./n, eavlColor(0.3198, 0.8492, 0.3956)));
            pts.push_back(ColorControlPoint(11./n, eavlColor(0.3602, 0.8896, 0.2919)));
            pts.push_back(ColorControlPoint(12./n, eavlColor(0.4568, 0.9136, 0.3018)));
            pts.push_back(ColorControlPoint(13./n, eavlColor(0.6033, 0.9255, 0.3295)));
            pts.push_back(ColorControlPoint(14./n, eavlColor(0.7066, 0.9255, 0.3414)));
            pts.push_back(ColorControlPoint(15./n, eavlColor(0.8000, 0.9255, 0.3529)));
        }
        else if (name == "LinearL")
        {
            double n = 15;
            pts.push_back(ColorControlPoint(0./n,  eavlColor(0.0143, 0.0143, 0.0143)));
            pts.push_back(ColorControlPoint(1./n,  eavlColor(0.1413, 0.0555, 0.1256)));
            pts.push_back(ColorControlPoint(2./n,  eavlColor(0.1761, 0.0911, 0.2782)));
            pts.push_back(ColorControlPoint(3./n,  eavlColor(0.1710, 0.1314, 0.4540)));
            pts.push_back(ColorControlPoint(4./n,  eavlColor(0.1074, 0.2234, 0.4984)));
            pts.push_back(ColorControlPoint(5./n,  eavlColor(0.0686, 0.3044, 0.5068)));
            pts.push_back(ColorControlPoint(6./n,  eavlColor(0.0008, 0.3927, 0.4267)));
            pts.push_back(ColorControlPoint(7./n,  eavlColor(0.0000, 0.4763, 0.3464)));
            pts.push_back(ColorControlPoint(8./n,  eavlColor(0.0000, 0.5565, 0.2469)));
            pts.push_back(ColorControlPoint(9./n,  eavlColor(0.0000, 0.6381, 0.1638)));
            pts.push_back(ColorControlPoint(10./n, eavlColor(0.2167, 0.6966, 0.0000)));
            pts.push_back(ColorControlPoint(11./n, eavlColor(0.3898, 0.7563, 0.0000)));
            pts.push_back(ColorControlPoint(12./n, eavlColor(0.6912, 0.7795, 0.0000)));
            pts.push_back(ColorControlPoint(13./n, eavlColor(0.8548, 0.8041, 0.4555)));
            pts.push_back(ColorControlPoint(14./n, eavlColor(0.9712, 0.8429, 0.7287)));
            pts.push_back(ColorControlPoint(15./n, eavlColor(0.9692, 0.9273, 0.8961)));
        }
        else if (name == "LinLhot")
        {
            double n = 15;
            pts.push_back(ColorControlPoint(0./n,  eavlColor(0.0225, 0.0121, 0.0121)));
            pts.push_back(ColorControlPoint(1./n,  eavlColor(0.1927, 0.0225, 0.0311)));
            pts.push_back(ColorControlPoint(2./n,  eavlColor(0.3243, 0.0106, 0.0000)));
            pts.push_back(ColorControlPoint(3./n,  eavlColor(0.4463, 0.0000, 0.0091)));
            pts.push_back(ColorControlPoint(4./n,  eavlColor(0.5706, 0.0000, 0.0737)));
            pts.push_back(ColorControlPoint(5./n,  eavlColor(0.6969, 0.0000, 0.1337)));
            pts.push_back(ColorControlPoint(6./n,  eavlColor(0.8213, 0.0000, 0.1792)));
            pts.push_back(ColorControlPoint(7./n,  eavlColor(0.8636, 0.0000, 0.0565)));
            pts.push_back(ColorControlPoint(8./n,  eavlColor(0.8821, 0.2555, 0.0000)));
            pts.push_back(ColorControlPoint(9./n,  eavlColor(0.8720, 0.4182, 0.0000)));
            pts.push_back(ColorControlPoint(10./n, eavlColor(0.8424, 0.5552, 0.0000)));
            pts.push_back(ColorControlPoint(11./n, eavlColor(0.8031, 0.6776, 0.0000)));
            pts.push_back(ColorControlPoint(12./n, eavlColor(0.7659, 0.7870, 0.0000)));
            pts.push_back(ColorControlPoint(13./n, eavlColor(0.8170, 0.8296, 0.0000)));
            pts.push_back(ColorControlPoint(14./n, eavlColor(0.8853, 0.8896, 0.4113)));
            pts.push_back(ColorControlPoint(15./n, eavlColor(0.9481, 0.9486, 0.7165)));
        }
        else 
            THROW(eavlException, "Unknown color table");
    }
};

#endif

