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
    class ColorControlPoint
    {
      public:
        float position;
        eavlColor color;
        ColorControlPoint(float p, const eavlColor &c) : position(p), color(c) { }
    };

  protected:
    string uniquename;
    bool smooth;
    vector<ColorControlPoint> pts;
  public:
    const string &GetName() const
    {
        return uniquename;
    }
    bool GetSmooth() const
    {
        return smooth;
    }
    void Sample(int n, float *colors) const
    {
        for (int i=0; i<n; i++)
        {
            eavlColor c = Map(float(i)/float(n-1));
            colors[3*i+0] = c.c[0];
            colors[3*i+1] = c.c[1];
            colors[3*i+2] = c.c[2];
        }
    }
    void Sample(int n, double *colors) const
    {
        for (int i=0; i<n; i++)
        {
            eavlColor c = Map(float(i)/float(n-1));
            colors[3*i+0] = c.c[0];
            colors[3*i+1] = c.c[1];
            colors[3*i+2] = c.c[2];
        }
    }
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
    eavlColorTable() : 
        uniquename(""), smooth(false)
    {
    }
    eavlColorTable(bool smooth_) : 
        uniquename("")
    {
      smooth = smooth_;
    }
    eavlColorTable(const eavlColorTable &ct) : 
        uniquename(ct.uniquename), smooth(ct.smooth), pts(ct.pts.begin(), ct.pts.end())
    {
    }
    void operator=(const eavlColorTable &ct)
    {
        uniquename = ct.uniquename;
        smooth = ct.smooth;
        pts.clear();
        pts.insert(pts.end(), ct.pts.begin(), ct.pts.end());
    }
    void Clear()
    {
        pts.clear();
    }
    void AddControlPoint(double v, eavlColor c)
    {
        pts.push_back(ColorControlPoint(v, c));
    }
    void Reverse()
    {
        // copy old control points
        vector<ColorControlPoint> tmp(pts.begin(), pts.end());
        Clear();
        for (int i=tmp.size()-1; i>=0; --i)
            AddControlPoint(1.0 - tmp[i].position, tmp[i].color);

        if (uniquename[1] == '0')
            uniquename[1] = '1';
        else
            uniquename[1] = '0';
    }
    eavlColorTable(string name)
    {
        if (name == "" || name == "default")
            name = "dense";

        smooth = true;
        if (name == "grey" || name == "gray")
        {
            AddControlPoint(0.0, eavlColor( 0, 0, 0));
            AddControlPoint(1.0, eavlColor( 1, 1, 1));
        }
        else if (name == "blue")
        {
            AddControlPoint(0.00, eavlColor( 0, 0, 0));
            AddControlPoint(0.33, eavlColor( 0, 0,.5));
            AddControlPoint(0.66, eavlColor( 0,.5, 1));
            AddControlPoint(1.00, eavlColor( 1, 1, 1));
        }
        else if (name == "orange")
        {
            AddControlPoint(0.00, eavlColor( 0, 0, 0));
            AddControlPoint(0.33, eavlColor(.5, 0, 0));
            AddControlPoint(0.66, eavlColor( 1,.5, 0));
            AddControlPoint(1.00, eavlColor( 1, 1, 1));
        }
        else if (name == "temperature")
        {
            AddControlPoint(0.05, eavlColor( 0, 0, 1));
            AddControlPoint(0.35, eavlColor( 0, 1, 1));
            AddControlPoint(0.50, eavlColor( 1, 1, 1));
            AddControlPoint(0.65, eavlColor( 1, 1, 0));
            AddControlPoint(0.95, eavlColor( 1, 0, 0));
        }
        else if (name == "rainbow")
        {
            AddControlPoint(0.00, eavlColor( 0, 0, 1));
            AddControlPoint(0.20, eavlColor( 0, 1, 1));
            AddControlPoint(0.45, eavlColor( 0, 1, 0));
            AddControlPoint(0.55, eavlColor( .7, 1, 0));
            AddControlPoint(0.6,  eavlColor( 1, 1, 0));
            AddControlPoint(0.75, eavlColor( 1, .5, 0));
            AddControlPoint(0.9,  eavlColor( 1, 0, 0));
            AddControlPoint(0.98, eavlColor( 1, 0, .5));
            AddControlPoint(1.0,  eavlColor( 1, 0, 1));
        }
        else if (name == "levels")
        {
            AddControlPoint(0.0, eavlColor( 0, 0, 1));
            AddControlPoint(0.2, eavlColor( 0, 0, 1));
            AddControlPoint(0.2, eavlColor( 0, 1, 1));
            AddControlPoint(0.4, eavlColor( 0, 1, 1));
            AddControlPoint(0.4, eavlColor( 0, 1, 0));
            AddControlPoint(0.6, eavlColor( 0, 1, 0));
            AddControlPoint(0.6, eavlColor( 1, 1, 0));
            AddControlPoint(0.8, eavlColor( 1, 1, 0));
            AddControlPoint(0.8, eavlColor( 1, 0, 0));
            AddControlPoint(1.0, eavlColor( 1, 0, 0));
        }
        else if (name == "dense" || name == "sharp")
        {
            smooth = (name == "dense") ? true : false;
            AddControlPoint(0.0, eavlColor(0.26, 0.22, 0.92));
            AddControlPoint(0.1, eavlColor(0.00, 0.00, 0.52));
            AddControlPoint(0.2, eavlColor(0.00, 1.00, 1.00));
            AddControlPoint(0.3, eavlColor(0.00, 0.50, 0.00));
            AddControlPoint(0.4, eavlColor(1.00, 1.00, 0.00));
            AddControlPoint(0.5, eavlColor(0.60, 0.47, 0.00));
            AddControlPoint(0.6, eavlColor(1.00, 0.47, 0.00));
            AddControlPoint(0.7, eavlColor(0.61, 0.18, 0.00));
            AddControlPoint(0.8, eavlColor(1.00, 0.03, 0.17));
            AddControlPoint(0.9, eavlColor(0.63, 0.12, 0.34));
            AddControlPoint(1.0, eavlColor(1.00, 0.40, 1.00));
        }
        else if (name == "thermal")
        {
            AddControlPoint(0.0, eavlColor(0.30, 0.00, 0.00));
            AddControlPoint(0.25,eavlColor(1.00, 0.00, 0.00));
            AddControlPoint(0.50,eavlColor(1.00, 1.00, 0.00));
            AddControlPoint(0.55,eavlColor(0.80, 0.55, 0.20));
            AddControlPoint(0.60,eavlColor(0.60, 0.37, 0.40));
            AddControlPoint(0.65,eavlColor(0.40, 0.22, 0.60));
            AddControlPoint(0.75,eavlColor(0.00, 0.00, 1.00));
            AddControlPoint(1.00,eavlColor(1.00, 1.00, 1.00));
        }
        // The following five tables are perceeptually linearized colortables
        // (4 rainbow, one heatmap) from BSD-licensed code by Matteo Niccoli.
        // See: http://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/
        else if (name == "IsoL")
        {
            double n = 5;
            AddControlPoint(0./n,  eavlColor(0.9102, 0.2236, 0.8997));
            AddControlPoint(1./n,  eavlColor(0.4027, 0.3711, 1.0000));
            AddControlPoint(2./n,  eavlColor(0.0422, 0.5904, 0.5899));
            AddControlPoint(3./n,  eavlColor(0.0386, 0.6206, 0.0201));
            AddControlPoint(4./n,  eavlColor(0.5441, 0.5428, 0.0110));
            AddControlPoint(5./n,  eavlColor(1.0000, 0.2288, 0.1631));
        }
        else if (name == "CubicL")
        {
            double n = 15;
            AddControlPoint(0./n,  eavlColor(0.4706, 0.0000, 0.5216));
            AddControlPoint(1./n,  eavlColor(0.5137, 0.0527, 0.7096));
            AddControlPoint(2./n,  eavlColor(0.4942, 0.2507, 0.8781));
            AddControlPoint(3./n,  eavlColor(0.4296, 0.3858, 0.9922));
            AddControlPoint(4./n,  eavlColor(0.3691, 0.5172, 0.9495));
            AddControlPoint(5./n,  eavlColor(0.2963, 0.6191, 0.8515));
            AddControlPoint(6./n,  eavlColor(0.2199, 0.7134, 0.7225));
            AddControlPoint(7./n,  eavlColor(0.2643, 0.7836, 0.5756));
            AddControlPoint(8./n,  eavlColor(0.3094, 0.8388, 0.4248));
            AddControlPoint(9./n,  eavlColor(0.3623, 0.8917, 0.2858));
            AddControlPoint(10./n, eavlColor(0.5200, 0.9210, 0.3137));
            AddControlPoint(11./n, eavlColor(0.6800, 0.9255, 0.3386));
            AddControlPoint(12./n, eavlColor(0.8000, 0.9255, 0.3529));
            AddControlPoint(13./n, eavlColor(0.8706, 0.8549, 0.3608));
            AddControlPoint(14./n, eavlColor(0.9514, 0.7466, 0.3686));
            AddControlPoint(15./n, eavlColor(0.9765, 0.5887, 0.3569));
        }
        else if (name == "CubicYF")
        {
            double n = 15;
            AddControlPoint(0./n,  eavlColor(0.5151, 0.0482, 0.6697));
            AddControlPoint(1./n,  eavlColor(0.5199, 0.1762, 0.8083));
            AddControlPoint(2./n,  eavlColor(0.4884, 0.2912, 0.9234));
            AddControlPoint(3./n,  eavlColor(0.4297, 0.3855, 0.9921));
            AddControlPoint(4./n,  eavlColor(0.3893, 0.4792, 0.9775));
            AddControlPoint(5./n,  eavlColor(0.3337, 0.5650, 0.9056));
            AddControlPoint(6./n,  eavlColor(0.2795, 0.6419, 0.8287));
            AddControlPoint(7./n,  eavlColor(0.2210, 0.7123, 0.7258));
            AddControlPoint(8./n,  eavlColor(0.2468, 0.7612, 0.6248));
            AddControlPoint(9./n,  eavlColor(0.2833, 0.8125, 0.5069));
            AddControlPoint(10./n, eavlColor(0.3198, 0.8492, 0.3956));
            AddControlPoint(11./n, eavlColor(0.3602, 0.8896, 0.2919));
            AddControlPoint(12./n, eavlColor(0.4568, 0.9136, 0.3018));
            AddControlPoint(13./n, eavlColor(0.6033, 0.9255, 0.3295));
            AddControlPoint(14./n, eavlColor(0.7066, 0.9255, 0.3414));
            AddControlPoint(15./n, eavlColor(0.8000, 0.9255, 0.3529));
        }
        else if (name == "LinearL")
        {
            double n = 15;
            AddControlPoint(0./n,  eavlColor(0.0143, 0.0143, 0.0143));
            AddControlPoint(1./n,  eavlColor(0.1413, 0.0555, 0.1256));
            AddControlPoint(2./n,  eavlColor(0.1761, 0.0911, 0.2782));
            AddControlPoint(3./n,  eavlColor(0.1710, 0.1314, 0.4540));
            AddControlPoint(4./n,  eavlColor(0.1074, 0.2234, 0.4984));
            AddControlPoint(5./n,  eavlColor(0.0686, 0.3044, 0.5068));
            AddControlPoint(6./n,  eavlColor(0.0008, 0.3927, 0.4267));
            AddControlPoint(7./n,  eavlColor(0.0000, 0.4763, 0.3464));
            AddControlPoint(8./n,  eavlColor(0.0000, 0.5565, 0.2469));
            AddControlPoint(9./n,  eavlColor(0.0000, 0.6381, 0.1638));
            AddControlPoint(10./n, eavlColor(0.2167, 0.6966, 0.0000));
            AddControlPoint(11./n, eavlColor(0.3898, 0.7563, 0.0000));
            AddControlPoint(12./n, eavlColor(0.6912, 0.7795, 0.0000));
            AddControlPoint(13./n, eavlColor(0.8548, 0.8041, 0.4555));
            AddControlPoint(14./n, eavlColor(0.9712, 0.8429, 0.7287));
            AddControlPoint(15./n, eavlColor(0.9692, 0.9273, 0.8961));
        }
        else if (name == "LinLhot")
        {
            double n = 15;
            AddControlPoint(0./n,  eavlColor(0.0225, 0.0121, 0.0121));
            AddControlPoint(1./n,  eavlColor(0.1927, 0.0225, 0.0311));
            AddControlPoint(2./n,  eavlColor(0.3243, 0.0106, 0.0000));
            AddControlPoint(3./n,  eavlColor(0.4463, 0.0000, 0.0091));
            AddControlPoint(4./n,  eavlColor(0.5706, 0.0000, 0.0737));
            AddControlPoint(5./n,  eavlColor(0.6969, 0.0000, 0.1337));
            AddControlPoint(6./n,  eavlColor(0.8213, 0.0000, 0.1792));
            AddControlPoint(7./n,  eavlColor(0.8636, 0.0000, 0.0565));
            AddControlPoint(8./n,  eavlColor(0.8821, 0.2555, 0.0000));
            AddControlPoint(9./n,  eavlColor(0.8720, 0.4182, 0.0000));
            AddControlPoint(10./n, eavlColor(0.8424, 0.5552, 0.0000));
            AddControlPoint(11./n, eavlColor(0.8031, 0.6776, 0.0000));
            AddControlPoint(12./n, eavlColor(0.7659, 0.7870, 0.0000));
            AddControlPoint(13./n, eavlColor(0.8170, 0.8296, 0.0000));
            AddControlPoint(14./n, eavlColor(0.8853, 0.8896, 0.4113));
            AddControlPoint(15./n, eavlColor(0.9481, 0.9486, 0.7165));
        }
        // ColorBrewer tables here.  (See LICENSE.txt)
        else if (name == "PuRd")
        {
            AddControlPoint(0.0000, eavlColor(0.9686, 0.9569, 0.9765));
            AddControlPoint(0.1250, eavlColor(0.9059, 0.8824, 0.9373));
            AddControlPoint(0.2500, eavlColor(0.8314, 0.7255, 0.8549));
            AddControlPoint(0.3750, eavlColor(0.7882, 0.5804, 0.7804));
            AddControlPoint(0.5000, eavlColor(0.8745, 0.3961, 0.6902));
            AddControlPoint(0.6250, eavlColor(0.9059, 0.1608, 0.5412));
            AddControlPoint(0.7500, eavlColor(0.8078, 0.0706, 0.3373));
            AddControlPoint(0.8750, eavlColor(0.5961, 0.0000, 0.2627));
            AddControlPoint(1.0000, eavlColor(0.4039, 0.0000, 0.1216));
        }
        else if (name == "Accent")
        {
            AddControlPoint(0.0000, eavlColor(0.4980, 0.7882, 0.4980));
            AddControlPoint(0.1429, eavlColor(0.7451, 0.6824, 0.8314));
            AddControlPoint(0.2857, eavlColor(0.9922, 0.7529, 0.5255));
            AddControlPoint(0.4286, eavlColor(1.0000, 1.0000, 0.6000));
            AddControlPoint(0.5714, eavlColor(0.2196, 0.4235, 0.6902));
            AddControlPoint(0.7143, eavlColor(0.9412, 0.0078, 0.4980));
            AddControlPoint(0.8571, eavlColor(0.7490, 0.3569, 0.0902));
            AddControlPoint(1.0000, eavlColor(0.4000, 0.4000, 0.4000));
        }
        else if (name == "Blues")
        {
            AddControlPoint(0.0000, eavlColor(0.9686, 0.9843, 1.0000));
            AddControlPoint(0.1250, eavlColor(0.8706, 0.9216, 0.9686));
            AddControlPoint(0.2500, eavlColor(0.7765, 0.8588, 0.9373));
            AddControlPoint(0.3750, eavlColor(0.6196, 0.7922, 0.8824));
            AddControlPoint(0.5000, eavlColor(0.4196, 0.6824, 0.8392));
            AddControlPoint(0.6250, eavlColor(0.2588, 0.5725, 0.7765));
            AddControlPoint(0.7500, eavlColor(0.1294, 0.4431, 0.7098));
            AddControlPoint(0.8750, eavlColor(0.0314, 0.3176, 0.6118));
            AddControlPoint(1.0000, eavlColor(0.0314, 0.1882, 0.4196));
        }
        else if (name == "BrBG")
        {
            AddControlPoint(0.0000, eavlColor(0.3294, 0.1882, 0.0196));
            AddControlPoint(0.1000, eavlColor(0.5490, 0.3176, 0.0392));
            AddControlPoint(0.2000, eavlColor(0.7490, 0.5059, 0.1765));
            AddControlPoint(0.3000, eavlColor(0.8745, 0.7608, 0.4902));
            AddControlPoint(0.4000, eavlColor(0.9647, 0.9098, 0.7647));
            AddControlPoint(0.5000, eavlColor(0.9608, 0.9608, 0.9608));
            AddControlPoint(0.6000, eavlColor(0.7804, 0.9176, 0.8980));
            AddControlPoint(0.7000, eavlColor(0.5020, 0.8039, 0.7569));
            AddControlPoint(0.8000, eavlColor(0.2078, 0.5922, 0.5608));
            AddControlPoint(0.9000, eavlColor(0.0039, 0.4000, 0.3686));
            AddControlPoint(1.0000, eavlColor(0.0000, 0.2353, 0.1882));
        }
        else if (name == "BuGn")
        {
            AddControlPoint(0.0000, eavlColor(0.9686, 0.9882, 0.9922));
            AddControlPoint(0.1250, eavlColor(0.8980, 0.9608, 0.9765));
            AddControlPoint(0.2500, eavlColor(0.8000, 0.9255, 0.9020));
            AddControlPoint(0.3750, eavlColor(0.6000, 0.8471, 0.7882));
            AddControlPoint(0.5000, eavlColor(0.4000, 0.7608, 0.6431));
            AddControlPoint(0.6250, eavlColor(0.2549, 0.6824, 0.4627));
            AddControlPoint(0.7500, eavlColor(0.1373, 0.5451, 0.2706));
            AddControlPoint(0.8750, eavlColor(0.0000, 0.4275, 0.1725));
            AddControlPoint(1.0000, eavlColor(0.0000, 0.2667, 0.1059));
        }
        else if (name == "BuPu")
        {
            AddControlPoint(0.0000, eavlColor(0.9686, 0.9882, 0.9922));
            AddControlPoint(0.1250, eavlColor(0.8784, 0.9255, 0.9569));
            AddControlPoint(0.2500, eavlColor(0.7490, 0.8275, 0.9020));
            AddControlPoint(0.3750, eavlColor(0.6196, 0.7373, 0.8549));
            AddControlPoint(0.5000, eavlColor(0.5490, 0.5882, 0.7765));
            AddControlPoint(0.6250, eavlColor(0.5490, 0.4196, 0.6941));
            AddControlPoint(0.7500, eavlColor(0.5333, 0.2549, 0.6157));
            AddControlPoint(0.8750, eavlColor(0.5059, 0.0588, 0.4863));
            AddControlPoint(1.0000, eavlColor(0.3020, 0.0000, 0.2941));
        }
        else if (name == "Dark2")
        {
            AddControlPoint(0.0000, eavlColor(0.1059, 0.6196, 0.4667));
            AddControlPoint(0.1429, eavlColor(0.8510, 0.3725, 0.0078));
            AddControlPoint(0.2857, eavlColor(0.4588, 0.4392, 0.7020));
            AddControlPoint(0.4286, eavlColor(0.9059, 0.1608, 0.5412));
            AddControlPoint(0.5714, eavlColor(0.4000, 0.6510, 0.1176));
            AddControlPoint(0.7143, eavlColor(0.9020, 0.6706, 0.0078));
            AddControlPoint(0.8571, eavlColor(0.6510, 0.4627, 0.1137));
            AddControlPoint(1.0000, eavlColor(0.4000, 0.4000, 0.4000));
        }
        else if (name == "GnBu")
        {
            AddControlPoint(0.0000, eavlColor(0.9686, 0.9882, 0.9412));
            AddControlPoint(0.1250, eavlColor(0.8784, 0.9529, 0.8588));
            AddControlPoint(0.2500, eavlColor(0.8000, 0.9216, 0.7725));
            AddControlPoint(0.3750, eavlColor(0.6588, 0.8667, 0.7098));
            AddControlPoint(0.5000, eavlColor(0.4824, 0.8000, 0.7686));
            AddControlPoint(0.6250, eavlColor(0.3059, 0.7020, 0.8275));
            AddControlPoint(0.7500, eavlColor(0.1686, 0.5490, 0.7451));
            AddControlPoint(0.8750, eavlColor(0.0314, 0.4078, 0.6745));
            AddControlPoint(1.0000, eavlColor(0.0314, 0.2510, 0.5059));
        }
        else if (name == "Greens")
        {
            AddControlPoint(0.0000, eavlColor(0.9686, 0.9882, 0.9608));
            AddControlPoint(0.1250, eavlColor(0.8980, 0.9608, 0.8784));
            AddControlPoint(0.2500, eavlColor(0.7804, 0.9137, 0.7529));
            AddControlPoint(0.3750, eavlColor(0.6314, 0.8510, 0.6078));
            AddControlPoint(0.5000, eavlColor(0.4549, 0.7686, 0.4627));
            AddControlPoint(0.6250, eavlColor(0.2549, 0.6706, 0.3647));
            AddControlPoint(0.7500, eavlColor(0.1373, 0.5451, 0.2706));
            AddControlPoint(0.8750, eavlColor(0.0000, 0.4275, 0.1725));
            AddControlPoint(1.0000, eavlColor(0.0000, 0.2667, 0.1059));
        }
        else if (name == "Greys")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 1.0000, 1.0000));
            AddControlPoint(0.1250, eavlColor(0.9412, 0.9412, 0.9412));
            AddControlPoint(0.2500, eavlColor(0.8510, 0.8510, 0.8510));
            AddControlPoint(0.3750, eavlColor(0.7412, 0.7412, 0.7412));
            AddControlPoint(0.5000, eavlColor(0.5882, 0.5882, 0.5882));
            AddControlPoint(0.6250, eavlColor(0.4510, 0.4510, 0.4510));
            AddControlPoint(0.7500, eavlColor(0.3216, 0.3216, 0.3216));
            AddControlPoint(0.8750, eavlColor(0.1451, 0.1451, 0.1451));
            AddControlPoint(1.0000, eavlColor(0.0000, 0.0000, 0.0000));
        }
        else if (name == "Oranges")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 0.9608, 0.9216));
            AddControlPoint(0.1250, eavlColor(0.9961, 0.9020, 0.8078));
            AddControlPoint(0.2500, eavlColor(0.9922, 0.8157, 0.6353));
            AddControlPoint(0.3750, eavlColor(0.9922, 0.6824, 0.4196));
            AddControlPoint(0.5000, eavlColor(0.9922, 0.5529, 0.2353));
            AddControlPoint(0.6250, eavlColor(0.9451, 0.4118, 0.0745));
            AddControlPoint(0.7500, eavlColor(0.8510, 0.2824, 0.0039));
            AddControlPoint(0.8750, eavlColor(0.6510, 0.2118, 0.0118));
            AddControlPoint(1.0000, eavlColor(0.4980, 0.1529, 0.0157));
        }
        else if (name == "OrRd")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 0.9686, 0.9255));
            AddControlPoint(0.1250, eavlColor(0.9961, 0.9098, 0.7843));
            AddControlPoint(0.2500, eavlColor(0.9922, 0.8314, 0.6196));
            AddControlPoint(0.3750, eavlColor(0.9922, 0.7333, 0.5176));
            AddControlPoint(0.5000, eavlColor(0.9882, 0.5529, 0.3490));
            AddControlPoint(0.6250, eavlColor(0.9373, 0.3961, 0.2824));
            AddControlPoint(0.7500, eavlColor(0.8431, 0.1882, 0.1216));
            AddControlPoint(0.8750, eavlColor(0.7020, 0.0000, 0.0000));
            AddControlPoint(1.0000, eavlColor(0.4980, 0.0000, 0.0000));
        }
        else if (name == "Paired")
        {
            AddControlPoint(0.0000, eavlColor(0.6510, 0.8078, 0.8902));
            AddControlPoint(0.0909, eavlColor(0.1216, 0.4706, 0.7059));
            AddControlPoint(0.1818, eavlColor(0.6980, 0.8745, 0.5412));
            AddControlPoint(0.2727, eavlColor(0.2000, 0.6275, 0.1725));
            AddControlPoint(0.3636, eavlColor(0.9843, 0.6039, 0.6000));
            AddControlPoint(0.4545, eavlColor(0.8902, 0.1020, 0.1098));
            AddControlPoint(0.5455, eavlColor(0.9922, 0.7490, 0.4353));
            AddControlPoint(0.6364, eavlColor(1.0000, 0.4980, 0.0000));
            AddControlPoint(0.7273, eavlColor(0.7922, 0.6980, 0.8392));
            AddControlPoint(0.8182, eavlColor(0.4157, 0.2392, 0.6039));
            AddControlPoint(0.9091, eavlColor(1.0000, 1.0000, 0.6000));
            AddControlPoint(1.0000, eavlColor(0.6941, 0.3490, 0.1569));
        }
        else if (name == "Pastel1")
        {
            AddControlPoint(0.0000, eavlColor(0.9843, 0.7059, 0.6824));
            AddControlPoint(0.1250, eavlColor(0.7020, 0.8039, 0.8902));
            AddControlPoint(0.2500, eavlColor(0.8000, 0.9216, 0.7725));
            AddControlPoint(0.3750, eavlColor(0.8706, 0.7961, 0.8941));
            AddControlPoint(0.5000, eavlColor(0.9961, 0.8510, 0.6510));
            AddControlPoint(0.6250, eavlColor(1.0000, 1.0000, 0.8000));
            AddControlPoint(0.7500, eavlColor(0.8980, 0.8471, 0.7412));
            AddControlPoint(0.8750, eavlColor(0.9922, 0.8549, 0.9255));
            AddControlPoint(1.0000, eavlColor(0.9490, 0.9490, 0.9490));
        }
        else if (name == "Pastel2")
        {
            AddControlPoint(0.0000, eavlColor(0.7020, 0.8863, 0.8039));
            AddControlPoint(0.1429, eavlColor(0.9922, 0.8039, 0.6745));
            AddControlPoint(0.2857, eavlColor(0.7961, 0.8353, 0.9098));
            AddControlPoint(0.4286, eavlColor(0.9569, 0.7922, 0.8941));
            AddControlPoint(0.5714, eavlColor(0.9020, 0.9608, 0.7882));
            AddControlPoint(0.7143, eavlColor(1.0000, 0.9490, 0.6824));
            AddControlPoint(0.8571, eavlColor(0.9451, 0.8863, 0.8000));
            AddControlPoint(1.0000, eavlColor(0.8000, 0.8000, 0.8000));
        }
        else if (name == "PiYG")
        {
            AddControlPoint(0.0000, eavlColor(0.5569, 0.0039, 0.3216));
            AddControlPoint(0.1000, eavlColor(0.7725, 0.1059, 0.4902));
            AddControlPoint(0.2000, eavlColor(0.8706, 0.4667, 0.6824));
            AddControlPoint(0.3000, eavlColor(0.9451, 0.7137, 0.8549));
            AddControlPoint(0.4000, eavlColor(0.9922, 0.8784, 0.9373));
            AddControlPoint(0.5000, eavlColor(0.9686, 0.9686, 0.9686));
            AddControlPoint(0.6000, eavlColor(0.9020, 0.9608, 0.8157));
            AddControlPoint(0.7000, eavlColor(0.7216, 0.8824, 0.5255));
            AddControlPoint(0.8000, eavlColor(0.4980, 0.7373, 0.2549));
            AddControlPoint(0.9000, eavlColor(0.3020, 0.5725, 0.1294));
            AddControlPoint(1.0000, eavlColor(0.1529, 0.3922, 0.0980));
        }
        else if (name == "PRGn")
        {
            AddControlPoint(0.0000, eavlColor(0.2510, 0.0000, 0.2941));
            AddControlPoint(0.1000, eavlColor(0.4627, 0.1647, 0.5137));
            AddControlPoint(0.2000, eavlColor(0.6000, 0.4392, 0.6706));
            AddControlPoint(0.3000, eavlColor(0.7608, 0.6471, 0.8118));
            AddControlPoint(0.4000, eavlColor(0.9059, 0.8314, 0.9098));
            AddControlPoint(0.5000, eavlColor(0.9686, 0.9686, 0.9686));
            AddControlPoint(0.6000, eavlColor(0.8510, 0.9412, 0.8275));
            AddControlPoint(0.7000, eavlColor(0.6510, 0.8588, 0.6275));
            AddControlPoint(0.8000, eavlColor(0.3529, 0.6824, 0.3804));
            AddControlPoint(0.9000, eavlColor(0.1059, 0.4706, 0.2157));
            AddControlPoint(1.0000, eavlColor(0.0000, 0.2667, 0.1059));
        }
        else if (name == "PuBu")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 0.9686, 0.9843));
            AddControlPoint(0.1250, eavlColor(0.9255, 0.9059, 0.9490));
            AddControlPoint(0.2500, eavlColor(0.8157, 0.8196, 0.9020));
            AddControlPoint(0.3750, eavlColor(0.6510, 0.7412, 0.8588));
            AddControlPoint(0.5000, eavlColor(0.4549, 0.6627, 0.8118));
            AddControlPoint(0.6250, eavlColor(0.2118, 0.5647, 0.7529));
            AddControlPoint(0.7500, eavlColor(0.0196, 0.4392, 0.6902));
            AddControlPoint(0.8750, eavlColor(0.0157, 0.3529, 0.5529));
            AddControlPoint(1.0000, eavlColor(0.0078, 0.2196, 0.3451));
        }
        else if (name == "PuBuGn")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 0.9686, 0.9843));
            AddControlPoint(0.1250, eavlColor(0.9255, 0.8863, 0.9412));
            AddControlPoint(0.2500, eavlColor(0.8157, 0.8196, 0.9020));
            AddControlPoint(0.3750, eavlColor(0.6510, 0.7412, 0.8588));
            AddControlPoint(0.5000, eavlColor(0.4039, 0.6627, 0.8118));
            AddControlPoint(0.6250, eavlColor(0.2118, 0.5647, 0.7529));
            AddControlPoint(0.7500, eavlColor(0.0078, 0.5059, 0.5412));
            AddControlPoint(0.8750, eavlColor(0.0039, 0.4235, 0.3490));
            AddControlPoint(1.0000, eavlColor(0.0039, 0.2745, 0.2118));
        }
        else if (name == "PuOr")
        {
            AddControlPoint(0.0000, eavlColor(0.4980, 0.2314, 0.0314));
            AddControlPoint(0.1000, eavlColor(0.7020, 0.3451, 0.0235));
            AddControlPoint(0.2000, eavlColor(0.8784, 0.5098, 0.0784));
            AddControlPoint(0.3000, eavlColor(0.9922, 0.7216, 0.3882));
            AddControlPoint(0.4000, eavlColor(0.9961, 0.8784, 0.7137));
            AddControlPoint(0.5000, eavlColor(0.9686, 0.9686, 0.9686));
            AddControlPoint(0.6000, eavlColor(0.8471, 0.8549, 0.9216));
            AddControlPoint(0.7000, eavlColor(0.6980, 0.6706, 0.8235));
            AddControlPoint(0.8000, eavlColor(0.5020, 0.4510, 0.6745));
            AddControlPoint(0.9000, eavlColor(0.3294, 0.1529, 0.5333));
            AddControlPoint(1.0000, eavlColor(0.1765, 0.0000, 0.2941));
        }
        else if (name == "PuRd")
        {
            AddControlPoint(0.0000, eavlColor(0.9686, 0.9569, 0.9765));
            AddControlPoint(0.1250, eavlColor(0.9059, 0.8824, 0.9373));
            AddControlPoint(0.2500, eavlColor(0.8314, 0.7255, 0.8549));
            AddControlPoint(0.3750, eavlColor(0.7882, 0.5804, 0.7804));
            AddControlPoint(0.5000, eavlColor(0.8745, 0.3961, 0.6902));
            AddControlPoint(0.6250, eavlColor(0.9059, 0.1608, 0.5412));
            AddControlPoint(0.7500, eavlColor(0.8078, 0.0706, 0.3373));
            AddControlPoint(0.8750, eavlColor(0.5961, 0.0000, 0.2627));
            AddControlPoint(1.0000, eavlColor(0.4039, 0.0000, 0.1216));
        }
        else if (name == "Purples")
        {
            AddControlPoint(0.0000, eavlColor(0.9882, 0.9843, 0.9922));
            AddControlPoint(0.1250, eavlColor(0.9373, 0.9294, 0.9608));
            AddControlPoint(0.2500, eavlColor(0.8549, 0.8549, 0.9216));
            AddControlPoint(0.3750, eavlColor(0.7373, 0.7412, 0.8627));
            AddControlPoint(0.5000, eavlColor(0.6196, 0.6039, 0.7843));
            AddControlPoint(0.6250, eavlColor(0.5020, 0.4902, 0.7294));
            AddControlPoint(0.7500, eavlColor(0.4157, 0.3176, 0.6392));
            AddControlPoint(0.8750, eavlColor(0.3294, 0.1529, 0.5608));
            AddControlPoint(1.0000, eavlColor(0.2471, 0.0000, 0.4902));
        }
        else if (name == "RdBu")
        {
            AddControlPoint(0.0000, eavlColor(0.4039, 0.0000, 0.1216));
            AddControlPoint(0.1000, eavlColor(0.6980, 0.0941, 0.1686));
            AddControlPoint(0.2000, eavlColor(0.8392, 0.3765, 0.3020));
            AddControlPoint(0.3000, eavlColor(0.9569, 0.6471, 0.5098));
            AddControlPoint(0.4000, eavlColor(0.9922, 0.8588, 0.7804));
            AddControlPoint(0.5000, eavlColor(0.9686, 0.9686, 0.9686));
            AddControlPoint(0.6000, eavlColor(0.8196, 0.8980, 0.9412));
            AddControlPoint(0.7000, eavlColor(0.5725, 0.7725, 0.8706));
            AddControlPoint(0.8000, eavlColor(0.2627, 0.5765, 0.7647));
            AddControlPoint(0.9000, eavlColor(0.1294, 0.4000, 0.6745));
            AddControlPoint(1.0000, eavlColor(0.0196, 0.1882, 0.3804));
        }
        else if (name == "RdGy")
        {
            AddControlPoint(0.0000, eavlColor(0.4039, 0.0000, 0.1216));
            AddControlPoint(0.1000, eavlColor(0.6980, 0.0941, 0.1686));
            AddControlPoint(0.2000, eavlColor(0.8392, 0.3765, 0.3020));
            AddControlPoint(0.3000, eavlColor(0.9569, 0.6471, 0.5098));
            AddControlPoint(0.4000, eavlColor(0.9922, 0.8588, 0.7804));
            AddControlPoint(0.5000, eavlColor(1.0000, 1.0000, 1.0000));
            AddControlPoint(0.6000, eavlColor(0.8784, 0.8784, 0.8784));
            AddControlPoint(0.7000, eavlColor(0.7294, 0.7294, 0.7294));
            AddControlPoint(0.8000, eavlColor(0.5294, 0.5294, 0.5294));
            AddControlPoint(0.9000, eavlColor(0.3020, 0.3020, 0.3020));
            AddControlPoint(1.0000, eavlColor(0.1020, 0.1020, 0.1020));
        }
        else if (name == "RdPu")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 0.9686, 0.9529));
            AddControlPoint(0.1250, eavlColor(0.9922, 0.8784, 0.8667));
            AddControlPoint(0.2500, eavlColor(0.9882, 0.7725, 0.7529));
            AddControlPoint(0.3750, eavlColor(0.9804, 0.6235, 0.7098));
            AddControlPoint(0.5000, eavlColor(0.9686, 0.4078, 0.6314));
            AddControlPoint(0.6250, eavlColor(0.8667, 0.2039, 0.5922));
            AddControlPoint(0.7500, eavlColor(0.6824, 0.0039, 0.4941));
            AddControlPoint(0.8750, eavlColor(0.4784, 0.0039, 0.4667));
            AddControlPoint(1.0000, eavlColor(0.2863, 0.0000, 0.4157));
        }
        else if (name == "RdYlBu")
        {
            AddControlPoint(0.0000, eavlColor(0.6471, 0.0000, 0.1490));
            AddControlPoint(0.1000, eavlColor(0.8431, 0.1882, 0.1529));
            AddControlPoint(0.2000, eavlColor(0.9569, 0.4275, 0.2627));
            AddControlPoint(0.3000, eavlColor(0.9922, 0.6824, 0.3804));
            AddControlPoint(0.4000, eavlColor(0.9961, 0.8784, 0.5647));
            AddControlPoint(0.5000, eavlColor(1.0000, 1.0000, 0.7490));
            AddControlPoint(0.6000, eavlColor(0.8784, 0.9529, 0.9725));
            AddControlPoint(0.7000, eavlColor(0.6706, 0.8510, 0.9137));
            AddControlPoint(0.8000, eavlColor(0.4549, 0.6784, 0.8196));
            AddControlPoint(0.9000, eavlColor(0.2706, 0.4588, 0.7059));
            AddControlPoint(1.0000, eavlColor(0.1922, 0.2118, 0.5843));
        }
        else if (name == "RdYlGn")
        {
            AddControlPoint(0.0000, eavlColor(0.6471, 0.0000, 0.1490));
            AddControlPoint(0.1000, eavlColor(0.8431, 0.1882, 0.1529));
            AddControlPoint(0.2000, eavlColor(0.9569, 0.4275, 0.2627));
            AddControlPoint(0.3000, eavlColor(0.9922, 0.6824, 0.3804));
            AddControlPoint(0.4000, eavlColor(0.9961, 0.8784, 0.5451));
            AddControlPoint(0.5000, eavlColor(1.0000, 1.0000, 0.7490));
            AddControlPoint(0.6000, eavlColor(0.8510, 0.9373, 0.5451));
            AddControlPoint(0.7000, eavlColor(0.6510, 0.8510, 0.4157));
            AddControlPoint(0.8000, eavlColor(0.4000, 0.7412, 0.3882));
            AddControlPoint(0.9000, eavlColor(0.1020, 0.5961, 0.3137));
            AddControlPoint(1.0000, eavlColor(0.0000, 0.4078, 0.2157));
        }
        else if (name == "Reds")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 0.9608, 0.9412));
            AddControlPoint(0.1250, eavlColor(0.9961, 0.8784, 0.8235));
            AddControlPoint(0.2500, eavlColor(0.9882, 0.7333, 0.6314));
            AddControlPoint(0.3750, eavlColor(0.9882, 0.5725, 0.4471));
            AddControlPoint(0.5000, eavlColor(0.9843, 0.4157, 0.2902));
            AddControlPoint(0.6250, eavlColor(0.9373, 0.2314, 0.1725));
            AddControlPoint(0.7500, eavlColor(0.7961, 0.0941, 0.1137));
            AddControlPoint(0.8750, eavlColor(0.6471, 0.0588, 0.0824));
            AddControlPoint(1.0000, eavlColor(0.4039, 0.0000, 0.0510));
        }
        else if (name == "Set1")
        {
            AddControlPoint(0.0000, eavlColor(0.8941, 0.1020, 0.1098));
            AddControlPoint(0.1250, eavlColor(0.2157, 0.4941, 0.7216));
            AddControlPoint(0.2500, eavlColor(0.3020, 0.6863, 0.2902));
            AddControlPoint(0.3750, eavlColor(0.5961, 0.3059, 0.6392));
            AddControlPoint(0.5000, eavlColor(1.0000, 0.4980, 0.0000));
            AddControlPoint(0.6250, eavlColor(1.0000, 1.0000, 0.2000));
            AddControlPoint(0.7500, eavlColor(0.6510, 0.3373, 0.1569));
            AddControlPoint(0.8750, eavlColor(0.9686, 0.5059, 0.7490));
            AddControlPoint(1.0000, eavlColor(0.6000, 0.6000, 0.6000));
        }
        else if (name == "Set2")
        {
            AddControlPoint(0.0000, eavlColor(0.4000, 0.7608, 0.6471));
            AddControlPoint(0.1429, eavlColor(0.9882, 0.5529, 0.3843));
            AddControlPoint(0.2857, eavlColor(0.5529, 0.6275, 0.7961));
            AddControlPoint(0.4286, eavlColor(0.9059, 0.5412, 0.7647));
            AddControlPoint(0.5714, eavlColor(0.6510, 0.8471, 0.3294));
            AddControlPoint(0.7143, eavlColor(1.0000, 0.8510, 0.1843));
            AddControlPoint(0.8571, eavlColor(0.8980, 0.7686, 0.5804));
            AddControlPoint(1.0000, eavlColor(0.7020, 0.7020, 0.7020));
        }
        else if (name == "Set3")
        {
            AddControlPoint(0.0000, eavlColor(0.5529, 0.8275, 0.7804));
            AddControlPoint(0.0909, eavlColor(1.0000, 1.0000, 0.7020));
            AddControlPoint(0.1818, eavlColor(0.7451, 0.7294, 0.8549));
            AddControlPoint(0.2727, eavlColor(0.9843, 0.5020, 0.4471));
            AddControlPoint(0.3636, eavlColor(0.5020, 0.6941, 0.8275));
            AddControlPoint(0.4545, eavlColor(0.9922, 0.7059, 0.3843));
            AddControlPoint(0.5455, eavlColor(0.7020, 0.8706, 0.4118));
            AddControlPoint(0.6364, eavlColor(0.9882, 0.8039, 0.8980));
            AddControlPoint(0.7273, eavlColor(0.8510, 0.8510, 0.8510));
            AddControlPoint(0.8182, eavlColor(0.7373, 0.5020, 0.7412));
            AddControlPoint(0.9091, eavlColor(0.8000, 0.9216, 0.7725));
            AddControlPoint(1.0000, eavlColor(1.0000, 0.9294, 0.4353));
        }
        else if (name == "Spectral")
        {
            AddControlPoint(0.0000, eavlColor(0.6196, 0.0039, 0.2588));
            AddControlPoint(0.1000, eavlColor(0.8353, 0.2431, 0.3098));
            AddControlPoint(0.2000, eavlColor(0.9569, 0.4275, 0.2627));
            AddControlPoint(0.3000, eavlColor(0.9922, 0.6824, 0.3804));
            AddControlPoint(0.4000, eavlColor(0.9961, 0.8784, 0.5451));
            AddControlPoint(0.5000, eavlColor(1.0000, 1.0000, 0.7490));
            AddControlPoint(0.6000, eavlColor(0.9020, 0.9608, 0.5961));
            AddControlPoint(0.7000, eavlColor(0.6706, 0.8667, 0.6431));
            AddControlPoint(0.8000, eavlColor(0.4000, 0.7608, 0.6471));
            AddControlPoint(0.9000, eavlColor(0.1961, 0.5333, 0.7412));
            AddControlPoint(1.0000, eavlColor(0.3686, 0.3098, 0.6353));
        }
        else if (name == "YlGnBu")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 1.0000, 0.8510));
            AddControlPoint(0.1250, eavlColor(0.9294, 0.9725, 0.6941));
            AddControlPoint(0.2500, eavlColor(0.7804, 0.9137, 0.7059));
            AddControlPoint(0.3750, eavlColor(0.4980, 0.8039, 0.7333));
            AddControlPoint(0.5000, eavlColor(0.2549, 0.7137, 0.7686));
            AddControlPoint(0.6250, eavlColor(0.1137, 0.5686, 0.7529));
            AddControlPoint(0.7500, eavlColor(0.1333, 0.3686, 0.6588));
            AddControlPoint(0.8750, eavlColor(0.1451, 0.2039, 0.5804));
            AddControlPoint(1.0000, eavlColor(0.0314, 0.1137, 0.3451));
        }
        else if (name == "YlGn")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 1.0000, 0.8980));
            AddControlPoint(0.1250, eavlColor(0.9686, 0.9882, 0.7255));
            AddControlPoint(0.2500, eavlColor(0.8510, 0.9412, 0.6392));
            AddControlPoint(0.3750, eavlColor(0.6784, 0.8667, 0.5569));
            AddControlPoint(0.5000, eavlColor(0.4706, 0.7765, 0.4745));
            AddControlPoint(0.6250, eavlColor(0.2549, 0.6706, 0.3647));
            AddControlPoint(0.7500, eavlColor(0.1373, 0.5176, 0.2627));
            AddControlPoint(0.8750, eavlColor(0.0000, 0.4078, 0.2157));
            AddControlPoint(1.0000, eavlColor(0.0000, 0.2706, 0.1608));
        }
        else if (name == "YlOrBr")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 1.0000, 0.8980));
            AddControlPoint(0.1250, eavlColor(1.0000, 0.9686, 0.7373));
            AddControlPoint(0.2500, eavlColor(0.9961, 0.8902, 0.5686));
            AddControlPoint(0.3750, eavlColor(0.9961, 0.7686, 0.3098));
            AddControlPoint(0.5000, eavlColor(0.9961, 0.6000, 0.1608));
            AddControlPoint(0.6250, eavlColor(0.9255, 0.4392, 0.0784));
            AddControlPoint(0.7500, eavlColor(0.8000, 0.2980, 0.0078));
            AddControlPoint(0.8750, eavlColor(0.6000, 0.2039, 0.0157));
            AddControlPoint(1.0000, eavlColor(0.4000, 0.1451, 0.0235));
        }
        else if (name == "YlOrRd")
        {
            AddControlPoint(0.0000, eavlColor(1.0000, 1.0000, 0.8000));
            AddControlPoint(0.1250, eavlColor(1.0000, 0.9294, 0.6275));
            AddControlPoint(0.2500, eavlColor(0.9961, 0.8510, 0.4627));
            AddControlPoint(0.3750, eavlColor(0.9961, 0.6980, 0.2980));
            AddControlPoint(0.5000, eavlColor(0.9922, 0.5529, 0.2353));
            AddControlPoint(0.6250, eavlColor(0.9882, 0.3059, 0.1647));
            AddControlPoint(0.7500, eavlColor(0.8902, 0.1020, 0.1098));
            AddControlPoint(0.8750, eavlColor(0.7412, 0.0000, 0.1490));
            AddControlPoint(1.0000, eavlColor(0.5020, 0.0000, 0.1490));
        }
        else 
            THROW(eavlException, "Unknown color table");

        uniquename = string("00") + name;
        if (smooth)
            uniquename[0] = '1';
    }
};

#endif

