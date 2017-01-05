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
        else if (name == "cool2warm")
        {
            AddControlPoint(0.0f, eavlColor(0.3347f, 0.2830f, 0.7564f));
            AddControlPoint(0.0039f, eavlColor(0.3389f, 0.2901f, 0.7627f));
            AddControlPoint(0.0078f, eavlColor(0.3432f, 0.2972f, 0.7688f));
            AddControlPoint(0.0117f, eavlColor(0.3474f, 0.3043f, 0.7749f));
            AddControlPoint(0.0156f, eavlColor(0.3516f, 0.3113f, 0.7809f));
            AddControlPoint(0.0196f, eavlColor(0.3558f, 0.3183f, 0.7869f));
            AddControlPoint(0.0235f, eavlColor(0.3600f, 0.3253f, 0.7928f));
            AddControlPoint(0.0274f, eavlColor(0.3642f, 0.3323f, 0.7986f));
            AddControlPoint(0.0313f, eavlColor(0.3684f, 0.3392f, 0.8044f));
            AddControlPoint(0.0352f, eavlColor(0.3727f, 0.3462f, 0.8101f));
            AddControlPoint(0.0392f, eavlColor(0.3769f, 0.3531f, 0.8157f));
            AddControlPoint(0.0431f, eavlColor(0.3811f, 0.3600f, 0.8213f));
            AddControlPoint(0.0470f, eavlColor(0.3853f, 0.3669f, 0.8268f));
            AddControlPoint(0.0509f, eavlColor(0.3896f, 0.3738f, 0.8322f));
            AddControlPoint(0.0549f, eavlColor(0.3938f, 0.3806f, 0.8375f));
            AddControlPoint(0.0588f, eavlColor(0.3980f, 0.3874f, 0.8428f));
            AddControlPoint(0.0627f, eavlColor(0.4023f, 0.3942f, 0.8480f));
            AddControlPoint(0.0666f, eavlColor(0.4065f, 0.4010f, 0.8531f));
            AddControlPoint(0.0705f, eavlColor(0.4108f, 0.4078f, 0.8582f));
            AddControlPoint(0.0745f, eavlColor(0.4151f, 0.4145f, 0.8632f));
            AddControlPoint(0.0784f, eavlColor(0.4193f, 0.4212f, 0.8680f));
            AddControlPoint(0.0823f, eavlColor(0.4236f, 0.4279f, 0.8729f));
            AddControlPoint(0.0862f, eavlColor(0.4279f, 0.4346f, 0.8776f));
            AddControlPoint(0.0901f, eavlColor(0.4321f, 0.4412f, 0.8823f));
            AddControlPoint(0.0941f, eavlColor(0.4364f, 0.4479f, 0.8868f));
            AddControlPoint(0.0980f, eavlColor(0.4407f, 0.4544f, 0.8913f));
            AddControlPoint(0.1019f, eavlColor(0.4450f, 0.4610f, 0.8957f));
            AddControlPoint(0.1058f, eavlColor(0.4493f, 0.4675f, 0.9001f));
            AddControlPoint(0.1098f, eavlColor(0.4536f, 0.4741f, 0.9043f));
            AddControlPoint(0.1137f, eavlColor(0.4579f, 0.4805f, 0.9085f));
            AddControlPoint(0.1176f, eavlColor(0.4622f, 0.4870f, 0.9126f));
            AddControlPoint(0.1215f, eavlColor(0.4666f, 0.4934f, 0.9166f));
            AddControlPoint(0.1254f, eavlColor(0.4709f, 0.4998f, 0.9205f));
            AddControlPoint(0.1294f, eavlColor(0.4752f, 0.5061f, 0.9243f));
            AddControlPoint(0.1333f, eavlColor(0.4796f, 0.5125f, 0.9280f));
            AddControlPoint(0.1372f, eavlColor(0.4839f, 0.5188f, 0.9317f));
            AddControlPoint(0.1411f, eavlColor(0.4883f, 0.5250f, 0.9352f));
            AddControlPoint(0.1450f, eavlColor(0.4926f, 0.5312f, 0.9387f));
            AddControlPoint(0.1490f, eavlColor(0.4970f, 0.5374f, 0.9421f));
            AddControlPoint(0.1529f, eavlColor(0.5013f, 0.5436f, 0.9454f));
            AddControlPoint(0.1568f, eavlColor(0.5057f, 0.5497f, 0.9486f));
            AddControlPoint(0.1607f, eavlColor(0.5101f, 0.5558f, 0.9517f));
            AddControlPoint(0.1647f, eavlColor(0.5145f, 0.5618f, 0.9547f));
            AddControlPoint(0.1686f, eavlColor(0.5188f, 0.5678f, 0.9577f));
            AddControlPoint(0.1725f, eavlColor(0.5232f, 0.5738f, 0.9605f));
            AddControlPoint(0.1764f, eavlColor(0.5276f, 0.5797f, 0.9633f));
            AddControlPoint(0.1803f, eavlColor(0.5320f, 0.5856f, 0.9659f));
            AddControlPoint(0.1843f, eavlColor(0.5364f, 0.5915f, 0.9685f));
            AddControlPoint(0.1882f, eavlColor(0.5408f, 0.5973f, 0.9710f));
            AddControlPoint(0.1921f, eavlColor(0.5452f, 0.6030f, 0.9733f));
            AddControlPoint(0.1960f, eavlColor(0.5497f, 0.6087f, 0.9756f));
            AddControlPoint(0.2f, eavlColor(0.5541f, 0.6144f, 0.9778f));
            AddControlPoint(0.2039f, eavlColor(0.5585f, 0.6200f, 0.9799f));
            AddControlPoint(0.2078f, eavlColor(0.5629f, 0.6256f, 0.9819f));
            AddControlPoint(0.2117f, eavlColor(0.5673f, 0.6311f, 0.9838f));
            AddControlPoint(0.2156f, eavlColor(0.5718f, 0.6366f, 0.9856f));
            AddControlPoint(0.2196f, eavlColor(0.5762f, 0.6420f, 0.9873f));
            AddControlPoint(0.2235f, eavlColor(0.5806f, 0.6474f, 0.9890f));
            AddControlPoint(0.2274f, eavlColor(0.5850f, 0.6528f, 0.9905f));
            AddControlPoint(0.2313f, eavlColor(0.5895f, 0.6580f, 0.9919f));
            AddControlPoint(0.2352f, eavlColor(0.5939f, 0.6633f, 0.9932f));
            AddControlPoint(0.2392f, eavlColor(0.5983f, 0.6685f, 0.9945f));
            AddControlPoint(0.2431f, eavlColor(0.6028f, 0.6736f, 0.9956f));
            AddControlPoint(0.2470f, eavlColor(0.6072f, 0.6787f, 0.9967f));
            AddControlPoint(0.2509f, eavlColor(0.6116f, 0.6837f, 0.9976f));
            AddControlPoint(0.2549f, eavlColor(0.6160f, 0.6887f, 0.9985f));
            AddControlPoint(0.2588f, eavlColor(0.6205f, 0.6936f, 0.9992f));
            AddControlPoint(0.2627f, eavlColor(0.6249f, 0.6984f, 0.9999f));
            AddControlPoint(0.2666f, eavlColor(0.6293f, 0.7032f, 1.0004f));
            AddControlPoint(0.2705f, eavlColor(0.6337f, 0.7080f, 1.0009f));
            AddControlPoint(0.2745f, eavlColor(0.6381f, 0.7127f, 1.0012f));
            AddControlPoint(0.2784f, eavlColor(0.6425f, 0.7173f, 1.0015f));
            AddControlPoint(0.2823f, eavlColor(0.6469f, 0.7219f, 1.0017f));
            AddControlPoint(0.2862f, eavlColor(0.6513f, 0.7264f, 1.0017f));
            AddControlPoint(0.2901f, eavlColor(0.6557f, 0.7308f, 1.0017f));
            AddControlPoint(0.2941f, eavlColor(0.6601f, 0.7352f, 1.0016f));
            AddControlPoint(0.2980f, eavlColor(0.6645f, 0.7395f, 1.0014f));
            AddControlPoint(0.3019f, eavlColor(0.6688f, 0.7438f, 1.0010f));
            AddControlPoint(0.3058f, eavlColor(0.6732f, 0.7480f, 1.0006f));
            AddControlPoint(0.3098f, eavlColor(0.6775f, 0.7521f, 1.0001f));
            AddControlPoint(0.3137f, eavlColor(0.6819f, 0.7562f, 0.9995f));
            AddControlPoint(0.3176f, eavlColor(0.6862f, 0.7602f, 0.9988f));
            AddControlPoint(0.3215f, eavlColor(0.6905f, 0.7641f, 0.9980f));
            AddControlPoint(0.3254f, eavlColor(0.6948f, 0.7680f, 0.9971f));
            AddControlPoint(0.3294f, eavlColor(0.6991f, 0.7718f, 0.9961f));
            AddControlPoint(0.3333f, eavlColor(0.7034f, 0.7755f, 0.9950f));
            AddControlPoint(0.3372f, eavlColor(0.7077f, 0.7792f, 0.9939f));
            AddControlPoint(0.3411f, eavlColor(0.7119f, 0.7828f, 0.9926f));
            AddControlPoint(0.3450f, eavlColor(0.7162f, 0.7864f, 0.9912f));
            AddControlPoint(0.3490f, eavlColor(0.7204f, 0.7898f, 0.9897f));
            AddControlPoint(0.3529f, eavlColor(0.7246f, 0.7932f, 0.9882f));
            AddControlPoint(0.3568f, eavlColor(0.7288f, 0.7965f, 0.9865f));
            AddControlPoint(0.3607f, eavlColor(0.7330f, 0.7998f, 0.9848f));
            AddControlPoint(0.3647f, eavlColor(0.7372f, 0.8030f, 0.9829f));
            AddControlPoint(0.3686f, eavlColor(0.7413f, 0.8061f, 0.9810f));
            AddControlPoint(0.3725f, eavlColor(0.7455f, 0.8091f, 0.9789f));
            AddControlPoint(0.3764f, eavlColor(0.7496f, 0.8121f, 0.9768f));
            AddControlPoint(0.3803f, eavlColor(0.7537f, 0.8150f, 0.9746f));
            AddControlPoint(0.3843f, eavlColor(0.7577f, 0.8178f, 0.9723f));
            AddControlPoint(0.3882f, eavlColor(0.7618f, 0.8205f, 0.9699f));
            AddControlPoint(0.3921f, eavlColor(0.7658f, 0.8232f, 0.9674f));
            AddControlPoint(0.3960f, eavlColor(0.7698f, 0.8258f, 0.9648f));
            AddControlPoint(0.4f, eavlColor(0.7738f, 0.8283f, 0.9622f));
            AddControlPoint(0.4039f, eavlColor(0.7777f, 0.8307f, 0.9594f));
            AddControlPoint(0.4078f, eavlColor(0.7817f, 0.8331f, 0.9566f));
            AddControlPoint(0.4117f, eavlColor(0.7856f, 0.8353f, 0.9536f));
            AddControlPoint(0.4156f, eavlColor(0.7895f, 0.8375f, 0.9506f));
            AddControlPoint(0.4196f, eavlColor(0.7933f, 0.8397f, 0.9475f));
            AddControlPoint(0.4235f, eavlColor(0.7971f, 0.8417f, 0.9443f));
            AddControlPoint(0.4274f, eavlColor(0.8009f, 0.8437f, 0.9410f));
            AddControlPoint(0.4313f, eavlColor(0.8047f, 0.8456f, 0.9376f));
            AddControlPoint(0.4352f, eavlColor(0.8085f, 0.8474f, 0.9342f));
            AddControlPoint(0.4392f, eavlColor(0.8122f, 0.8491f, 0.9306f));
            AddControlPoint(0.4431f, eavlColor(0.8159f, 0.8507f, 0.9270f));
            AddControlPoint(0.4470f, eavlColor(0.8195f, 0.8523f, 0.9233f));
            AddControlPoint(0.4509f, eavlColor(0.8231f, 0.8538f, 0.9195f));
            AddControlPoint(0.4549f, eavlColor(0.8267f, 0.8552f, 0.9156f));
            AddControlPoint(0.4588f, eavlColor(0.8303f, 0.8565f, 0.9117f));
            AddControlPoint(0.4627f, eavlColor(0.8338f, 0.8577f, 0.9076f));
            AddControlPoint(0.4666f, eavlColor(0.8373f, 0.8589f, 0.9035f));
            AddControlPoint(0.4705f, eavlColor(0.8407f, 0.8600f, 0.8993f));
            AddControlPoint(0.4745f, eavlColor(0.8441f, 0.8610f, 0.8950f));
            AddControlPoint(0.4784f, eavlColor(0.8475f, 0.8619f, 0.8906f));
            AddControlPoint(0.4823f, eavlColor(0.8508f, 0.8627f, 0.8862f));
            AddControlPoint(0.4862f, eavlColor(0.8541f, 0.8634f, 0.8817f));
            AddControlPoint(0.4901f, eavlColor(0.8574f, 0.8641f, 0.8771f));
            AddControlPoint(0.4941f, eavlColor(0.8606f, 0.8647f, 0.8724f));
            AddControlPoint(0.4980f, eavlColor(0.8638f, 0.8651f, 0.8677f));
            AddControlPoint(0.5019f, eavlColor(0.8673f, 0.8645f, 0.8626f));
            AddControlPoint(0.5058f, eavlColor(0.8710f, 0.8627f, 0.8571f));
            AddControlPoint(0.5098f, eavlColor(0.8747f, 0.8609f, 0.8515f));
            AddControlPoint(0.5137f, eavlColor(0.8783f, 0.8589f, 0.8459f));
            AddControlPoint(0.5176f, eavlColor(0.8818f, 0.8569f, 0.8403f));
            AddControlPoint(0.5215f, eavlColor(0.8852f, 0.8548f, 0.8347f));
            AddControlPoint(0.5254f, eavlColor(0.8885f, 0.8526f, 0.8290f));
            AddControlPoint(0.5294f, eavlColor(0.8918f, 0.8504f, 0.8233f));
            AddControlPoint(0.5333f, eavlColor(0.8949f, 0.8480f, 0.8176f));
            AddControlPoint(0.5372f, eavlColor(0.8980f, 0.8456f, 0.8119f));
            AddControlPoint(0.5411f, eavlColor(0.9010f, 0.8431f, 0.8061f));
            AddControlPoint(0.5450f, eavlColor(0.9040f, 0.8405f, 0.8003f));
            AddControlPoint(0.5490f, eavlColor(0.9068f, 0.8378f, 0.7944f));
            AddControlPoint(0.5529f, eavlColor(0.9096f, 0.8351f, 0.7886f));
            AddControlPoint(0.5568f, eavlColor(0.9123f, 0.8322f, 0.7827f));
            AddControlPoint(0.5607f, eavlColor(0.9149f, 0.8293f, 0.7768f));
            AddControlPoint(0.5647f, eavlColor(0.9174f, 0.8263f, 0.7709f));
            AddControlPoint(0.5686f, eavlColor(0.9198f, 0.8233f, 0.7649f));
            AddControlPoint(0.5725f, eavlColor(0.9222f, 0.8201f, 0.7590f));
            AddControlPoint(0.5764f, eavlColor(0.9245f, 0.8169f, 0.7530f));
            AddControlPoint(0.5803f, eavlColor(0.9266f, 0.8136f, 0.7470f));
            AddControlPoint(0.5843f, eavlColor(0.9288f, 0.8103f, 0.7410f));
            AddControlPoint(0.5882f, eavlColor(0.9308f, 0.8068f, 0.7349f));
            AddControlPoint(0.5921f, eavlColor(0.9327f, 0.8033f, 0.7289f));
            AddControlPoint(0.5960f, eavlColor(0.9346f, 0.7997f, 0.7228f));
            AddControlPoint(0.6f, eavlColor(0.9363f, 0.7960f, 0.7167f));
            AddControlPoint(0.6039f, eavlColor(0.9380f, 0.7923f, 0.7106f));
            AddControlPoint(0.6078f, eavlColor(0.9396f, 0.7884f, 0.7045f));
            AddControlPoint(0.6117f, eavlColor(0.9412f, 0.7845f, 0.6984f));
            AddControlPoint(0.6156f, eavlColor(0.9426f, 0.7806f, 0.6923f));
            AddControlPoint(0.6196f, eavlColor(0.9439f, 0.7765f, 0.6861f));
            AddControlPoint(0.6235f, eavlColor(0.9452f, 0.7724f, 0.6800f));
            AddControlPoint(0.6274f, eavlColor(0.9464f, 0.7682f, 0.6738f));
            AddControlPoint(0.6313f, eavlColor(0.9475f, 0.7640f, 0.6677f));
            AddControlPoint(0.6352f, eavlColor(0.9485f, 0.7596f, 0.6615f));
            AddControlPoint(0.6392f, eavlColor(0.9495f, 0.7552f, 0.6553f));
            AddControlPoint(0.6431f, eavlColor(0.9503f, 0.7508f, 0.6491f));
            AddControlPoint(0.6470f, eavlColor(0.9511f, 0.7462f, 0.6429f));
            AddControlPoint(0.6509f, eavlColor(0.9517f, 0.7416f, 0.6368f));
            AddControlPoint(0.6549f, eavlColor(0.9523f, 0.7369f, 0.6306f));
            AddControlPoint(0.6588f, eavlColor(0.9529f, 0.7322f, 0.6244f));
            AddControlPoint(0.6627f, eavlColor(0.9533f, 0.7274f, 0.6182f));
            AddControlPoint(0.6666f, eavlColor(0.9536f, 0.7225f, 0.6120f));
            AddControlPoint(0.6705f, eavlColor(0.9539f, 0.7176f, 0.6058f));
            AddControlPoint(0.6745f, eavlColor(0.9541f, 0.7126f, 0.5996f));
            AddControlPoint(0.6784f, eavlColor(0.9542f, 0.7075f, 0.5934f));
            AddControlPoint(0.6823f, eavlColor(0.9542f, 0.7023f, 0.5873f));
            AddControlPoint(0.6862f, eavlColor(0.9541f, 0.6971f, 0.5811f));
            AddControlPoint(0.6901f, eavlColor(0.9539f, 0.6919f, 0.5749f));
            AddControlPoint(0.6941f, eavlColor(0.9537f, 0.6865f, 0.5687f));
            AddControlPoint(0.6980f, eavlColor(0.9534f, 0.6811f, 0.5626f));
            AddControlPoint(0.7019f, eavlColor(0.9529f, 0.6757f, 0.5564f));
            AddControlPoint(0.7058f, eavlColor(0.9524f, 0.6702f, 0.5503f));
            AddControlPoint(0.7098f, eavlColor(0.9519f, 0.6646f, 0.5441f));
            AddControlPoint(0.7137f, eavlColor(0.9512f, 0.6589f, 0.5380f));
            AddControlPoint(0.7176f, eavlColor(0.9505f, 0.6532f, 0.5319f));
            AddControlPoint(0.7215f, eavlColor(0.9496f, 0.6475f, 0.5258f));
            AddControlPoint(0.7254f, eavlColor(0.9487f, 0.6416f, 0.5197f));
            AddControlPoint(0.7294f, eavlColor(0.9477f, 0.6358f, 0.5136f));
            AddControlPoint(0.7333f, eavlColor(0.9466f, 0.6298f, 0.5075f));
            AddControlPoint(0.7372f, eavlColor(0.9455f, 0.6238f, 0.5015f));
            AddControlPoint(0.7411f, eavlColor(0.9442f, 0.6178f, 0.4954f));
            AddControlPoint(0.7450f, eavlColor(0.9429f, 0.6117f, 0.4894f));
            AddControlPoint(0.7490f, eavlColor(0.9415f, 0.6055f, 0.4834f));
            AddControlPoint(0.7529f, eavlColor(0.9400f, 0.5993f, 0.4774f));
            AddControlPoint(0.7568f, eavlColor(0.9384f, 0.5930f, 0.4714f));
            AddControlPoint(0.7607f, eavlColor(0.9368f, 0.5866f, 0.4654f));
            AddControlPoint(0.7647f, eavlColor(0.9350f, 0.5802f, 0.4595f));
            AddControlPoint(0.7686f, eavlColor(0.9332f, 0.5738f, 0.4536f));
            AddControlPoint(0.7725f, eavlColor(0.9313f, 0.5673f, 0.4477f));
            AddControlPoint(0.7764f, eavlColor(0.9293f, 0.5607f, 0.4418f));
            AddControlPoint(0.7803f, eavlColor(0.9273f, 0.5541f, 0.4359f));
            AddControlPoint(0.7843f, eavlColor(0.9251f, 0.5475f, 0.4300f));
            AddControlPoint(0.7882f, eavlColor(0.9229f, 0.5407f, 0.4242f));
            AddControlPoint(0.7921f, eavlColor(0.9206f, 0.5340f, 0.4184f));
            AddControlPoint(0.7960f, eavlColor(0.9182f, 0.5271f, 0.4126f));
            AddControlPoint(0.8f, eavlColor(0.9158f, 0.5203f, 0.4069f));
            AddControlPoint(0.8039f, eavlColor(0.9132f, 0.5133f, 0.4011f));
            AddControlPoint(0.8078f, eavlColor(0.9106f, 0.5063f, 0.3954f));
            AddControlPoint(0.8117f, eavlColor(0.9079f, 0.4993f, 0.3897f));
            AddControlPoint(0.8156f, eavlColor(0.9052f, 0.4922f, 0.3841f));
            AddControlPoint(0.8196f, eavlColor(0.9023f, 0.4851f, 0.3784f));
            AddControlPoint(0.8235f, eavlColor(0.8994f, 0.4779f, 0.3728f));
            AddControlPoint(0.8274f, eavlColor(0.8964f, 0.4706f, 0.3672f));
            AddControlPoint(0.8313f, eavlColor(0.8933f, 0.4633f, 0.3617f));
            AddControlPoint(0.8352f, eavlColor(0.8901f, 0.4559f, 0.3561f));
            AddControlPoint(0.8392f, eavlColor(0.8869f, 0.4485f, 0.3506f));
            AddControlPoint(0.8431f, eavlColor(0.8836f, 0.4410f, 0.3452f));
            AddControlPoint(0.8470f, eavlColor(0.8802f, 0.4335f, 0.3397f));
            AddControlPoint(0.8509f, eavlColor(0.8767f, 0.4259f, 0.3343f));
            AddControlPoint(0.8549f, eavlColor(0.8732f, 0.4183f, 0.3289f));
            AddControlPoint(0.8588f, eavlColor(0.8696f, 0.4106f, 0.3236f));
            AddControlPoint(0.8627f, eavlColor(0.8659f, 0.4028f, 0.3183f));
            AddControlPoint(0.8666f, eavlColor(0.8622f, 0.3950f, 0.3130f));
            AddControlPoint(0.8705f, eavlColor(0.8583f, 0.3871f, 0.3077f));
            AddControlPoint(0.8745f, eavlColor(0.8544f, 0.3792f, 0.3025f));
            AddControlPoint(0.8784f, eavlColor(0.8505f, 0.3712f, 0.2973f));
            AddControlPoint(0.8823f, eavlColor(0.8464f, 0.3631f, 0.2921f));
            AddControlPoint(0.8862f, eavlColor(0.8423f, 0.3549f, 0.2870f));
            AddControlPoint(0.8901f, eavlColor(0.8381f, 0.3467f, 0.2819f));
            AddControlPoint(0.8941f, eavlColor(0.8339f, 0.3384f, 0.2768f));
            AddControlPoint(0.8980f, eavlColor(0.8295f, 0.3300f, 0.2718f));
            AddControlPoint(0.9019f, eavlColor(0.8251f, 0.3215f, 0.2668f));
            AddControlPoint(0.9058f, eavlColor(0.8207f, 0.3129f, 0.2619f));
            AddControlPoint(0.9098f, eavlColor(0.8162f, 0.3043f, 0.2570f));
            AddControlPoint(0.9137f, eavlColor(0.8116f, 0.2955f, 0.2521f));
            AddControlPoint(0.9176f, eavlColor(0.8069f, 0.2866f, 0.2472f));
            AddControlPoint(0.9215f, eavlColor(0.8022f, 0.2776f, 0.2424f));
            AddControlPoint(0.9254f, eavlColor(0.7974f, 0.2685f, 0.2377f));
            AddControlPoint(0.9294f, eavlColor(0.7925f, 0.2592f, 0.2329f));
            AddControlPoint(0.9333f, eavlColor(0.7876f, 0.2498f, 0.2282f));
            AddControlPoint(0.9372f, eavlColor(0.7826f, 0.2402f, 0.2236f));
            AddControlPoint(0.9411f, eavlColor(0.7775f, 0.2304f, 0.2190f));
            AddControlPoint(0.9450f, eavlColor(0.7724f, 0.2204f, 0.2144f));
            AddControlPoint(0.9490f, eavlColor(0.7672f, 0.2102f, 0.2098f));
            AddControlPoint(0.9529f, eavlColor(0.7620f, 0.1997f, 0.2053f));
            AddControlPoint(0.9568f, eavlColor(0.7567f, 0.1889f, 0.2009f));
            AddControlPoint(0.9607f, eavlColor(0.7514f, 0.1777f, 0.1965f));
            AddControlPoint(0.9647f, eavlColor(0.7459f, 0.1662f, 0.1921f));
            AddControlPoint(0.9686f, eavlColor(0.7405f, 0.1541f, 0.1877f));
            AddControlPoint(0.9725f, eavlColor(0.7349f, 0.1414f, 0.1834f));
            AddControlPoint(0.9764f, eavlColor(0.7293f, 0.1279f, 0.1792f));
            AddControlPoint(0.9803f, eavlColor(0.7237f, 0.1134f, 0.1750f));
            AddControlPoint(0.9843f, eavlColor(0.7180f, 0.0975f, 0.1708f));
            AddControlPoint(0.9882f, eavlColor(0.7122f, 0.0796f, 0.1667f));
            AddControlPoint(0.9921f, eavlColor(0.7064f, 0.0585f, 0.1626f));
            AddControlPoint(0.9960f, eavlColor(0.7005f, 0.0315f, 0.1585f));
            AddControlPoint(1.0f, eavlColor(0.6946f, 0.0029f, 0.1545f));
        }
        else 
            THROW(eavlException, "Unknown color table");

        uniquename = string("00") + name;
        if (smooth)
            uniquename[0] = '1';
    }
};

#endif

