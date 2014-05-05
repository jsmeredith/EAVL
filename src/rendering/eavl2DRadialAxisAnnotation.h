// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_2D_RADIAL_AXIS_ANNOTATION_H
#define EAVL_2D_RADIAL_AXIS_ANNOTATION_H

#include "eavlUtility.h"

// ****************************************************************************
// Class:  eavl2DRadialAxisAnnotation
//
// Purpose:
///   A 2D axis.
//
// Programmer:  Jeremy Meredith
// Creation:    March 20, 2013
//
// Modifications:
// ****************************************************************************
class eavl2DRadialAxisAnnotation : public eavlAnnotation
{
  protected:
    double maj_ts, maj_toff;
    double min_ts, min_toff;
    double xorigin, yorigin, radius;
    int    nsegments;
    double lower, upper;
    double fontscale;
    int    linewidth;
    eavlColor color;
    vector<eavlBillboardTextAnnotation*> labels;

    vector<double> maj_positions;
    vector<double> maj_proportions;

    vector<double> min_positions;
    vector<double> min_proportions;

    int moreOrLessTickAdjustment;
  public:
    eavl2DRadialAxisAnnotation(eavlWindow *win) :
        eavlAnnotation(win)
    {
        fontscale = 0.05;
        linewidth = 1;
        color = eavlColor::white;
        moreOrLessTickAdjustment = 0;
        nsegments = 1000;
    }
    void SetMoreOrLessTickAdjustment(int offset)
    {
        moreOrLessTickAdjustment = offset;
    }
    void SetColor(eavlColor c)
    {
        color = c;
    }
    void SetLineWidth(int lw)
    {
        linewidth = lw;
    }
    void SetMajorTickSize(double len, double offset)
    {
        /// offset of 0 means the tick is inside the frame
        /// offset of 1 means the tick is outside the frame
        /// offset of 0.5 means the tick is centered on the frame
        maj_ts=len;
        maj_toff = offset;
    }
    void SetMinorTickSize(double len, double offset)
    {
        min_ts=len;
        min_toff = offset;
    }
    void SetScreenPosition(double xo, double yo, double rad)
    {
        xorigin = xo;
        yorigin = yo;
        radius  = rad;
    }
    void SetLabelFontScale(float s)
    {
        fontscale = s;
        for (unsigned int i=0; i<labels.size(); i++)
            labels[i]->SetScale(s);
    }
    void SetRangeForAutoTicks(double l, double u)
    {
        lower = l;
        upper = u;

        CalculateTicks(lower, upper, false, maj_positions, maj_proportions, moreOrLessTickAdjustment);
        CalculateTicks(lower, upper, true,  min_positions, min_proportions, moreOrLessTickAdjustment);
    }
    void SetMajorTicks(const vector<double> &pos, const vector<double> &prop)
    {
        maj_positions.clear();
        maj_positions.insert(maj_positions.begin(), pos.begin(), pos.end());

        maj_proportions.clear();
        maj_proportions.insert(maj_proportions.begin(), prop.begin(), prop.end());
    }
    void SetMinorTicks(const vector<double> &pos, const vector<double> &prop)
    {
        min_positions.clear();
        min_positions.insert(min_positions.begin(), pos.begin(), pos.end());

        min_proportions.clear();
        min_proportions.insert(min_proportions.begin(), prop.begin(), prop.end());
    }
    virtual void Render(eavlView &view)
    {
        view.SetupForWorldSpace();

        glDisable(GL_LIGHTING);
        glColor3fv(color.c);
        if (linewidth > 0)
        {
            glLineWidth(linewidth);
            glBegin(GL_LINES);
            for (int i=0; i<nsegments; i++)
            {
                double a0 = double(i)/double(nsegments);
                double a1 = double(i+1)/double(nsegments);
                double q0 = lower + (upper-lower)*a0;
                double q1 = lower + (upper-lower)*a1;
                double x0 = xorigin + radius * cos(q0*M_PI/180.);
                double x1 = xorigin + radius * cos(q1*M_PI/180.);
                double y0 = yorigin + radius * sin(q0*M_PI/180.);
                double y1 = yorigin + radius * sin(q1*M_PI/180.);
                glVertex2d(x0, y0);
                glVertex2d(x1, y1);
            }
            glEnd();
        }

        glLineWidth(1);
        glBegin(GL_LINES);
        // major ticks
        int nmajor = maj_proportions.size();
        while ((int)labels.size() < nmajor)
        {
            labels.push_back(new eavlBillboardTextAnnotation(win,"test",
                                                             color,
                                                             fontscale,
                                                             0,0,0, true));
        }
        for (int i=0; i<nmajor; ++i)
        {
            double p = maj_positions[i];

            double qrad = p*M_PI/180.;
            
            double dx = cos(qrad);
            double dy = sin(qrad);

            double xc = xorigin + radius * dx;
            double yc = yorigin + radius * dy;

            float xs = xc - radius*dx*maj_ts*maj_toff;
            float xe = xc + radius*dx*maj_ts*(1. - maj_toff);
            float ys = yc - radius*dy*maj_ts*maj_toff;
            float ye = yc + radius*dy*maj_ts*(1. - maj_toff);

            glVertex2d(xs, ys);
            glVertex2d(xe, ye);

            // a little extra space for the label
            xe += radius*dx*fontscale*.2;
            //ye += dy*fontscale*.1;

            char val[256];
            snprintf(val, 256, "%g", maj_positions[i]);
            labels[i]->SetText(val);
            if (fabs(maj_positions[i]) < 1e-10)
                labels[i]->SetText("0");
            labels[i]->SetPosition(xe,ye, 0);
            /*
              // NOTE: THIS COMMENTED-OUT CODE IS OLD
              // AND USED A [0,1] ANCHOR RANGE INSTEAD
              // OF THE NEW [-1,1] ANCHOR RANGE.
            double anchorx = .5;
            double anchory = .5;
            if (p < 60 || p > 300)
            {
                double pp = p > 300 & p-360 : p;
                anchorx = 0;
            }
            if (p > 120 && p < 240)
                anchorx = 1;
            if (p >= 30 && p <= 150)
                anchory = 0;
            if (p >= 210 && p <= 330)
                anchory = 1;
                */
            double dxsign = dx<0 ? -1 : +1;
            double dxmag  = fabs(dx);
            double newdx  = dxsign * sqrt(dxmag);
            labels[i]->SetRawAnchor(-newdx, -dy);
        }

        // minor ticks
        if (min_ts != 0)
        {
            int nminor = min_proportions.size();
            for (int i=0; i<nminor; ++i)
            {
                double p = min_positions[i];

                double qrad = p*M_PI/180.;
            
                double dx = cos(qrad);
                double dy = sin(qrad);

                double xc = xorigin + radius * dx;
                double yc = yorigin + radius * dy;

                float xs = xc - radius*dx*min_ts*min_toff;
                float xe = xc + radius*dx*min_ts*(1. - min_toff);
                float ys = yc - radius*dy*min_ts*min_toff;
                float ye = yc + radius*dy*min_ts*(1. - min_toff);

                glVertex2d(xs, ys);
                glVertex2d(xe, ye);
            }
        }
        glEnd();

        for (int i=0; i<nmajor; ++i)
        {
            labels[i]->Render(view);
        }

    }    
};


#endif
