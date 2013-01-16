// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_2D_AXIS_ANNOTATION_H
#define EAVL_2D_AXIS_ANNOTATION_H

#include "eavlUtility.h"

// ****************************************************************************
// Class:  eavl2DAxisAnnotation
//
// Purpose:
///   A 2D axis.
//
// Programmer:  Jeremy Meredith
// Creation:    January 15, 2013
//
// Modifications:
// ****************************************************************************
class eavl2DAxisAnnotation : public eavlScreenSpaceAnnotation
{
  protected:
    double maj_tx, maj_ty, maj_toff;
    double min_tx, min_ty, min_toff;
    double x0, y0, x1, y1;
    double anchorx, anchory;
    double lower, upper;
    double fontscale;
    int    linewidth;
    vector<eavlScreenTextAnnotation*> labels;

    vector<double> maj_positions;
    vector<double> maj_proportions;

    vector<double> min_positions;
    vector<double> min_proportions;
  public:
    eavl2DAxisAnnotation(eavlWindow *win) :
        eavlScreenSpaceAnnotation(win)
    {
        anchorx = anchory = 0;
        fontscale = 0.05;
        linewidth = 1;
    }
    void SetLineWidth(int lw)
    {
        linewidth = lw;
    }
    void SetMajorTickSize(double xlen, double ylen, double offset)
    {
        /// offset of 0 means the tick is inside the frame
        /// offset of 1 means the tick is outside the frame
        /// offset of 0.5 means the tick is centered on the frame
        maj_tx=xlen;
        maj_ty=ylen;
        maj_toff = offset;
    }
    void SetMinorTickSize(double xlen, double ylen, double offset)
    {
        min_tx=xlen;
        min_ty=ylen;
        min_toff = offset;
    }
    void SetScreenPosition(double x0_, double y0_,
                           double x1_, double y1_)
    {
        x0 = x0_;
        y0 = y0_;

        x1 = x1_;
        y1 = y1_;
    }
    void SetLabelAnchor(float ax, float ay)
    {
        anchorx = ax;
        anchory = ay;
    }
    void SetLabelFontScale(float s)
    {
        fontscale = s;
    }
    void SetRangeForAutoTicks(double l, double u)
    {
        lower = l;
        upper = u;

        CalculateTicks(lower, upper, false, maj_positions, maj_proportions);
        CalculateTicks(lower, upper, true,  min_positions, min_proportions);
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
    virtual void Render()
    {
        glDisable(GL_LIGHTING);
        glColor3f(1,1,1);
        if (linewidth > 0)
        {
            glLineWidth(linewidth);
            glBegin(GL_LINES);
            glVertex2d(x0, y0);
            glVertex2d(x1, y1);
            glEnd();
        }

        glLineWidth(1);
        glBegin(GL_LINES);

        vector<double> positions;
        vector<double> proportions;
        // major ticks
        int nmajor = maj_proportions.size();
        while (labels.size() < nmajor)
        {
            labels.push_back(new eavlScreenTextAnnotation(win,"test",
                                                          eavlColor::white,
                                                          fontscale,
                                                          0,0, 0));
        }
        for (int i=0; i<nmajor; ++i)
        {
            float xc = x0 + (x1-x0) * maj_proportions[i];
            float yc = y0 + (y1-y0) * maj_proportions[i];
            float xs = xc - maj_tx*maj_toff;
            float xe = xc + maj_tx*(1. - maj_toff);
            float ys = yc - maj_ty*maj_toff;
            float ye = yc + maj_ty*(1. - maj_toff);

            glVertex2d(xs, ys);
            glVertex2d(xe, ye);

            if (maj_ty == 0)
            {
                xs -= (maj_tx<0?-1.:+1.) * fontscale * .1;
            }

            char val[256];
            snprintf(val, 256, "%g", maj_positions[i]);
            labels[i]->SetText(val);
            if (fabs(maj_positions[i]) < 1e-10)
                labels[i]->SetText("0");
            labels[i]->SetPosition(xs,ys);
            labels[i]->SetAnchor(anchorx,anchory);
        }

        // minor ticks
        if (min_tx != 0 && min_ty != 0)
        {
            int nminor = min_proportions.size();
            for (int i=0; i<nminor; ++i)
            {
                float xc = x0 + (x1-x0) * min_proportions[i];
                float yc = y0 + (y1-y0) * min_proportions[i];
                float xs = xc - min_tx*min_toff;
                float xe = xc + min_tx*(1. - min_toff);
                float ys = yc - min_ty*min_toff;
                float ye = yc + min_ty*(1. - min_toff);

                glVertex2d(xs, ys);
                glVertex2d(xe, ye);
            }
        }

        glEnd();

        for (int i=0; i<nmajor; ++i)
        {
            labels[i]->Setup(win->view);
            labels[i]->Render();
        }

    }    
};


#endif
