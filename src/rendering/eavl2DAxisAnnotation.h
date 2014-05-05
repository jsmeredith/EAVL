// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_2D_AXIS_ANNOTATION_H
#define EAVL_2D_AXIS_ANNOTATION_H

#include "eavlUtility.h"
#include "eavlTextAnnotation.h"

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
class eavl2DAxisAnnotation : public eavlAnnotation
{
  protected:
    double maj_tx, maj_ty, maj_toff;
    double min_tx, min_ty, min_toff;
    double x0, y0, x1, y1;
    eavlTextAnnotation::HorizontalAlignment halign;
    eavlTextAnnotation::VerticalAlignment valign;
    double lower, upper;
    double fontscale;
    int    linewidth;
    eavlColor color;
    bool   logarithmic;
    vector<eavlTextAnnotation*> labels;

    vector<double> maj_positions;
    vector<double> maj_proportions;

    vector<double> min_positions;
    vector<double> min_proportions;
    
    bool worldSpace;

    int moreOrLessTickAdjustment;
  public:
    eavl2DAxisAnnotation(eavlWindow *win) :
        eavlAnnotation(win)
    {
        halign = eavlTextAnnotation::HCenter;
        valign = eavlTextAnnotation::VCenter;
        fontscale = 0.05;
        linewidth = 1;
        color = eavlColor::white;
        logarithmic = false;
        moreOrLessTickAdjustment = 0;
        worldSpace = false;
    }
    void SetLogarithmic(bool l)
    {
        logarithmic = l;
    }
    void SetWorldSpace(bool ws)
    {
        worldSpace = ws;
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
    ///\todo: rename, since it might be screen OR world position?
    void SetScreenPosition(double x0_, double y0_,
                           double x1_, double y1_)
    {
        x0 = x0_;
        y0 = y0_;

        x1 = x1_;
        y1 = y1_;
    }
    void SetLabelAlignment(eavlTextAnnotation::HorizontalAlignment h,
                           eavlTextAnnotation::VerticalAlignment v)
    {
        halign = h;
        valign = v;
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

        if (logarithmic)
        {
            CalculateTicksLogarithmic(lower, upper, false, maj_positions, maj_proportions, moreOrLessTickAdjustment);
            CalculateTicksLogarithmic(lower, upper, true,  min_positions, min_proportions, moreOrLessTickAdjustment);
        }
        else
        {
            CalculateTicks(lower, upper, false, maj_positions, maj_proportions, moreOrLessTickAdjustment);
            CalculateTicks(lower, upper, true,  min_positions, min_proportions, moreOrLessTickAdjustment);
        }
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
        if (worldSpace)
            view.SetupForWorldSpace();
        else
            view.SetupForScreenSpace();

        glDisable(GL_LIGHTING);
        glColor3fv(color.c);
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

        // major ticks
        int nmajor = maj_proportions.size();
        while ((int)labels.size() < nmajor)
        {
            if (worldSpace)
            {
                labels.push_back(new eavlBillboardTextAnnotation(win,"test",
                                                                 color,
                                                                 fontscale,
                                                                 0,0,0, true));
            }
            else
            {
                labels.push_back(new eavlScreenTextAnnotation(win,"test",
                                                              color,
                                                              fontscale,
                                                              0,0, 0));
            }
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
                // slight shift to space between label and tick
                xs -= (maj_tx<0?-1.:+1.) * fontscale * .1;
            }

            char val[256];
            snprintf(val, 256, "%g", maj_positions[i]);
            labels[i]->SetText(val);
            //if (fabs(maj_positions[i]) < 1e-10)
            //    labels[i]->SetText("0");
            if (worldSpace)
                ((eavlBillboardTextAnnotation*)(labels[i]))->SetPosition(xs,ys,0);
            else
                ((eavlScreenTextAnnotation*)(labels[i]))->SetPosition(xs,ys);

            labels[i]->SetAlignment(halign,valign);
        }

        // minor ticks
        if (min_tx != 0 || min_ty != 0)
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
            labels[i]->Render(view);
        }

    }    
};


#endif
