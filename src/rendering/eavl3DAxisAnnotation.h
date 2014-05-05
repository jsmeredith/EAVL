// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_3D_AXIS_ANNOTATION_H
#define EAVL_3D_AXIS_ANNOTATION_H

#include "eavlUtility.h"

// ****************************************************************************
// Class:  eavl3DAxisAnnotation
//
// Purpose:
///   A 3D axis.
//
// Programmer:  Jeremy Meredith
// Creation:    January 15, 2013
//
// Modifications:
// ****************************************************************************
class eavl3DAxisAnnotation : public eavlAnnotation
{
  protected:
    double maj_size, maj_toff;
    double min_size, min_toff;
    int    axis;
    float invertx, inverty, invertz;
    double x0, y0, z0,   x1, y1, z1;
    double lower, upper;
    double fontscale;
    eavlColor color;
    vector<eavlBillboardTextAnnotation*> labels;
    int moreOrLessTickAdjustment;
  public:
    eavl3DAxisAnnotation(eavlWindow *win) :
        eavlAnnotation(win)
    {
        axis = 0;
        color = eavlColor::white;
    }
    void SetMoreOrLessTickAdjustment(int offset)
    {
        moreOrLessTickAdjustment = offset;
    }
    void SetColor(eavlColor c)
    {
        color = c;
    }
    void SetAxis(int a)
    {
        axis = a;
    }
    void SetTickInvert(bool x, bool y, bool z)
    {
        invertx = x ? +1 : -1;
        inverty = y ? +1 : -1;
        invertz = z ? +1 : -1;
    }
    void SetMajorTickSize(double size, double offset)
    {
        /// offset of 0 means the tick is inside the frame
        /// offset of 1 means the tick is outside the frame
        /// offset of 0.5 means the tick is centered on the frame
        maj_size = size;
        maj_toff = offset;
    }
    void SetMinorTickSize(double size, double offset)
    {
        min_size = size;
        min_toff = offset;
    }
    void SetWorldPosition(double x0_, double y0_, double z0_,
                          double x1_, double y1_, double z1_)
    {
        x0 = x0_;
        y0 = y0_;
        z0 = z0_;

        x1 = x1_;
        y1 = y1_;
        z1 = z1_;
    }
    void SetLabelFontScale(float s)
    {
        fontscale = s;
        for (unsigned int i=0; i<labels.size(); i++)
            labels[i]->SetScale(s);
    }
    void SetRange(double l, double u)
    {
        lower = l;
        upper = u;
    }
    virtual void Render(eavlView &view)
    {
        view.SetupForWorldSpace();

        glDisable(GL_LIGHTING);
        glLineWidth(1);
        glColor3fv(color.c);
        glBegin(GL_LINES);
        glVertex3d(x0, y0, z0);
        glVertex3d(x1, y1, z1);

        vector<double> positions;
        vector<double> proportions;
        // major ticks
        CalculateTicks(lower, upper, false, positions, proportions, moreOrLessTickAdjustment);
        int nmajor = proportions.size();
        while ((int)labels.size() < nmajor)
        {
            labels.push_back(new eavlBillboardTextAnnotation(win,"test",
                                                          color,
                                                          fontscale,
                                                          0,0,0, false, 0));
        }
        for (int i=0; i<nmajor; ++i)
        {
            float xc = x0 + (x1-x0) * proportions[i];
            float yc = y0 + (y1-y0) * proportions[i];
            float zc = z0 + (z1-z0) * proportions[i];
            for (int pass=0; pass<=1; pass++)
            {
                float tx=0, ty=0, tz=0;
                switch (axis)
                {
                  case 0: if (pass==0) ty=maj_size; else tz=maj_size; break;
                  case 1: if (pass==0) tx=maj_size; else tz=maj_size; break;
                  case 2: if (pass==0) tx=maj_size; else ty=maj_size; break;
                }
                tx *= invertx;
                ty *= inverty;
                tz *= invertz;
                float xs = xc - tx*maj_toff;
                float xe = xc + tx*(1. - maj_toff);
                float ys = yc - ty*maj_toff;
                float ye = yc + ty*(1. - maj_toff);
                float zs = zc - tz*maj_toff;
                float ze = zc + tz*(1. - maj_toff);

                glVertex3d(xs, ys, zs);
                glVertex3d(xe, ye, ze);
            }

            float tx=0, ty=0, tz=0;
            const float s = 0.4;
            switch (axis)
            {
              case 0: ty=s*fontscale; tz=s*fontscale; break;
              case 1: tx=s*fontscale; tz=s*fontscale; break;
              case 2: tx=s*fontscale; ty=s*fontscale; break;
            }
            tx *= invertx;
            ty *= inverty;
            tz *= invertz;
            char val[256];
            snprintf(val, 256, "%g", positions[i]);
            labels[i]->SetText(val);
            //if (fabs(positions[i]) < 1e-10)
            //    labels[i]->SetText("0");
            labels[i]->SetPosition(xc - tx, yc - ty, zc - tz);
            labels[i]->SetAlignment(eavlTextAnnotation::HCenter,
                                    eavlTextAnnotation::VCenter);
        }

        // minor ticks
        CalculateTicks(lower, upper, true, positions, proportions, moreOrLessTickAdjustment);
        int nminor = proportions.size();
        for (int i=0; i<nminor; ++i)
        {
            float xc = x0 + (x1-x0) * proportions[i];
            float yc = y0 + (y1-y0) * proportions[i];
            float zc = z0 + (z1-z0) * proportions[i];
            for (int pass=0; pass<=1; pass++)
            {
                float tx=0, ty=0, tz=0;
                switch (axis)
                {
                  case 0: if (pass==0) ty=min_size; else tz=min_size; break;
                  case 1: if (pass==0) tx=min_size; else tz=min_size; break;
                  case 2: if (pass==0) tx=min_size; else ty=min_size; break;
                }
                tx *= invertx;
                ty *= inverty;
                tz *= invertz;
                float xs = xc - tx*min_toff;
                float xe = xc + tx*(1. - min_toff);
                float ys = yc - ty*min_toff;
                float ye = yc + ty*(1. - min_toff);
                float zs = zc - tz*min_toff;
                float ze = zc + tz*(1. - min_toff);

                glVertex3d(xs, ys, zs);
                glVertex3d(xe, ye, ze);
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
