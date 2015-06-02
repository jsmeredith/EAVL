// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COLOR_BAR_ANNOTATION_H
#define EAVL_COLOR_BAR_ANNOTATION_H

#include "eavlView.h"

#include "eavl2DAxisAnnotation.h"
#include "eavlColorTable.h"

// ****************************************************************************
// Class:  eavlColorBarAnnotation
//
// Purpose:
///   Annotation which renders a colortable to the screen.
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavlColorBarAnnotation : public eavlAnnotation
{
  protected:
    eavlColorTable colortable;
    eavl2DAxisAnnotation *axis;
  public:
    eavlColorBarAnnotation(eavlWindow *win) : eavlAnnotation(win)
    {
        axis = new eavl2DAxisAnnotation(win);
    }
    virtual ~eavlColorBarAnnotation()
    {
    }
    void SetColorTable(const eavlColorTable &ct)
    {
        colortable = ct;
    }
    void SetRange(double l, double h, int nticks, bool logscale)
    {
        vector<double> pos, prop;
        axis->SetMinorTicks(pos, prop); // clear the minor ticks

        for (int i=0; i<nticks; ++i)
        {
            double p = double(i) / double(nticks-1);
            double v = l + p*(h-l);
            if (logscale)
            {
                double ll = log(l);
                double lh = log(h);
                double lv = ll + p*(lh-ll);
                v = exp(lv);
            }
            pos.push_back(v);
            prop.push_back(p);
        }
        axis->SetMajorTicks(pos, prop);
    }
    void SetAxisColor(eavlColor c)
    {
        axis->SetColor(c);
    }
    virtual void Render(eavlView &view)
    {
        win->SetupForScreenSpace();

        float l = -0.88, r = +0.88;
        float b = +0.87, t = +0.92;

        win->surface->AddColorBar(l,t, r-l, b-t,
                                  colortable, true);

        axis->SetLineWidth(0);
        axis->SetScreenPosition(l,b, r,b);
        axis->SetMajorTickSize(0, .02, 1.0);
        axis->SetMinorTickSize(0,0,0); // no minor ticks
        axis->SetLabelAlignment(eavlTextAnnotation::HCenter,
                                eavlTextAnnotation::Top);
        axis->Render(view);

    }
};

#endif
