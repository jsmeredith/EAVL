// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COLOR_BAR_ANNOTATION_H
#define EAVL_COLOR_BAR_ANNOTATION_H

#include "eavlView.h"

#include "eavlColor.h"

// ****************************************************************************
// Class:  eavlColorLegendAnnotation
//
// Purpose:
///   Annotation which renders a set of discrete colors and values.
//
// Programmer:  Jeremy Meredith
// Creation:    May  1, 2014
//
// Modifications:
// ****************************************************************************
class eavlColorLegendAnnotation : public eavlAnnotation
{
  protected:
    vector<string> labels;
    vector<eavlColor> colors;
    vector<eavlScreenTextAnnotation*> annot;
  public:
    eavlColorLegendAnnotation(eavlWindow *win) : eavlAnnotation(win)
    {
    }
    virtual ~eavlColorLegendAnnotation()
    {
    }
    void Clear()
    {
        labels.clear();
        colors.clear();
    }
    void AddItem(const string &label, eavlColor color)
    {
        labels.push_back(label);
        colors.push_back(color);
    }
    virtual void Render(eavlView &view)
    {
        win->SetupForScreenSpace();

        float l = -0.95, r = -0.90;
        float b = +0.90, t = +0.95;
        float spacing = 0.07;

        for (unsigned int i=0; i<colors.size(); ++i)
        {
            win->surface->AddRectangle(l,
                                       t - spacing*float(i),
                                       r-l,
                                       b-t,
                                       colors[i]);
        }

        // reset positions
        l = -0.95; r = -0.90;
        b = +0.90; t = +0.95;

        const float fontscale = 0.05;

        while (annot.size() < labels.size())
        {
            annot.push_back(new eavlScreenTextAnnotation(win, "text",
                                                         eavlColor::black,
                                                         fontscale,
                                                         0,0,
                                                         0));
        }

        for (unsigned int i=0; i<labels.size(); ++i)
        {
            eavlScreenTextAnnotation *txt = annot[i];
            txt->SetText(labels[i]);
            txt->SetPosition(r + .02, (b+t)/2. - spacing*float(i));
            txt->SetAlignment(eavlTextAnnotation::Left,
                              eavlTextAnnotation::VCenter);
            txt->Render(view);
        }

    }
};

#endif
