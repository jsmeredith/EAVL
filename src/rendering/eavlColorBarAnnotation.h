// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COLOR_BAR_ANNOTATION_H
#define EAVL_COLOR_BAR_ANNOTATION_H

#include "eavlView.h"

#include "eavl2DAxisAnnotation.h"

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
    string ctname;
    eavl2DAxisAnnotation *axis;
  public:
    GLuint texid;
    eavlColorBarAnnotation(eavlWindow *win) : eavlAnnotation(win)
    {
        texid = 0;
        axis = new eavl2DAxisAnnotation(win);
    }
    void SetColorTable(const string &colortablename)
    {
        if (ctname == colortablename)
            return;
        ctname = colortablename;
    }
    void SetRange(double l, double h, int nticks)
    {
        vector<double> pos, prop;
        axis->SetMinorTicks(pos, prop); // clear the minor ticks

        for (int i=0; i<nticks; ++i)
        {
            double p = double(i) / double(nticks-1);
            pos.push_back(l + p*(h-l));
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
        view.SetupForScreenSpace();

        eavlTexture *tex = win->GetTexture(ctname);
        if (!tex )
        {
            if (!tex)
                tex = new eavlTexture;
            tex->CreateFromColorTable(eavlColorTable(ctname));
            win->SetTexture(ctname, tex);
        }

        tex->Enable();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glColor3fv(eavlColor::white.c);

        glBegin(GL_QUADS);

        float l = -0.88, r = +0.88;
        float b = +0.87, t = +0.92;

        glTexCoord1f(0);
        glVertex3f(l, b, .99);
        glVertex3f(l, t, .99);

        glTexCoord1f(1);
        glVertex3f(r, t, .99);
        glVertex3f(r, b, .99);

        glEnd();

        tex->Disable();

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
