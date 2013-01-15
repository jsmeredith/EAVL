// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COLOR_BAR_ANNOTATION_H
#define EAVL_COLOR_BAR_ANNOTATION_H

#include "eavlView.h"

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
class eavlColorBarAnnotation : public eavlScreenSpaceAnnotation
{
  protected:
    string ctname;
    bool needsUpdate;
  public:
    GLuint texid;
    eavlColorBarAnnotation(eavlWindow *win) : eavlScreenSpaceAnnotation(win)
    {
        texid == 0;
        needsUpdate = true;
    }
    void SetColorTable(const string &colortablename)
    {
        if (ctname == colortablename)
            return;
        ctname = colortablename;
        needsUpdate = true;
    }
    virtual void Render()
    {
        eavlTexture *tex = win->GetTexture("colorbar");
        if (!tex || needsUpdate)
        {
            if (!tex)
                tex = new eavlTexture;
            tex->CreateFromColorTable(eavlColorTable(ctname));
            win->SetTexture("colorbar", tex);
        }
        needsUpdate = false;

        tex->Enable();

        glDisable(GL_LIGHTING);
        glColor3fv(eavlColor::white.c);

        glBegin(GL_QUADS);

        glTexCoord1f(0);
        glVertex3f(-.9, .90 ,.99);
        glVertex3f(-.9, .95 ,.99);

        glTexCoord1f(1);
        glVertex3f(+.9, .95 ,.99);
        glVertex3f(+.9, .90 ,.99);

        glEnd();

        tex->Disable();
    }
};

#endif
