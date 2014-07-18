// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_SIMPLE_GL_H
#define EAVL_SCENE_RENDERER_SIMPLE_GL_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlTexture.h"

// ****************************************************************************
// Class:  eavlSceneRendererSimpleGL
//
// Purpose:
///   A very simple, though not necessarily fast, implementation of
///   an OpenGL scene renderer.
//
// Programmer:  Jeremy Meredith
// Creation:    July 14, 2014
//
// Modifications:
//
// ****************************************************************************
class eavlSceneRendererSimpleGL : public eavlSceneRenderer
{
    std::map<std::string,eavlTexture*> textures;

  public:
    virtual ~eavlSceneRendererSimpleGL()
    {
        for (std::map<std::string,eavlTexture*>::iterator i = textures.begin();
             i != textures.end() ; ++i)
            delete i->second;
        textures.clear();
    }

    eavlTexture *GetTexture(const std::string &s)
    {
        return textures[s];
    }
    void SetTexture(const std::string &s, eavlTexture *tex)
    {
        textures[s] = tex;
    }


    // we're not caching anything; always say we need it
    virtual bool NeedsGeometryForPlot(eavlPlot*)
    {
        return true;
    }

    virtual void SetActiveColor(eavlColor c)
    {
        glColor3fv(c.c);
        glDisable(GL_TEXTURE_1D);
    }
    virtual void SetActiveColorTable(string ctname)
    {
        glColor3fv(eavlColor::white.c);

        eavlTexture *tex = GetTexture(ctname);
        if (!tex)
        {
            tex = new eavlTexture;
            tex->CreateFromColorTable(eavlColorTable(ctname));
            SetTexture(ctname, tex);
        }
        tex->Enable();
    }


    // ------------------------------------------------------------------------
    virtual void StartTriangles()
    {
        glBegin(GL_TRIANGLES);
    }

    virtual void EndTriangles()
    {
        glEnd();
        ///\todo: small hack -- should use tex->Disable() instead
        glDisable(GL_TEXTURE_1D);
    }

    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2)
    {
        glNormal3d(u0,v0,w0);
        glTexCoord1f(s0);
        glVertex3d(x0,y0,z0);

        glNormal3d(u1,v1,w1);
        glTexCoord1f(s1);
        glVertex3d(x1,y1,z1);

        glNormal3d(u2,v2,w2);
        glTexCoord1f(s2);
        glVertex3d(x2,y2,z2);
    }


    // ------------------------------------------------------------------------

    virtual void StartPoints()
    {
        glDisable(GL_LIGHTING);
        glPointSize(3);
        glBegin(GL_POINTS);
    }

    virtual void EndPoints()
    {
        glEnd();
        glDisable(GL_TEXTURE_1D);
    }

    virtual void AddPointVs(double x, double y, double z, double r, double s)
    {
        glTexCoord1f(s);
        glVertex3d(x,y,z);
    }

    // ------------------------------------------------------------------------

    virtual void StartLines()
    {
        glDisable(GL_LIGHTING);
        glLineWidth(2);
        glBegin(GL_LINES);
    }

    virtual void EndLines()
    {
        glEnd();
        glDisable(GL_TEXTURE_1D);
    }

    virtual void AddLineVs(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double s0, double s1)
    {
        glTexCoord1f(s0);
        glVertex3d(x0,y0,z0);

        glTexCoord1f(s1);
        glVertex3d(x1,y1,z1);
    }


    // ------------------------------------------------------------------------
    virtual void Render(eavlView v)
    {
        glFinish();
    }

};


#endif
