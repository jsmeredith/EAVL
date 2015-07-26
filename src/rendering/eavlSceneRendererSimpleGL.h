// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_SIMPLE_GL_H
#define EAVL_SCENE_RENDERER_SIMPLE_GL_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"
#include "eavlGLTexture.h"

//#define USE_DISPLAY_LISTS
//#define USE_VERTEX_BUFFERS

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
    std::map<std::string,eavlGLTexture*> textures;

#ifdef USE_DISPLAY_LISTS
    GLuint mylist;
#endif
#ifdef USE_VERTEX_BUFFERS
    vector<float> vb_vertex;
    vector<float> vb_normal;
    vector<float> vb_tex;
#endif

  public:
    eavlSceneRendererSimpleGL()
    {
#ifdef USE_DISPLAY_LISTS
        mylist = 0;
#endif
    }
    virtual ~eavlSceneRendererSimpleGL()
    {
        for (std::map<std::string,eavlGLTexture*>::iterator i = textures.begin();
             i != textures.end() ; ++i)
            delete i->second;
        textures.clear();
    }

    eavlGLTexture *GetTexture(const std::string &s)
    {
        return textures[s];
    }
    void SetTexture(const std::string &s, eavlGLTexture *tex)
    {
        textures[s] = tex;
    }

#ifdef USE_DISPLAY_LISTS
    virtual void StartScene()
    {
        eavlSceneRenderer::StartScene();
        if (mylist > 0)
            glDeleteLists(mylist, 1);
        mylist = glGenLists(1);
        glNewList(mylist, GL_COMPILE_AND_EXECUTE);
    }

    virtual void EndScene()
    {
        eavlSceneRenderer::EndScene();
        glEndList();
    }
#else
 #ifdef USE_VERTEX_BUFFERS
    virtual void StartScene()
    {
        eavlSceneRenderer::StartScene();
        vb_vertex.clear();
        vb_normal.clear();
        vb_tex.clear();
        //cerr << "Reset\n";
    }

    virtual void EndScene()
    {
        eavlSceneRenderer::EndScene();
    }
 #else
    // we're not caching anything; always say we need it
    virtual bool NeedsGeometryForPlot(int)
    {
        return true;
    }
    virtual void StartScene()
    {
        eavlSceneRenderer::StartScene();
        // we're rendering in immediate mode and
        // will send triangles to OpenGL instead of 
        // caching; set up the rendering stuff now.
        SetupForRendering();
    }
 #endif
#endif

    virtual void SetActiveColor(eavlColor c)
    {
        glColor3fv(c.c);
        glDisable(GL_TEXTURE_1D);
    }
    virtual void SetActiveColorTable(eavlColorTable ct)
    {
        glColor3fv(eavlColor::white.c);

        eavlGLTexture *tex = GetTexture(ct.GetName());
        if (!tex)
        {
            tex = new eavlGLTexture;
            tex->CreateFromColorTable(ct);
            SetTexture(ct.GetName(), tex);
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

#define CONVERT_TETS_TO_TRIANGLES
#ifdef CONVERT_TETS_TO_TRIANGLES 
    virtual void StartTetrahedra()
    {
        //glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);
        StartTriangles();
    }

    virtual void EndTetrahedra()
    {
        EndTriangles();
        glDisable(GL_CULL_FACE);
    }


    virtual void AddTetrahedronVs(double x0, double y0, double z0,
                                  double x1, double y1, double z1,
                                  double x2, double y2, double z2,
                                  double x3, double y3, double z3,
                                  double s0, double s1, double s2, double s3)
    {
        AddTriangleVs(x1,y1,z1,
                      x0,y0,z0,
                      x2,y2,z2,
                      s1,s0,s2);
        AddTriangleVs(x0,y0,z0,
                      x1,y1,z1,
                      x3,y3,z3,
                      s0,s1,s3);
        AddTriangleVs(x1,y1,z1,
                      x2,y2,z2,
                      x3,y3,z3,
                      s1,s2,s3);
        AddTriangleVs(x2,y2,z2,
                      x0,y0,z0,
                      x3,y3,z3,
                      s2,s0,s3);
    }
#endif

    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2)
    {
#ifdef USE_VERTEX_BUFFERS
        vb_vertex.push_back(x0);
        vb_vertex.push_back(y0);
        vb_vertex.push_back(z0);
        vb_vertex.push_back(x1);
        vb_vertex.push_back(y1);
        vb_vertex.push_back(z1);
        vb_vertex.push_back(x2);
        vb_vertex.push_back(y2);
        vb_vertex.push_back(z2);

        vb_normal.push_back(u0);
        vb_normal.push_back(v0);
        vb_normal.push_back(w0);
        vb_normal.push_back(u1);
        vb_normal.push_back(v1);
        vb_normal.push_back(w1);
        vb_normal.push_back(u2);
        vb_normal.push_back(v2);
        vb_normal.push_back(w2);

        vb_tex.push_back(s0);
        vb_tex.push_back(s1);
        vb_tex.push_back(s2);
#else
        glNormal3d(u0,v0,w0);
        glTexCoord1f(s0);
        glVertex3d(x0,y0,z0);

        glNormal3d(u1,v1,w1);
        glTexCoord1f(s1);
        glVertex3d(x1,y1,z1);

        glNormal3d(u2,v2,w2);
        glTexCoord1f(s2);
        glVertex3d(x2,y2,z2);
#endif
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
    void SetupForRendering()
    {
        if (view.viewtype == eavlView::EAVL_VIEW_3D)
        {
            if (eyeLight)
            {
                // We need to set lighting without the view matrix (for an eye
                // light) so load the identity matrix into modelview temporarily.
                glMatrixMode( GL_MODELVIEW );
                glLoadIdentity();
            }

            eavlColor ambient(Ka,Ka,Ka);
            eavlColor diffuse(Kd,Kd,Kd);
            eavlColor specular(Ks,Ks,Ks);
            float     shininess = 8.0;

            bool twoSidedLighting = true;
            //glShadeModel(GL_SMOOTH);
            glEnable(GL_LIGHTING);
            glEnable(GL_LIGHT0);
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, twoSidedLighting?1:0);
            glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient.c);
            // needed to get white specular highlights on textured polygons
            glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);
            glLightfv(GL_LIGHT0, GL_AMBIENT, eavlColor::black.c);
            glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse.c);
            glLightfv(GL_LIGHT0, GL_SPECULAR, eavlColor::white.c);

            //float lightdir[4] = {0, 0, 1, 0};
            float lightdir[4] = {Lx, Ly, Lz, 0};
            glLightfv(GL_LIGHT0, GL_POSITION, lightdir);

            glEnable(GL_COLOR_MATERIAL);
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE) ;
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular.c);
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);


            if (eyeLight)
            {
                // Okay, set the view matrix back now that lighting's done.
                glLoadMatrixf(view.V.GetOpenGLMatrix4x4());
            }

            glEnable(GL_DEPTH_TEST);
        }
        else
        {
            glDisable(GL_LIGHTING);
            glDisable(GL_DEPTH_TEST);
        }
    }

    virtual void Render()
    {
        SetupForRendering();
        
        //cerr << "Render\n";
#ifdef USE_DISPLAY_LISTS
        glCallList(mylist);
#endif
#ifdef USE_VERTEX_BUFFERS
        int ntris = vb_vertex.size() / 9;
        //cerr << "ntris="<<ntris<<"\n";
        glNormalPointer(GL_FLOAT, 0, &(vb_normal[0]));
        glTexCoordPointer(1, GL_FLOAT, 0, &(vb_tex[0]));
        glVertexPointer(3, GL_FLOAT, 0, &(vb_vertex[0]));
        glEnableClientState(GL_NORMAL_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(GL_TRIANGLES, 0, ntris*3);
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
#endif
        glFinish();
    }

};


#endif
