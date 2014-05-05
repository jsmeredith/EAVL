// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TEXT_ANNOTATION_H
#define EAVL_TEXT_ANNOTATION_H

#include <eavlBitmapFont.h>
#include <eavlBitmapFontFactory.h>
#include <eavlPNGImporter.h>
#include <eavlTexture.h>
#include <eavlMatrix4x4.h>
#include <eavlAnnotation.h>

#ifdef _WIN32
 #include "GL/glext.h"
#endif

// ****************************************************************************
// Class:  eavlTextAnnotation
//
// Purpose:
///   Allows 2D or 3D text.
//
// Programmer:  Jeremy Meredith
// Creation:    January  9, 2013
//
// Modifications:
// ****************************************************************************
class eavlTextAnnotation : public eavlAnnotation
{
  public:
    enum HorizontalAlignment
    {
        Left,
        HCenter,
        Right
    };
    enum VerticalAlignment
    {
        Bottom,
        VCenter,
        Top
    };

  protected:
    string text;
    eavlColor color;
    double  scale;
    ///\todo: change anchor range to [-1,+1] (center=0) instead of [0,1] (center=.5)
    double anchorx, anchory;

  public:
    eavlTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, double s)
        : eavlAnnotation(w), text(txt), color(c), scale(s)
    {
        anchorx = 0;
        anchory = 0;
    }
    void SetText(const string &txt)
    {
        text = txt;
    }
    void SetRawAnchor(double h, double v)
    {
        anchorx = h;
        anchory = v;
    }
    void SetAlignment(HorizontalAlignment h, VerticalAlignment v)
    {
        switch (h)
        {
          case Left:    anchorx = -1.0; break;
          case HCenter: anchorx =  0.0; break;
          case Right:   anchorx = +1.0; break;
        }

        // For vertical alignment, "center" is generally the center
        // of only the above-baseline contents of the font, so we
        // use a value slightly off of zero for VCenter.
        // (We don't use an offset value instead of -1.0 for the 
        // bottom value, because generally we want a true minimum
        // extent, e.g. to have text sitting at the bottom of a
        // window, and in that case, we need to keep all the text,
        // including parts that descend below the baseline, above
        // the bottom of the window.
        switch (v)
        {
          case Bottom:  anchory = -1.0;  break;
          case VCenter: anchory = -0.06; break;
          case Top:     anchory = +1.0;  break;
        }
    }
    void SetScale(double s)
    {
        scale = s;
    }
  protected:
    void RenderText()
    {
        // set up a texture for the font if needed
        eavlBitmapFont *fnt = eavlBitmapFontFactory::GetDefaultFont();
        eavlTexture *tex = win->GetTexture(fnt->name);
        if (!tex)
        {
            eavlDataSet *img = (eavlDataSet*)fnt->userPointer;
            if (!img)
            {
                string ftype;
                vector<unsigned char> &rawpngdata = fnt->GetRawImageData(ftype);
                if (ftype != "png")
                    cerr << "Error: expected PNG type for font image data\n";
                eavlPNGImporter *pngimp = new eavlPNGImporter(&rawpngdata[0],
                                                          rawpngdata.size());
                img = pngimp->GetMesh("mesh",0);
                img->AddField(pngimp->GetField("a","mesh",0));
                fnt->userPointer = img;
            }

            tex = new eavlTexture;
            tex->CreateFromDataSet(img, false,false,false,true);
            win->SetTexture(fnt->name, tex);
        }

        // Kerning causes overlapping polygons.  Ideally only draw
        // pixels where alpha>0, though this causes problems for alpha
        // between 0 and 1 (unless can solve with different blend
        // func??) when text isn't rendered last.  Simpler is to just
        // disable z-writing entirely, but then must draw all text
        // last anyway.  Or maybe draw text with two passes, once to
        // update Z (when alpha==1) and once to draw pixels (when alpha>0).
        if (true)
        {
            glDepthMask(GL_FALSE);
        }
        else
        {
            glAlphaFunc(GL_GREATER, 0);
            glEnable(GL_ALPHA_TEST);
        }

        tex->Enable();
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
        glDisable(GL_LIGHTING);
        glTexEnvf(GL_TEXTURE_FILTER_CONTROL, GL_TEXTURE_LOD_BIAS, -.5);
        //glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

        glColor3fv(color.c);

        glBegin(GL_QUADS);

        double textwidth = fnt->GetTextWidth(text);

        double fx = -(.5 + .5*anchorx) * textwidth;
        double fy = -(.5 + .5*anchory);
        double fz = 0;
        for (unsigned int i=0; i<text.length(); ++i)
        {
            char c = text[i];
            char nextchar = (i < text.length()-1) ? text[i+1] : 0;

            double vl,vr,vt,vb;
            double tl,tr,tt,tb;
            fnt->GetCharPolygon(c, fx, fy,
                                vl, vr, vt, vb,
                                tl, tr, tt, tb, nextchar);

            glTexCoord2f(tl, 1-tt);
            glVertex3f(scale*vl, scale*vt, fz);

            glTexCoord2f(tl, 1-tb);
            glVertex3f(scale*vl, scale*vb, fz);

            glTexCoord2f(tr, 1-tb);
            glVertex3f(scale*vr, scale*vb, fz);

            glTexCoord2f(tr, 1-tt);
            glVertex3f(scale*vr, scale*vt, fz);
            /*cerr << "tl="<<tl<<endl;
              cerr << "tr="<<tr<<endl;
              cerr << "tt="<<tt<<endl;
              cerr << "tb="<<tb<<endl;*/
        }

        glEnd();

        glTexEnvf(GL_TEXTURE_FILTER_CONTROL, GL_TEXTURE_LOD_BIAS, 0);
        glDepthMask(GL_TRUE);
        glDisable(GL_ALPHA_TEST);
        tex->Disable();
    }
};

// ****************************************************************************
// Class:  eavlScreenTextAnnotation
//
// Purpose:
///   Text location and height are in normalized screen space coordinates.
///   At the default angle (0.0), text is oriented upright.
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavlScreenTextAnnotation : public eavlTextAnnotation
{
  protected:
    double x,y;
    double angle;
  public:
    eavlScreenTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, double s,
                             double ox, double oy, double angleDeg = 0.)
        : eavlTextAnnotation(w,txt,c,s)
    {
        x = ox;
        y = oy;
        angle = angleDeg;
    }
    void SetPosition(double ox, double oy)
    {
        x = ox;
        y = oy;
    }
    virtual void Render(eavlView &view)
    {
        view.SetupForScreenSpace();

        eavlMatrix4x4 mtx;
        mtx.CreateTranslate(x, y, 0);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        mtx.CreateScale(1./view.windowaspect, 1, 1);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        mtx.CreateRotateZ(angle * M_PI / 180.);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        RenderText();
    }
};

// ****************************************************************************
// Class:  eavlWorldTextAnnotation
//
// Purpose:
///   Text location, orientation, and size are in world space coordinates.
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavlWorldTextAnnotation : public eavlTextAnnotation
{
  protected:
    eavlMatrix4x4 mtx;
  public:
    eavlWorldTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, double s,
                            double ox, double oy, double oz,
                            double nx, double ny, double nz,
                            double ux, double uy, double uz)
        : eavlTextAnnotation(w,txt,c,s)
    {
        mtx.CreateRBT(eavlPoint3(ox,oy,oz),
                      eavlPoint3(ox,oy,oz) - eavlVector3(nx,ny,nz),
                      eavlVector3(ux,uy,uz));

    }
    virtual void Render(eavlView &view)
    {
        view.SetupForWorldSpace();

        eavlMatrix4x4 M = view.V * mtx;

        if (view.viewtype == eavlView::EAVL_VIEW_2D)
        {
            eavlMatrix4x4 S;
            S.CreateScale(1. / view.view2d.xscale, 1, 1);
            M = view.V * mtx * S;
        }

        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf(M.GetOpenGLMatrix4x4());

        RenderText();
    }
};

// ****************************************************************************
// Class:  eavlBillboardTextAnnotation
//
// Purpose:
///   Text location origin is in world space, but the text is rotated so it
///   is always facing towards the user and always at the same orientation
///   (e.g. upright if angle==0).
///   Height can either be in screen space height (so it doesn't change
///   apparent size as the view moves), or in world space height (so
///   it gets bigger and smaller based on distance to the viewer).
//
// Programmer:  Jeremy Meredith
// Creation:    January 10, 2013
//
// Modifications:
// ****************************************************************************
class eavlBillboardTextAnnotation : public eavlTextAnnotation
{
  protected:
    double x,y,z;
    bool fixed2Dscale;
    double angle;
  public:
    eavlBillboardTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, double s,
                                double ox, double oy, double oz,
                                bool scaleIsScreenSpace,
                                double angleDeg = 0.)
        : eavlTextAnnotation(w,txt,c,s)
    {
        x = ox;
        y = oy;
        z = oz;
        angle = angleDeg;
        fixed2Dscale = scaleIsScreenSpace;
    }
    void SetPosition(double ox, double oy, double oz)
    {
        x = ox;
        y = oy;
        z = oz;
    }
    virtual void Render(eavlView &view)
    {
        view.SetupViewportForWorld();

        if (fixed2Dscale)
        {
            view.SetupMatricesForScreen();

            eavlPoint3 p = view.P * view.V * eavlPoint3(x,y,z);

            eavlMatrix4x4 mtx;
            mtx.CreateTranslate(p.x, p.y, -p.z);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());

            mtx.CreateScale(1./view.windowaspect, 1, 1);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());

            //if (view.viewtype == eavlView::EAVL_VIEW_2D)
            {
                double vl, vr, vt, vb;
                view.GetRealViewport(vl,vr,vb,vt);
                double xs = (vr-vl);
                double ys = (vt-vb);
                eavlMatrix4x4 S;
                S.CreateScale(2./xs, 2./ys, 1);
                glMultMatrixf(S.GetOpenGLMatrix4x4());
            }

            mtx.CreateRotateZ(angle * M_PI / 180.);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());
        }
        else
        {
            view.SetupMatricesForWorld();

            eavlMatrix4x4 mtx;
            if (view.viewtype == eavlView::EAVL_VIEW_2D)
            {
                mtx.CreateRBT(eavlPoint3(x,y,z),
                              eavlPoint3(x,y,z) - eavlVector3(0,0,-1),
                              eavlVector3(0,1,0));
                glMultMatrixf(mtx.GetOpenGLMatrix4x4());

                mtx.CreateScale(1. / view.view2d.xscale, 1, 1);
                glMultMatrixf(mtx.GetOpenGLMatrix4x4());
            }
            else
            {
                mtx.CreateRBT(eavlPoint3(x,y,z),
                              eavlPoint3(x,y,z) - (view.view3d.from-view.view3d.at),
                              view.view3d.up);
                glMultMatrixf(mtx.GetOpenGLMatrix4x4());
            }

            mtx.CreateRotateZ(angle * M_PI / 180.);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        }

        RenderText();
    }
};

// ****************************************************************************
// Class:  eavlViewportAnchoredScreenTextAnnotation
//
// Purpose:
///   Screen text is anchored to a normalized viewport location instead of
///   window location, then offset in (aspect-independent) window
///   coordinates.  The aspect-independence means that (like font size
///   making sense no matter the window aspect ratio and text rotation)
///   the x and y offset units are both in terms of window height.
//
// Programmer:  Jeremy Meredith
// Creation:    May  5, 2014
//
// Modifications:
// ****************************************************************************
class eavlViewportAnchoredScreenTextAnnotation : public eavlScreenTextAnnotation
{
  protected:
    double vx, vy; // normalized viewport coords
    double dx, dy; // aspect-independent window coordinate offset
  public:
    eavlViewportAnchoredScreenTextAnnotation(eavlWindow *w, const string &txt,
                                             eavlColor c, double s,
                                             double vx, double vy,
                                             double dx, double dy,
                                             double angleDeg = 0.)
        : eavlScreenTextAnnotation(w,txt,c,s,0,0,angleDeg),
          vx(vx), vy(vy), dx(dx), dy(dy)
    {
    }
    virtual void Render(eavlView &view)
    {
        view.SetupForScreenSpace();

        double vl, vr, vb, vt;
        view.GetRealViewport(vl,vr,vb,vt);

        // SetPosition
        x = dx/view.windowaspect + (vl+vr)/2. + vx * (vr-vl)/2.;
        y = dy + (vb+vt)/2. + vy * (vt-vb)/2.;

        eavlScreenTextAnnotation::Render(view);
    }
};

#endif
