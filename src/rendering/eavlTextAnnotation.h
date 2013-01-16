// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TEXT_ANNOTATION_H
#define EAVL_TEXT_ANNOTATION_H

#include <eavlBitmapFont.h>
#include <eavlBitmapFontFactory.h>
#include <eavlPNGImporter.h>
#include <eavlTexture.h>
#include <eavlMatrix4x4.h>
#include <eavlAnnotation.h>

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
  protected:
    float  scale;
    string text;
    eavlColor color;
    float anchorx, anchory;
  public:
    eavlTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, float s)
        : eavlAnnotation(w), text(txt), color(c), scale(s)
    {
        anchorx = 0;
        anchory = 0;
    }
    void SetText(const string &txt)
    {
        text = txt;
    }
    void SetAnchor(float h, float v)
    {
        anchorx = h;
        anchory = v;
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

        float textwidth = fnt->GetTextWidth(text);
        float fx = -anchorx * textwidth, fy = -anchory, fz = 0;
        for (int i=0; i<text.length(); ++i)
        {
            char c = text[i];
            char nextchar = (i < text.length()-1) ? text[i+1] : 0;

            float vl,vr,vt,vb;
            float tl,tr,tt,tb;
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
    float x,y;
    float angle;
  public:
    eavlScreenTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, float s,
                             float ox, float oy, float angleDeg = 0.)
        : eavlTextAnnotation(w,txt,c,s)
    {
        x = ox;
        y = oy;
        angle = angleDeg;
    }
    void SetPosition(float ox, float oy)
    {
        x = ox;
        y = oy;
    }
    virtual void Setup(eavlView &view)
    {
        
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glOrtho(-1,1, -1,1, -1,1);

        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        eavlMatrix4x4 mtx;
        mtx.CreateTranslate(x, y, 0);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        mtx.CreateScale(1./view.aspect, 1, 1);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        mtx.CreateRotateZ(angle * M_PI / 180.);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());
    }
    virtual void Render()
    {
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
    eavlWorldTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, float s,
                            float ox, float oy, float oz,
                            float nx, float ny, float nz,
                            float ux, float uy, float uz)
        : eavlTextAnnotation(w,txt,c,s)
    {
        mtx.CreateRBT(eavlPoint3(ox,oy,oz),
                      eavlPoint3(ox,oy,oz) - eavlVector3(nx,ny,nz),
                      eavlVector3(ux,uy,uz));

    }
    virtual void Setup(eavlView &view)
    {
        glMatrixMode( GL_PROJECTION );
        glLoadMatrixf(view.P.GetOpenGLMatrix4x4());

        glMatrixMode( GL_MODELVIEW );
        glLoadMatrixf(view.V.GetOpenGLMatrix4x4());

        glMultMatrixf(mtx.GetOpenGLMatrix4x4());
    }
    virtual void Render()
    {
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
    float x,y,z;
    bool fixed2Dscale;
    float angle;
  public:
    eavlBillboardTextAnnotation(eavlWindow *w, const string &txt, eavlColor c, float s,
                                float ox, float oy, float oz,
                                bool scaleIsScreenSpace,
                                float angleDeg = 0.)
        : eavlTextAnnotation(w,txt,c,s)
    {
        x = ox;
        y = oy;
        z = oz;
        angle = angleDeg;
        fixed2Dscale = scaleIsScreenSpace;
    }
    void SetPosition(float ox, float oy, float oz)
    {
        x = ox;
        y = oy;
        z = oz;
    }
    virtual void Setup(eavlView &view)
    {
        if (fixed2Dscale)
        {
            eavlPoint3 p = view.P * view.V * eavlPoint3(x,y,z);

            glMatrixMode( GL_PROJECTION );
            glLoadIdentity();
            glOrtho(-1,1, -1,1, -1,1);

            glMatrixMode( GL_MODELVIEW );
            glLoadIdentity();

            eavlMatrix4x4 mtx;
            mtx.CreateTranslate(p.x, p.y, -p.z);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());

            mtx.CreateScale(1./view.aspect, 1, 1);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());

            // height given in (0,1) vert range, but we
            // are currently in -1,1 range, so scale up by 2.0
            mtx.CreateScale(2.0, 2.0, 1.0);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());

            mtx.CreateRotateZ(angle * M_PI / 180.);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());
        }
        else
        {
            glMatrixMode( GL_PROJECTION );
            glLoadMatrixf(view.P.GetOpenGLMatrix4x4());

            glMatrixMode( GL_MODELVIEW );
            glLoadMatrixf(view.V.GetOpenGLMatrix4x4());

            eavlMatrix4x4 mtx;
            mtx.CreateRBT(eavlPoint3(x,y,z),
                          eavlPoint3(x,y,z) - (view.view3d.from-view.view3d.at),
                          view.view3d.up);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());

            mtx.CreateRotateZ(angle * M_PI / 180.);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        }
    }
    virtual void Render()
    {
        RenderText();
    }
};

#endif
