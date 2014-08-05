// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_RENDER_SURFACE_GL_H
#define EAVL_RENDER_SURFACE_GL_H

#include "eavlRenderSurface.h"
#include "eavlTexture.h"

class eavlRenderSurfaceGL : public eavlRenderSurface
{
  protected:
    std::map<std::string,eavlTexture*> textures;
    eavlTexture *GetTexture(const std::string &s)
    {
        return textures[s];
    }
    void SetTexture(const std::string &s, eavlTexture *tex)
    {
        textures[s] = tex;
    }

  public:
    eavlRenderSurfaceGL()
    {
    }
    // todo: leave these pure virtual and
    // create an OpenGL version?
    virtual void Initialize()
    {
    }
    virtual void Resize(int w, int h)
    {
    }
    virtual void Activate()
    {
    }
    virtual void Finish()
    {
        glFinish();
    }
    virtual void Clear(eavlColor bg)
    {
        glClearColor(bg.c[0], bg.c[1], bg.c[2], 1.0); ///< c[3] instead of 1.0?
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    }

    virtual void AddRectangle(float x, float y, 
                              float w, float h,
                              eavlColor c)
    {
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glColor3fv(c.c);

        float depth = 0.99;
        glBegin(GL_QUADS);
        glVertex3f(x,y,depth);
        glVertex3f(x+w,y,depth);
        glVertex3f(x+w,y+h,depth);
        glVertex3f(x,y+h,depth);
        glEnd();
    }
    virtual void AddLine(float x0, float y0,
                         float x1, float y1,
                         float linewidth,
                         eavlColor c)
    {
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glColor3fv(c.c);

        glLineWidth(linewidth);

        glBegin(GL_LINES);
        glVertex2f(x0,y0);
        glVertex2f(x1,y1);
        glEnd();
    }
    virtual void AddColorBar(float x, float y, 
                             float w, float h,
                             string ctname,
                             bool horizontal)
    {
        glDisable(GL_DEPTH_TEST);

        eavlTexture *tex = GetTexture(ctname);
        if (!tex )
        {
            if (!tex)
                tex = new eavlTexture;
            tex->CreateFromColorTable(eavlColorTable(ctname));
            SetTexture(ctname, tex);
        }

        tex->Enable();

        glColor3fv(eavlColor::white.c);

        float depth = 0.99;
        glBegin(GL_QUADS);
        glTexCoord1f(0);
        glVertex3f(x,y,depth);
        glTexCoord1f(horizontal ? 1 : 0);
        glVertex3f(x+w,y,depth);
        glTexCoord1f(1);
        glVertex3f(x+w,y+h,depth);
        glTexCoord1f(horizontal ? 0 : 1);
        glVertex3f(x,y+h,depth);
        glEnd();

        tex->Disable();
    }

    virtual void AddText(float x, float y,
                         float scale,
                         float angle,
                         float windowaspect,
                         float anchorx, float anchory,
                         eavlColor color,
                         string text) ///<\todo: better way to get view here!
    {
        eavlMatrix4x4 mtx;
        mtx.CreateTranslate(x, y, 0);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        mtx.CreateScale(1./windowaspect, 1, 1);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        mtx.CreateRotateZ(angle * M_PI / 180.);
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        glColor3fv(color.c);

        RenderText(scale, anchorx, anchory, text);
    }

  private:
    void RenderText(float scale, float anchorx, float anchory, string text)
    {
        // set up a texture for the font if needed
        eavlBitmapFont *fnt = eavlBitmapFontFactory::GetDefaultFont();
        eavlTexture *tex = GetTexture(fnt->name);
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
            SetTexture(fnt->name, tex);
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

#endif