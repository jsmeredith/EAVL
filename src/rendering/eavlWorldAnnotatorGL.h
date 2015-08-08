// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_WORLD_ANNOTATOR_GL_H
#define EAVL_WORLD_ANNOTATOR_GL_H

#include "eavlWorldAnnotator.h"

#include <eavlBitmapFont.h>
#include <eavlBitmapFontFactory.h>
#include <eavlPNGImporter.h>
#include <eavlGLTexture.h>

class eavlWorldAnnotatorGL : public eavlWorldAnnotator
{
  protected:
    ///\todo: duplication with render surface
    std::map<std::string,eavlGLTexture*> textures;
    eavlGLTexture *GetTexture(const std::string &s)
    {
        return textures[s];
    }
    void SetTexture(const std::string &s, eavlGLTexture *tex)
    {
        textures[s] = tex;
    }

  public:
    eavlWorldAnnotatorGL() : eavlWorldAnnotator()
    {
    }
    ~eavlWorldAnnotatorGL()
    {
      std::map<std::string,eavlGLTexture*>::iterator it;
      for(it= textures.begin(); it != textures.end(); it++) 
      {
         delete it->second;
      }
    }
    virtual void AddLine(float x0, float y0, float z0,
                         float x1, float y1, float z1,
                         float linewidth,
                         eavlColor c,
                         bool infront)
    {
        if (infront)
            glDepthRange(-.0001,.9999);

        glDisable(GL_LIGHTING);

        glColor3fv(c.c);

        glLineWidth(linewidth);

        glBegin(GL_LINES);
        glVertex3f(x0,y0,z0);
        glVertex3f(x1,y1,z1);
        glEnd();

        if (infront)
            glDepthRange(0,1);

    }
    virtual void AddText(float ox, float oy, float oz,
                         float rx, float ry, float rz,
                         float ux, float uy, float uz,
                         float scale,
                         float anchorx, float anchory,
                         eavlColor color,
                         string text)
    {
        eavlPoint3 o(ox,oy,oz);
        eavlVector3 r(rx,ry,rz);
        eavlVector3 u(ux,uy,uz);
        eavlVector3 n = (r % u).normalized();

        eavlMatrix4x4 mtx;
        mtx.CreateWorld(o, r, u, n);
        //cerr << "scale="<<scale<<" anchor="<<anchorx<<","<<anchory<<"  mtx=\n"<<mtx<<endl;

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glMultMatrixf(mtx.GetOpenGLMatrix4x4());

        glColor3fv(color.c);

        RenderText(scale, anchorx, anchory, text);

        glPopMatrix();
    }


  private:
    ///\todo: duplication with render surface
    void RenderText(float scale, float anchorx, float anchory, string text)
    {
        // set up a texture for the font if needed
        eavlBitmapFont *fnt = eavlBitmapFontFactory::GetDefaultFont();
        eavlGLTexture *tex = GetTexture(fnt->name);
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
                delete pngimp;
            }

            tex = new eavlGLTexture;
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

