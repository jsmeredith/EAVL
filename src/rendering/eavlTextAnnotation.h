#ifndef EAVL_TEXT_ANNOTATION_H
#define EAVL_TEXT_ANNOTATION_H

#include <eavlBitmapFont.h>
#include <eavlBitmapFontFactory.h>
#include <eavlPNGImporter.h>
#include <eavlTexture.h>
#include <eavlMatrix4x4.h>

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
class eavlTextAnnotation
{
  protected:
    string text;
    bool twod;
    bool billboard;
    bool fixed2Dscale;
    float scale;
    float x, y, z;
    eavlMatrix4x4 mtx;
    eavlColor color;
  public:
    eavlTextAnnotation(const string &txt, eavlColor c, float s,
                       float ox, float oy)
    {
        text = txt;
        color = c;
        scale = s;
        x = ox;
        y = oy;
        z = 0;
        twod = true;
        billboard = false;
        mtx.CreateIdentity();
    }
    eavlTextAnnotation(const string &txt, eavlColor c, float s,
                       float ox, float oy, float oz,
                       bool scale_is_fixed_in_2D)
    {
        text = txt;
        color = c;
        scale = s;
        x = ox;
        y = oy;
        z = oz;
        twod = false;
        billboard = true;
        fixed2Dscale = scale_is_fixed_in_2D;
        mtx.CreateIdentity();
    }
    eavlTextAnnotation(const string &txt, eavlColor c, float s,
                       float ox, float oy, float oz,
                       float nx, float ny, float nz,
                       float ux, float uy, float uz)
    {
        text = txt;
        color = c;
        scale = s;
        twod = false;
        billboard = false;
        x = y = z = 0;
        mtx.CreateRBT(eavlPoint3(ox,oy,oz),
                      eavlPoint3(ox,oy,oz) - eavlVector3(nx,ny,nz),
                      eavlVector3(ux,uy,uz));

    }

    virtual void Render(eavlCamera &camera)
    {
        eavlBitmapFont *fnt = eavlBitmapFontFactory::GetDefaultFont();
        eavlTexture *tex = (eavlTexture*)fnt->userPointer;
        if (!tex)
        {
            string ftype;
            vector<unsigned char> &rawpngdata = fnt->GetRawImageData(ftype);
            if (ftype != "png")
                cerr << "Error: expected PNG type for font image data\n";
            eavlPNGImporter *pngimp = new eavlPNGImporter(&rawpngdata[0],
                                                          rawpngdata.size());
            eavlDataSet *img = pngimp->GetMesh("mesh",0);
            img->AddField(pngimp->GetField("a","mesh",0));

            tex = new eavlTexture;
            tex->CreateFromDataSet(img, false,false,false,true);
            fnt->userPointer = tex;
        }

        if (billboard)
        {
            if (fixed2Dscale)
            {
                eavlPoint3 p = camera.P * camera.V * eavlPoint3(x,y,z);

                glMatrixMode( GL_PROJECTION );
                glLoadIdentity();
                glOrtho(-1,1, -1,1, -1,1);

                glMatrixMode( GL_MODELVIEW );
                glLoadIdentity();
                mtx.CreateTranslate(p.x, p.y, -p.z);
                glMultMatrixf(mtx.GetOpenGLMatrix4x4());
                mtx.CreateScale(1./camera.aspect, 1, 1);
                glMultMatrixf(mtx.GetOpenGLMatrix4x4());
                // height given in (0,1) vert range, but we
                // are currently in -1,1 range, so scale up by 2.0
                mtx.CreateScale(2.0, 2.0, 1.0);
                glMultMatrixf(mtx.GetOpenGLMatrix4x4());
            }
            else
            {
                glMatrixMode( GL_PROJECTION );
                glLoadIdentity();
                glMultMatrixf(camera.P.GetOpenGLMatrix4x4());

                glMatrixMode( GL_MODELVIEW );
                glLoadIdentity();
                glMultMatrixf(camera.V.GetOpenGLMatrix4x4());

                mtx.CreateRBT(eavlPoint3(x,y,z),
                              eavlPoint3(x,y,z) - (camera.from-camera.at),
                              camera.up);
                glMultMatrixf(mtx.GetOpenGLMatrix4x4());

            }
        }
        else if (twod)
        {
            glMatrixMode( GL_PROJECTION );
            glLoadIdentity();
            glOrtho(0,1, 0,1, -1,1);

            glMatrixMode( GL_MODELVIEW );
            glLoadIdentity();
            mtx.CreateTranslate(x, y, 0);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());
            mtx.CreateScale(1./camera.aspect, 1, 1);
            glMultMatrixf(mtx.GetOpenGLMatrix4x4());
        }
        else
        {
            glMatrixMode( GL_PROJECTION );
            glLoadIdentity();
            glMultMatrixf(camera.P.GetOpenGLMatrix4x4());

            glMatrixMode( GL_MODELVIEW );
            glLoadIdentity();
            glMultMatrixf(camera.V.GetOpenGLMatrix4x4());

            glMultMatrixf(mtx.GetOpenGLMatrix4x4());
        }

        glColor3fv(color.c);

        //glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

        // Kerning causing overlapping polygons.  Ideally only draw pixels where alpha>0,
        // though this causes problems for alpha between 0 and 1 (unless can solve with different blend func??).
        // Simpler is to just disable depth test entirely, but then must draw all text last.
        // Or maybe draw text with two passes, once to update Z (when alpha==1) and once to draw pixels (when alpha>0).
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

        glBegin(GL_QUADS);

        float fx = 0, fy = 0, fz = 0;
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

#endif
