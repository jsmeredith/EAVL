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
    }

    virtual void AddRectangle(float x, float y, 
                              float w, float h,
                              eavlColor c)
    {
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
};

#endif
