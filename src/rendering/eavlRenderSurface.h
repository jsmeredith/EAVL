#ifndef EAVL_RENDER_SURFACE_H
#define EAVL_RENDER_SURFACE_H


class eavlRenderSurface
{
  protected:
  public:
    eavlRenderSurface()
    {
    }
    virtual void Initialize() = 0;
    virtual void Resize(int w, int h) = 0;
    virtual void Activate() = 0;
    virtual void Finish() = 0;
    //virtual unsigned char *GetRGBA()
};

#endif
