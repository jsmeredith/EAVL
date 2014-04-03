#ifndef EAVL_RENDER_SURFACE_OSMESA_H
#define EAVL_RENDER_SURFACE_OSMESA_H

#include "eavlRenderSurface.h"
#include "eavlException.h"

#include "GL/gl_mangle.h"
#include "GL/osmesa.h"
#include "GL/gl.h"

#include <climits>

class eavlRenderSurfaceOSMesa : public eavlRenderSurface
{
  protected:
    OSMesaContext ctx;
    vector<unsigned char> rgba;
    vector<float> zbuff;
    int width, height;
  public:
    eavlRenderSurfaceOSMesa() : eavlRenderSurface(), width(0),height(0)
    {
        ctx = NULL;
    }
    virtual ~eavlRenderSurfaceOSMesa()
    {
        if (ctx)
            OSMesaDestroyContext(ctx);
    }
    virtual void Initialize()
    {
        ctx = OSMesaCreateContextExt( OSMESA_RGBA, 32, 0, 0, NULL );
        if (!ctx)
            THROW(eavlException, "Could not create OSMesa context");

    }
    virtual void Resize(int w, int h)
    {
        width = w;
        height = h;
        rgba.resize(w*h*4);
    }
    virtual void Activate()
    {
        if (!OSMesaMakeCurrent( ctx, &rgba[0], GL_UNSIGNED_BYTE, width, height))
            THROW(eavlException,
                  "Couldn't make framebuffer current for osmesa context");

        if (false)
        {
            int z, s, a;
            glGetIntegerv(GL_DEPTH_BITS, &z);
            glGetIntegerv(GL_STENCIL_BITS, &s);
            glGetIntegerv(GL_ACCUM_RED_BITS, &a);
            printf("Depth=%d Stencil=%d Accum=%d\n", z, s, a);
        }
    }
    virtual const unsigned char *GetRGBABuffer()
    {
        unsigned char *raw_rgbabuff;
        int format, w, h;
        OSMesaGetColorBuffer(ctx, &w, &h, &format, (void**)&raw_rgbabuff);
        // note: this is really just returning our internal rgba array
        return raw_rgbabuff;
    }
    virtual const float *GetZBuffer()
    {
        // we told mesa to use a 32-bit zbuffer, which means unsigned int
        unsigned int *raw_zbuff;
        int zbytes, w, h;
        OSMesaGetDepthBuffer(ctx, &w, &h, &zbytes, (void**)&raw_zbuff);
        // but caller wants a float....
        int npixels = w*h;
        zbuff.resize(npixels);
        for (int i=0; i<npixels; ++i)
            zbuff[i] = raw_zbuff[i] / float(UINT_MAX);
        return &zbuff[0];
    }
    virtual void Finish()
    {
        glFinish();
    }
};

#endif
