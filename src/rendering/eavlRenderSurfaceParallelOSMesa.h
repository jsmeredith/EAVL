#ifndef EAVL_RENDER_SURFACE_PARALLEL_OSMESA_H
#define EAVL_RENDER_SURFACE_PARALLEL_OSMESA_H

#include "eavlRenderSurfaceOSMesa.h"
#include "eavlColor.h"
#include "eavlCompositor.h"

class eavlRenderSurfaceParallelOSMesa : public eavlRenderSurfaceOSMesa
{
  protected:
    vector<unsigned char> composited_rgba;
    vector<float> composited_zbuff;
    boost::mpi::communicator &comm;
    eavlColor bg;
  public:
    eavlRenderSurfaceParallelOSMesa(boost::mpi::communicator &mpicomm,
                                    eavlColor &bgcolor)
        : eavlRenderSurfaceOSMesa(), comm(mpicomm), bg(bgcolor)
    {
    }
    virtual void Finish()
    {
        eavlRenderSurfaceOSMesa::Finish();

        int npixels = width * height;
        composited_rgba.resize(4*npixels);
        composited_zbuff.resize(npixels);

        const unsigned char *rgba = GetRGBABuffer();
        const float *zbuff = GetZBuffer();

        MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);                    
        ParallelZComposite(comm,
                           npixels,
                           zbuff, rgba,
                           &composited_zbuff[0], &composited_rgba[0],
                           bg.GetComponentAsByte(0),
                           bg.GetComponentAsByte(1),
                           bg.GetComponentAsByte(2));

        glDrawPixels(width,height, GL_RGBA,GL_UNSIGNED_BYTE, &composited_rgba[0]);
    }
};


#endif

