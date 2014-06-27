// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_COMPOSITOR_H
#define EAVL_COMPOSITOR_H

#include "eavlConfig.h"
#ifdef HAVE_MPI
#include "mpi.h"

void ParallelZComposite(const MPI_Comm &comm,
                        int npixels,
                        const float *inz, const unsigned char *inrgba,
                        float *outz, unsigned char *outrgba,
                        unsigned char bgr, unsigned char bgg, unsigned char bgb);

 
#endif

#endif
