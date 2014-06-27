// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
// This file contains code from VisIt, (c) 2000-2012 LLNS.  See COPYRIGHT.txt.
#include "eavlCompositor.h"

#include "STL.h"

#ifdef HAVE_MPI
// ****************************************************************************
// Method:  ParallelZComposite
//
// Purpose:
///   
//
// Note: modified from a chain of VisIt/MeshTV code written and modified over
//       the years by Kat Price, Mark Miller, and other developers
//
// Arguments:
//   
//
// Programmer:  Jeremy Meredith
// Creation:    January 24, 2013
//
// Modifications:
// ****************************************************************************

struct Pixel
{
    float         z;
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};;

static MPI_Datatype  mpiTypePixel;
static MPI_Op        mpiOpMergePixelBuffers;
static unsigned char local_bg[3];


static void
MergePixelBuffersOp(void *ibuf, void *iobuf, int *count, MPI_Datatype *)
{
    Pixel *in_pixels    = (Pixel *) ibuf;
    Pixel *inout_pixels = (Pixel *) iobuf;

    const int amount = *count;
    const unsigned char local_bg_r = local_bg[0];
    const unsigned char local_bg_g = local_bg[1];
    const unsigned char local_bg_b = local_bg[2];

    for (int i = 0; i < amount; i++)
    {
        ///\todo: since we have alpha, do we want to do some
        /// smarter sort of blending based on Z values?
        /// this code assumes full opacity.
        if ( in_pixels[i].z < inout_pixels[i].z )
        {
            inout_pixels[i] = in_pixels[i];
        }
        else if (in_pixels[i].z == inout_pixels[i].z)
        {
            if ((inout_pixels[i].r == local_bg_r) &&
                (inout_pixels[i].g == local_bg_g) &&
                (inout_pixels[i].b == local_bg_b))
            {
                // Since 'inout' is background color, take whatever
                // is in 'in' even if it too is background color
                inout_pixels[i].r = in_pixels[i].r;
                inout_pixels[i].g = in_pixels[i].g;
                inout_pixels[i].b = in_pixels[i].b;
                inout_pixels[i].a = in_pixels[i].a;
            }
            else if ((in_pixels[i].r != local_bg_r) || 
                     (in_pixels[i].g != local_bg_g) || 
                     (in_pixels[i].b != local_bg_b))
            {
                // Neither 'inout' nor 'in' is the background color.
                // So, average them.
                float newr = float(in_pixels[i].r) + float(inout_pixels[i].r); 
                float newg = float(in_pixels[i].g) + float(inout_pixels[i].g); 
                float newb = float(in_pixels[i].b) + float(inout_pixels[i].b); 
                float newa = float(in_pixels[i].a) + float(inout_pixels[i].a); 
                inout_pixels[i].r = (unsigned char) (newr * 0.5); 
                inout_pixels[i].g = (unsigned char) (newg * 0.5);
                inout_pixels[i].b = (unsigned char) (newb * 0.5); 
                inout_pixels[i].a = (unsigned char) (newa * 0.5); 
            }
        }
    }
}


static void 
InitializeMPIStuff(void)
{
    const int n = 5;
    int          lengths[n]       = {1, 1, 1, 1, 1};
    MPI_Aint     displacements[n] = {0, 0, 0, 0, 0};
    MPI_Datatype types[n] = {MPI_FLOAT,
                             MPI_UNSIGNED_CHAR,
                             MPI_UNSIGNED_CHAR,
                             MPI_UNSIGNED_CHAR,
                             MPI_UNSIGNED_CHAR};

    // create the MPI data type for Pixel
    Pixel onePixel;
    MPI_Address(&onePixel.z, &displacements[0]);
    MPI_Address(&onePixel.r, &displacements[1]);
    MPI_Address(&onePixel.g, &displacements[2]);
    MPI_Address(&onePixel.b, &displacements[3]);
    MPI_Address(&onePixel.a, &displacements[4]);
    for (int i = n-1; i >= 0; i--)
        displacements[i] -= displacements[0];
    MPI_Type_struct(n, lengths, displacements, types,
                    &mpiTypePixel);
    MPI_Type_commit(&mpiTypePixel);

    // and the merge operation for a reduction
    MPI_Op_create((MPI_User_function *)MergePixelBuffersOp, 1,
                  &mpiOpMergePixelBuffers);
}

static void
FinalizeMPIStuff(void)
{
    MPI_Op_free(&mpiOpMergePixelBuffers);
    MPI_Type_free(&mpiTypePixel);
}

void
ParallelZComposite(const MPI_Comm &comm,
                   int npixels,
                   const float *inz, const unsigned char *inrgba,
                   float *outz, unsigned char *outrgba,
                   unsigned char bgr, unsigned char bgg, unsigned char bgb)
{
    static bool MPIStuffInitialized = false;
    if (!MPIStuffInitialized)
    {
        InitializeMPIStuff();
        MPIStuffInitialized = true;
    }

    const int chunksize = 1 << 20;
    std::vector<Pixel> inpixels(chunksize);
    std::vector<Pixel> outpixels(chunksize);

    local_bg[0] = bgr;
    local_bg[1] = bgg;
    local_bg[2] = bgb;

    //cerr << "merging "<<npixels<<" pixels, bg="<<int(bgr)<<","<<int(bgg)<<","<<int(bgb)<<"\n";
    //cerr << "inpixel[0] = "<<int(inrgba[0])<<","<<int(inrgba[1])<<","<<int(inrgba[2])<<","<<int(inrgba[3])<<"\n";
    //cerr << "inzbuff[0] = "<<inz[0]<<endl;

    int i_in = 0, i_out = 0;
    while (npixels > 0)
    {
        int chunk = npixels < chunksize ? npixels : chunksize;

        for (int i=0; i<chunk; ++i, ++i_in)
        {
            inpixels[i].z = inz[i_in];
            inpixels[i].r = inrgba[i_in*4 + 0];
            inpixels[i].g = inrgba[i_in*4 + 1];
            inpixels[i].b = inrgba[i_in*4 + 2];
            inpixels[i].a = inrgba[i_in*4 + 3];
        }

        int err = MPI_Allreduce(&inpixels[0],  &outpixels[0], chunk,
                                mpiTypePixel, mpiOpMergePixelBuffers, comm);
        if (err != MPI_SUCCESS)
        {
            int errclass;
            MPI_Error_class(err,&errclass);
            char err_buffer[4096];
            int resultlen;
            MPI_Error_string(err,err_buffer,&resultlen);
            cerr << err_buffer << endl;
        }


        for (int i=0; i<chunk; ++i, ++i_out)
        {
            outz[i_out]          = outpixels[i].z;
            outrgba[i_out*4 + 0] = outpixels[i].r;
            outrgba[i_out*4 + 1] = outpixels[i].g;
            outrgba[i_out*4 + 2] = outpixels[i].b;
            outrgba[i_out*4 + 3] = outpixels[i].a;
        }

        npixels -= chunk;
    }

}
#endif

