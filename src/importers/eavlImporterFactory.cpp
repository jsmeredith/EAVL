// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlImporterFactory.h"

#include "eavlADIOSImporter.h" //This must be before anything that includes mpi
#include "eavlPixieImporter.h" //This must be before anything that includes mpi
#include "eavlNetCDFImporter.h"
#include "eavlNetCDFDecomposingImporter.h"
#include "eavlVTKImporter.h"
#include "eavlSiloImporter.h"
#include "eavlChimeraImporter.h"
#include "eavlMADNESSImporter.h"
#include "eavlBOVImporter.h"
#include "eavlPDBImporter.h"

#include "eavlException.h"

eavlImporter *
eavlImporterFactory::GetImporterForFile(const std::string &filename)
{
    eavlImporter *importer = NULL;

    int flen = filename.length();
    if (flen>4 && filename.substr(flen-4) == ".vtk")
    {
        importer = new eavlVTKImporter(filename);
    }
    else if (flen>8 && filename.substr(flen-8) == ".madness")
    {
        importer = new eavlMADNESSImporter(filename);
    }
    else if (flen>4 && filename.substr(flen-4) == ".bov")
    {
        importer = new eavlBOVImporter(filename);
    }
    else if (flen>4 && filename.substr(flen-4) == ".pdb")
    {
        importer = new eavlPDBImporter(filename);
    }
#ifdef HAVE_NETCDF
    else if (flen>3 && filename.substr(flen-3) == ".nc")
    {
        importer = new eavlNetCDFImporter(filename);

        // this is the auto-decomposing importer when we're ready for it
        //int NUMDOMAINS = 8;
        //importer = new eavlNetCDFDecomposingImporter(NUMDOMAINS,
        //                                             filename);
    }
#endif
#ifdef HAVE_HDF5
    else if (flen>3 && filename.substr(flen-3) == ".h5")
    {
        THROW(eavlException,"Error: HDF5 not implemented yet\n");
    }
#endif
#ifdef HAVE_SILO
    else if (flen>5 && filename.substr(flen-5) == ".silo")
    {
        importer = new eavlSiloImporter(filename);
    }
    else if (flen>4 && filename.substr(flen-4) == ".chi")
    {
        importer = new eavlChimeraImporter(filename);
    }
#endif
#ifdef HAVE_ADIOS
    else if (flen>3 && filename.substr(flen-3) == ".bp")
    {
        importer = new eavlADIOSImporter(filename);
    }
    else if (flen>6 && filename.substr(flen-6) == ".pixie")
    {
        importer = new eavlPixieImporter(filename);
    }
#endif

    return importer;
}
