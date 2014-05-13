// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlImporterFactory.h"

#include "eavlADIOSImporter.h" //This must be before anything that includes mpi
#include "eavlPixieImporter.h" //This must be before anything that includes mpi
#include "eavlXGCImporter.h" //This must be before anything that includes mpi
#include "eavlNetCDFImporter.h"
#include "eavlNetCDFDecomposingImporter.h"
#include "eavlVTKImporter.h"
#include "eavlSiloImporter.h"
#include "eavlChimeraImporter.h"
#include "eavlMADNESSImporter.h"
#include "eavlBOVImporter.h"
#include "eavlPDBImporter.h"
#include "eavlPNGImporter.h"
#include "eavlCurveImporter.h"
#include "eavlLAMMPSDumpImporter.h"

#include "eavlException.h"

#include <cctype>

eavlImporter *
eavlImporterFactory::GetImporterForFile(const std::string &fn_orig)
{
    string filename(fn_orig);
    std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);

    eavlImporter *importer = NULL;

    int flen = filename.length();
    if (flen>4 && filename.substr(flen-4) == ".vtk")
    {
        importer = new eavlVTKImporter(fn_orig);
    }
    else if (flen>8 && filename.substr(flen-8) == ".madness")
    {
        importer = new eavlMADNESSImporter(fn_orig);
    }
    else if (flen>4 && filename.substr(flen-4) == ".bov")
    {
        importer = new eavlBOVImporter(fn_orig);
    }
    else if (flen>4 && filename.substr(flen-4) == ".pdb")
    {
        importer = new eavlPDBImporter(fn_orig);
    }
    else if (flen>4 && filename.substr(flen-4) == ".png")
    {
        importer = new eavlPNGImporter(fn_orig);
    }
    else if (flen>6 && filename.substr(flen-6) == ".curve")
    {
        importer = new eavlCurveImporter(fn_orig);
    }
    else if (flen>5 && filename.substr(flen-5) == ".dump")
    {
        importer = new eavlLAMMPSDumpImporter(fn_orig);
    }
#ifdef HAVE_NETCDF
    else if (flen>3 && filename.substr(flen-3) == ".nc")
    {
        importer = new eavlNetCDFImporter(fn_orig);

        // this is the auto-decomposing importer when we're ready for it
        //int NUMDOMAINS = 8;
        //importer = new eavlNetCDFDecomposingImporter(NUMDOMAINS,
        //                                             fn_orig);
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
        importer = new eavlSiloImporter(fn_orig);
    }
    else if (flen>4 && filename.substr(flen-4) == ".chi")
    {
        importer = new eavlChimeraImporter(fn_orig);
    }
#endif
#ifdef HAVE_ADIOS
    else if (flen>3 && filename.substr(flen-3) == ".bp")
    {
	if (filename.find("xgc.3d") != string::npos)
	    importer = new eavlXGCImporter(fn_orig);
	else
	    importer = new eavlADIOSImporter(fn_orig);
    }
    /*
    else if (flen>6 && filename.substr(flen-6) == ".pixie")
    {
        importer = new eavlPixieImporter(fn_orig);
    }
    */
#endif

    return importer;
}
