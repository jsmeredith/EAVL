// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_IMPORTER_FACTORY_H
#define EAVL_IMPORTER_FACTORY_H

#include "eavlImporter.h"

// ****************************************************************************
// Class:  eavlImporterFactory
//
// Purpose:
///   Determine and create the appropriate importer for a given file.
//
// Programmer:  Jeremy Meredith
// Creation:    July 20, 2012
//
// Modifications:
// ****************************************************************************
class eavlImporterFactory
{
  public:
    static eavlImporter *GetImporterForFile(const std::string &filename);
};

#endif
