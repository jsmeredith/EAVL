// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlVTKDataSet.h"

#include "eavl.h"
#include "eavlException.h"
#include "eavlVTKImporter.h"
#include "eavlVTKExporter.h"

#include <vtkDataSet.h>
#include <vtkDataSetReader.h>
#include <vtkDataSetWriter.h>

#ifdef HAVE_VTK

#include "eavlDataSet.h"

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
vtkDataSet *ConvertEAVLToVTK(eavlDataSet *in)
{
    ostringstream stream;
    eavlVTKExporter exporter(in);
    exporter.Export(stream);
    string str = stream.str();

    // Note: VisIt does this: (I ask because we're getting a 1-byte
    // invalid read in valgrind; maybe this fixes it?):
    //vtkCharArray *charArray = vtkCharArray::New();
    //int iOwnIt = 1;  // 1 means we own it -- you don't delete it.
    //charArray->SetArray((char *) asCharTmp, asCharLengthTmp, iOwnIt);
    //reader->SetReadFromInputString(1);
    //reader->SetInputArray(charArray);

    vtkDataSetReader *rdr = vtkDataSetReader::New();
    rdr->SetReadFromInputString(1);
    rdr->SetInputString(str.c_str());

    vtkDataSet *out = rdr->GetOutput();
    rdr->Update();
    out->Register(NULL);
    rdr->Delete();
    return out;
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
eavlDataSet *ConvertVTKToEAVL(vtkDataSet *in)
{
    vtkDataSetWriter *wrtr = vtkDataSetWriter::New();
    wrtr->WriteToOutputStringOn();
    wrtr->SetFileTypeToBinary();
    wrtr->SetInputData(in);
    wrtr->Write();

    eavlVTKImporter importer(wrtr->GetOutputString(),
                             wrtr->GetOutputStringLength());
    int chunk = 0; // only one domain, of course
    string meshname = "mesh"; // unused for VTK importer; one mesh per file
    eavlDataSet *out = importer.GetMesh(meshname,chunk);
    vector<string> allvars = importer.GetFieldList(meshname);
    for (unsigned int i=0; i<allvars.size(); i++)
        out->AddField(importer.GetField(allvars[i], meshname, chunk));

    wrtr->Delete();
    return out;
}

#else

vtkDataSet *ConvertEAVLToVTK(eavlDataSet *in)
{
    THROW(eavlException, "Cannot call EAVL<->VTK conversion; VTK library not compiled in.");
}

eavlDataSet *ConvertVTKToEAVL(vtkDataSet *in)
{
    THROW(eavlException, "Cannot call EAVL<->VTK conversion; VTK library not compiled in.");
}

#endif

