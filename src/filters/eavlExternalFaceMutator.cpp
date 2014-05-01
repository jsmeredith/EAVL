// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlExternalFaceMutator.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlCellComponents.h"
#include "eavlCellSetAllFacesOfStructured.h"
#include "eavlCellSetAllFacesOfExplicit.h"

eavlExternalFaceMutator::eavlExternalFaceMutator()
{
}

void
eavlExternalFaceMutator::Execute()
{
    int inCellSetIndex = dataset->GetCellSetIndex(cellsetname);
    eavlCellSet *inCells = dataset->GetCellSet(cellsetname);

    int nf = inCells->GetNumFaces();
    vector<int> faceCount(nf,0);
    vector<int> faceCell(nf,-1);
    int nc = inCells->GetNumCells();
    for (int i=0; i<nc; i++)
    {
        eavlCell faces = inCells->GetCellFaces(i);
        for (int j=0; j<faces.numIndices; j++)
        {
            faceCount[faces.indices[j]] ++;
            faceCell[faces.indices[j]] = i;
        }
    }

    int n_ext = 0;
    for (int i=0; i<nf; i++)
    {
        if (faceCount[i] == 1)
            n_ext++;
    }

    
    ///\todo: UGH: I don't like this ugly logic here.
    ///       <aybe eavlCellSet should just have a GetFace method
    ///       instead of having to create an allFaces cell set?
    ///       Upside: gets rid of logic like this.
    ///       Downside: what is the field data stored on??? plus, these
    ///                 all-faces structures have no problem-sized data
    ///                 so they're actually rather efficient.
    ///       Best solution: probably just find a way to simplify this.
    ///       Possible implementation: GetExternalFaces gets
    ///       handed a face cell set as well as the volumetric one???
    eavlCellSetAllStructured *cellsAllStruc =
        dynamic_cast<eavlCellSetAllStructured*>(inCells);
    eavlCellSetExplicit *cellsExplicit =
        dynamic_cast<eavlCellSetExplicit*>(inCells);
    eavlCellSet *faceCellSet = NULL;

    if (cellsAllStruc)
    {
        eavlCellSetAllFacesOfStructured *faceCellSetAllSruc =
            new eavlCellSetAllFacesOfStructured(cellsAllStruc);
        faceCellSet = faceCellSetAllSruc;
    }
    else if (cellsExplicit)
    {
        eavlCellSetAllFacesOfExplicit *faceCellSetAllExpl =
            new eavlCellSetAllFacesOfExplicit(cellsExplicit);
        faceCellSet = faceCellSetAllExpl;
    }
    else
    {
        // do nothing for now....
        //THROW(eavlException, "Unsupported cell set type");
    }

    eavlCellSetExplicit *outCells =
        new eavlCellSetExplicit(string("extface_of_")+inCells->GetName(), 2);
    eavlExplicitConnectivity conn;
    for (int i=0; i<nf; i++)
    {
        if (faceCount[i] == 1)
        {
            eavlCell face = faceCellSet->GetCellNodes(i);
            conn.shapetype.push_back(face.type);
            conn.connectivity.push_back(face.numIndices);
            for (int j=0; j<face.numIndices; j++)
            {
                conn.connectivity.push_back(face.indices[j]);
            }
        }
    }
    outCells->SetCellNodeConnectivity(conn);
    dataset->AddCellSet(outCells);

    // copy any cell fields
    int nOldFields = dataset->GetNumFields();
    for (int i=0; i < nOldFields; ++i)
    {
        eavlField *inField = dataset->GetField(i);
        if (inField->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            inField->GetAssocCellSet() == dataset->GetCellSet(inCellSetIndex)->GetName())
        {
            eavlArray *inArray = inField->GetArray();
            int n = n_ext;
            int nc = inArray->GetNumberOfComponents();
            // I guess it's most appropriate to re-use the input 
            // field name directly?
            eavlFloatArray *outArray =
                new eavlFloatArray(/*string("extface_of_") + */
                                   inArray->GetName(), nc);
            outArray->SetNumberOfTuples(n);
            int outindex = 0;
            for (int i=0; i<nf; i++)
            {
                if (faceCount[i] == 1)
                {
                    for (int c=0; c<nc; c++)
                    {
                        outArray->SetComponentFromDouble(outindex, c,
                                inArray->GetComponentAsDouble(faceCell[i],c));
                    }
                    outindex++;
                }
            }
            eavlField *outField = new eavlField(0, outArray,
                                                eavlField::ASSOC_CELL_SET,
                                                outCells->GetName());            
            dataset->AddField(outField);
        }
    }
}
