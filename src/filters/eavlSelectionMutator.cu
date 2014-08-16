// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlSelectionMutator.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllPoints.h"
#include "eavlExecutor.h"
#include "eavlException.h"
#include "eavlMapOp.h"
#include "eavlPrefixSumOp_1.h"
#include "eavlReduceOp_1.h"
#include "eavlReverseIndexOp.h"
#include "eavlSimpleReverseIndexOp.h"

//compare function for qsort
int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}


//eavl map functor
struct FindSelection
{
    eavlIntArray *checkArray;
    FindSelection(eavlIntArray *inArray) : checkArray(inArray) { }
    EAVL_FUNCTOR int operator()(int x) 
    { 
        for(int i = 0; i < checkArray->GetNumberOfTuples(); i++)
        {   
            if(checkArray->GetValue(i) == x)
            {
                return 1; 
            }
            else if(checkArray->GetValue(i) > x)
            {
               return 0;
            }
        }
        return 0;
    }
};


eavlSelectionMutator::eavlSelectionMutator()
{
}

    
void
eavlSelectionMutator::Execute()
{
    int inCellSetIndex = dataset->GetCellSetIndex(cellsetname);
    eavlCellSet *inCells = dataset->GetCellSet(cellsetname);
    eavlField *inField = dataset->GetField(fieldname);
    eavlArray *arrayToOperateOn = inField->GetArray();
    eavlIntArray *mapOutput = new eavlIntArray("inSelection", 1, arrayToOperateOn->GetNumberOfTuples());

    //----dump interesting paticles to array that qsort can use
    int array[chosenElements->GetNumberOfTuples()];
    for(int n = 0 ; n < chosenElements->GetNumberOfTuples(); n++ ) 
    {
        array[n] = chosenElements->GetValue(n);

    }
    qsort(array, chosenElements->GetNumberOfTuples(), sizeof(int), cmpfunc);
    for(int n = 0 ; n < chosenElements->GetNumberOfTuples(); n++ ) 
    {
        chosenElements->SetValue(n, array[n]);
    }
    //--
   

    //----perform map with custom selection fuctor
    eavlExecutor::AddOperation(new_eavlMapOp( eavlOpArgs(arrayToOperateOn),
                                              eavlOpArgs(mapOutput),
                                              FindSelection(chosenElements)
                                             ),
                                "test if in selection"
                               );
    eavlExecutor::Go();
    //--


	//----perform exclusive scan on 0/1 array
	eavlIntArray *exclusiveScanOut = new eavlIntArray("exclusive scan out", 1,  mapOutput->GetNumberOfTuples());
    eavlExecutor::AddOperation(new eavlPrefixSumOp_1(
                                                      mapOutput,
                                                      exclusiveScanOut,
                                                      false
                                                     ),
                                "scan to generate indexes of items to keep"
                               );
    eavlExecutor::Go();
    //--
	

	//----perform reduction
	eavlIntArray *totalInterestingIds = new eavlIntArray("totalInterestingIds", 1, 1);
    eavlExecutor::AddOperation(new eavlReduceOp_1<eavlAddFunctor<int> >
                                                (mapOutput,
                                                 totalInterestingIds,
                                                 eavlAddFunctor<int>()
                                                 ),
                                "sumreduce to count output vals"
                               );
    eavlExecutor::Go();
    //--
    

    //----perform reverse index
    eavlIntArray *interestingIndexes = new eavlIntArray("interestingIndexes", 
                                                        1, 
                                                        totalInterestingIds->GetValue(0)
                                                       ); 

    if(totalInterestingIds->GetValue(0) > 0)
    {
        eavlExecutor::AddOperation(new eavlSimpleReverseIndexOp
                                                   (
                                                    mapOutput,
                                                    exclusiveScanOut,
                                                    interestingIndexes
                                                   ),
                                "generate reverse lookup"
                               );
        eavlExecutor::Go();
    }
    //--
    
    
    //----Create cell set from the selection
    unsigned int numnewcells = totalInterestingIds->GetValue(0);
    eavlExplicitConnectivity conn;
    for (unsigned int i=0; i<numnewcells; ++i)
    {
        eavlCell cell = inCells->GetCellNodes(interestingIndexes->GetValue(i));
        conn.AddElement(cell);
    }

    eavlCellSetExplicit *subset = new eavlCellSetExplicit(string("selection_of_")+inCells->GetName(),
                                                          inCells->GetDimensionality());
    subset->SetCellNodeConnectivity(conn);
    dataset->AddCellSet(subset);

    for (int i=0; i<dataset->GetNumFields(); i++)
    {
        eavlField *f = dataset->GetField(i);
        if (f->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            f->GetAssocCellSet() == dataset->GetCellSet(inCellSetIndex)->GetName())
        {
            int numcomp = f->GetArray()->GetNumberOfComponents();
            eavlFloatArray *a = new eavlFloatArray(
                                 string("selection_of_")+f->GetArray()->GetName(),
                                 numcomp, numnewcells);
            for (unsigned int j=0; j < numnewcells; j++)
            {
                int e = interestingIndexes->GetValue(j);
                for (int k=0; k<numcomp; ++k)
                    a->SetComponentFromDouble(j,k, f->GetArray()->GetComponentAsDouble(e,k));
            }

            eavlField *newfield = new eavlField(f->GetOrder(), a,
                                                eavlField::ASSOC_CELL_SET,
                                                subset->GetName());
            dataset->AddField(newfield);
        }
    }
    //--
}

