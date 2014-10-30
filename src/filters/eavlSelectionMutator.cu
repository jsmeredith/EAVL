// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlSelectionMutator.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllPoints.h"
#include "eavlNewIsoTables.h"
#include "eavlExecutor.h"
#include "eavlException.h"
#include "eavlMapOp.h"
#include "eavlPrefixSumOp_1.h"
#include "eavlReduceOp_1.h"
#include "eavlReverseIndexOp.h"
#include "eavlSimpleReverseIndexOp.h"

#define INIT(TYPE, TABLE, COUNT)                \
    {                                           \
        TABLE = new TYPE(TABLE ## _raw, COUNT); \
    }

//compare function for qsort
int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}


//eavl map functor
struct FindSelection
{
    int size;
    eavlConstArray<int> checkArray;
    FindSelection(eavlConstArray<int> *inArray, int _size) : checkArray(*inArray), size(_size) { }
    EAVL_FUNCTOR int operator()(int key)
    {
        int minIndex, maxIndex, midPoint;
        minIndex = 0;
        maxIndex = size;
        while(maxIndex >= minIndex)
        {
            midPoint = minIndex + ((maxIndex - minIndex) / 2);
            if(checkArray[midPoint] == key)
                return 1;
            else if(checkArray[midPoint] < key)
                minIndex = midPoint + 1;
            else
                maxIndex = midPoint - 1;
        }
        return 0;
    }
};


eavlSelectionMutator::eavlSelectionMutator()
{
    presorted = false;
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
    int *constArray_raw = (int *)malloc(sizeof(int *)*chosenElements->GetNumberOfTuples());
    #pragma omp parallel for
    for(int n = 0 ; n < chosenElements->GetNumberOfTuples(); n++ )
    {
        constArray_raw[n] = chosenElements->GetValue(n);

    }

    if(!presorted)
    {
        qsort(constArray_raw, chosenElements->GetNumberOfTuples(), sizeof(int), cmpfunc);
    }

    //----put elements in const array for use in fuctor
    eavlConstArray<int> *constArray;
    INIT(eavlConstArray<int>,  constArray,  chosenElements->GetNumberOfTuples());
    //--

    //----perform map with custom selection fuctor
    eavlExecutor::AddOperation(new_eavlMapOp( eavlOpArgs(arrayToOperateOn),
                                              eavlOpArgs(mapOutput),
                                              FindSelection(constArray, chosenElements->GetNumberOfTuples())
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

    delete interestingIndexes;
	delete totalInterestingIds;
	delete exclusiveScanOut;
	delete mapOutput;
	delete constArray;
	free(constArray_raw);
}
