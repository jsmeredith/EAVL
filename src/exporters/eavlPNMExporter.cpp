// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlPNMExporter.h"
#include <iostream>

void
eavlPNMExporter::Export(ostream &out, int w, int h, eavlByteArray *array)
{
    if(array->GetNumberOfComponents() < 3)
        THROW(eavlException, "Error: PNM files need an R, G, and B component!");

    const byte *tuple; 

    out<<"P6"<<endl<<w<<" "<<h<<endl<<255<<endl;
    for(int i = h-1; i >= 0; i--)
        for(int j = 0; j < w; j++)
        {
            tuple = array->GetTuple(i*w + j);
            out<<tuple[0]<<tuple[1]<<tuple[2];
        }
}

void
eavlPNMExporter::ConvertAndExport(ostream &out, int w, int h, eavlFloatArray *array)
{
    int ncomponents = array->GetNumberOfComponents();
    int ntuples     = array->GetNumberOfTuples();
    float min, max;
    double value;


    max = min = array->GetComponentAsDouble(0, 0);

    for(int i = 0; i < ntuples; i++)
        for(int j = 0; j < ncomponents; j++)
        {
            value = array->GetComponentAsDouble(i, j);
            if(value > max)
                max = value;
            else if(value < min)
                min = value;
        }

    const float *tuple;
    byte  *convertedTuple = new byte [ncomponents];

    out<<"P6"<<endl<<w<<" "<<h<<endl<<255<<endl;
    for(int i = h-1; i >= 0; i--)
        for(int j = 0; j < w; j++)
        {
            tuple = array->GetTuple(i*w + j);
            for(int c = 0; c < 3; c++)
                convertedTuple[c] = (byte)((tuple[c]-min)/(max-min)*255);

            out<<convertedTuple[0]
               <<convertedTuple[1]
               <<convertedTuple[2];
        }

    delete [] convertedTuple;
}
