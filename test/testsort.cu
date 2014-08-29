// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlRadixSortOp.h"
#include "eavlExecutor.h"
#include <algorithm>

using namespace std;
int main(int argc, char *argv[])
{
    try
    {   



        int size = atoi(argv[1]);
        cout<<"Sorting "<<size<<" random elements"<<endl;
       


        int * verify = new int[size];

        int s = size;
        eavlIntArray * ins =  new eavlIntArray("",1,s);
        eavlIntArray * outs =  new eavlIntArray("",1,s);
        for(int j = 0; j<1; j++)
        {
            for( int i = 0; i<s ; i++)
            {
                int val = rand() % 20;
                ins->SetValue(i,val);
                verify[i] = val;
            }
            cout<<endl;

            int tGPU  = eavlTimer::Start();
            eavlExecutor::AddOperation(new_eavlRadixSortOp(eavlOpArgs(ins),
                                                             eavlOpArgs(outs)),
                                                             "");
            eavlExecutor::Go();
            cout<<"GPU SORT  RUNTIME: "<<eavlTimer::Stop(tGPU,"")<<endl;

            int tCPU  = eavlTimer::Start();
            std::sort(verify, verify + size);
            cout<<"CPU SORT  RUNTIME: "<<eavlTimer::Stop(tCPU,"")<<endl;

            cout<<"Verify"<<endl;
            for( int i = 0; i<s ; i++)
            {
                if( verify[i] != ins->GetValue(i))  cout<<"Baseline varies at "<<i<<" vals b "<<verify[i]<<" != "<< ins->GetValue(i)<<" "<<endl;

            }
            cout<<endl;
        }
        


        
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        cerr << "\nUsage: number of elements to sort "<<endl;
        return 1;
    }


    return 0;
}