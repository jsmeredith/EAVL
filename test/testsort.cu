// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl.h"
#include "eavlCUDA.h"
#include "eavlRadixSortOp.h"
#include "eavlExecutor.h"
#include <algorithm>
#include <time.h>

using namespace std;
void printUsage()
{
    cout <<"\nUsage: testsort numElements [-cpu]"<<endl;
}
int main(int argc, char *argv[])
{
    try 
    {   
        bool cpu = false;
        if(argc > 3 || argc == 1)
        {
            printUsage();
            exit(0);
        }
        int size = atoi(argv[1]);
        if(argc == 3)
        {
            if(strcmp (argv[2],"-cpu") == 0)
            {
                cpu = true;
            }
            else 
            {
                cout<<"Unknown option: "<<argv[2]<<endl;
                cout<<"Defaulting to prefer GPU"<<endl;
                printUsage();
            }
        }
        

        if(size < 1) 
        {
            size = 1000000;
            cout<<"Invalid size. Using size of 1,000,000.\n";
        }
        if(cpu) eavlExecutor::SetExecutionMode(eavlExecutor::ForceCPU);

        cout<<"Sorting "<<size<<" random elements"<<endl;
        uint * verify = new uint[size];
        eavlIntArray * keys   =  new eavlIntArray("",1,size);
        eavlIntArray * values =  new eavlIntArray("",1,size);
        srand (time(NULL));
        for(int j = 0; j<1; j++)
        {
            for( int i = 0; i<size ; i++)
            {
                uint val = rand();
                if(i == 0) val = UINT_MAX; // Throw an edge case in there
                keys->SetValue(i,val);
                values->SetValue(i,val);
                verify[i] = val;
            }
            cout<<endl;

            int tGPU  = eavlTimer::Start();
            eavlExecutor::AddOperation(new_eavlRadixSortOp(eavlOpArgs(keys),
                                                           eavlOpArgs(values), false),
                                                           "");
            eavlExecutor::Go();
            if(cpu) cout<<"CPU SORT   RUNTIME: "<<eavlTimer::Stop(tGPU,"")<<endl;
            else cout<<"GPU SORT   RUNTIME: "<<eavlTimer::Stop(tGPU,"")<<endl;

            int tCPU  = eavlTimer::Start();
            std::sort(verify, verify + size);
            cout<<"STD::SORT  RUNTIME: "<<eavlTimer::Stop(tCPU,"")<<endl;

            cout<<"Verifying"<<endl;
            bool sorted = true;
            for( int i = 0; i < size; i++)
            {
                if( verify[i] != (uint)keys->GetValue(i) || verify[i] != (uint)values->GetValue(i) )
                {
                    cout<<"Baseline varies at "<<i<<" std::Sort "<<verify[i]<<" Key "
                        <<(uint)keys->GetValue(i)<<" value "<<(uint)values->GetValue(i)<<endl;
                    cout<<"Suppressing the rest of the output."<<endl;
                    sorted = false;
                    break; 
                }

            }
            if(sorted) cout<<"Verified.\n";
            else cout<<"Verification failed.\n";
        }
        


        
    }
    catch (const eavlException &e)
    {
        cerr << e.GetErrorText() << endl;
        printUsage();
        return 1;
    }


    return 0;
}