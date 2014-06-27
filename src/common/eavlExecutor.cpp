// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlExecutor.h"

eavlExecutor::ExecutionMode eavlExecutor::executionMode = PreferGPU;
eavlExecutor *eavlExecutor::instance = NULL;


void
eavlExecutor::real_Go()
{
    for (unsigned int i=0; i<plan.size(); i++)
    {
        //cerr << "Executing "<<opnames[i]<<endl;
        int th = eavlTimer::Start();
#ifdef HAVE_CUDA
        switch (executionMode)
        {
          case PreferGPU:
            try {
                plan[i]->GoGPU();
            }
            catch (eavlException &e)
            {
                cerr << "Warning: failed GPU, trying CPU, error was "<<e.GetErrorText()<<"\n";
                try {
                    plan[i]->GoCPU();
                }
                catch (eavlException &e2)
                {
                    cerr << "Error: both GPU and CPU ops failed\n";
                    cerr << "   GPU error was: " << e.GetErrorText() << endl;
                    cerr << "   CPU error was: " << e2.GetErrorText() << endl;
                }
            }
            break;
          case ForceGPU:
            plan[i]->GoGPU();
            break;
          case ForceCPU:
            plan[i]->GoCPU();
            break;
        }
#else
        switch (executionMode)
        {
          case PreferGPU:
            try {
                plan[i]->GoCPU();
            }
            catch (eavlException &e)
            {
                cerr << "Error: no GPU implementation, and CPU op failed\n";
                cerr << "   CPU error was: " << e.GetErrorText() << endl;
            }
            break;
          case ForceGPU:
            THROW(eavlException, "GPU support was not compiled in.");
          case ForceCPU:
            plan[i]->GoCPU();
            break;
        }
#endif
        eavlTimer::Stop(th, opnames[i]);
    }

    for (unsigned int i=0; i<plan.size(); i++)
        delete plan[i];

    plan.clear();
    opnames.clear();
}


void
eavlExecutor::real_AddOperation(eavlOperation *op, const std::string &name)
{
    plan.push_back(op);
    opnames.push_back(name);
}
