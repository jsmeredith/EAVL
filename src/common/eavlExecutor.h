// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_EXECUTOR_H
#define EAVL_EXECUTOR_H

// ****************************************************************************
// Class:  eavlExecutor
//
// Purpose:
///   Execute a sequence of eavlOperations.  For now, this is simplistic,
///   but eventually could incorporate heterogeneous and overlapped
///   execution, or other types of intelligence.  When it executes a plan,
///   it can also take other actions (like collecting detailed timing).
//
// Programmer:  Jeremy Meredith, Dave Pugmire, Sean Ahern, Rob Sisneros
// Creation:    August 29, 2011
//
// ****************************************************************************

#include "STL.h"
#include "eavlOperation.h"
#include "eavlConfig.h"
#include "eavlException.h"
#include "eavlTimer.h"

class eavlExecutor
{
  public:
    enum ExecutionMode
    {
        PreferGPU,
        ForceGPU,
        ForceCPU
    };
  public:
    static void SetExecutionMode(ExecutionMode em)
    {
        Instance()->executionMode = em;
    }
    static ExecutionMode GetExecutionMode()
    {
        return Instance()->executionMode;
    }
    static void Go()
    {
        Instance()->real_Go();
    }
    static void AddOperation(eavlOperation *op,
                             const std::string &name)
    {
        Instance()->real_AddOperation(op,name);
    }


  protected:
    static eavlExecutor *Instance()
    {
        if (!instance)
            instance = new eavlExecutor;
        return instance;
    }
    void real_Go();
    void real_AddOperation(eavlOperation *op, const std::string &name);
    
  protected:
    static eavlExecutor    *instance;
    static ExecutionMode    executionMode;
    vector<eavlOperation *> plan;
    vector<string>          opnames;
};

#endif
