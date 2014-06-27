// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlTimer.h"
#include <stdio.h>
#include <algorithm>
#include <ctime>

#include "eavl.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlCUDA.h"

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

using std::cerr;
using std::endl;
using std::max;

// ----------------------------------------------------------------------------

eavlTimer *eavlTimer::instance = NULL;

// ----------------------------------------------------------------------------
static double
DiffTime(const struct TIMEINFO &startTime, const struct TIMEINFO &endTime)
{
#if defined(_WIN32)
    // 
    // Figure out how many milliseconds between start and end times 
    //
    int ms = (int) difftime(endTime.time, startTime.time);
    if (ms == 0)
    {
        ms = endTime.millitm - startTime.millitm;
    }
    else
    {
        ms =  ((ms - 1) * 1000);
        ms += (1000 - startTime.millitm) + endTime.millitm;
    }

    return (ms/1000.);
#else
    double seconds = double(endTime.tv_sec - startTime.tv_sec) + 
                     double(endTime.tv_usec - startTime.tv_usec) / 1000000.;
                     
    return seconds;
#endif
}

static void
GetCurrentTimeInfo(struct TIMEINFO &timeInfo)
{
#if defined(_WIN32)
    _ftime(&timeInfo);
#else
    gettimeofday(&timeInfo, 0);
#endif
}



// ****************************************************************************
//  Constructor:  eavlTimer::eavlTimer
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
eavlTimer::eavlTimer()
{
    // Initialize some timer methods and reserve some space.
    startTimes.reserve(1000);
    timeLengths.reserve(1000);
    descriptions.reserve(1000);
    currentActiveTimers = 0;
    suspended = false;
}

// ****************************************************************************
//  Destructor:  
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
eavlTimer::~eavlTimer()
{
    // nothing to do
}

// ****************************************************************************
//  Method:  eavlTimer::Instance
//
//  Purpose:
///   Return the timer singleton.
//
//  Arguments:
//    
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
eavlTimer *eavlTimer::Instance()
{
    if (!instance)
    {
        instance = new eavlTimer;
    }
    return instance;
}

// ****************************************************************************
//  Method:  eavlTimer::Start
//
//  Purpose:
///   Start a timer, and return a handle.
//
//  Arguments:
//    none
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
int eavlTimer::Start()
{
    return Instance()->real_Start();
}

// ****************************************************************************
//  Method:  eavlTimer::Stop
//
//  Purpose:
///   Stop a timer and add its length to our list.
//
//  Arguments:
//    handle       a timer handle returned by eavlTimer::Start
//    desription   a description for the event timed
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
double eavlTimer::Stop(int handle, const std::string &description)
{
    return Instance()->real_Stop(handle, description);
}

// ****************************************************************************
//  Method:  eavlTimer::Insert
//
//  Purpose:
///   Add a user-generated (e.g. calculated) timing to the list
//
//  Arguments:
//    desription   a description for the event timed
//    value        the runtime to insert
//
//  Programmer:  Jeremy Meredith
//  Creation:    October 22, 2007
//
// ****************************************************************************
void eavlTimer::Insert(const std::string &description, double value)
{
    Instance()->real_Insert(description, value);
}

// ****************************************************************************
//  Method:  eavlTimer::Dump
//
//  Purpose:
///   Add timings to on ostream.
//
//  Arguments:
//    out        the stream to print to.
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
void eavlTimer::Dump(std::ostream &out)
{
    return Instance()->real_Dump(out);
}

// ****************************************************************************
//  Method:  eavlTimer::real_Start
//
//  Purpose:
///   the true start routine
//
//  Arguments:
//    none
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
int eavlTimer::real_Start()
{
    if (suspended)
        return -1;

#pragma omp barrier
#ifdef HAVE_CUDA
    if (eavlExecutor::GetExecutionMode() != eavlExecutor::ForceCPU)
    {
        cudaThreadSynchronize();
        CUDA_CHECK_ERROR();
    }
#endif

    int handle = startTimes.size();
    currentActiveTimers++;

    struct TIMEINFO t;
    GetCurrentTimeInfo(t);
    startTimes.push_back(t);

    return handle;
}

// ****************************************************************************
//  Method:  eavlTimer::real_Stop
//
//  Purpose:
///   the true stop routine
//
//  Arguments:
//    handle       a timer handle returned by eavlTimer::Start
//    desription   a description for the event timed
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
double eavlTimer::real_Stop(int handle, const std::string &description)
{
    if (suspended || handle<0)
        return 0;

#pragma omp barrier
#ifdef HAVE_CUDA
    if (eavlExecutor::GetExecutionMode() != eavlExecutor::ForceCPU)
    {
        cudaThreadSynchronize();
        CUDA_CHECK_ERROR();
    }
#endif

    if ((unsigned int)handle > startTimes.size())
    {
        cerr << "Invalid timer handle '"<<handle<<"'\n";
        exit(1);
    }

    struct TIMEINFO t;
    GetCurrentTimeInfo(t);
    double length = DiffTime(startTimes[handle], t);
    timeLengths.push_back(length);

    char str[2048];
    sprintf(str, "%*s%s", currentActiveTimers*3, " ", description.c_str());
    descriptions.push_back(str);

    currentActiveTimers--;
    return length;
}

// ****************************************************************************
//  Method:  eavlTimer::real_Insert
//
//  Purpose:
///   the true insert routine
//
//  Arguments:
//    desription   a description for the event timed
//    value        the run time to insert
//
//  Programmer:  Jeremy Meredith
//  Creation:    October 22, 2007
//
// ****************************************************************************
void eavlTimer::real_Insert(const std::string &description, double value)
{
#if 0 // can disable inserting just to make sure it isn't broken
    cerr << description << " " << value << endl;
#else
    timeLengths.push_back(value);

    char str[2048];
    sprintf(str, "%*s[%s]",
            (currentActiveTimers+1)*3, " ", description.c_str());
    descriptions.push_back(str);
#endif
}

// ****************************************************************************
//  Method:  eavlTimer::real_Dump
//
//  Purpose:
///   the true dump routine
//
//  Arguments:
//    out        the stream to print to.
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  9, 2004
//
// ****************************************************************************
void eavlTimer::real_Dump(std::ostream &out)
{
    size_t maxlen = 0;
    for (unsigned int i=0; i<descriptions.size(); i++)
        maxlen = max(maxlen, descriptions[i].length());

    out << "\nTimings\n-------\n";
    for (unsigned int i=0; i<descriptions.size(); i++)
    {
        char desc[10000];
        sprintf(desc, "%-*s", (int)maxlen, descriptions[i].c_str());
        out << desc << " took " << timeLengths[i] << endl;
    }
}



// ****************************************************************************
// Method:  eavlTimer::Suspend
//
// Purpose:
///   Temporarily suspend collecting timings. (Timings have some performance
///   and synchronization penalties, which suspending collection minimizes.)
//
// Programmer:  Jeremy Meredith
// Creation:    July 16, 2012
//
// Modifications:
// ****************************************************************************
void
eavlTimer::Suspend()
{
    Instance()->suspended = true;
}

// ****************************************************************************
// Method:  eavlTimer::Resume
//
// Purpose:
///   Resule collecting timings. (Timings have some performance and
///   synchronization penalties, which suspending collection minimizes.)
//
// Programmer:  Jeremy Meredith
// Creation:    July 16, 2012
//
// Modifications:
// ****************************************************************************
void
eavlTimer::Resume()
{
    Instance()->suspended = false;
}
