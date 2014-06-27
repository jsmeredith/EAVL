// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_TIMER_H
#define EAVL_TIMER_H

#include <vector>
#include <string>
#include <iostream>

#if defined(_WIN32)
#include <time.h>
#include <sys/timeb.h>
#else
#include <sys/time.h>
#include <sys/timeb.h>
#endif

#if defined(_WIN32)
#    define TIMEINFO _timeb
#else
#    define TIMEINFO timeval
#endif

// ****************************************************************************
//  Class:  eavlTimer
//
//  Purpose:
///   Encapsulated a set of hierarchical timers.  Starting a timer
///   returns a handle to a timer.  Pass this handle, and a description,
///   into the timer Stop routine.  Timers can nest and output will
///   be displayed in a tree format.
//
//  Programmer:  Jeremy Meredith
//  Creation:    August  6, 2004
//
// ****************************************************************************
class eavlTimer
{
  public:
    static eavlTimer *Instance();

    static int    Start();
    static double Stop(int handle, const std::string &descr);
    static void   Insert(const std::string &descr, double value);

    static void   Dump(std::ostream&);

    static void   Suspend();
    static void   Resume();

  private:

    int    real_Start();
    double real_Stop(int, const std::string &);
    void   real_Insert(const std::string &descr, double value);
    void   real_Dump(std::ostream&);

    eavlTimer();
    ~eavlTimer();

    static eavlTimer *instance;

    bool                     suspended;
    std::vector<TIMEINFO>    startTimes;
    std::vector<double>      timeLengths;
    std::vector<std::string> descriptions;
    int                      currentActiveTimers;
};

#endif
