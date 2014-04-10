// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlUtility.h"
#include <cmath>

// ****************************************************************************
// Method:  CalculateTicks
//
// Purpose:
///   Given a lower an upper range, find a reasonable set of tick
///   positions (either major or minor) and their normalized
///   distance within the given range.
//
// Note: modified from Hank Childs' original algorithm in VisIt
//
// Arguments:
//   lower, upper   the range
//
// Programmer:  Jeremy Meredith
// Creation:    January 15, 2013
//
// Modifications:
//   Jeremy Meredith, Thu Mar  7 13:22:43 EST 2013
//   Give a little wiggle room so that ranges like (0,1) get the 0 and 1 major
//   tick marks.  Also, allow an offset to the tick quantity (useful values
//   are -1 for fewer, 0 for normal, and +1 for more ticks).  This latter
//   feature is useful if you have axis of rather different length.
//
// ****************************************************************************
inline double ffix(double value)
{
  int ivalue = (int)(value);
  double v = (value - ivalue);
  if (v > 0.9999)
  {
    ivalue++;
  }
  return (double) ivalue;
}

void CalculateTicks(double lower, double upper, bool minor,
                    vector<double> &positions,
                    vector<double> &proportions,
                    int modifyTickQuantity)
{
    positions.clear();
    proportions.clear();

    double sortedRange[2];
    sortedRange[0] = lower < upper ? lower : upper;
    sortedRange[1] = lower > upper ? lower : upper;

    double range = sortedRange[1] - sortedRange[0];

    // Find the integral points.
    double pow10 = log10(range);

    // Build in numerical tolerance    
    double eps = 10.0e-10;
    pow10 += eps;


    // ffix moves you in the wrong direction if pow10 is negative.
    if (pow10 < 0.)
    {
        pow10 = pow10 - 1.;
    }

    double fxt = pow(10., ffix(pow10));
    
    // Find the number of integral points in the interval.
    int numTicks = ffix(range/fxt) + 1;

    // We should get about major 10 ticks on a range that's near
    // the power of 10.  (e.g. range=1000).  If the range is small
    // enough we have less than 5 ticks (e.g. range=400), then 
    // divide the step by 2, or if it's about 2 ticks (e.g. range=150)
    // or less, then divide the step by 5.  That gets us back to
    // about 10 major ticks.
    //
    // But we might want more or less.  To adjust this up by
    // approximately a factor of 2, instead of the default 
    // 1/2/5 dividers, use 2/5/10, and to adjust it down by
    // about a factor of two, use .5/1/2 as the dividers.
    // (We constrain to 1s, 2s, and 5s, for the obvious reason
    // that only those values are factors of 10.....)
    double divs[5] = { 0.5, 1, 2, 5, 10 };
    int divindex = (numTicks >= 5) ? 1 : (numTicks >= 3 ? 2 : 3);
    divindex += modifyTickQuantity;

    double div = divs[divindex];

    // If there aren't enough major tick points in this decade, use the next
    // decade.
    double majorStep = fxt / div;
    double minorStep = (fxt/div) / 10.;

    // When we get too close, we lose the tickmarks. Run some special case code.
    if (numTicks <= 1)
    {
        if (minor)
        {
            // no minor ticks
            return;
        }
        else
        {
            positions.resize(3);
            proportions.resize(3);
            positions[0] = lower;
            positions[1] = (lower+upper) / 2.;
            positions[2] = upper;
            proportions[0] = 0.0;
            proportions[1] = 0.5;
            proportions[2] = 1.0;
            return;
        }
    }

    // Figure out the first major and minor tick locations, relative to the
    // start of the axis.
    double majorStart, minorStart;
    if (sortedRange[0] < 0.)
    {
        majorStart = majorStep*(ffix(sortedRange[0]*(1./majorStep)));
        minorStart = minorStep*(ffix(sortedRange[0]*(1./minorStep)));
    }
    else
    {
        majorStart = majorStep*(ffix(sortedRange[0]*(1./majorStep) + .999));
        minorStart = minorStep*(ffix(sortedRange[0]*(1./minorStep) + .999));
    }

    // Create all of the minor ticks
    const int max_count_cutoff = 1000;
    numTicks = 0;
    double location = minor ? minorStart : majorStart;
    double step = minor ? minorStep : majorStep;
    while (location <= sortedRange[1] && numTicks < max_count_cutoff)
    {
        positions.push_back(location);
        proportions.push_back((location - sortedRange[0]) / range);
        numTicks++;
        location += step;
    }

    if (sortedRange[0] != lower)
    {
        // We must reverse all of the proportions.
        for (unsigned int j = 0 ; j < proportions.size(); j++)
        {
            proportions[j] = 1. - proportions[j];
        }
    }
}


void CalculateTicksLogarithmic(double lower, double upper, bool minor,
                               vector<double> &positions,
                               vector<double> &proportions,
                               int modifyTickQuantity)
{
    positions.clear();
    proportions.clear();

    double sortedRange[2];
    sortedRange[0] = lower < upper ? lower : upper;
    sortedRange[1] = lower > upper ? lower : upper;

    double range = sortedRange[1] - sortedRange[0];

    double first_log = ceil(sortedRange[0]);
    double last_log = floor(sortedRange[1]);
    if (last_log <= first_log)
        last_log = first_log+1;
    double diff_log = last_log - first_log;
    int step = (diff_log + 9) / 10;

    if (minor)
    {
        first_log -= step;
        last_log += step;
    }

    for (int i=first_log; i<=last_log; i += step)
    {
        double logpos = i;
        double pos = pow(10, logpos);
        if (minor)
        {
            // If we're showing major tickmarks for every power of 10,
            // then show 2x10^n, 3x10^n, ..., 9x10^n for minor ticks.
            // If we're skipping some powers of 10, then use the minor
            // ticks to show where those skipped ones are.  (Beyond
            // a range of 100 orders of magnitude, we get more than 10
            // minor ticks per major tick, but that's awfully rare.)
            if (step == 1)
            {
                for (int j=1; j<10; ++j)
                {
                    double minor_pos = double(j) * double(pos);
                    double minor_logpos = log10(minor_pos);
                    if (minor_logpos < sortedRange[0] ||
                        minor_logpos > sortedRange[1])
                    {
                        continue;
                    }
                    positions.push_back(minor_pos);
                    proportions.push_back((minor_logpos - sortedRange[0]) / (sortedRange[1]-sortedRange[0]));
                }
            }
            else
            {
                for (int j=1; j<step; ++j)
                {
                    double minor_logpos = logpos + j;
                    double minor_pos = pow(10., minor_logpos);
                    if (minor_logpos < sortedRange[0] ||
                        minor_logpos > sortedRange[1])
                    {
                        continue;
                    }
                    positions.push_back(minor_pos);
                    proportions.push_back((minor_logpos - sortedRange[0]) / (sortedRange[1]-sortedRange[0]));
                }
            }
        }
        else
        {
            if (logpos > sortedRange[1])
                break;
            positions.push_back(pos);
            proportions.push_back((logpos - sortedRange[0]) / (sortedRange[1]-sortedRange[0]));
        }
    }
}
