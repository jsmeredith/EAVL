// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlMatrix4x4.h"

ostream &operator<<(ostream& out, const eavlMatrix4x4 &r)
{
    out << r.m[0][0] << "," << r.m[0][1] << "," << r.m[0][2] << "," << r.m[0][3] << "\n";
    out << r.m[1][0] << "," << r.m[1][1] << "," << r.m[1][2] << "," << r.m[1][3] << "\n";
    out << r.m[2][0] << "," << r.m[2][1] << "," << r.m[2][2] << "," << r.m[2][3] << "\n";
    out << r.m[3][0] << "," << r.m[3][1] << "," << r.m[3][2] << "," << r.m[3][3] << "\n";
    return out;
}

