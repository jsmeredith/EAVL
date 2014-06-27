// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlPoint3.h"

ostream &operator<<(ostream& out, const eavlPoint3 &r)
{
    out << "<" << r.x << "," << r.y << "," << r.z << ">";
    return out;
}

