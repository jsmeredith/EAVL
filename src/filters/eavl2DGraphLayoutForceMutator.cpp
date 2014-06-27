// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavl2DGraphLayoutForceMutator.h"
#include "eavlCellSetExplicit.h"

#ifdef _WIN32
 #define random rand
#endif

eavl2DGraphLayoutForceMutator::eavl2DGraphLayoutForceMutator()
{
    niter = 100;
    startdist = 1.0;
    finaldist = 0.005;
    areaconstant = 1.0;
    gravityconstant = 0.3;
}

void
eavl2DGraphLayoutForceMutator::Execute()
{
    eavlCellSet *cs = dataset->GetCellSet(cellsetname);
    eavlCellSetExplicit *cse = dynamic_cast<eavlCellSetExplicit*>(cs);
    if (!cse)
        THROW(eavlException, "Expected explicit cell set.");

    int npts = dataset->GetNumPoints();

    //
    // add coordinates, or clobber the old ones
    //
    eavlCoordinatesCartesian *coords =
        new eavlCoordinatesCartesian(NULL,
                                     eavlCoordinatesCartesian::X,
                                     eavlCoordinatesCartesian::Y);
    coords->SetAxis(0, new eavlCoordinateAxisField("newx",0));
    coords->SetAxis(1, new eavlCoordinateAxisField("newy",0));
    if (dataset->GetNumCoordinateSystems() == 0)
        dataset->AddCoordinateSystem(coords);
    else
        dataset->SetCoordinateSystem(0, coords);

    eavlFloatArray *x = new eavlFloatArray("newx", 1, npts);
    eavlFloatArray *y = new eavlFloatArray("newy", 1, npts);
    dataset->AddField(new eavlField(1, x, eavlField::ASSOC_POINTS));
    dataset->AddField(new eavlField(1, y, eavlField::ASSOC_POINTS));

    //
    // coordinate initialization
    //

#if 0 // first option: arrange them in a circle
    for (int i=0; i<npts; ++i)
    {
        float a = float(i)/float(npts);
        x->SetValue(i, cos(a * 2 * M_PI));
        y->SetValue(i, sin(a * 2 * M_PI));
    }
#elif 0 // second option: in a grid
    int sq = sqrt(npts);
    for (int i=0; i<npts; ++i)
    {
        int xx = i % sq;
        int yy = i / sq;
        x->SetValue(i, float(xx)/float(sq));
        y->SetValue(i, float(yy)/float(sq));
    }
#else // third option: random
    for (int i=0; i<npts; ++i)
    {
        float xx = double(random()) / double(RAND_MAX);
        float yy = double(random()) / double(RAND_MAX);
        x->SetValue(i, 2.*xx-1.);
        y->SetValue(i, 2.*yy-1.);
    }
#endif

    //
    // do the force-directed layout in 2D
    //

    float k = sqrt(areaconstant / npts);
    float k2 = k*k;
    vector<float> vx(npts, 0);
    vector<float> vy(npts, 0);
    double distchange = finaldist / startdist;
    double distdelta  = niter==1 ? 1 : pow(distchange, 1. / double(niter-1));
    for (int iter = 0 ; iter < niter ; ++iter)
    {
        // cooling schedule determines current maxdist
        float maxdist = startdist * pow(distdelta, iter);

        // init vels to 0
        for (int i=0; i<npts; ++i)
        {
            vx[i] = 0;
            vy[i] = 0;
        }
        
#if 1
        // repulsive force
        for (int i=0; i<npts; ++i)
        {
            float ix = x->GetValue(i);
            float iy = y->GetValue(i);
            for (int j=i+1; j<npts; ++j)
            {
                float dx = ix - x->GetValue(j);
                float dy = iy - y->GetValue(j);
                ///\todo: if dx==dy==0, assume some small random displacement

                /*
                float len = sqrt(dx*dx + dy*dy);
                float udx = dx/len;
                float udy = dy/len;

                float frx = udx * (k*k / len);
                float fry = udy * (k*k / len);
                */
                // same equation, just more efficient via simplification
                float len2 = dx*dx + dy*dy;
                float frx = dx * k2 / len2;
                float fry = dy * k2 / len2;

                vx[i] += frx;
                vy[i] += fry;

                vx[j] -= frx;
                vy[j] -= fry;
            }
        }
#endif

        // attractive force
        for (int c=0; c<cs->GetNumCells(); ++c)
        {
            eavlCell cell = cs->GetCellNodes(c);
            if (cell.numIndices != 2)
                continue;
            int i = cell.indices[0];
            int j = cell.indices[1];

            float dx = x->GetValue(i) - x->GetValue(j);
            float dy = y->GetValue(i) - y->GetValue(j);
            ///\todo: if dx==dy==0, assume some small random displacement
            float len = sqrt(dx*dx + dy*dy);
            float udx = dx/len;
            float udy = dy/len;

            float fax = udx * (len*len / k);
            float fay = udy * (len*len / k);
            vx[i] -= fax;
            vy[i] -= fay;

            vx[j] += fax;
            vy[j] += fay;
           
        }

#if 1
        // attract to center to keep trees from escaping forest
        for (int i=0; i<npts; ++i)
        {
            float dx = x->GetValue(i);
            float dy = y->GetValue(i);
            if (dx == 0 && dy == 0)
                continue;

            float len = sqrt(dx*dx + dy*dy);
            float udx = dx/len;
            float udy = dy/len;

            float fcx = gravityconstant * udx * len / k;
            float fcy = gravityconstant * udy * len / k;
            vx[i] -= fcx;
            vy[i] -= fcy;
        }
#endif

        // clamp any distance to current maxdist (defined by cooling schedule)
        for (int i=0; i<npts; ++i)
        {
            float len = sqrt(vx[i]*vx[i] + vy[i]*vy[i]);
            if (len > maxdist)
            {
                vx[i] = maxdist * vx[i] / len;
                vy[i] = maxdist * vy[i] / len;
            }
        }

        // update point locations
        for (int i=0; i<npts; ++i)
        {
            x->SetValue(i, x->GetValue(i) + vx[i]);
            y->SetValue(i, y->GetValue(i) + vy[i]);
        }
    }
}
