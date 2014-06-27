// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlPointDistanceFieldFilter.h"
#include "eavlDataSet.h"
#include "eavlException.h"
#include "eavlExecutor.h"
#include "eavlCellSetAllStructured.h"
#include "eavlArray.h"
#include "eavlLogicalStructureRegular.h"
#include "eavlCoordinates.h"
#include <algorithm>

eavlPointDistanceFieldFilter::eavlPointDistanceFieldFilter()
{
    exact = false;
    niter = 3;
    dim = 1;
    xmin = ymin = zmin = -1;
    xmax = ymax = zmax = +1;
    ni = nj = nk = 1;
}

eavlPointDistanceFieldFilter::~eavlPointDistanceFieldFilter()
{
}

void eavlPointDistanceFieldFilter::Execute()
{
    //
    // Fix up limits and extents so they make sense
    //
    if (xmin>xmax)
    {
        std::swap(xmin, xmax);
    }
    else if (xmin==xmax)
    {
        xmin -= 1.;
        xmax += 1.;
    }

    if (ymin>ymax)
    {
        std::swap(ymin, ymax);
    }
    else if (ymin==ymax)
    {
        ymin -= 1.;
        ymax += 1.;
    }

    if (zmin>zmax)
    {
        std::swap(zmin, zmax);
    }
    else if (zmin==zmax)
    {
        zmin -= 1.;
        zmax += 1.;
    }

    if (ni < 1)
        ni = 1;
    if (nj < 1)
        nj = 1;
    if (nk < 1)
        nk = 1;

    //
    // Create the output mesh
    //

    // set the number of points
    int npts = ni;
    if (dim>=2)
        npts *= nj;
    if (dim>=3)
        npts *= nk;
    output->SetNumPoints(npts);

    eavlRegularStructure reg;
    if (dim==1)
        reg.SetNodeDimension1D(ni);
    else if (dim==2)
        reg.SetNodeDimension2D(ni, nj);
    else // (dim==3)
        reg.SetNodeDimension3D(ni, nj, nk);

    // set the logical structure
    eavlLogicalStructure *log = new eavlLogicalStructureRegular(reg.dimension,
                                                                reg);
    output->SetLogicalStructure(log);

    // create coordinates
    eavlCoordinates *coords;
    if (dim==1)
        coords = new eavlCoordinatesCartesian(log,
                                              eavlCoordinatesCartesian::X);
    else if (dim==2)
        coords = new eavlCoordinatesCartesian(log,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y);
    else // (dim==3)
        coords = new eavlCoordinatesCartesian(log,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);

    // create the coordinate axes
    eavlFloatArray *x = new eavlFloatArray("x", 1, ni);
    for (int i=0; i<ni; ++i)
        x->SetValue(i, xmin + (xmax-xmin)*double(i)/double(ni-1));
    output->AddField(new eavlField(1, x, eavlField::ASSOC_LOGICALDIM, 0));
    coords->SetAxis(0, new eavlCoordinateAxisField("x"));

    eavlFloatArray *y = NULL;
    if (dim >= 2)
    {
        y = new eavlFloatArray("y", 1, nj);
        for (int j=0; j<nj; ++j)
            y->SetValue(j, ymin + (ymax-ymin)*double(j)/double(nj-1));
        output->AddField(new eavlField(1, y, eavlField::ASSOC_LOGICALDIM, 1));
        coords->SetAxis(1, new eavlCoordinateAxisField("y"));
    }

    eavlFloatArray *z = NULL;
    if (dim >= 3)
    {
        z = new eavlFloatArray("z", 1, nk);
        for (int k=0; k<nk; ++k)
            z->SetValue(k, zmin + (zmax-zmin)*double(k)/double(nk-1));
        output->AddField(new eavlField(1, z, eavlField::ASSOC_LOGICALDIM, 2));
        coords->SetAxis(2, new eavlCoordinateAxisField("z"));
    }

    // set the coordinates
    output->AddCoordinateSystem(coords);

    // create a cell set implicitly covering the entire regular structure
    eavlCellSet *cells = new eavlCellSetAllStructured("cells", reg);
    output->AddCellSet(cells);

    //
    // Create the distance field
    //
    eavlFloatArray *dist = new eavlFloatArray("dist", 1, npts);
    for (int i=0; i<npts; ++i)
        dist->SetValue(i, -1);
    output->AddField(new eavlField(1, dist, eavlField::ASSOC_POINTS));

    // closest point field
    eavlIntArray *cp = new eavlIntArray("cp", 1, npts);
    for (int i=0; i<npts; ++i)
        cp->SetValue(i, -1);

    if (exact)
    {
        //
        // We were asked to create the "exact" results.
        // This is a naive, slow, O(n*m) algorithm.
        // (where n = num input points and m = num output mesh nodes)
        //
        for (int k=0; k<nk; ++k)
        {
            for (int j=0; j<nj; ++j)
            {
                for (int i=0; i<ni; ++i)
                {
                    const int myindex = i + j*ni + k*ni*nj;
                    const float myx = x->GetValue(i);
                    const float myy = y ? y->GetValue(j) : 0;
                    const float myz = z ? z->GetValue(k) : 0;
                    float old_dist = dist->GetValue(myindex);
                    for (int p=0; p<input->GetNumPoints(); ++p)
                    {
                        double px=0,py=0,pz=0;
                        px = input->GetPoint(p, 0);
                        if (dim>=2)
                            py = input->GetPoint(p, 1);
                        if (dim>=3)
                            pz = input->GetPoint(p, 2);

                        double dx = px - myx;
                        double dy = py - myy;
                        double dz = pz - myz;
                        double new_dist = sqrt(dx*dx + dy*dy + dz*dz);

                        if (old_dist < 0 || new_dist < old_dist)
                        {
                            dist->SetValue(myindex, new_dist);
                            old_dist = new_dist;
                        }
                    }
                }
            }
        }
    }
    else
    {
        //
        // We can do the fast, approximate algorithm.
        // This is a much better runtim O(n + m) algorithm.
        // (where n = num input points and m = num output mesh nodes)
        //

        // It's iterative, i.e. accuracy increases with more iterations.
        // However, it's very good with only 2 iterations.  The default is
        // 3 iterations, which is close enough for almost any purpose.
        // Five iterations is almost indistinguishable from 10 or more.
        //

        //
        // First step:
        //
        // Step through input points tag neighboring mesh points with
        // their location and starting distance
        //
        for (int p=0; p<input->GetNumPoints(); ++p)
        {
            double px=0,py=0,pz=0;
            px = input->GetPoint(p, 0);
            if (dim>=2)
                py = input->GetPoint(p, 1);
            if (dim>=3)
                pz = input->GetPoint(p, 2);

            double ix = double(ni) * (px - xmin) / double(xmax - xmin);
            double iy = double(nj) * (py - ymin) / double(ymax - ymin);
            double iz = double(nk) * (pz - zmin) / double(zmax - zmin);
            //cerr << "px="<<px<<" ix="<<ix<<" cx="<<cx<<endl;
            for (int jitterx=0; jitterx <= 1; ++jitterx)
            {
                for (int jittery=0; jittery <= 1; ++jittery)
                {
                    for (int jitterz=0; jitterz <= 1; ++jitterz)
                    {
                        int iix = jitterx ? ceil(ix) : floor(ix);
                        int iiy = jittery ? ceil(iy) : floor(iy);
                        int iiz = jitterz ? ceil(iz) : floor(iz);
                        int cx = (ix<=0) ? 0 : (ix >= ni-1) ? ni-1 : iix;
                        int cy = (iy<=0) ? 0 : (iy >= nj-1) ? nj-1 : iiy;
                        int cz = (iz<=0) ? 0 : (iz >= nk-1) ? nk-1 : iiz;
                        double xx = x->GetValue(cx);
                        double yy = y ? y->GetValue(cy) : 0;
                        double zz = z ? z->GetValue(cz) : 0;
                        double dx = px-xx;
                        double dy = py-yy;
                        double dz = pz-zz;
                        double new_dist = sqrt(dx*dx + dy*dy + dz*dz);
                        int index = cx + cy*ni + cz*ni*nj;
                        float old_dist = dist->GetValue(index);

                        if (old_dist < 0 || new_dist < old_dist)
                        {
                            cp->SetValue(index, p);
                            dist->SetValue(index, new_dist);
                        }
                    }
                }
            }
        }

        //
        // Second step:
        //
        // Iterate sweeps to propagate closest points through neighbors.
        // A single iteration (on sweep per axis per direction, e.g. in 2D 
        // this is an up, then right, then down, then left sweep) will
        // propagate reasonable values to every point in the mesh.
        // Accuracy improves with each iteration, and very quickly at
        // the early iterations (2, 3, 4).  That's because the largest
        // errors occur where curvature is high, i.e. the nodes
        // very near the original points, and each iteration is (in a way)
        // expanding the exact answer one more grid cell away from the 
        // source points.
        //
        for (int iter=0; iter<niter; ++iter)
        {
            for (int direction = -1 ; direction <= 1; direction += 2)
            {
                for (int axis = 0; axis < dim ; ++axis)
                {
                    for (int kk=0; kk<nk; ++kk)
                    {
                        for (int jj=0; jj<nj; ++jj)
                        {
                            for (int ii=0; ii<ni; ++ii)
                            {
                                int i = (axis==0 && direction>0) ? ni-1-ii : ii;
                                int j = (axis==1 && direction>0) ? nj-1-jj : jj;
                                int k = (axis==2 && direction>0) ? nk-1-kk : kk;

                                const int myindex = i + j*ni + k*ni*nj;
                                const float myx = x->GetValue(i);
                                const float myy = y ? y->GetValue(j) : 0;
                                const float myz = z ? z->GetValue(k) : 0;
                                float old_dist = dist->GetValue(myindex);

                                int srcindex = myindex;
                                if (axis == 0)
                                    srcindex += direction * 1;
                                else if (axis == 1)
                                    srcindex += direction * ni;
                                else // (axis == 2)
                                    srcindex += direction * ni*nj;

                                // We're being lazy with this check; we're allowing
                                // it to wrap around the boundary in a nonsensical way.
                                // But since the algorithm is conservative, no
                                // harm done except wasted computation.
                                if (srcindex < 0 || srcindex >= npts)
                                    continue;

                                int src_cp = cp->GetValue(srcindex);
                                if (src_cp < 0)
                                    continue;
                            
                                double px=0,py=0,pz=0;
                                px = input->GetPoint(src_cp, 0);
                                if (dim>=2)
                                    py = input->GetPoint(src_cp, 1);
                                if (dim>=3)
                                    pz = input->GetPoint(src_cp, 2);

                                double dx = px - myx;
                                double dy = py - myy;
                                double dz = pz - myz;
                                double new_dist = sqrt(dx*dx + dy*dy + dz*dz);

                                if (old_dist < 0 || new_dist < old_dist)
                                {
                                    cp->SetValue(myindex, src_cp);
                                    dist->SetValue(myindex, new_dist);
                                    old_dist = new_dist;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    delete cp;
}
