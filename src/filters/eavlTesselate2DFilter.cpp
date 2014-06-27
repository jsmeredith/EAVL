// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlTesselate2DFilter.h"
#include "eavlCoordinates.h"
#include "eavlCellSetExplicit.h"
#include "eavlException.h"
#include <cmath>

static float Legendre(int i, float x)
{
  float scale = sqrt(2. * i + 1.);
    switch (i)
    {
      case 0:    return scale * 1;
      case 1:    return scale * x;
      case 2:    return scale * (3. * x*x -1) / 2.;
    }
    return -999999999;
}

static float EvalLegendre(float scale, float xx, float yy, eavlArray *arr, int index)
{
    float sum = 0;
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            float v = arr->GetComponentAsDouble(index, i*3 + j) * Legendre(i, xx) * Legendre(j, yy);
            //cerr << "i="<<i<<" j="<<j<<" val="<<v<<endl;
            sum += v;
        }
    }
    ///\todo: first, scale as calculated has a square root, so squaring it here seems wasteful.
    ///       furthermore, it's based on the level of the quadtree; that's no good!
    ///       the right thing is probably to just scale the darn coefficients in the MADNESS reader.
    return sum * scale*scale;
}


eavlTesselate2DFilter::eavlTesselate2DFilter()
{
}

void
eavlTesselate2DFilter::Execute()
{
    eavlCellSet *inCells = input->GetCellSet(cellsetname);

    //
    // set up output mesh
    //
    eavlFloatArray *coords = new eavlFloatArray("coords",3);
    
    eavlField *coordField = new eavlField(1, coords, eavlField::ASSOC_POINTS);
    output->AddField(coordField);

    eavlCellSetExplicit *outCellSet = new eavlCellSetExplicit("tesselated",2);
    output->AddCellSet(outCellSet);

    //
    // calculate number of edges
    ///\todo: (NOT DOING EDGE MATCHING YET; NEEDS TO SUPPORT QUADTREES WHICH
    ///        THEMSELVES CAN'T DO THAT YET)
    //
    int total_nedge = 0;
    int in_ncells = inCells->GetNumCells();
    for (int e=0; e<in_ncells; e++)
    {
        eavlCell cell = inCells->GetCellNodes(e);
        switch (cell.type)
        {
          case EAVL_TRI:     total_nedge += 3; break;
          case EAVL_QUAD:    total_nedge += 4; break;
          case EAVL_PIXEL:   total_nedge += 4; break;
          case EAVL_POLYGON: total_nedge += cell.numIndices; break;
          default: break;
        }
    }

    //
    // calculate total number of points
    //
    output->SetNumPoints(input->GetNumPoints() + in_ncells + total_nedge);

    //
    // tesselate high-order cell arrays and fields to single-component nodal scalars
    //
    vector<pair<eavlArray*,eavlArray*> > legendre3x3_arrays;
    for (int i=0; i<input->GetNumFields(); i++)
    {
        if (input->GetField(i)->GetAssociation() == eavlField::ASSOC_CELL_SET &&
            input->GetField(i)->GetOrder() == 2 &&
            input->GetField(i)->GetArray()->GetNumberOfComponents() == 9)
        {
            eavlFloatArray *arr = new eavlFloatArray(input->GetField(i)->GetArray()->GetName(),
                                                       1); // single-component, now
            arr->SetNumberOfTuples(output->GetNumPoints());
            legendre3x3_arrays.push_back(pair<eavlArray*,eavlArray*>(input->GetField(i)->GetArray(), arr));

            eavlField *f = new eavlField(1, arr, eavlField::ASSOC_POINTS);
            output->AddField(f);
        }
    }

    //
    // create nodal arrays and fields
    //
    vector<pair<eavlArray*,eavlArray*> > nodal_arrays;
    for (int i=0; i<input->GetNumFields(); i++)
    {
        if (input->GetField(i)->GetAssociation() == eavlField::ASSOC_POINTS &&
            input->GetField(i)->GetOrder() == 1)
        {
            eavlFloatArray *arr = new eavlFloatArray(input->GetField(i)->GetArray()->GetName(),
                                                       input->GetField(i)->GetArray()->GetNumberOfComponents());
            arr->SetNumberOfTuples(output->GetNumPoints());
            nodal_arrays.push_back(pair<eavlArray*,eavlArray*>(input->GetField(i)->GetArray(), arr));

            eavlField *f = new eavlField(input->GetField(i)->GetOrder(), arr,
                                         eavlField::ASSOC_POINTS);
            output->AddField(f);
        }
    }

    //
    // copy old points and point values
    //
    coords->SetNumberOfTuples(output->GetNumPoints());
    for (int i=0; i<input->GetNumPoints(); i++)
    {
        double x = input->GetPoint(i, 0);
        double y = input->GetPoint(i, 1);
        double z = input->GetPoint(i, 2);
        coords->SetComponentFromDouble(i, 0, x);
        coords->SetComponentFromDouble(i, 1, y);
        coords->SetComponentFromDouble(i, 2, z);
    }
    for (size_t j=0; j<nodal_arrays.size(); j++)
    {
        int nc = nodal_arrays[j].first->GetNumberOfComponents();
        for (int c=0; c<nc; c++)
            for (int p=0; p<input->GetNumPoints(); p++)
                nodal_arrays[j].second->SetComponentFromDouble(p, c,
                            nodal_arrays[j].first->GetComponentAsDouble(p,c));
    }

    //
    // do the tesselation:
    // create new cells and new points
    //
    eavlExplicitConnectivity conn;
    int new_point_index = input->GetNumPoints();
    for (int e=0; e<in_ncells; e++)
    {
        eavlCell cell = inCells->GetCellNodes(e);

        int nedges = 0;
        signed char (*edges)[2] = NULL;
        switch (cell.type)
        {
          case EAVL_TRI:
            nedges  = 3;
            edges   = eavlTriEdges;
            break;
          case EAVL_QUAD:
            nedges  = 4;
            edges   = eavlQuadEdges;
            break;
          case EAVL_PIXEL:
            nedges  = 4;
            edges   = eavlPixelEdges;
            break;
          case EAVL_POLYGON:
            if (cell.numIndices == 3)
            {
                nedges  = 3;
                edges   = eavlTriEdges;
            }
            else if (cell.numIndices == 4)
            {
                nedges  = 4;
                edges   = eavlQuadEdges;
            }
            else
            {
                THROW(eavlException,"Don't know what to do with this shape\n");
            }
            break;
          default:
            THROW(eavlException,"Don't know what to do with this shape\n");
        }

        ///\todo: not a great way to calculate scale; maybe doing it
        ///       in the MADNESS reader is a better idea.
        float cell_size_a = fabs(input->GetPoint(cell.indices[1], 0) -
                                 input->GetPoint(cell.indices[0], 0));
        float cell_size_b = fabs(input->GetPoint(cell.indices[1], 1) -
                                 input->GetPoint(cell.indices[0], 1));
        float cell_size = (cell_size_a > cell_size_b) ? cell_size_a : cell_size_b;
        float legendre_scale = sqrt(1. / cell_size);
        //cerr << "legendre_scale = "<<legendre_scale<<endl;

        //
        // create a new point at the centroid of the cell
        //
        int centroid_point_index = new_point_index;

        // do the coordinates
        double x = 0;
        double y = 0;
        double z = 0;
        for (int j=0; j<cell.numIndices; j++)
        {
            x += input->GetPoint(cell.indices[j], 0);
            y += input->GetPoint(cell.indices[j], 1);
            z += input->GetPoint(cell.indices[j], 2);
        }
        x /= double(cell.numIndices);
        y /= double(cell.numIndices);
        z /= double(cell.numIndices);
        coords->SetComponentFromDouble(new_point_index, 0, x);
        coords->SetComponentFromDouble(new_point_index, 1, y);
        coords->SetComponentFromDouble(new_point_index, 2, z);

        // do the nodal arrays
        for (size_t k=0; k<nodal_arrays.size(); k++)
        {
            int nc = nodal_arrays[k].first->GetNumberOfComponents();
            for (int c=0; c<nc; c++)
            {
                double value = 0;
                for (int j=0; j<cell.numIndices; j++)
                    value += nodal_arrays[k].first->GetComponentAsDouble(cell.indices[j], c);
                value /= double(cell.numIndices);

                nodal_arrays[k].second->SetComponentFromDouble(new_point_index,
                                                               c, value);
            }
        }

        // tesselate the high-order ones at this centroid
        // note: the xx,yy values range from -1 to +1
        //       it must be a quadrilateral cell
        //       and the centroid is at 0,0
        for (size_t k=0; k<legendre3x3_arrays.size(); k++)
        {
            if (nedges != 4)
                THROW(eavlException,"We've got 3x3 legendre arrays for non-quadrilateral cells?!");
            eavlArray *inarr = legendre3x3_arrays[k].first;
            eavlArray *outarr = legendre3x3_arrays[k].second;
            float value = EvalLegendre(legendre_scale, 0, 0, inarr, e);
            outarr->SetComponentFromDouble(new_point_index, 0, value);
        }

        //
        // create new points midway along each edge
        //

        // coordinates
        new_point_index = centroid_point_index + 1;// note: reset 'new_point_index'
        for (int j=0; j<nedges; j++)
        {
            double x = (input->GetPoint(cell.indices[edges[j][0]], 0) + 
                        input->GetPoint(cell.indices[edges[j][1]], 0)) / 2.;
            double y = (input->GetPoint(cell.indices[edges[j][0]], 1) + 
                        input->GetPoint(cell.indices[edges[j][1]], 1)) / 2.;
            double z = (input->GetPoint(cell.indices[edges[j][0]], 2) + 
                        input->GetPoint(cell.indices[edges[j][1]], 2)) / 2.;
            coords->SetComponentFromDouble(new_point_index, 0, x);
            coords->SetComponentFromDouble(new_point_index, 1, y);
            coords->SetComponentFromDouble(new_point_index, 2, z);
            new_point_index++;
        }

        // nodal arrays
        for (size_t k=0; k<nodal_arrays.size(); k++)
        {
            new_point_index = centroid_point_index + 1;// note: reset 'new_point_index'
            int nc = nodal_arrays[k].first->GetNumberOfComponents();
            for (int j=0; j<nedges; j++)
            {
                for (int c=0; c<nc; c++)
                {
                    double value = (nodal_arrays[k].first->GetComponentAsDouble(cell.indices[edges[j][0]], c) +
                                    nodal_arrays[k].first->GetComponentAsDouble(cell.indices[edges[j][1]], c)) / 2.;
                                    
                    nodal_arrays[k].second->SetComponentFromDouble(new_point_index,
                                                                   c, value);
                }
                new_point_index++;
            }
        }

        // tesselate legendre3x3 arrays at the edges
        for (size_t k=0; k<legendre3x3_arrays.size(); k++)
        {
            new_point_index = centroid_point_index + 1;// note: reset 'new_point_index'
            eavlArray *inarr = legendre3x3_arrays[k].first;
            eavlArray *outarr = legendre3x3_arrays[k].second;
            // note: we're assuming a particular edge ordering
            float value0 = EvalLegendre(legendre_scale, 0, -1, inarr, e);
            float value1 = EvalLegendre(legendre_scale, +1, 0, inarr, e);
            float value2 = EvalLegendre(legendre_scale, 0, +1, inarr, e);
            float value3 = EvalLegendre(legendre_scale, -1, 0, inarr, e);
            outarr->SetComponentFromDouble(new_point_index + 0, 0, value0);
            outarr->SetComponentFromDouble(new_point_index + 1, 0, value1);
            outarr->SetComponentFromDouble(new_point_index + 2, 0, value2);
            outarr->SetComponentFromDouble(new_point_index + 3, 0, value3);
            new_point_index += 4;
        }

        // tesselate legendre3x3 arrays at the nodes, too
        // note: inefficient in the general case since we may have
        // shared nodes, but it's not too bad
        for (size_t k=0; k<legendre3x3_arrays.size(); k++)
        {
            eavlArray *inarr = legendre3x3_arrays[k].first;
            eavlArray *outarr = legendre3x3_arrays[k].second;
            // note: we're assuming a particular edge ordering again
            float value0 = EvalLegendre(legendre_scale, -1, -1, inarr, e);
            float value1 = EvalLegendre(legendre_scale, +1, -1, inarr, e);
            float value2 = EvalLegendre(legendre_scale, +1, +1, inarr, e);
            float value3 = EvalLegendre(legendre_scale, -1, +1, inarr, e);
            outarr->SetComponentFromDouble(cell.indices[edges[0][0]], 0, value0);
            outarr->SetComponentFromDouble(cell.indices[edges[1][0]], 0, value1);
            outarr->SetComponentFromDouble(cell.indices[edges[2][0]], 0, value2);
            outarr->SetComponentFromDouble(cell.indices[edges[3][0]], 0, value3);
        }

        // create the cells for each
        for (int j=0; j<nedges; j++)
        {
            conn.shapetype.push_back((int)EAVL_QUAD);
            conn.connectivity.push_back(4);
            conn.connectivity.push_back(centroid_point_index);
            conn.connectivity.push_back(centroid_point_index + 1 + j);
            conn.connectivity.push_back(cell.indices[edges[j][1]]);
            conn.connectivity.push_back(centroid_point_index + 1 + (j+1)%nedges);
        }
    }
    outCellSet->SetCellNodeConnectivity(conn);

    eavlCoordinatesCartesian *coordsys =
        new eavlCoordinatesCartesian(NULL,
                                     eavlCoordinatesCartesian::X,
                                     eavlCoordinatesCartesian::Y,
                                     eavlCoordinatesCartesian::Z);
    ///\todo: assuming 3D coords
    coordsys->SetAxis(0, new eavlCoordinateAxisField("coords", 0));
    coordsys->SetAxis(1, new eavlCoordinateAxisField("coords", 1));
    coordsys->SetAxis(2, new eavlCoordinateAxisField("coords", 2));
    output->AddCoordinateSystem(coordsys);
}
