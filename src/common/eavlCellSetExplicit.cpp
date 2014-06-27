// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlCellSetExplicit.h"

using std::multimap;

class eavlEdge
{
 public:
    int a;
    int b;
    bool operator<(const eavlEdge &e) const
    {
        if (a < e.a)
            return true;
        if (a > e.a)
            return false;
        if (b < e.b)
            return true;
        return false;
    }
    eavlEdge() { }
    eavlEdge(const eavlEdge &e) : a(e.a), b(e.b) { }
    eavlEdge(int p0, int p1)
    {
        if (p0 < p1)
        {
            a = p0;
            b = p1;
        }
        else
        {
            a = p1;
            b = p0;
        }
    }
};


void eavlCellSetExplicit::BuildEdgeConnectivity()
{
    if (numEdges >= 0)
        return; // already done!

    numEdges = 0;

    map<eavlEdge,int> mapEdgesToEdgeIDs; ///<\todo: horribly inefficient

    cellEdgeConnectivity.shapetype.clear();
    cellEdgeConnectivity.connectivity.clear();
    edgeNodeConnectivity.shapetype.clear();
    edgeNodeConnectivity.connectivity.clear();

    int nCells = GetNumCells();
    for (int i=0; i<nCells; i++)
    {
        eavlCell el = GetCellNodes(i);

        int nedges = 0;
        signed char (*edges)[2];
        switch (el.type)
        {
          case EAVL_TET:
            edges = eavlTetEdges;
            nedges = 6;
            break;
          case EAVL_PYRAMID:
            edges = eavlPyramidEdges;
            nedges = 8;
            break;
          case EAVL_WEDGE:
            edges = eavlWedgeEdges;
            nedges = 9;
            break;
          case EAVL_HEX:
            edges = eavlHexEdges;
            nedges = 12;
            break;
          case EAVL_VOXEL:
            edges = eavlVoxEdges;
            nedges = 12;
            break;
          case EAVL_TRI:
            edges = eavlTriEdges;
            nedges = 3;
            break;
          case EAVL_QUAD:
            edges = eavlQuadEdges;
            nedges = 4;
            break;
          case EAVL_PIXEL:
            edges = eavlPixelEdges;
            nedges = 4;
            break;
                
          default:
            edges = NULL;
            break;
        }
        cellEdgeConnectivity.shapetype.push_back(cellNodeConnectivity.shapetype[i]);
        cellEdgeConnectivity.connectivity.push_back(nedges);
        for (int j=0; j<nedges; j++)
        {
            eavlEdge edge(el.indices[edges[j][0]],
                          el.indices[edges[j][1]]);
            int index = numEdges;
            if (mapEdgesToEdgeIDs.count(edge) == 0)
            {
                mapEdgesToEdgeIDs[edge] = index;
                numEdges++;
                edgeNodeConnectivity.shapetype.push_back(EAVL_BEAM);
                edgeNodeConnectivity.connectivity.push_back(2);
                edgeNodeConnectivity.connectivity.push_back(edge.a);
                edgeNodeConnectivity.connectivity.push_back(edge.b);
            }
            else
            {
                index = mapEdgesToEdgeIDs[edge];
            }
            cellEdgeConnectivity.connectivity.push_back(index);
        }
    }

    cellEdgeConnectivity.CreateReverseIndex();
    edgeNodeConnectivity.CreateReverseIndex();

    //debug
    //cout << "--- CELL EDGE CONNECTIVITY =\n";
    //cellEdgeConnectivity.PrintSummary(cout);
    //cout << "--- EDGE NODE CONNECTIVITY =\n";
    //edgeNodeConnectivity.PrintSummary(cout);
}

class eavlFace
{
  public:
    int n;      ///< number of points (3 or 4)
    int a,b,c;  ///< IDs used for indexing only
    int ids[4]; ///< actual IDs in clockwise order for one of the adjacent cells
    bool operator<(const eavlFace &o) const
    {
        if (a < o.a)
            return true;
        if (a > o.a)
            return false;
        if (b < o.b)
            return true;
        if (b > o.b)
            return false;
        if (c < o.c)
            return true;
        return false;
    }
    eavlFace() { }
    eavlFace(const eavlFace &f) : n(f.n), a(f.a), b(f.b), c(f.c)
    {
        ids[0] = f.ids[0];
        ids[1] = f.ids[1];
        ids[2] = f.ids[2];
        ids[3] = f.ids[3];
    }
    eavlFace(int x, int y, int z)
    {
        n = 3;
        ids[0] = x;
        ids[1] = y;
        ids[2] = z;
        ids[3] = 0;
        SetFromThree(x,y,z);
    }
    eavlFace(int w, int x, int y, int z)
    {
        n = 4;
        ids[0] = w;
        ids[1] = x;
        ids[2] = y;
        ids[3] = z;
        SetFromFour(w,x,y,z);
    }
    void SetFromFour(int w, int x, int y, int z)
    {
        if (w>x && w>y && w>z) // w is highest
            SetFromThree(x,y,z);
        else if (x>y && x>z)   // x is highest
            SetFromThree(w,y,z);
        else if (y>z)          // y is highest
            SetFromThree(w,x,z);
        else                   // z is highest
            SetFromThree(w,x,y);
    }
    void SetFromThree(int x, int y, int z)
    {
        if (x<y && x<z) // x is lowest
        {
            if (y<z) // z is highest
            {
                a=x; b=y; c=z;
            }
            else // y is highest
            {
                a=x; b=z; c=y;
            }
        }
        else if (y<z)
        {
            // y is lowest
            if (x<z) // z is highest
            {
                a=y; b=x; c=z;
            }
            else // x is highest
            {
                a=y; b=z; c=x;
            }
        }
        else // z is lowest
        {
            if (x<y) // y is highest
            {
                a=z; b=x; c=y;
            }
            else // x is highest
            {
                a=z; b=y; c=x;
            }
        }
    }
};

void eavlCellSetExplicit::BuildFaceConnectivity()
{
    if (numFaces >= 0)
        return; // already done!

    numFaces = 0;

    map<eavlFace,int> mapFacesToFaceIDs; ///<\todo: horribly inefficient

    cellFaceConnectivity.shapetype.clear();
    cellFaceConnectivity.connectivity.clear();
    faceNodeConnectivity.shapetype.clear();
    faceNodeConnectivity.connectivity.clear();

    int nCells = GetNumCells();
    for (int i=0; i<nCells; i++)
    {
        eavlCell el = GetCellNodes(i);

        int ntris = 0;
        signed char (*tris)[3] = NULL;
        int nquads = 0;
        signed char (*quads)[4] = NULL;
        switch (el.type)
        {
          case EAVL_HEX:
            nquads = 6;
            quads = eavlHexQuadFaces;
            break;
          case EAVL_VOXEL:
            nquads = 6;
            quads = eavlVoxQuadFaces;
            break;
          case EAVL_TET:
            ntris = 4;
            tris = eavlTetTriangleFaces;
            break;
          case EAVL_PYRAMID:
            ntris = 4;
            tris = eavlPyramidTriangleFaces;
            nquads = 1;
            quads = eavlPyramidQuadFaces;
            break;
          case EAVL_WEDGE:
            ntris = 2;
            tris = eavlWedgeTriangleFaces;
            nquads = 3;
            quads = eavlWedgeQuadFaces;
            break;
          default:
            break; // do nothing
        }

        cellFaceConnectivity.shapetype.push_back(cellNodeConnectivity.shapetype[i]);
        cellFaceConnectivity.connectivity.push_back(ntris + nquads);
        for (int f=0; f<ntris; f++)
        {
            eavlFace face(el.indices[tris[f][0]],
                          el.indices[tris[f][1]],
                          el.indices[tris[f][2]]);

            int index = numFaces;
            if (mapFacesToFaceIDs.count(face) == 0)
            {
                mapFacesToFaceIDs[face] = index;
                numFaces++;
                faceNodeConnectivity.shapetype.push_back(EAVL_TRI);
                faceNodeConnectivity.connectivity.push_back(3);
                faceNodeConnectivity.connectivity.push_back(face.ids[0]);
                faceNodeConnectivity.connectivity.push_back(face.ids[1]);
                faceNodeConnectivity.connectivity.push_back(face.ids[2]);
            }
            else
            {
                index = mapFacesToFaceIDs[face];
            }
            cellFaceConnectivity.connectivity.push_back(index);
        }
        for (int f=0; f<nquads; f++)
        {
            eavlFace face(el.indices[quads[f][0]],
                          el.indices[quads[f][1]],
                          el.indices[quads[f][2]],
                          el.indices[quads[f][3]]);

            int index = numFaces;
            if (mapFacesToFaceIDs.count(face) == 0)
            {
                mapFacesToFaceIDs[face] = index;
                numFaces++;
                faceNodeConnectivity.shapetype.push_back(EAVL_QUAD);
                faceNodeConnectivity.connectivity.push_back(4);
                faceNodeConnectivity.connectivity.push_back(face.ids[0]);
                faceNodeConnectivity.connectivity.push_back(face.ids[1]);
                faceNodeConnectivity.connectivity.push_back(face.ids[2]);
                faceNodeConnectivity.connectivity.push_back(face.ids[3]);
            }
            else
            {
                index = mapFacesToFaceIDs[face];
            }
            cellFaceConnectivity.connectivity.push_back(index);
        }
    }

    cellFaceConnectivity.CreateReverseIndex();
    faceNodeConnectivity.CreateReverseIndex();

    //debug
    //cout << "--- CELL FACE CONNECTIVITY =\n";
    //cellFaceConnectivity.PrintSummary(cout);
    //cout << "--- FACE NODE CONNECTIVITY =\n";
    //faceNodeConnectivity.PrintSummary(cout);
}

void eavlCellSetExplicit::BuildNodeCellConnectivity()
{
    nodeCellConnectivity.shapetype.clear();
    nodeCellConnectivity.connectivity.clear();

    multimap<int,int> cells_of_nodes;

    int maxNodeID = 0;
    int numCells = GetNumCells();
    for (int cell = 0, cindex = 0; cell < numCells; ++cell)
    {
        int npts = cellNodeConnectivity.connectivity[cindex++];
        for (int pt=0; pt<npts; ++pt)
        {
            int index = cellNodeConnectivity.connectivity[cindex++];
            if (index > maxNodeID)
                maxNodeID = index;
            cells_of_nodes.insert(pair<int,int>(index,cell));
        }
    }

    int filled_array_to_node = 0;
    int cur_node_connstart = 0;
    for (multimap<int,int>::iterator iter = cells_of_nodes.begin();
         iter != cells_of_nodes.end(); iter++)
    {
        int node = iter->first;
        while (filled_array_to_node <= node)
        {
            // add empty spots to skip nodes not referenced by our cells
            // but also create an empty one that we can start adding
            // connectivity items to for ones that are referenced.
            ++filled_array_to_node;
            nodeCellConnectivity.shapetype.push_back(EAVL_POINT);
            cur_node_connstart = nodeCellConnectivity.connectivity.size();
            nodeCellConnectivity.connectivity.push_back(0);
        }
        int cell = iter->second;
        nodeCellConnectivity.connectivity.push_back(cell);
        ++nodeCellConnectivity.connectivity[cur_node_connstart];
    }
    while (filled_array_to_node < dataset_numpoints)
    {
        // add empty spots for tail nodes not referenced by our cells
        ++filled_array_to_node;
        nodeCellConnectivity.shapetype.push_back(EAVL_POINT);
        cur_node_connstart = nodeCellConnectivity.connectivity.size();
        nodeCellConnectivity.connectivity.push_back(0);
    }

    nodeCellConnectivity.CreateReverseIndex();
}
