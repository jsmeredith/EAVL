// Copyright 2010-2012 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_LOGICAL_STRUCTURE_QUADTREE_H
#define EAVL_LOGICAL_STRUCTURE_QUADTREE_H

#include "eavlException.h"

class eavlLogicalStructureQuadTree : public eavlLogicalStructure
{
  public:
    class QuadTreeCell
    {
      public:
        int lvl, x, y;
        float xmin, xmax, ymin, ymax;
        std::vector<QuadTreeCell> children;
#define K 3
        float coeffs[K][K];
        void Print(std::ostream&,int=0);
        bool  HasValue(float x, float y);
        float GetValue(float x, float y);
        int GetNumCells(bool leafOnly);
        QuadTreeCell *GetNthCell(int i);
    };
    QuadTreeCell root;
    vector<QuadTreeCell*> celllist;
    void BuildLeafCellList()
    {
        int n = root.GetNumCells(true);
        for (int i=0; i<n; i++)
        {
            QuadTreeCell *cell = root.GetNthCell(i);
            celllist.push_back(cell);
        }
    }

    eavlLogicalStructureQuadTree() : eavlLogicalStructure(1) { }
    virtual void PrintSummary(ostream &out)
    {
        out << "   eavlLogicalStructureQuadTree:"<<endl;
        out << "     total number of cells = "<<root.GetNumCells(true)<<endl;
    }
};

///\todo: This isn't a clean inheritance from eavlCoordinates;
///       the base class functionality is totally ignored and changed.
class eavlCoordinatesQuadTree : public eavlCoordinates
{
    ///\todo: a specific example: do we need the logical structure
    /// passed into eavlCoordinates constructure?  NULL is a 
    /// horrible idea here; need to change that, too.
  public:
    eavlCoordinatesQuadTree() : eavlCoordinates(2, NULL)
    {
        SetAxis(0, new eavlCoordinateAxisRegular(0, 0.0, 1.0));
        SetAxis(1, new eavlCoordinateAxisRegular(1, 0.0, 1.0));
    }
    virtual double GetCartesianPoint(int i, int c,
                                     eavlLogicalStructure *log,
                                     vector<eavlField*>&fd)
    {
        eavlLogicalStructureQuadTree *l = dynamic_cast<eavlLogicalStructureQuadTree*>(log);
        if (!l)
            THROW(eavlException,"Expected eavlLogicalStructureQuadTree in GetPoint");
        if (l->celllist.size() == 0)
            THROW(eavlException,"Haven't yet built leaf cell list for logical structure");
        if ((i/4) >= l->celllist.size())
            THROW(eavlException,"Asked for more cells than we have in quad tree");
        //cerr << "Asking for point "<<i<<" cell "<<(i/4)<<":\n";
        eavlLogicalStructureQuadTree::QuadTreeCell *cell = l->celllist[i/4];
        int which = i%4;
        if (c == 0) // x
        {
            if (which==0 || which==2)
                return cell->xmin;
            else
                return cell->xmax;
        }
        else if (c == 1) // y
        {
            if (which==0 || which==1)
                return cell->ymin;
            else
                return cell->ymax;
        }
        else
        {
            ///\todo: throw: why is someone asking for this?
            return 0;
        }
    }
    virtual int GetDimension() { return 2; }
    virtual void PrintSummary(ostream &out)
    {
        out << "    eavlCoordinatesQuadTree"<<endl;
    }
};

/*
// note: this is the code for the case where we're using internal tree nodes, too
eavlLogicalStructureQuadTree::QuadTreeCell*
eavlLogicalStructureQuadTree::QuadTreeCell::GetNthCell(int i)
{
    if (i == 0)
        return this;
    i -= 1; // not this node
    for (int j=0; j<children.size(); j++)
    {
        int n = children[j].GetNumCells(false);
        if (i < n)
            return children[j].GetNthCell(i);
        i -= n;
    }

    THROW(eavlException,"not enough cells in the mesh!");
}
*/

eavlLogicalStructureQuadTree::QuadTreeCell*
eavlLogicalStructureQuadTree::QuadTreeCell::GetNthCell(int i)
{
    //cerr << "  i="<<i<<endl;
    for (int j=0; j<children.size(); j++)
    {
        int n = children[j].GetNumCells(true);
        //cerr << "    child #"<<j<<" has "<<n<<" cells\n";
        if (i < n)
        {
            //cerr << "      -- descending\n";
            return children[j].GetNthCell(i);
        }
        i -= n;
    }
    if (i == 0)
    {
        //cerr << "      -- found it\n";
        return this;
    }
    
    THROW(eavlException,"not enough cells in the mesh!");
}

void
eavlLogicalStructureQuadTree::QuadTreeCell::Print(std::ostream &out,int lvl)
{
    out << string(lvl*3,' ');
    out << "("<<lvl<<",["<<x<<","<<y<<"])  "
        <<"    extents="<<xmin<<","<<xmax<<","<<ymin<<","<<ymax<<"\n";
    if (children.size() > 0)
    {
        for (int i=0; i<children.size(); i++)
            children[i].Print(out,lvl+1);
    }
    else
    {
        for (int i=0; i<3; i++)
        {
            out << string(lvl*3,' ');
            out << " coeffs["<<i<<",*] = ";
            for (int j=0; j<3; j++)
            {
                out << coeffs[i][j]<<" ";
            }
            out << endl;
        }
    }
}

bool
eavlLogicalStructureQuadTree::QuadTreeCell::HasValue(float x, float y)
{
    bool val = (x>=xmin &&
                x<=xmax &&
                y>=ymin &&
                y<=ymax);
    //cerr << "hasvalue, level="<<lvl<<"  extents="<<xmin<<","<<xmax<<","<<ymin<<","<<ymax<<"\n";
    return val;
}
 
float Legendre(int i, float x)
{
    float scale = sqrt(2 * i + 1);
    switch (i)
    {
      case 0:    return scale * 1;
      case 1:    return scale * x;
      case 2:    return scale * (3. * x*x -1) / 2.;
    }
    return -9999999999;
}

int
eavlLogicalStructureQuadTree::QuadTreeCell::GetNumCells(bool leafOnly)
{
    int subTree = 0;
    for (int i=0; i<children.size(); i++)
        subTree += children[i].GetNumCells(leafOnly);
    if (!leafOnly || children.size() == 0)
        subTree++;
    return subTree;
}

float
eavlLogicalStructureQuadTree::QuadTreeCell::GetValue(float X, float Y)
{
    if (children.size() > 0)
    {
        for (int i=0; i<children.size(); i++)
        {
            if (children[i].HasValue(X,Y))
                return children[i].GetValue(X,Y);
        }
        //cerr << "eh?\n";
    }
    //cerr << "Evaluating at level="<<lvl<<"  x,y="<<x<<","<<y<<"\n";
    //Print(cerr);
    // no children had it, so it's gotta be us
    float xx = -1 + 2. * (X - xmin) / (xmax - xmin);
    float yy = -1 + 2. * (Y - ymin) / (ymax - ymin);
    float scale = sqrt(1 << (lvl-1));
    float sum = 0;
    /*float x0 = Legendre(0, xx);
    float x1 = Legendre(1, xx);
    float x2 = Legendre(2, xx);
    float y0 = Legendre(0, yy);
    float y1 = Legendre(1, yy);
    float y2 = Legendre(2, yy);*/

    /*
    cerr << "scale="<<scale<<endl;
    for (int i=0; i<3; i++)
        cerr << "Legendre("<<i<<", x="<<xx<<") = "<<Legendre(i,xx)<<endl;
    for (int j=0; j<3; j++)
        cerr << "Legendre("<<j<<", y="<<yy<<") = "<<Legendre(j,yy)<<endl;

    for (int i=0; i<3; i++)
        cerr << "scaled Legendre("<<i<<", x="<<xx<<") = "<<Legendre(i,xx)*scale<<endl;
    for (int j=0; j<3; j++)
        cerr << "scaled Legendre("<<j<<", y="<<yy<<") = "<<Legendre(j,yy)*scale<<endl;
    */

    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            float v = coeffs[i][j] * Legendre(i, xx) * Legendre(j, yy) *scale*scale;
            //cerr << "i="<<i<<" j="<<j<<" val="<<v<<endl;
            sum += v;
        }
    }
    return sum;
}

#endif
