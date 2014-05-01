// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlMADNESSImporter.h"

#include "eavlCoordinates.h"
#include "eavlCellSetAllQuadTree.h"
#include "eavlCellSetExplicit.h"
#include "eavlCellSetAllStructured.h"
#include "eavlException.h"

static bool
ParseNodeHeader(ifstream &in,
                int &l, int &x, int &y,
                bool &coeff, bool &children)
{
    string buff;
    char c;

    // find the open paren
    c = in.get();
    while (in.good() && c != '(')
        c = in.get();
    if (!in.good())
        return false;

    // parse the number until the comma, it's the level
    buff = "";
    c = in.get();
    while (in.good() && c != ',')
    {
        buff += c;
        c = in.get();
    }
    if (!in.good())
        return false;
    l = atoi(buff.c_str());

    // get the open bracket
    c = in.get();
    if (!in.good() || c != '[')
        return false;

    // parse the number until the comma, it's the X index
    buff = "";
    c = in.get();
    while (in.good() && c != ',')
    {
        buff += c;
        c = in.get();
    }
    if (!in.good())
        return false;
    x = atoi(buff.c_str());

    // parse the number until the close bracket, it's the Y index
    buff = "";
    c = in.get();
    while (in.good() && c != ']')
    {
        buff += c;
        c = in.get();
    }
    if (!in.good())
        return false;
    y = atoi(buff.c_str());

    // get the close paren
    c = in.get();
    if (!in.good() || c != ')')
        return false;

    // find the open paren for the flags
    while (in.good() && c != '(')
        c = in.get();
    if (!in.good())
        return false;

    // parse the text until the equal, it's the text "has_coeff"
    buff = "";
    c = in.get();
    while (in.good() && c != '=')
    {
        buff += c;
        c = in.get();
    }
    if (!in.good() || buff != "has_coeff")
        return false;

    // parse the text until the comma, it's either "true" or "false"
    buff = "";
    c = in.get();
    while (in.good() && c != ',')
    {
        buff += c;
        c = in.get();
    }
    if (!in.good())
        return false;
    if (buff == "true")
        coeff = true;
    else if (buff == "false")
        coeff = false;
    else
        return false; // error

    // skip whitespace
    c = in.get();
    while (in.good() && c == ' ')
    {
        c = in.get();
    }
    if (!in.good())
        return false;

    // parse the text until the equal, it's the text "has_children"
    buff = "";
    while (in.good() && c != '=')
    {
        buff += c;
        c = in.get();
    }
    if (!in.good() || buff != "has_children")
        return false;

    // parse the text until the comma, it's either "true" or "false"
    buff = "";
    c = in.get();
    while (in.good() && c != ',')
    {
        buff += c;
        c = in.get();
    }
    if (!in.good())
        return false;
    if (buff == "true")
        children = true;
    else if (buff == "false")
        children = false;
    else
        return false; // error

    // skip the rest of the line
    char tmp[4096];
    in.getline(tmp,4096);
    return true;
}

static bool
ParseCoeffRow(ifstream &in, float *coeffs)
{
    char c;

    // find the open bracket
    c = in.get();
    while (in.good() && c != '[')
        c = in.get();
    if (!in.good())
        return false;
    // and the close
    while (in.good() && c != ']')
        c = in.get();
    if (!in.good())
        return false;

    // get each coeff
    string buff;
    for (int i=0; i<K; i++)
    {
        // skip whitespace
        c = in.get();
        while (in.good() && c == ' ')
            c = in.get();
        if (!in.good())
            return false;

        // get data
        while (in.good() && c != ' ' && c != '\n' && c != '\r' && c != '\t')
        {
            buff += c;
            c = in.get();
        }
        if (!in.good())
            return false;
        //cerr << "buff="<<buff<<endl;
        coeffs[i] = atof(buff.c_str());
        buff = "";
    }
    return true;
}

static bool
ParseNode(ifstream &in, eavlLogicalStructureQuadTree::QuadTreeCell &node)
{
    int level, x, y;
    bool hascoeff, haschildren;
    if (!ParseNodeHeader(in, level,x,y, hascoeff, haschildren))
        return false;

    node.lvl = level;
    node.x   = x;
    node.y   = y;
    //cerr << "parsing node "<<level<<", ["<<x<<","<<y<<"]: coeff="<<hascoeff<<endl;

    if (hascoeff)
    {
        for (int i=0; i<K; i++)
        {
            ParseCoeffRow(in, node.coeffs[i]);
        }
    }
    else
    {
        // skip the "empty tensor" line
        char buff[4096];
        in.getline(buff,4096);
        for (int i=0; i<K; i++)
            for (int j=0; j<K; j++)
                node.coeffs[i][j]=0;
    }

    if (haschildren)
    {
        node.children.resize(4);

        node.children[0].xmin = node.xmin;
        node.children[0].xmax = (node.xmin+node.xmax)/2.;
        node.children[0].ymin = node.ymin;
        node.children[0].ymax = (node.ymin+node.ymax)/2.;

        node.children[1].xmin = (node.xmin+node.xmax)/2.;
        node.children[1].xmax = node.xmax;
        node.children[1].ymin = node.ymin;
        node.children[1].ymax = (node.ymin+node.ymax)/2.;

        node.children[2].xmin = node.xmin;
        node.children[2].xmax = (node.xmin+node.xmax)/2.;
        node.children[2].ymin = (node.ymin+node.ymax)/2.;
        node.children[2].ymax = node.ymax;

        node.children[3].xmin = (node.xmin+node.xmax)/2.;
        node.children[3].xmax = node.xmax;
        node.children[3].ymin = (node.ymin+node.ymax)/2.;
        node.children[3].ymax = node.ymax;

        ParseNode(in, node.children[0]);
        ParseNode(in, node.children[1]);
        ParseNode(in, node.children[2]);
        ParseNode(in, node.children[3]);
    }

    return true;
}

eavlMADNESSImporter::eavlMADNESSImporter(const string &fn)
{
    log = new eavlLogicalStructureQuadTree();
    log->root.xmin = -1;
    log->root.xmax = +1;
    log->root.ymin = -1;
    log->root.ymax = +1;

    ifstream in(fn.c_str());
    if (!in)
        THROW(eavlException,"Error opening given filename in MADNESS importer.");

    char buff[4096];
    in.getline(buff,4096);

    bool success = ParseNode(in, log->root);
    log->BuildLeafCellList();
    if (!success)
        THROW(eavlException,"Error parsing MADNESS file");

    //log->root.Print(cout);
    //cerr << "MADNESS TREE: "
    //     << log->root.GetNumCells(true)<<" leaf cells, "
    //     << log->root.GetNumCells(false)<<" total cell count\n";

    in.close();
}

eavlMADNESSImporter::~eavlMADNESSImporter()
{
}

vector<string> 
eavlMADNESSImporter::GetFieldList(const string &mesh)
{
    vector<string> fields;
    fields.push_back("levels");
    fields.push_back("cell_const");
    fields.push_back("node_linear");
    fields.push_back("cell_biquadratic");
    return fields;
}

int
eavlMADNESSImporter::GetNumChunks(const string &mesh)
{
    return 1;
}

eavlDataSet*
eavlMADNESSImporter::GetMesh(const string &mesh, int)
{
    eavlCoordinatesQuadTree *coords = new eavlCoordinatesQuadTree();

    eavlDataSet *data = new eavlDataSet;
    data->SetNumPoints(log->root.GetNumCells(true) * 4);
    data->SetLogicalStructure(log);
    data->AddCoordinateSystem(coords);

    eavlCellSetAllQuadTree *el = new eavlCellSetAllQuadTree("AllQuadTreeCells",log);
    data->AddCellSet(el);

    return data;
}

eavlField *
eavlMADNESSImporter::GetField(const string &name, const string &mesh, int chunk)
{
    if (name == "levels")
    {
        int ncells = log->root.GetNumCells(true);
        eavlFloatArray *arr = new eavlFloatArray(name,1);
        arr->SetNumberOfTuples(ncells);
        for (int i=0; i<ncells; i++)
        {
            eavlLogicalStructureQuadTree::QuadTreeCell *n = log->root.GetNthCell(i);
            arr->SetComponentFromDouble(i, 0, n->lvl);
        }
     
        eavlField *field = new eavlField(0, arr, eavlField::ASSOC_CELL_SET, "AllQuadTreeCells");

        return field;
    }
    else if (name == "cell_const")
    {
        int ncells = log->root.GetNumCells(true);
        eavlFloatArray *arr = new eavlFloatArray(name,1);
        arr->SetNumberOfTuples(ncells);
        for (int i=0; i<ncells; i++)
        {
            eavlLogicalStructureQuadTree::QuadTreeCell *n = log->root.GetNthCell(i);
            // evaluate the legendre polynomials at the center of the node
            float v = n->GetValue((n->xmin + n->xmax)/2.,
                                  (n->ymin + n->ymax)/2.);
            arr->SetComponentFromDouble(i, 0, v);
        }
     
        eavlField *field = new eavlField(0, arr, eavlField::ASSOC_CELL_SET, "AllQuadTreeCells");
        return field;
    }
    else if (name == "node_linear")
    {
        int ncells = log->root.GetNumCells(true);
        int nnodes = ncells*4;
        eavlFloatArray *arr = new eavlFloatArray(name,1);
        arr->SetNumberOfTuples(nnodes);
        for (int i=0; i<ncells; i++)
        {
            eavlLogicalStructureQuadTree::QuadTreeCell *n = log->root.GetNthCell(i);
            arr->SetComponentFromDouble(i*4+0, 0,
                                        n->GetValue(n->xmin, n->ymin));
            arr->SetComponentFromDouble(i*4+1, 0,
                                        n->GetValue(n->xmax, n->ymin));
            arr->SetComponentFromDouble(i*4+2, 0,
                                        n->GetValue(n->xmin, n->ymax));
            arr->SetComponentFromDouble(i*4+3, 0,
                                        n->GetValue(n->xmax, n->ymax));
        }
     
        eavlField *field = new eavlField(1, arr, eavlField::ASSOC_POINTS);
        return field;
    }
    else if (name == "cell_biquadratic")
    {
        int ncells = log->root.GetNumCells(true);
        eavlFloatArray *arr = new eavlFloatArray(name,9);
        arr->SetNumberOfTuples(ncells);
        for (int i=0; i<ncells; i++)
        {
            eavlLogicalStructureQuadTree::QuadTreeCell *n = log->root.GetNthCell(i);
            for (int j=0; j<9; j++)
                arr->SetComponentFromDouble(i, j, n->coeffs[j/3][j%3]);
        }
     
        eavlField *field = new eavlField(2, arr, eavlField::ASSOC_CELL_SET, "AllQuadTreeCells");
        return field;
    }

    THROW(eavlException,"unknown var");
}
