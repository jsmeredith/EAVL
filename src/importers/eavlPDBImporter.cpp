// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
// This file contains code from VisIt, (c) 2000-2012 LLNS.  See COPYRIGHT.txt.
#include "eavlPDBImporter.h"

#include "eavlAtomicProperties.h"
#include "eavlCellSetExplicit.h"
#include <cstring>

eavlPDBImporter::eavlPDBImporter(const string &fn)
{
    filename = fn;
    OpenFileAtBeginning();

    nmodels = 0;
    metadata_read = false;
    dbTitle = "";
}

eavlPDBImporter::~eavlPDBImporter()
{
    bonds.clear();
    for (size_t i=0; i<allatoms.size(); i++)
    {
        allatoms[i].clear();
    }
    allatoms.clear();

    compoundNames.clear();

    nmodels = 0;
    metadata_read = false;
}

vector<string>
eavlPDBImporter::GetFieldList(const string &mesh)
{
    ReadAllMetaData();

    vector<string> retval;

    retval.push_back("element");
    if (compoundNames.size() > 1)
        retval.push_back("compound");
    retval.push_back("restype");
    retval.push_back("resseq");
    retval.push_back("backbone");
    retval.push_back("occupancy");
    retval.push_back("tempfactor");

    return retval;
}

eavlDataSet *
eavlPDBImporter::GetMesh(const string &mesh, int)
{
    int model = 0;
    ReadAtomsForModel(model);
    vector<Atom> &atoms = allatoms[model];
    int natoms = atoms.size();

    eavlDataSet *data = new eavlDataSet;
    data->SetNumPoints(natoms);

    // points
    eavlCoordinatesCartesian *coords = new eavlCoordinatesCartesian(NULL,
                                              eavlCoordinatesCartesian::X,
                                              eavlCoordinatesCartesian::Y,
                                              eavlCoordinatesCartesian::Z);

    data->AddCoordinateSystem(coords);
    coords->SetAxis(0,new eavlCoordinateAxisField("xcoord",0));
    coords->SetAxis(1,new eavlCoordinateAxisField("ycoord",0));
    coords->SetAxis(2,new eavlCoordinateAxisField("zcoord",0));

    eavlArray *axisValues[3] = {
        new eavlFloatArray("xcoord",1, natoms),
        new eavlFloatArray("ycoord",1, natoms),
        new eavlFloatArray("zcoord",1, natoms)
    };
    for (int i=0; i<natoms; i++)
    {
        axisValues[0]->SetComponentFromDouble(i, 0, atoms[i].x);
        axisValues[1]->SetComponentFromDouble(i, 0, atoms[i].y);
        axisValues[2]->SetComponentFromDouble(i, 0, atoms[i].z);
    }

    for (int d=0; d<3; d++)
    {
        eavlField *field = new eavlField(1, axisValues[d], eavlField::ASSOC_POINTS);
        data->AddField(field);
    }

    // atom cell set:
    // skip for now

    // bonds cell set:
    eavlCellSetExplicit *bondcells = new eavlCellSetExplicit("bonds", 1);
    eavlExplicitConnectivity bondconn;
    for (size_t i=0; i<bonds.size(); i++)
    {
        int ix[2] = {bonds[i].first,bonds[i].second};
        bondconn.AddElement(EAVL_BEAM, 2, ix);
    }
    bondcells->SetCellNodeConnectivity(bondconn);
    data->AddCellSet(bondcells);

    return data;
}

eavlField *
eavlPDBImporter::GetField(const string &varname, const string &mesh, int chunk)
{
    int model = 0;
    ReadAtomsForModel(model);
    vector<Atom> &atoms = allatoms[model];
    int natoms = atoms.size();

    eavlFloatArray *arr = new eavlFloatArray(varname, 1, natoms);
    
    if (string(varname) == "element")
    {
        for (int i=0; i<natoms; i++)
            arr->SetValue(i,atoms[i].atomicnumber);
    }

    if (string(varname) == "restype")
    {
        for (int i=0; i<natoms; i++)
            arr->SetValue(i,atoms[i].residuenumber);
    }

    if (string(varname) == "resseq")
    {
        for (int i=0; i<natoms; i++)
            arr->SetValue(i,atoms[i].resseq);
    }

    if (string(varname) == "backbone")
    {
        for (int i=0; i<natoms; i++)
            arr->SetValue(i,atoms[i].backbone ? 1 : 0);
    }

    if (string(varname) == "compound")
    {
        for (int i=0; i<natoms; i++)
            arr->SetValue(i,atoms[i].compound);
    }

    if (string(varname) == "occupancy")
    {
        for (int i=0; i<natoms; i++)
            arr->SetValue(i,atoms[i].occupancy);
    }

    if (string(varname) == "tempfactor")
    {
        for (int i=0; i<natoms; i++)
            arr->SetValue(i,atoms[i].tempfactor);
    }

    return new eavlField(0, arr, eavlField::ASSOC_POINTS);
}


// ****************************************************************************
//  Function:  AtomsShouldBeBonded
//
//  Purpose:
///   Simple but effective test to see if two atoms are bonded.
//
//  Arguments:
//    atoms      all atoms
//    a1,a2      two atom indices
//
//  Programmer:  Jeremy Meredith
//  Creation:    March 23, 2006
//
// ****************************************************************************

bool
eavlPDBImporter::AtomsShouldBeBonded(const vector<eavlPDBImporter::Atom> &atoms, int a1, int a2)
{
    float dx = atoms[a1].x - atoms[a2].x;
    float dy = atoms[a1].y - atoms[a2].y;
    float dz = atoms[a1].z - atoms[a2].z;
    float dist2 = dx*dx + dy*dy + dz*dz;
    if (dist2 > .4*.4)
    {
        if (atoms[a1].atomicnumber==1 ||
            atoms[a2].atomicnumber==1)
        {
            if (dist2 < 1.2*1.2)
            {
                return true;
            }
        }
        else
        {
            if (dist2 < 1.9*1.9)
            {
                return true;
            }
        }
    }
    return false;
}

// ****************************************************************************
//  Method:  eavlPDBImporter::CreateBondsFromModel_Slow
//
//  Purpose:
///   Search all appropriate atom pairs for bonds using a slower algorithm.
//
//  Arguments:
//    model      the model index to use for distance tests
//
//  Programmer:  Jeremy Meredith
//  Creation:    March 23, 2006
//
//  Modifications:
//    Jeremy Meredith, Mon Aug 28 17:49:30 EDT 2006
//    Bonds are now line segment cells, not an atom-centered 4-comp array.
//
// ****************************************************************************
void
eavlPDBImporter::CreateBondsFromModel_Slow(int model)
{
    // We should only have to create bonds once for all models
    if (bonds.size() > 0)
        return;

    vector<Atom> &atoms = allatoms[model];
    int natoms = atoms.size();
    bonds.reserve(natoms);  // just a guess

    //
    // This is an N^2 algorithm.  Slow, but safe.
    // Don't use it unless there's something wrong
    // with the fast one.
    //
    for (int i=0; i<natoms; i++)
    {
        for (int j=0; j<i; j++)
        {
            if (AtomsShouldBeBonded(atoms,i,j))
            {
                bonds.push_back(pair<int,int>(i,j));
            }
        }
    }
}

// ****************************************************************************
//  Method:  eavlPDBImporter::CreateBondsFromModel_Fast
//
//  Purpose:
///   Search all appropriate atom pairs for bonds using a faster algorithm.
//
//  Arguments:
//    model      the model index to use for distance tests
//
//  Programmer:  Jeremy Meredith
//  Creation:    March 23, 2006
//
//  Modifications:
//    Jeremy Meredith, Mon Aug 28 17:49:30 EDT 2006
//    Bonds are now line segment cells, not an atom-centered 4-comp array.
//
//    Jeremy Meredith, Wed Apr 18 11:02:04 EDT 2007
//    Account for potentially zero atoms.  This only seemed to appear
//    when there was a parsing problem with the file, so maybe it should
//    be changed to throw an error in ReadAtomsForModel()?
//
// ****************************************************************************
void
eavlPDBImporter::CreateBondsFromModel_Fast(int model)
{
    // We should only have to create bonds once for all models
    if (bonds.size() > 0)
        return;

    vector<Atom> &atoms = allatoms[model];
    int natoms = atoms.size();
    if (natoms <= 0)
        return;

    //
    // The strategy here is to divide atoms into 3D spatial bins
    // and compare atoms in some (i,j,k) bin with all atoms in
    // the 27 surrounding bins -- i.e. (i-1,j-1,k-1) thru
    // (i+1,j+1,k+1) -- to find ones that should be bonded.
    //
    // This means that the size of each bin, determined by
    // the value "maxBondDist", should truly at least as
    // large as the maximum bond distance, or else this
    // will fail to catch some bonds.
    //
    // Simultaneously, setting maxBondDist too large means
    // that too many atoms must be compared for bonds, and
    // will likely slow down the algorithm.
    //
    float maxBondDist = 3.0;

    float minx =  FLT_MAX;
    float maxx = -FLT_MAX;
    float miny =  FLT_MAX;
    float maxy = -FLT_MAX;
    float minz =  FLT_MAX;
    float maxz = -FLT_MAX;

    for (int a=0; a<natoms; a++)
    {
        Atom &atom = atoms[a];
        if (atom.x < minx)
            minx = atom.x;
        if (atom.x > maxx)
            maxx = atom.x;
        if (atom.y < miny)
            miny = atom.y;
        if (atom.y > maxy)
            maxy = atom.y;
        if (atom.z < minz)
            minz = atom.z;
        if (atom.z > maxz)
            maxz = atom.z;
    }

    float szx = maxx - minx;
    float szy = maxy - miny;
    float szz = maxz - minz;

    int ni = 1 + int(szx / maxBondDist);
    int nj = 1 + int(szy / maxBondDist);
    int nk = 1 + int(szz / maxBondDist);

    //
    // I know -- I'm using a grid of STL vectors, and this
    // could potentially be inefficient, but I'll wait until
    // I see a problem with this strategy before I change it.
    //
    typedef vector<int> intvec;
    intvec *atomgrid = new intvec[ni*nj*nk];

    for (int a=0; a<natoms; a++)
    {
        Atom &atom = atoms[a];
        int ix = int((atom.x - minx) / maxBondDist);
        int jx = int((atom.y - miny) / maxBondDist);
        int kx = int((atom.z - minz) / maxBondDist);
        atomgrid[ix + ni*(jx + nj*(kx))].push_back(a);
    }

    for (int i=0; i<ni; i++)
    {
        for (int j=0; j<nj; j++)
        {
            for (int k=0; k<nk; k++)
            {
                int index1 = i + ni*(j + nj*(k));
                int na = atomgrid[index1].size();
                for (int a=0; a<na; a++)
                {
                    int ctr = 0;
                    int a1 = atomgrid[index1][a];
                    for (int p=-1; p<=1 && ctr<4; p++)
                    {
                        int ii = i+p;
                        if (ii<0 || ii>=ni)
                            continue;

                        for (int q=-1; q<=1 && ctr<4; q++)
                        {
                            int jj = j+q;
                            if (jj<0 || jj>=nj)
                                continue;

                            for (int r=-1; r<=1 && ctr<4; r++)
                            {
                                int kk = k+r;
                                if (kk<0 || kk>=nk)
                                    continue;

                                int index2 = ii + ni*(jj + nj*(kk));
                                int naa = atomgrid[index2].size();
                                for (int aa=0; aa<naa && ctr<4; aa++)
                                {
                                    if (index1==index2 && a==aa)
                                        continue;

                                    int a2 = atomgrid[index2][aa];

                                    // Only create one direction of
                                    // each bond pair
                                    if (a1 > a2)
                                        continue;

                                    if (AtomsShouldBeBonded(atoms,a1,a2))
                                    {
                                        bonds.push_back(pair<int,int>(a1,a2));
                                        ctr++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    delete[] atomgrid;
}


// ****************************************************************************
//  Method:  eavlPDBImporter::ReadAllMetaData
//
//  Purpose:
///   Open the file and read the meta-data it contains.
///   There isn't much to do for this file format other than
///   count the number of models.
//
//  Arguments:
//    none
//
//  Programmer:  Jeremy Meredith
//  Creation:    March 23, 2006
//
//  Modifications:
//    Jeremy Meredith, Wed Apr 18 10:59:48 EDT 2007
//    Files with non-unixy text formatting (^M's at the end of every line)
//    required allowing for an extra character in getline.
//
//    Jeremy Meredith, Wed Oct 17 11:27:10 EDT 2007
//    Added compound support.
//
//    Jeremy Meredith, Thu Oct 18 16:31:20 EDT 2007
//    COMPND records can be multi-line; ignore all but the first line.
//
//    Jeremy Meredith, Mon Oct 22 12:58:00 EDT 2007
//    Explicitly make "no compound" part of the compound name array.
//    This makes getting the name for any particular compound number easier.
//
//    Jeremy Meredith, Thu Feb 12 12:33:54 EST 2009
//    Moved HETNAM parsing into this function so that we can create the
//    enumerated scalar to include new residue types defined in this file.
//
//    Jeremy Meredith, Thu Jan  7 13:00:03 EST 2010
//    Error on non-ascii data.
//
//    Jeremy Meredith, Fri Jan  8 16:38:33 EST 2010
//    Only check for ASCII data in strict mode (for performance reasons).
//    In strict mode, also check record types against a "complete" set.
//
// ****************************************************************************
void
eavlPDBImporter::ReadAllMetaData()
{
    if (metadata_read)
        return;

    OpenFileAtBeginning();

    metadata_read = true;

    char line[82];
    in.getline(line, 82);
    nmodels = 0;
    int titleLineCount = 0, sourceLineCount = 0;
    bool canReadCompounds = true;
    std::string hetnam, longhetnam;
    bool readingHetnam = false;
    std::string source;
    while (in)
    {
        string record(line,0,6);
        if (readingHetnam && record != "HETNAM")
        {
            AddResiduename(hetnam.c_str(), longhetnam.c_str());
            hetnam = "";
            longhetnam = "";
            readingHetnam = false;
        }

        if (record == "MODEL ")
        {
            // Count the models
            nmodels++;
            // Only read compound names once, even if there are multiple models
            if (compoundNames.size() > 0)
                canReadCompounds = false;
        }
        else if (record == "TITLE ")
        {
            dbTitle += "\n\t";
            dbTitle += string(line + ((titleLineCount > 0) ? 11 : 10));
            titleLineCount++;
        }
        else if (record == "SOURCE")
        {
            source += "\n\t";
            source += string(line + 10);
            sourceLineCount++;
        }
        else if (canReadCompounds &&
                 record == "COMPND" && line[8]==' ' && line[9]==' ')
        {
            if (compoundNames.size() == 0)
            {
                compoundNames.push_back("No compound");
            }
            compoundNames.push_back(string(line + 10));
        }
        else if (record == "HETNAM")
        {
            char het[4];
            memcpy(het, line + 11, 3);
            het[3] = '\0';

            if(hetnam != std::string(het))
            {
                if(readingHetnam)
                {
                    AddResiduename(hetnam.c_str(), longhetnam.c_str());
                    longhetnam = "";
                }

                readingHetnam = true;
            }

            if(longhetnam.size() > 0)
                longhetnam += "\n";
            char *c = line + strlen(line) - 1;
            while (*c == ' ' && c >= line+15)
                *c-- = '\0';
            longhetnam += line + 15;
            hetnam = het;
        }
        in.getline(line, 82);
    }

    if(titleLineCount == 0 && sourceLineCount > 0)
        dbTitle = source;

    allatoms.resize(nmodels==0 ? 1 : nmodels);

    OpenFileAtBeginning();
}

// ****************************************************************************
//  Method:  eavlPDBImporter::OpenFileAtBeginning
//
//  Purpose:
///   We don't want to close and re-open the file every time we want to
///   start back at the beginning, so we encapsulate the logic to both
///   ensure the file is still opened (in case it got closed or was never
///   opened) and to reset the flags and seek back to the beginning, in 
///   this function.
//
//  Arguments:
//    
//
//  Programmer:  Jeremy Meredith
//  Creation:    March 23, 2006
//
// ****************************************************************************
void
eavlPDBImporter::OpenFileAtBeginning()
{
    if (!in.is_open())
    {
        in.open(filename.c_str());
        if (!in)
        {
            THROW(eavlException, "Couldn't open file" + filename);
        }
    }
    else
    {
        in.clear();
        in.seekg(0, ios::beg);
    }
}

// ****************************************************************************
//  Method:  eavlPDBImporter::ReadAtomsForModel
//
//  Purpose:
///   Reads the atom records for the given model.
//
//  Arguments:
//    model      the zero-origin index of the model to read
//
//  Programmer:  Jeremy Meredith
//  Creation:    March 23, 2006
//
//  Modifications:
//    Brad Whitlock, Thu Mar 23 18:27:48 PST 2006
//    Added support for HETNAM.
//
//    Jeremy Meredith, Mon Aug 28 17:53:21 EDT 2006
//    Added support for CONECT records.
//
//    Jeremy Meredith, Wed Apr 18 10:59:48 EDT 2007
//    Files with non-unixy text formatting (^M's at the end of every line)
//    required allowing for an extra character in getline.
//
//    Jeremy Meredith, Wed Oct 17 11:27:10 EDT 2007
//    Added compound support.
//
//    Jeremy Meredith, Thu Oct 18 16:31:20 EDT 2007
//    COMPND records can be multi-line; ignore all but the first line.
//
//    Jeremy Meredith, Thu Feb 12 12:33:54 EST 2009
//    Moved HETNAM parsing out of this function, and into the meta-data
//    reading so that we can create the enumerated scalar with new residues.
//
// ****************************************************************************

void
eavlPDBImporter::ReadAtomsForModel(int model)
{
    ReadAllMetaData();
    OpenFileAtBeginning();

    if (allatoms[model].size() > 0)
        return;

    vector<Atom> &atoms = allatoms[model];    
    atoms.clear();

    char line[82];
    in.getline(line, 82);

    if (nmodels != 0)
    {
        int curmodel = -1;
        while (in && curmodel < model)
        {
            string record(line,0,6);
            if (record == "MODEL ")
            {
                curmodel++;
            }
            in.getline(line, 82);
        }
    }

    int compound = 0;

    while (in)
    {
        string record(line,0,6);

        if (record == "ATOM  ")
        {
            Atom a(line, compound);
            atoms.push_back(a);
        }
        else if (record == "HETATM")
        {
            Atom a(line, compound);
            atoms.push_back(a);
        }
        else if (record == "ENDMDL")
        {
            break;
        }
        else if (record == "CONECT")
        {
            ConnectRecord c(line);
            connect.push_back(c);
            //c.Print(cout);
        }
        else if (record == "COMPND" && line[8]==' ' && line[9]==' ')
        {
            compound++;
        }
        else
        {
            // ignoring record type 'record'
        }

        in.getline(line, 82);
    }

    CreateBondsFromModel(model);
}

// ****************************************************************************
//  Method:  eavlPDBImporter::CreateBondsFromModel
//
//  Purpose:
///   Create the bonds using a distance method.
///   It's disabled right now, but this is also where we would
///   add the bonds from the CONECT records.
//
//  Arguments:
//    model      the model index
//
//  Programmer:  Jeremy Meredith
//  Creation:    August 28, 2006
//
// ****************************************************************************
void
eavlPDBImporter::CreateBondsFromModel(int model)
{
    CreateBondsFromModel_Fast(model);
    
#if 0 // to generate bonds from CONECT records, re-enable this

    // NOTE: this needs to be updated to create bonds
    // as line segments instead of as a 4-comp cell array
    // before it will work.

    // ALSO: the conect records appear to reference atoms by
    // number only within the current compound; this should be
    // checked using a file with >1 compound
    for (int i=0; i<connect.size(); i++)
    {
        const ConnectRecord &c = connect[i];
        int a = c.a - 1; // ASSUME 1-origin atom sequence numbers

        int q = 0;
        for (int q=0; q < 4 && c.b[q] != -1; q++)
        {
            int b = c.b[q] - 1; // ASSUME 1-origin atom sequence numbers
            for (int p=0; p<4; p++)
            {
                if (bonds[p][a] == b)
                {
                    break;
                }

                if (bonds[p][a] == -1)
                {
                    bonds[p][a] = b;
                    break;
                }
            }
        }
    }
#endif
}

// ****************************************************************************
//  Method:  static Scan* functions
//
//  Purpose:
///   Fast functions to get the characters in a line by position.
//
//  Arguments:
//    line       input
//    len        lengths of input line
//    start      index of first character to extract
//    end        index of last character to extract
//    val        where to store the result
//
//  Programmer:  Jeremy Meredith
//  Creation:    August 28, 2006
//
// ****************************************************************************
static inline void
ScanString(const char *line, int len, int start, int end, char *val)
{
    int i;
    int first = start - 1;
    for (i=first; i<end && i<len; i++)
    {
        val[i - first] = line[i];
    }
    val[i - first] = '\0';
}

static char tmpbuff[1024];

static inline void
ScanInt(const char *line, int len, int start, int end, int *val)
{
    int i;
    int first = start - 1;
    for (i=first; i<end && i<len; i++)
    {
        tmpbuff[i - first] = line[i];
    }
    tmpbuff[i - first] = '\0';
    *val = atoi(tmpbuff);
}

static inline void
ScanChar(const char *line, int len, int start, char *val)
{
    if (len < start)
        *val = '\0';
    else
        *val = tmpbuff[start-1];
}

static inline void
ScanFloat(const char *line, int len, int start, int end, float *val)
{
    int i;
    int first = start - 1;
    for (i=first; i<end && i<len; i++)
    {
        tmpbuff[i - first] = line[i];
    }
    tmpbuff[i - first] = '\0';

    //sscanf(tmpbuff, "%f", val);
    *val = atof(tmpbuff);
}

// ****************************************************************************
//  Constructor:  Atom::Atom
//
//  Arguments:
//    line       the line of text in a PDB file
//
//  Programmer:  Jeremy Meredith
//  Creation:    March 23, 2006
//
//  Modifications:
//    Brad Whitlock, Fri Jun 2 13:15:47 PST 2006
//    Added Jeremy's fix for yet another style of ATOM line.
//
//    Jeremy Meredith, Mon Aug 28 17:58:02 EDT 2006
//    Changed the scanning to (a) match the PDB spec document more 
//    effectively, (b) be faster, and (c) handle some missing elements
//    (short lines) better.
//
//    Jeremy Meredith, Wed Oct 17 11:27:10 EDT 2007
//    Added compound support.
//
// ****************************************************************************
eavlPDBImporter::Atom::Atom(const char *line, int cmpnd)
{
    char record[7];
    int len = strlen(line);
    ScanString(line, len,  1,  6,  record);
    ScanInt   (line, len,  7, 11, &serial);
    ScanString(line, len, 13, 16,  name);
    ScanChar  (line, len, 17,     &altloc);
    ScanString(line, len, 18, 20,  resname);
    ScanChar  (line, len, 22,     &chainid);
    ScanInt   (line, len, 23, 26, &resseq);
    ScanChar  (line, len, 27,     &icode);
    ScanFloat (line, len, 31, 38, &x);
    ScanFloat (line, len, 39, 46, &y);
    ScanFloat (line, len, 47, 54, &z);
    ScanFloat (line, len, 55, 60, &occupancy);
    ScanFloat (line, len, 61, 66, &tempfactor);
    ScanString(line, len, 73, 76,  segid);
    ScanString(line, len, 77, 78,  element);
    ScanString(line, len, 79, 80,  charge);

    // Left-justify element names
    if (element[0] == ' ')
    {
        element[0] = element[1];
        element[1] = '\0';
    }

    if((atomicnumber = ElementNameToAtomicNumber(element)) < 0)
    {
        // We have a weird file that does not keep the element name in
        // the place designated by the ATOM record. Files like this seem
        // to use the space for a line number. Check columns 12,13
        // for the atom number.
        if(line[12] == ' ' || (line[12] >= '0' && line[12] <= '9'))
        {
            element[0] = line[13];
            element[1] = '\0';
        }
        else if (line[13] >= '0' && line[13] <= '9')
        {
            element[0] = line[12];
            element[1] = '\0';
        }
        else
        {
            element[0] = line[12];
            element[1] = line[13];
        }

        atomicnumber = ElementNameToAtomicNumber(element);
        if (atomicnumber < 0 &&
            element[1] != '\0')
        {
            element[1] = '\0';
            atomicnumber = ElementNameToAtomicNumber(element);
        }

        if (atomicnumber < 0)
        {
            char msg[2000];
            snprintf(msg, 2000, "Unknown element name <%s> in line: %s",
                     element, line);
            THROW(eavlException, msg);
        }
    }

    // Shift spaces out of the resname.
    if(resname[0] == ' ')
    {
        if(resname[1] == ' ')
        {
            resname[0] = resname[2];
            resname[1] = '\0';
        }
        else
        {
            resname[0] = resname[1];
            resname[1] = resname[2];
            resname[2] = '\0';
        }
    }
    // Look up the residue number from the name.
    if((residuenumber = ResiduenameToNumber(resname)) < 0)
    {
        residuenumber = 0;
    }

    backbone = false;
    if (strcmp(name, " N  ")==0 ||
        strcmp(name, " C  ")==0 ||
        strcmp(name, " CA ")==0)
    {
        backbone = true;
    }

    compound = cmpnd;
}

// ****************************************************************************
//  Method:  Atom::Print
//
//  Purpose:
///   Print the atom to a stream.
//
//  Arguments:
//    out        the ostream.
//
//  Programmer:  Jeremy Meredith
//  Creation:    March 23, 2006
//
// ****************************************************************************
void eavlPDBImporter::Atom::Print(ostream &out)
{
    out << "Atom:\n"
        << " serial   ="<<serial<<endl
        << " name     ="<<name<<endl
        << " altloc   ="<<altloc<<endl
        << " resname  ="<<resname<<endl
        << " chainid  ="<<chainid<<endl
        << " resseq   ="<<resseq<<endl
        << " icode    ="<<icode<<endl
        << " x        ="<<x<<endl
        << " y        ="<<y<<endl
        << " z        ="<<z<<endl
        << " occupancy="<<occupancy<<endl
        << " tempfact ="<<tempfactor<<endl
        << " segid    ="<<segid<<endl
        << " element  ="<<element<<endl
        << " charge   ="<<charge<<endl;
}


// ****************************************************************************
//  Constructor:  ConnectRecord::ConnectRecord
//
//  Programmer:  Jeremy Meredith
//  Creation:    August 28, 2006
//
// ****************************************************************************
eavlPDBImporter::ConnectRecord::ConnectRecord(const char *origline)
{
    // We need to prevent this from trying to
    // skip over whitespace, as the last three
    // of these fields are optional, but there
    // may be more stuff later on the line.  This
    // probably means sscanf is not the best way
    // to accomplish this.
    char line[82];
    strcpy(line, origline);
    line[31] = '\0';

    char record[7];
    b[0] = -1;
    b[1] = -1;
    b[2] = -1;
    b[3] = -1;
    sscanf(line, "%6c%5d%5d%5d%5d%5d",
           record,
           &a,
           &b[0], &b[1], &b[2], &b[3]);
}

// ****************************************************************************
//  Method:  ConnectRecord::Print
//
//  Purpose:
///   Print the connect record contets.
//
//  Programmer:  Jeremy Meredith
//  Creation:    August 28, 2006
//
// ****************************************************************************
void
eavlPDBImporter::ConnectRecord::Print(ostream &out)
{
    out << "Connect Record:\n"
        << "a  = "<<a<<endl
        << "b1 = "<<b[0]<<endl
        << "b2 = "<<b[1]<<endl
        << "b3 = "<<b[2]<<endl
        << "b4 = "<<b[3]<<endl;
}
