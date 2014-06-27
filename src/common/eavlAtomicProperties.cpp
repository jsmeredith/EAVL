// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
// This file contains code from VisIt, (c) 2000-2012 LLNS.  See COPYRIGHT.txt.
#include "eavl.h"
#include "eavlAtomicProperties.h"
#include "STL.h"
#include <cstring>

static std::map<std::string, int> elementname_to_atomicnumber;
static std::map<std::string, int> residuename_to_number;
static std::map<int, std::string> number_to_residuename;
static std::map<std::string, int> residuename_to_abbr;
static std::map<std::string, std::string> residuename_to_longname;

float covalent_radius[MAX_ELEMENT_NUMBER+1] = {
     .1f,  // ?  0
    0.32f, // H  1
    0.93f, // He 2
    1.23f, // Li 3
    0.90f, // Be 4
    0.82f, // B  5
    0.77f, // C  6
    0.75f, // N  7
    0.73f, // O  8
    0.72f, // F  9
    0.71f, // Ne 10
    1.54f, // Na 11
    1.36f, // Mg 12
    1.18f, // Al 13
    1.11f, // Si 14
    1.06f, // P  15
    1.02f, // S  16
    0.99f, // Cl 17
    0.98f, // Ar 18
    2.03f, // K  19
    1.74f, // Ca 20
    1.44f, // Sc 21
    1.32f, // Ti 22
    1.22f, // V  23
    1.18f, // Cr 24
    1.17f, // Mn 25
    1.17f, // Fe 26
    1.16f, // Co 27
    1.15f, // Ni 28
    1.17f, // Cu 29
    1.25f, // Zn 30
    1.26f, // Ga 31
    1.22f, // Ge 32
    1.20f, // As 33
    1.16f, // Se 34
    1.14f, // Br 35
    1.12f, // Kr 36
    2.16f, // Rb 37
    1.91f, // Sr 38
    1.62f, // Y  39
    1.45f, // Zr 40
    1.34f, // Nb 41
    1.30f, // Mo 42
    1.27f, // Tc 43
    1.25f, // Ru 44
    1.25f, // Rh 45
    1.28f, // Pd 46
    1.34f, // Ag 47
    1.48f, // Cd 48
    1.44f, // In 49
    1.41f, // Sn 50
    1.40f, // Sb 51
    1.36f, // Te 52
    1.33f, // I  53
    1.31f, // Xe 54
    2.35f, // Cs 55
    1.98f, // Ba 56
    1.69f, // La 57
    1.65f, // Ce 58
    1.65f, // Pr 59
    1.64f, // Nd 60
    1.63f, // Pm 61
    1.62f, // Sm 62
    1.85f, // Eu 63
    1.61f, // Gd 64
    1.59f, // Tb 65
    1.59f, // Dy 66
    1.58f, // Ho 67
    1.57f, // Er 68
    1.56f, // Tm 69
    1.74f, // Yb 70
    1.56f, // Lu 71
    1.44f, // Hf 72
    1.34f, // Ta 73
    1.30f, // W  74
    1.28f, // Re 75
    1.26f, // Os 76
    1.27f, // Ir 77
    1.30f, // Pt 78
    1.34f, // Au 79
    1.49f, // Hg 80
    1.48f, // Tl 81
    1.47f, // Pb 82
    1.46f, // Bi 83
    1.46f, // Po 84
    1.45f, // At 85
    1.50f, // Rn 86 // 86-96 from en.wikipedia.org/wiki/Covalent_radius 10/6/09
    2.60f, // Fr 87
    2.21f, // Ra 88
    2.15f, // Ac 89
    2.06f, // Th 90
    2.00f, // Pa 91
    1.96f, // U  92
    1.90f, // Np 93
    1.87f, // Pu 94
    1.80f, // Am 95
    1.69f, // Cm 96
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f
};

float atomic_radius[MAX_ELEMENT_NUMBER+1] = {
     .1f,  // ?  0
    0.79f, // H  1
    0.49f, // He 2
    2.05f, // Li 3
    1.40f, // Be 4
    1.17f, // B  5
    0.91f, // C  6
    0.75f, // N  7
    0.65f, // O  8
    0.57f, // F  9
    0.51f, // Ne 10
    2.23f, // Na 11
    1.72f, // Mg 12
    1.82f, // Al 13
    1.46f, // Si 14
    1.23f, // P  15
    1.09f, // S  16
    0.97f, // Cl 17
    0.88f, // Ar 18
    2.77f, // K  19
    2.23f, // Ca 20
    2.09f, // Sc 21
    2.00f, // Ti 22
    1.92f, // V  23
    1.85f, // Cr 24
    1.79f, // Mn 25
    1.72f, // Fe 26
    1.67f, // Co 27
    1.62f, // Ni 28
    1.57f, // Cu 29
    1.53f, // Zn 30
    1.81f, // Ga 31
    1.52f, // Ge 32
    1.33f, // As 33
    1.22f, // Se 34
    1.12f, // Br 35
    1.03f, // Kr 36
    2.98f, // Rb 37
    2.45f, // Sr 38
    2.27f, // Y  39
    2.16f, // Zr 40
    2.08f, // Nb 41
    2.01f, // Mo 42
    1.95f, // Tc 43
    1.89f, // Ru 44
    1.83f, // Rh 45
    1.79f, // Pd 46
    1.75f, // Ag 47
    1.71f, // Cd 48
    2.00f, // In 49
    1.72f, // Sn 50
    1.53f, // Sb 51
    1.42f, // Te 52
    1.32f, // I  53
    1.24f, // Xe 54
    3.34f, // Cs 55
    2.78f, // Ba 56
    2.74f, // La 57
    2.70f, // Ce 58
    2.67f, // Pr 59
    2.64f, // Nd 60
    2.62f, // Pm 61
    2.59f, // Sm 62
    2.56f, // Eu 63
    2.54f, // Gd 64
    2.51f, // Tb 65
    2.49f, // Dy 66
    2.47f, // Ho 67
    2.45f, // Er 68
    2.42f, // Tm 69
    2.40f, // Yb 70
    2.25f, // Lu 71
    2.16f, // Hf 72
    2.09f, // Ta 73
    2.02f, // W  74
    1.97f, // Re 75
    1.92f, // Os 76
    1.87f, // Ir 77
    1.83f, // Pt 78
    1.79f, // Au 79
    1.76f, // Hg 80
    2.08f, // Tl 81
    1.81f, // Pb 82
    1.63f, // Bi 83
    1.53f, // Po 84
    1.43f, // At 85
    1.34f, // Rn 86
    2.60f, // Fr 87 (using covalent radius just to get a number here)
    2.15f, // Ra 88 (88-99 from "http://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)", which references J.C. Slater, J. Chem. Phys. 1964, 41, 3199.)
    1.95f, // Ac 89
    1.80f, // Th 90
    1.80f, // Pa 91
    1.75f, // U  92
    1.75f, // Np 93
    1.75f, // Pu 94
    1.75f, // Am 95
    1.74f, // Cm 96 (96-99 use metallic radius to just get a number)
    1.70f, // Bk 97
    1.86f, // Cf 98
    1.86f, // Es 99
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f,
    .1f
};

const char *element_names[MAX_ELEMENT_NUMBER+1] = {
    "?",  // 0
    "H",  // 1
    "He", // 2
    "Li", // 3
    "Be", // 4
    "B",  // 5
    "C",  // 6
    "N",  // 7
    "O",  // 8
    "F",  // 9
    "Ne", // 10
    "Na", // 11
    "Mg", // 12
    "Al", // 13
    "Si", // 14
    "P",  // 15
    "S",  // 16
    "Cl", // 17
    "Ar", // 18
    "K",  // 19
    "Ca", // 20
    "Sc", // 21
    "Ti", // 22
    "V",  // 23
    "Cr", // 24
    "Mn", // 25
    "Fe", // 26
    "Co", // 27
    "Ni", // 28
    "Cu", // 29
    "Zn", // 30
    "Ga", // 31
    "Ge", // 32
    "As", // 33
    "Se", // 34
    "Br", // 35
    "Kr", // 36
    "Rb", // 37
    "Sr", // 38
    "Y",  // 39
    "Zr", // 40
    "Nb", // 41
    "Mo", // 42
    "Tc", // 43
    "Ru", // 44
    "Rh", // 45
    "Pd", // 46
    "Ag", // 47
    "Cd", // 48
    "In", // 49
    "Sn", // 50
    "Sb", // 51
    "Te", // 52
    "I",  // 53
    "Xe", // 54
    "Cs", // 55
    "Ba", // 56
    "La", // 57
    "Ce", // 58
    "Pr", // 59
    "Nd", // 60
    "Pm", // 61
    "Sm", // 62
    "Eu", // 63
    "Gd", // 64
    "Tb", // 65
    "Dy", // 66
    "Ho", // 67
    "Er", // 68
    "Tm", // 69
    "Yb", // 70
    "Lu", // 71
    "Hf", // 72
    "Ta", // 73
    "W",  // 74
    "Re", // 75
    "Os", // 76
    "Ir", // 77
    "Pt", // 78
    "Au", // 79
    "Hg", // 80
    "Tl", // 81
    "Pb", // 82
    "Bi", // 83
    "Po", // 84
    "At", // 85
    "Rn", // 86
    "Fr", // 87
    "Ra", // 88
    "Ac", // 89
    "Th", // 90
    "Pa", // 91
    "U",  // 92
    "Np", // 93
    "Pu", // 94
    "Am", // 95
    "Cm", // 96
    "Bk", // 97
    "Cf", // 98
    "Es", // 99
    "Fm", // 100
    "Md", // 101
    "No", // 102
    "Lr", // 103
    "Rf", // 104
    "Db", // 105
    "Sg", // 106
    "Bh", // 107
    "Hs", // 108
    "Mt", // 109
};

// These are sorted by the abbreviation to make the map
// tree more balanced.
const char *residue_names[KNOWN_AMINO_ACIDS] = {
    "UNK", //  0   Unknown
    "ALA", //  1   Alanine             A
    "ASX", //  2   ASP/ASN ambiguous   B
    "CYS", //  3   Cysteine            C
    "ASP", //  4   Aspartic acid       D
    "GLU", //  5   Glutamic acid       E
    "PHE", //  6   Phenylalanine       F
    "GLY", //  7   Glycine             G
    "HIS", //  8   Histidine           H
    "ILE", //  9   Isoleucine          I
    "LYS", // 10   Lysine              K
    "LEU", // 11   Leucine             L
    "MET", // 12   Methionine          M
    "ASN", // 13   Asparagine          N
    "PRO", // 14   Proline             P
    "GLN", // 15   Glutamine           Q
    "ARG", // 16   Arginine            R
    "SER", // 17   Serine              S
    "THR", // 18   Threonine           T
    "VAL", // 19   Valine              V
    "TRP", // 20   Tryptophan          W
    "TYR", // 21   Tyrosine            Y
    "GLX"  // 22   GLU/GLN ambiguous   Z
};

const char residue_abbrs[KNOWN_AMINO_ACIDS] = {
    '?', // UNK  Unknown
    'A', // ALA  Alanine
    'B', // ASX  ASP/ASN ambiguous
    'C', // CYS  Cysteine
    'D', // ASP  Aspartic acid
    'E', // GLU  Glutamic acid
    'F', // PHE  Phenylalanine
    'G', // GLY  Glycine
    'H', // HIS  Histidine
    'I', // ILE  Isoleucine
    'K', // LYS  Lysine
    'L', // LEU  Leucine
    'M', // MET  Methionine
    'N', // ASN  Asparagine
    'P', // PRO  Proline
    'Q', // GLN  Glutamine
    'R', // ARG  Arginine
    'S', // SER  Serine
    'T', // THR  Threonine
    'V', // VAL  Valine
    'W', // TRP  Tryptophan
    'Y', // TYR  Tyrosine
    'Z', // GLX  GLU/GLN ambiguous
};

static void
InitializeResidueNameToAbbrMap()
{
    // These are sorted by the abbreviation to make the map
    // tree more balanced.
    residuename_to_abbr["UNK"] = '?'; // Unknown             ?
    residuename_to_abbr["ALA"] = 'A'; // Alanine             A
    residuename_to_abbr["ASX"] = 'B'; // ASP/ASN ambiguous   B
    residuename_to_abbr["CYS"] = 'C'; // Cysteine            C
    residuename_to_abbr["ASP"] = 'D'; // Aspartic acid       D
    residuename_to_abbr["GLU"] = 'E'; // Glutamic acid       E
    residuename_to_abbr["PHE"] = 'F'; // Phenylalanine       F
    residuename_to_abbr["GLY"] = 'G'; // Glycine             G
    residuename_to_abbr["HIS"] = 'H'; // Histidine           H
    residuename_to_abbr["ILE"] = 'I'; // Isoleucine          I
    residuename_to_abbr["LYS"] = 'K'; // Lysine              K
    residuename_to_abbr["LEU"] = 'L'; // Leucine             L
    residuename_to_abbr["MET"] = 'M'; // Methionine          M
    residuename_to_abbr["ASN"] = 'N'; // Asparagine          N
    residuename_to_abbr["PRO"] = 'P'; // Proline             P
    residuename_to_abbr["GLN"] = 'Q'; // Glutamine           Q
    residuename_to_abbr["ARG"] = 'R'; // Arginine            R
    residuename_to_abbr["SER"] = 'S'; // Serine              S
    residuename_to_abbr["THR"] = 'T'; // Threonine           T
    residuename_to_abbr["VAL"] = 'V'; // Valine              V
    residuename_to_abbr["TRP"] = 'W'; // Tryptophan          W
    residuename_to_abbr["TYR"] = 'Y'; // Tyrosine            Y
    residuename_to_abbr["GLX"] = 'Z'; // GLU/GLN ambiguous   Z

    // Add water
    residuename_to_abbr["HOH"] = 'w'; // water
}

static void
InitializeResidueNameToNumberMap()
{
    int i, index = 0;
    for(i = 0; i < KNOWN_AMINO_ACIDS; ++i, ++index)
    {
        residuename_to_number[residue_names[i]] = index;
        number_to_residuename[index] = residue_names[i];
    }
 
    // Add DNA/RNA base pairs
    const char *dnarna_bases[] = {"A", "C", "T", "G", "U"};
    for(i = 0; i < 5; ++i, ++index)
    {
        residuename_to_number[dnarna_bases[i]] = index;
        number_to_residuename[index] = dnarna_bases[i];
    }

    // Add water
    residuename_to_number["HOH"] = index;
    number_to_residuename[index] = "HOH";
}

static void
InitializeResidueNameToLongNameMap()
{
    residuename_to_longname["UNK"] = "Unknown";
    residuename_to_longname["ALA"] = "Alanine";
    residuename_to_longname["ASX"] = "ASP/ASN ambiguous";
    residuename_to_longname["CYS"] = "Cysteine";
    residuename_to_longname["ASP"] = "Aspartic acid";
    residuename_to_longname["GLU"] = "Glutamic acid";
    residuename_to_longname["PHE"] = "Phenylalanine";
    residuename_to_longname["GLY"] = "Glycine";
    residuename_to_longname["HIS"] = "Histidine";
    residuename_to_longname["ILE"] = "Isoleucine";
    residuename_to_longname["LYS"] = "Lysine";
    residuename_to_longname["LEU"] = "Leucine";
    residuename_to_longname["MET"] = "Methionine";
    residuename_to_longname["ASN"] = "Asparagine";
    residuename_to_longname["PRO"] = "Proline";
    residuename_to_longname["GLN"] = "Glutamine";
    residuename_to_longname["ARG"] = "Arginine";
    residuename_to_longname["SER"] = "Serine";
    residuename_to_longname["THR"] = "Threonine";
    residuename_to_longname["VAL"] = "Valine";
    residuename_to_longname["TRP"] = "Tryptophan";
    residuename_to_longname["TYR"] = "Tyrosine";
    residuename_to_longname["GLX"] = "GLU/GLN ambiguous";

    // Add DNA/RNA base pairs
    residuename_to_longname["A"] = "Adenine";
    residuename_to_longname["C"] = "Guanine";
    residuename_to_longname["T"] = "Thymine";
    residuename_to_longname["G"] = "Cytosine";
    residuename_to_longname["U"] = "Uracil";

    // Add water
    residuename_to_longname["HOH"] = "Water";
}

static void
InitializeElementNameToAtomicNumberMap()
{
    // These are sorted by an unrelated number (basically covalent radius)
    // for randomiation make the map tree more balanced.
    elementname_to_atomicnumber["?"]  = 0;
    elementname_to_atomicnumber["H"]  = 1;
    elementname_to_atomicnumber["Ne"] = 10;
    elementname_to_atomicnumber["F"]  = 9;
    elementname_to_atomicnumber["O"]  = 8;
    elementname_to_atomicnumber["N"]  = 7;
    elementname_to_atomicnumber["C"]  = 6;
    elementname_to_atomicnumber["B"]  = 5;
    elementname_to_atomicnumber["Be"] = 4;
    elementname_to_atomicnumber["He"] = 2;
    elementname_to_atomicnumber["Ar"] = 18;
    elementname_to_atomicnumber["Cl"] = 17;
    elementname_to_atomicnumber["S"]  = 16;
    elementname_to_atomicnumber["P"]  = 15;
    elementname_to_atomicnumber["Si"] = 14;
    elementname_to_atomicnumber["Kr"] = 36;
    elementname_to_atomicnumber["Br"] = 35;
    elementname_to_atomicnumber["Ni"] = 28;
    elementname_to_atomicnumber["Se"] = 34;
    elementname_to_atomicnumber["Co"] = 27;
    elementname_to_atomicnumber["Cu"] = 29;
    elementname_to_atomicnumber["Fe"] = 26;
    elementname_to_atomicnumber["Mn"] = 25;
    elementname_to_atomicnumber["Al"] = 13;
    elementname_to_atomicnumber["Cr"] = 24;
    elementname_to_atomicnumber["As"] = 33;
    elementname_to_atomicnumber["Ge"] = 32;
    elementname_to_atomicnumber["V"]  = 23;
    elementname_to_atomicnumber["Li"] = 3;
    elementname_to_atomicnumber["Rh"] = 45;
    elementname_to_atomicnumber["Ru"] = 44;
    elementname_to_atomicnumber["Zn"] = 30;
    elementname_to_atomicnumber["Ga"] = 31;
    elementname_to_atomicnumber["Os"] = 76;
    elementname_to_atomicnumber["Ir"] = 77;
    elementname_to_atomicnumber["Tc"] = 43;
    elementname_to_atomicnumber["Re"] = 75;
    elementname_to_atomicnumber["Pd"] = 46;
    elementname_to_atomicnumber["W"]  = 74;
    elementname_to_atomicnumber["Pt"] = 78;
    elementname_to_atomicnumber["Mo"] = 42;
    elementname_to_atomicnumber["Xe"] = 54;
    elementname_to_atomicnumber["Ti"] = 22;
    elementname_to_atomicnumber["I"]  = 53;
    elementname_to_atomicnumber["Ta"] = 73;
    elementname_to_atomicnumber["Nb"] = 41;
    elementname_to_atomicnumber["Ag"] = 47;
    elementname_to_atomicnumber["Au"] = 79;
    elementname_to_atomicnumber["Te"] = 52;
    elementname_to_atomicnumber["Mg"] = 12;
    elementname_to_atomicnumber["Sb"] = 51;
    elementname_to_atomicnumber["Sn"] = 50;
    elementname_to_atomicnumber["U"]  = 92;
    elementname_to_atomicnumber["In"] = 49;
    elementname_to_atomicnumber["Sc"] = 21;
    elementname_to_atomicnumber["Hf"] = 72;
    elementname_to_atomicnumber["Zr"] = 40;
    elementname_to_atomicnumber["At"] = 85;
    elementname_to_atomicnumber["Bi"] = 83;
    elementname_to_atomicnumber["Po"] = 84;
    elementname_to_atomicnumber["Pb"] = 82;
    elementname_to_atomicnumber["Cd"] = 48;
    elementname_to_atomicnumber["Tl"] = 81;
    elementname_to_atomicnumber["Hg"] = 80;
    elementname_to_atomicnumber["Na"] = 11;
    elementname_to_atomicnumber["Tm"] = 69;
    elementname_to_atomicnumber["Lu"] = 71;
    elementname_to_atomicnumber["Er"] = 68;
    elementname_to_atomicnumber["Ho"] = 67;
    elementname_to_atomicnumber["Dy"] = 66;
    elementname_to_atomicnumber["Tb"] = 65;
    elementname_to_atomicnumber["Gd"] = 64;
    elementname_to_atomicnumber["Y"]  = 39;
    elementname_to_atomicnumber["Sm"] = 62;
    elementname_to_atomicnumber["Pm"] = 61;
    elementname_to_atomicnumber["Nd"] = 60;
    elementname_to_atomicnumber["Th"] = 90;
    elementname_to_atomicnumber["Ce"] = 58;
    elementname_to_atomicnumber["Pr"] = 59;
    elementname_to_atomicnumber["La"] = 57;
    elementname_to_atomicnumber["Yb"] = 70;
    elementname_to_atomicnumber["Ca"] = 20;
    elementname_to_atomicnumber["Eu"] = 63;
    elementname_to_atomicnumber["Sr"] = 38;
    elementname_to_atomicnumber["Ba"] = 56;
    elementname_to_atomicnumber["K"]  = 19;
    elementname_to_atomicnumber["Rb"] = 37;
    elementname_to_atomicnumber["Cs"] = 55;
    elementname_to_atomicnumber["Rn"] = 86;
    elementname_to_atomicnumber["Fr"] = 87;
    elementname_to_atomicnumber["Ra"] = 88;
    elementname_to_atomicnumber["Ac"] = 89;
    elementname_to_atomicnumber["Pa"] = 91;
    elementname_to_atomicnumber["Np"] = 93;
    elementname_to_atomicnumber["Pu"] = 94;
    elementname_to_atomicnumber["Am"] = 95;
    elementname_to_atomicnumber["Cm"] = 96;
    elementname_to_atomicnumber["Bk"] = 97;
    elementname_to_atomicnumber["Cf"] = 98;
    elementname_to_atomicnumber["Es"] = 99;
    elementname_to_atomicnumber["Fm"] = 100;
    elementname_to_atomicnumber["Md"] = 101;
    elementname_to_atomicnumber["No"] = 102;
    elementname_to_atomicnumber["Lr"] = 103;
    elementname_to_atomicnumber["Rf"] = 104;
    elementname_to_atomicnumber["Db"] = 105;
    elementname_to_atomicnumber["Sg"] = 106;
    elementname_to_atomicnumber["Bh"] = 107;
    elementname_to_atomicnumber["Hs"] = 108;
    elementname_to_atomicnumber["Mt"] = 109;
}

static bool maps_initialized = false;
void
InitializeAtomicPropertyMaps()
{
    if (maps_initialized)
        return;

    maps_initialized = true;
        
    InitializeElementNameToAtomicNumberMap();
    InitializeResidueNameToNumberMap();
    InitializeResidueNameToAbbrMap();
    InitializeResidueNameToLongNameMap();

    // PrintColorTablesFor_avtColorTables();
}

int ElementNameToAtomicNumber(const char *element)
{
    static char name[3];
    name[0] = element[0];
    name[1] = element[1];
    name[2] = '\0';

    InitializeAtomicPropertyMaps();

    // Make sure the first character is upper case.
    if(name[0] >= 'a' && name[0] <= 'z')
        name[0] -= ('a' - 'A');

    // Make sure the second character is lower case.
    if(name[1] >= 'A' && name[1] <= 'Z')
        name[1] += ('a' - 'A');

    switch (name[0])
    {
    case '?':
        {
            return 0;
        }
    case 'A':
        {
            switch (name[1])
            {
            case 'c': return 89;
            case 'g': return 47;
            case 'l': return 13;
            case 'm': return 95;
            case 'r': return 18;
            case 's': return 33;
            case 't': return 85;
            case 'u': return 79;
            }
        }
        break;

    case 'B':
        {
            switch (name[1])
            {
            case '\0': return 5;
            case 'a': return 56;
            case 'e': return 4;
            case 'h': return 107;
            case 'i': return 83;
            case 'k': return 97;
            case 'r': return 35;
            }
        }
        break;

    case 'C':
        {
            switch (name[1])
            {
            case '\0': return 6;
            case 'a': return 20;
            case 'd': return 48;
            case 'e': return 58;
            case 'f': return 98;
            case 'l': return 17;
            case 'm': return 96;
            case 'o': return 27;
            case 'r': return 24;
            case 's': return 55;
            case 'u': return 29;
            }
        }
        break;

    case 'D':
        {
            switch (name[1])
            {
            case 'b': return 105;
            case 'y': return 66;
            }
        }
        break;

    case 'E':
        {
            switch (name[1])
            {
            case 'r': return 68;
            case 's': return 99;
            case 'u': return 63;
            }
        }
        break;

    case 'F':
        {
            switch (name[1])
            {
            case '\0': return 9;
            case 'e': return 26;
            case 'm': return 100;
            case 'r': return 87;
            }
        }
        break;

    case 'G':
        {
            switch (name[1])
            {
            case 'a': return 31;
            case 'd': return 64;
            case 'e': return 32;
            }
        }
        break;

    case 'H':
        {
            switch (name[1])
            {
            case '\0': return 1;
            case 'e': return 2;
            case 'f': return 72;
            case 'g': return 80;
            case 'o': return 67;
            case 's': return 108;
            }
        }
        break;

    case 'I':
        {
            switch (name[1])
            {
            case '\0': return 53;
            case 'n': return 49;
            case 'r': return 77;
            }
        }
        break;

    case 'K':
        {
            switch (name[1])
            {
            case '\0': return 19;
            case 'r': return 36;
            }
        }
        break;

    case 'L':
        {
            switch (name[1])
            {
            case 'a': return 57;
            case 'i': return 3;
            case 'r': return 103;
            case 'u': return 71;
            }
        }
        break;

    case 'M':
        {
            switch (name[1])
            {
            case 'd': return 101;
            case 'g': return 12;
            case 'n': return 25;
            case 'o': return 42;
            case 't': return 109;
            }
        }
        break;

    case 'N':
        {
            switch (name[1])
            {
            case '\0': return 7;
            case 'a': return 11;
            case 'b': return 41;
            case 'd': return 60;
            case 'e': return 10;
            case 'i': return 28;
            case 'o': return 102;
            case 'p': return 93;
            }
        }
        break;

    case 'O':
        {
            switch (name[1])
            {
            case '\0': return 8;
            case 's': return 76;
            }
        }
        break;

    case 'P':
        {
            switch (name[1])
            {
            case '\0': return 15;
            case 'a': return 91;
            case 'b': return 82;
            case 'd': return 46;
            case 'm': return 61;
            case 'o': return 84;
            case 'r': return 59;
            case 't': return 78;
            case 'u': return 94;
            }
        }
        break;

    case 'R':
        {
            switch (name[1])
            {
            case 'a': return 88;
            case 'b': return 37;
            case 'e': return 75;
            case 'f': return 104;
            case 'h': return 45;
            case 'n': return 86;
            case 'u': return 44;
            }
        }
        break;

    case 'S':
        {
            switch (name[1])
            {
            case '\0': return 16;
            case 'b': return 51;
            case 'c': return 21;
            case 'e': return 34;
            case 'g': return 106;
            case 'i': return 14;
            case 'm': return 62;
            case 'n': return 50;
            case 'r': return 38;
            }
        }
        break;

    case 'T':
        {
            switch (name[1])
            {
            case 'a': return 73;
            case 'b': return 65;
            case 'c': return 43;
            case 'e': return 52;
            case 'h': return 90;
            case 'i': return 22;
            case 'l': return 81;
            case 'm': return 69;
            }
        }
        break;

    case 'U':
        {
            switch (name[1])
            {
            case '\0': return 92;
            }
        }
        break;

    case 'V':
        {
            switch (name[1])
            {
            case '\0': return 23;
            }
        }
        break;

    case 'W':
        {
            switch (name[1])
            {
            case '\0': return 74;
            }
        }
        break;

    case 'X':
        {
            switch (name[1])
            {
            case 'e': return 54;
            }
        }
        break;

    case 'Y':
        {
            switch (name[1])
            {
            case '\0': return 39;
            case 'b': return 70;
            }
        }
        break;

    case 'Z':
        {
            switch (name[1])
            {
            case 'n': return 30;
            case 'r': return 40;
            }
        }
        break;
    }
    return -1;
}


int ResiduenameToNumber(const char *name)
{
    // Try a cache since these numbers are often
    // looked up in groups.
    static char lastname[10] = "";
    static char lastnumber   = -1;
    if (strcmp(name, lastname) == 0)
    {
        return lastnumber;
    }

    // Nope -- cache the name for next time.
    snprintf(lastname,10,"%s",name);

    InitializeAtomicPropertyMaps();

    // Advance past leading spaces.
    const char *name2 = name;
    while(*name2 == ' ' && *name2 != '\0') ++name2;

    std::map<std::string, int>::const_iterator it = 
        residuename_to_number.find(name2);
    int retval = (it != residuename_to_number.end()) ? it->second : -1;

    // Cache the result for next time
    lastnumber = retval;

    return retval;
}

const char *NumberToResiduename(int num)
{
    InitializeAtomicPropertyMaps();

    std::map<int, std::string>::const_iterator it = 
        number_to_residuename.find(num);
    return (it != number_to_residuename.end()) ? it->second.c_str() : 0;
}

int NumberOfKnownResidues()
{
    InitializeAtomicPropertyMaps();

    return residuename_to_number.size();
}

int ResiduenameToAbbr(const char *name)
{
    InitializeAtomicPropertyMaps();

    // Advance past leading spaces.
    const char *name2 = name;
    while(*name2 == ' ' && *name2 != '\0') ++name2;

    std::map<std::string, int>::const_iterator it = 
        residuename_to_abbr.find(name2);
    return (it != residuename_to_abbr.end()) ? it->second : -1;
}

void AddResiduename(const char *name, const char *longname)
{
    InitializeAtomicPropertyMaps();

    // Advance past leading spaces.
    const char *name2 = name;
    while(*name2 == ' ' && *name2 != '\0') ++name2;
    const char *lname2 = longname;
    while(*lname2 == ' ' && *lname2 != '\0') ++lname2;

    std::map<std::string, int>::const_iterator it = 
        residuename_to_number.find(name2);
    if(it == residuename_to_number.end())
    {
        // Look for the largest number in the map.
        int m = 0;
        for(std::map<std::string, int>::const_iterator i =
            residuename_to_number.begin();
            i != residuename_to_number.end(); ++i)
        {
            if(i->second > m)
                m = i->second;
        }

        // Add the new residue name.
        residuename_to_number[name2] = m+1;
        number_to_residuename[m+1] = name2;

        // Store the long name too.
        residuename_to_longname[name2] = lname2;
    }
}

const char *ResiduenameToLongName(const char *name)
{
    InitializeAtomicPropertyMaps();

    std::map<std::string, std::string>::const_iterator it = 
        residuename_to_longname.find(name);
    return (it != residuename_to_longname.end()) ? it->second.c_str() : 0;
}

int ResidueLongnameMaxlen()
{
    InitializeAtomicPropertyMaps();

    unsigned int m = 0;
    for(std::map<std::string, std::string>::const_iterator i =
        residuename_to_longname.begin();
        i != residuename_to_longname.end(); ++i)
    {
        if(i->second.size() > m)
            m = i->second.size();
    }

    return m;
}

