// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
// This file contains code from VisIt, (c) 2000-2012 LLNS.  See COPYRIGHT.txt.
#ifndef EAVL_ATOMIC_PROPERTIES_H
#define EAVL_ATOMIC_PROPERTIES_H

#define MAX_ELEMENT_NUMBER 109
#define KNOWN_AMINO_ACIDS  23

extern float         atomic_radius[MAX_ELEMENT_NUMBER+1];
extern float         covalent_radius[MAX_ELEMENT_NUMBER+1];

extern const char   *element_names[MAX_ELEMENT_NUMBER+1];
extern const char   *residue_names[KNOWN_AMINO_ACIDS];

int  ElementNameToAtomicNumber(const char *element);
int  ResiduenameToNumber(const char *name);
const char *NumberToResiduename(int num);
int NumberOfKnownResidues();
int  ResiduenameToAbbr(const char *name);
void AddResiduename(const char *name, const char *longname);
const char *ResiduenameToLongName(const char *name);
int ResidueLongnameMaxlen();

#endif
