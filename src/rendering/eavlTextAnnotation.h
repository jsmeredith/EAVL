#ifndef EAVL_TEXT_ANNOTATION_H
#define EAVL_TEXT_ANNOTATION_H

#include <eavlBitmapFont.h>
#include <eavlBitmapFontFactory.h>
#include <eavlPNGImporter.h>
#include <eavlTexture.h>
#include <eavlMatrix4x4.h>

// ****************************************************************************
// Class:  eavlTextAnnotation
//
// Purpose:
///   Allows 2D or 3D text.
//
// Programmer:  Jeremy Meredith
// Creation:    January  9, 2013
//
// Modifications:
// ****************************************************************************
class eavlTextAnnotation
{
  protected:
    string text;
    bool billboard;
  public:
    eavlTextAnnotation(const string &txt, float scale,
                       float x, float y)
    {
        text = txt;
        billboard = false;
    }
    eavlTextAnnotation(const string &txt, float scale,
                       float ox, float oy, float oz)
    {
        text = txt;
        billboard = true;
    }
    eavlTextAnnotation(const string &txt, float scale,
                       float ox, float oy, float oz,
                       float nx, float ny, float nz,
                       float rx, float ry, float rz)
    {
        text = txt;
        billboard = false;
    }

};

#endif
