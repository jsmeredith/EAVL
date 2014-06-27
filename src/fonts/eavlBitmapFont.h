// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_BITMAP_FONT_H
#define EAVL_BITMAP_FONT_H

#include "STL.h"

// ****************************************************************************
// Class:  eavlBitmapFont
//
// Purpose:
///   Class encapsulating an image and character information necessary
///   to render a bitmap font.
//
// Programmer:  Jeremy Meredith
// Creation:    January  8, 2013
//
// Modifications:
// ****************************************************************************
class eavlBitmapFont
{
  public:
    struct Character
    {
        string id;
        char   c;
        int offx, offy;
        int x, y, w, h;
        int adv;
        int kern[256];
        Character()
        {
            ResetKerning();
        }
        Character(const string &id_, char c_, int offx_, int offy_,
                  int x_, int y_, int w_, int h_, int adv_)
            : id(id_), c(c_), offx(offx_), offy(offy_),
              x(x_), y(y_), w(w_), h(h_), adv(adv_)
        {
            ResetKerning();
        }
        Character(const string &id_, int metrics[])
            : id(id_), c(metrics[0]), offx(metrics[1]), offy(metrics[2]),
              x(metrics[3]), y(metrics[4]), w(metrics[5]), h(metrics[6]),
              adv(metrics[7])
        {
            ResetKerning();
        }
        void ResetKerning()
        {
            for (int i=0; i<256; i++)
                kern[i]=0;
        }
        void Print(ostream &out)
        {
            out << "char id='"<<id<<"' c(ord)="<<(int)(unsigned char)(c)
                <<" adv="<<adv<<" off="<<offx<<","<<offy
                <<" rect x,y="<<x<<","<<y<<" w,h="<<w<<","<<h<<endl;
        }
    };

    string name;
    string imgfile;
    int    height;
    int    ascender;
    int    descender;
    int    imgw, imgh;
    int    padl, padr, padt, padb;
    int    shortmap[256];
    vector<Character> chars;

    void      *userPointer;
    long long  userLong;

    vector<unsigned char> rawimagefiledata;

  public:
    eavlBitmapFont();
    void ReadFromNGLFile(const std::string &fn);
    void DumpToInitFile(const std::string &fn, const std::string &shortname);
    Character GetChar(char c)
    {
        return chars[shortmap[(unsigned char)c]];
    }
    vector<unsigned char> &GetRawImageData(string &type);
    double GetTextWidth(const std::string &text)
    {
        double width = 0;
        for (unsigned int i=0; i<text.length(); ++i)
        {
            Character c = GetChar(text[i]);
            char nextchar = (i < text.length()-1) ? text[i+1] : 0;

            const bool kerning = true;
            if (kerning && nextchar>0)
                width += double(c.kern[int(nextchar)]) / double(height);
            width += double(c.adv) / double(height);
        }
        return width;
    }
    void GetCharPolygon(char character, double &x, double &y,
                        double &vl, double &vr, double &vt, double &vb,
                        double &tl, double &tr, double &tt, double &tb,
                        char nextchar = 0)
    {
        Character c = GetChar(character);

        // By default, the origin for the font is at the
        // baseline.  That's nice, but we'd rather it
        // be at the actual bottom, so create an offset.
        double yoff = -double(descender) / double(height);

        tl =      double(c.x +       padl) / double(imgw);
        tr =      double(c.x + c.w - padr) / double(imgw);
        tt = 1. - double(c.y +       padt) / double(imgh);
        tb = 1. - double(c.y + c.h - padb) / double(imgh);

        vl =        x + double(c.offx +       padl) / double(height);
        vr =        x + double(c.offx + c.w - padr) / double(height);
        vt = yoff + y + double(c.offy -       padt) / double(height);
        vb = yoff + y + double(c.offy - c.h + padb) / double(height);

        const bool kerning = true;
        if (kerning && nextchar>0)
            x += double(c.kern[int(nextchar)]) / double(height);
        x += double(c.adv) / double(height);
    }
    void Print(ostream &out)
    {
        out << "name      = " << name << endl;
        out << "imgfile   = " << imgfile << endl;
        out << "height    = " << height << endl;
        out << "ascender  = " << ascender << endl;
        out << "descender = " << descender << endl;
        out << "imgw      = " << imgw << endl;
        out << "imgh      = " << imgh << endl;
        out << "padding   = " 
            << padl << "," << padr << "," << padt << "," << padb 
            << " (l,r,t,b)" << endl;
        for (unsigned int i=0; i<chars.size(); ++i)
        {
            Character &c = chars[i];
            out << "  char "<<i<<": ";
            c.Print(out);
        }
    }
};

#endif
