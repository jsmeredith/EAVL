// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
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
struct eavlBitmapFont
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
    void GetCharPolygon(char character, float &x, float &y,
                        float &vl, float &vr, float &vt, float &vb,
                        float &tl, float &tr, float &tt, float &tb,
                        char nextchar = 0)
    {
        Character c = GetChar(character);

        tl =      float(c.x +       padl) / float(imgw);
        tr =      float(c.x + c.w - padr) / float(imgw);
        tt = 1. - float(c.y +       padt) / float(imgh);
        tb = 1. - float(c.y + c.h - padb) / float(imgh);

        vl = x + float(c.offx +       padl) / float(height);
        vr = x + float(c.offx + c.w - padr) / float(height);
        vt = y + float(c.offy -       padt) / float(height);
        vb = y + float(c.offy - c.h + padb) / float(height);

        const bool kerning = true;
        if (kerning && nextchar>0)
            x += float(c.kern[nextchar]) / float(height);
        x += float(c.adv) / float(height);
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
        for (int i=0; i<chars.size(); ++i)
        {
            Character &c = chars[i];
            out << "  char "<<i<<": ";
            c.Print(out);
        }
    }
};

#endif
