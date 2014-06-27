// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "eavlBitmapFont.h"
#include <cctype>
#include <iomanip>

eavlBitmapFont::eavlBitmapFont()
{
    for (int i=0; i<256; ++i)
        shortmap[i] = 0;
    padl=padr=padt=padb=0;
    userPointer = NULL;
    userLong    = 0;
}

vector<unsigned char> &
eavlBitmapFont::GetRawImageData(string &type)
{
    type = "png"; ///<\todo: hardcoded to png
    return rawimagefiledata;
}


#ifdef HAVE_XML_TOOLS

#include "XMLTools.h"
class eavlNGLFontParser : public XMLParser
{
  public:
    eavlBitmapFont *font;
    eavlNGLFontParser(eavlBitmapFont *f) : XMLParser()
    {
        font = f;
    }

    virtual void beginElement(const string &name,const XMLAttributes &atts)
    {
        if (name == "description")
        {
            font->name = atts.GetValue("family");
        }
        else if (name == "metrics")
        {
            font->height = atoi(atts.GetValue("height").c_str());
            font->ascender = atoi(atts.GetValue("ascender").c_str());
            font->descender = atoi(atts.GetValue("descender").c_str());
        }
        else if (name == "texture")
        {
            font->imgfile = atts.GetValue("file");
            font->imgw = atoi(atts.GetValue("width").c_str());
            font->imgh = atoi(atts.GetValue("height").c_str());
        }
        else if (name == "padding")
        {
            font->padl = atoi(atts.GetValue("left").c_str());
            font->padt = atoi(atts.GetValue("top").c_str());
            font->padr = atoi(atts.GetValue("right").c_str());
            font->padb = atoi(atts.GetValue("bottom").c_str());
        }
        else if (name == "char")
        {
            eavlBitmapFont::Character c;
            c.id = atts.GetValue("id");
            c.c = char(c.id[0]);
            if (c.id.length() == 1)
            {
                font->shortmap[(int)(unsigned char)(c.c)] = font->chars.size();
            }
            c.offx = atoi(atts.GetValue("offset_x").c_str());
            c.offy = atoi(atts.GetValue("offset_y").c_str());
            c.x = atoi(atts.GetValue("rect_x").c_str());
            c.y = atoi(atts.GetValue("rect_y").c_str());
            c.w = atoi(atts.GetValue("rect_w").c_str());
            c.h = atoi(atts.GetValue("rect_h").c_str());
            c.adv = atoi(atts.GetValue("advance").c_str());
            font->chars.push_back(c);
        }
        else if (name == "kerning")
        {
            // we should now be parsing a character which has just
            // been added to the end of the list in the font
            eavlBitmapFont::Character &c = font->chars[font->chars.size()-1];
            string s = atts.GetValue("id");
            char shortid = char(s[0]);
            if (s.length() == 1)
            {
                c.kern[(int)(unsigned char)(shortid)] = atoi(atts.GetValue("advance").c_str());
                //cerr << "c.c="<<c.c<<" id="<<shortid<<" kern="<<c.kern[(int)(unsigned char)(shortid)]<<endl;
            }
        }
    }
    virtual void handleText(const string &text)
    {
    }
    virtual void endElement(const string &name)
    {
    }
};

void
eavlBitmapFont::ReadFromNGLFile(const std::string &fn)
{
    ifstream in(fn.c_str());
    eavlNGLFontParser *parser = new eavlNGLFontParser(this);
    parser->Initialize(in);
    parser->ParseAllEntities();
    in.close();
    //f->Print(cout);
    delete parser;

    // read the bitmap image file
    ifstream file(imgfile.c_str(), ios::in|ios::binary|ios::ate);
    std::streamsize size = 0;
    if(file.seekg(0, ios::end).good()) size = file.tellg();
    file.seekg(0, ios::beg);
    rawimagefiledata.resize(size);
    file.read((char*)(&rawimagefiledata[0]), size);
    file.close();
}
#else
#include "eavl.h"
#include "eavlException.h"
void
eavlBitmapFont::ReadFromNGLFile(const std::string &fn)
{
    THROW(eavlException, "Did not build with XML support");
}
#endif

void
eavlBitmapFont::DumpToInitFile(const std::string &fn,
                               const std::string &shortname)
{
    // write the cpp and header files
    ofstream cpp((fn + ".cpp").c_str(), ios::out);
    ofstream hpp((fn + ".h").c_str(), ios::out);

    string shortupper(shortname);
    std::transform(shortupper.begin(), shortupper.end(), shortupper.begin(),
                   toupper);
    hpp << "#ifndef EAVL_"<<shortupper<<"_H" << endl;
    hpp << "#define EAVL_"<<shortupper<<"_H" << endl;
    hpp << endl;
    hpp << "eavlBitmapFont *Create"<<shortname<<"Font();" << endl;
    hpp << endl;
    hpp << "#endif" << endl;

    cpp << "#include <eavlBitmapFont.h>" << endl;
    cpp << endl;
    cpp << "static int charmetrics[][8] = {" << endl;
    for (unsigned int i=0; i<chars.size(); i++)
    {
        Character &c = chars[i];
        cpp << "    {"
            << int((unsigned char)c.c) << ","
            << c.offx << "," << c.offy << ","
            << c.x << "," << c.y << ","
            << c.w << "," << c.h << ","
            << c.adv << "}";
        if (i < chars.size()) 
            cpp << ",";
        cpp << endl;
    }
    cpp << "};" << endl;
    cpp << "static const char *charids[] = {" << endl;
    for (unsigned int i=0; i<chars.size(); i++)
    {
        Character &c = chars[i];
        if (i % 10 == 0)
            cpp << endl << "    ";
        cpp << "\"" << ((c.c == '\"' || c.c == '\\') ? "\\" : "") << c.id << "\"";
        if (i < chars.size()-1)
            cpp << ",";
        else
            cpp << endl;
    }
    cpp << "};" << endl;
    cpp << endl;
    cpp << "static const unsigned char rawimage[] = {";
    for (unsigned int i=0; i<rawimagefiledata.size(); ++i)
    {
        if (i % 10 == 0)
            cpp << endl << "    ";
        cpp << "0x"
            << std::hex << std::setfill('0') << std::setw(2)
            << int((unsigned char)(rawimagefiledata[i]));
        if (i < rawimagefiledata.size()-1)
            cpp << ",";
        else
            cpp << endl;
    }
    cpp << std::dec;
    cpp << "};" << endl;
    cpp << endl;
    cpp << "eavlBitmapFont *Create"<<shortname<<"Font()" << endl;
    cpp << "{" << endl;
    cpp << "    eavlBitmapFont *font = new eavlBitmapFont;" << endl;
    cpp << "    font->name      = \"" << name << "\";" << endl;
    cpp << "    font->height    = " << height << ";" << endl;
    cpp << "    font->ascender  = " << ascender << ";" << endl;
    cpp << "    font->descender = " << descender << ";" << endl;
    cpp << "    font->imgw      = " << imgw << ";" << endl;
    cpp << "    font->imgh      = " << imgh << ";" << endl;
    cpp << "    font->padl      = " << padl << ";" << endl;
    cpp << "    font->padr      = " << padr << ";" << endl;
    cpp << "    font->padt      = " << padt << ";" << endl;
    cpp << "    font->padb      = " << padb << ";" << endl;
    cpp << "    font->rawimagefiledata.insert(" << endl
        << "        font->rawimagefiledata.begin()," << endl
        << "        rawimage, rawimage + "<<rawimagefiledata.size() << ");" << endl;
    cpp << "    for (int i=0; i<"<<chars.size()<<"; i++)" << endl;
    cpp << "    {" << endl;
    cpp << "        font->chars.push_back(eavlBitmapFont::Character("
        << "charids[i],charmetrics[i]));" << endl;
    cpp << "        font->shortmap[charmetrics[i][0]] = i;" << endl;
    cpp << "    }" << endl;
    cpp << endl;
    cpp << "    // Any kerning data follows..." << endl;
    cpp << endl;
    for (unsigned int i=0; i<chars.size(); i++)
    {
        Character &c = chars[i];
        bool header = false;
        for (int j=0; j<256; j++)
        {
            if (c.kern[j] != 0)
            {
                if (!header)
                    cpp << "    // Character " << c.id << endl;
                header = true;
                cpp << "    font->chars["<<i<<"].kern["<<j<<"] = " << c.kern[j] << ";" << endl;
            }
        }
    }
    cpp << "    return font;" << endl;
    cpp << "}" << endl;

    cpp << endl;

    cpp.close();
    hpp.close();
}
