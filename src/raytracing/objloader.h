#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H
//scratch pixel
//http://www.scratchapixel.com/lessons/3d-advanced-lessons/obj-file-format/reading-an-obj-file/


/*!
    \file objloader.cpp
    \brief load an OBJ file and store its geometry/material in memory

    This code was adapted from the project Embree from Intel.
    Copyright 2009-2012 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 
    Copyright 2013 Scratchapixel

    Compile with: clang++/c++ -o objloader objloader.cpp -O3 -Wall -std=c++0x
 */


#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <map>
#include <memory>
#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>
#include <cstring>

#define MAX_LINE_LENGTH 10000

template<typename T>
class Vec2
{
public:
    T x, y;
    Vec2() : x(0), y(0) {}
    Vec2(T xx, T yy) : x(xx), y(yy) {}
};

template<typename T>
class Vec3
{
public:
    T x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
    friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
    { os << v.x << ", " << v.y << ", " << v.z; return os; }
};

 typedef Vec3<float> Vec3fobj;
 typedef Vec3<int>   Vec3iobj;
 typedef Vec2<float> Vec2fobj;

inline Vec3fobj getVec3(std::ifstream &ifs) { float x, y, z; ifs >> x >> y >> z; return Vec3fobj(x, y, z); }

/*! returns the path of a file */
inline std::string getFilePath(const std::string &filename)
{
    size_t pos = filename.find_last_of('/');
    if (pos == std::string::npos) return filename;
    return filename.substr(0, pos);
}

/*! \struct Material
 *  \brief a simple structure to store material's properties
 */
struct Material
{
    Vec3fobj Ka, Kd, Ks;   /*! ambient, diffuse and specular rgb coefficients */
    float d;            /*! transparency */
    float Ns, Ni;       /*! specular exponent and index of refraction */
    Material()
    {   //default material needs to something other than black
        Ka.x=.5;
        Ka.y=.5;
        Ka.z=.5;

        Kd.x=.5;
        Kd.y=.5;
        Kd.z=.5;

        Ks.x=.5;
        Ks.y=.5;
        Ks.z=.5;

    }
    Material(Vec3fobj _Ka, Vec3fobj _Kd, Vec3fobj _Ks, float _d, float _Ns, float _Ni)
      : Ka(_Ka), Kd(_Kd), Ks(_Ks), d(_d), Ns(_Ns), Ni(_Ni) {}
};

/*! \class TriangleMesh
 *  \brief a basic class to store a triangle mesh data
 */
class TriangleMesh
{
public:
    Vec3fobj *positions;   /*! position/vertex array */
    Vec3fobj *normals;     /*! normal array (can be null) */
    Vec2fobj *texcoords;   /*! texture coordinates (can be null) */
    int numTriangles;   /*! number of triangles */
    int *triangles;     /*! triangle index list */
    TriangleMesh() : positions(NULL), normals(NULL), texcoords(NULL), triangles(NULL) {}
    ~TriangleMesh()
    {
        if (positions) delete [] positions;
        if (normals)   delete [] normals;
        if (texcoords) delete [] texcoords;
        if (triangles) delete [] triangles;
    }
};

// trim from start
static inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
}

/*! \class OBJPrimitive
 *  \brief a basic class to store a OBJPrimitive (defined by a mesh and a material)
 */
struct OBJPrimitive
{
    OBJPrimitive( TriangleMesh *m, Material *mat) : 
        mesh(m), material(mat) {}
     TriangleMesh* mesh;   /*! the object's geometry */
     Material* material;   /*! the object's material */
};

/*! Three-index vertex, indexing start at 0, -1 means invalid vertex. */
struct Vertex {
    int v, vt, vn;
    Vertex() {};
    Vertex(int v) : v(v), vt(v), vn(v) {};
    Vertex(int v, int vt, int vn) : v(v), vt(vt), vn(vn) {};
};

// need to declare this operator if we want to use Vertex in a map
static inline bool operator < ( const Vertex& a, const Vertex& b ) {
    if (a.v  != b.v)  return a.v  < b.v;
    if (a.vn != b.vn) return a.vn < b.vn;
    if (a.vt != b.vt) return a.vt < b.vt;
    return false;
}

/*! Parse separator. */
static inline const char* parseSep(const char*& token) {
    size_t sep = strspn(token, " \t");
    if (!sep) throw std::runtime_error("separator expected");
    return token+=sep;
}

/*! Read float from a string. */
static inline float getFloat(const char*& token) {
    token += strspn(token, " \t");
    float n = (float)atof(token);
    token += strcspn(token, " \t\r");
    return n;
}

/*! Read Vec2fobj from a string. */
static inline Vec2fobj getVec2fobj(const char*& token) {
    float x = getFloat(token);
    float y = getFloat(token);
    return Vec2fobj(x,y);
}

/*! Read Vec3fobj from a string. */
static inline Vec3fobj getVec3fobj(const char*& token) {
    float x = getFloat(token);
    float y = getFloat(token);
    float z = getFloat(token);
    return Vec3fobj(x, y, z);
}

/*! Parse optional separator. */
static inline const char* parseSepOpt(const char*& token) {
    return token+=strspn(token, " \t");
}

/*! Determine if character is a separator. */
static inline bool isSep(const char c) {
    return (c == ' ') || (c == '\t');
}

class ObjReader
{
public:
    ObjReader(const char *filename);
    
    inline Vertex getInt3(const char*& token);
    inline int fix_v(int index)  { return(index > 0 ? index - 1 : (index == 0 ? 0 : (int)v .size() + index)); }
    inline int fix_vt(int index) { return(index > 0 ? index - 1 : (index == 0 ? 0 : (int)vt.size() + index)); }
    inline int fix_vn(int index) { return(index > 0 ? index - 1 : (index == 0 ? 0 : (int)vn.size() + index)); }
    std::vector<Vec3fobj> v, vn;
    std::vector<Vec2fobj> vt;
    std::vector<std::vector<Vertex> > curGroup;
    std::map<std::string, Material*> materials;
    Material* curMaterial;
    inline void loadMTL(const std::string &mtlFilename);
    inline void flushFaceGroup();
    inline unsigned int getVertex(std::map<Vertex, unsigned int>&, std::vector<Vec3fobj>&, std::vector<Vec3fobj>&, std::vector<Vec2fobj>&, const Vertex&);
    std::vector<OBJPrimitive* > model;
    inline void printStats();
    inline void extractNormals();
    bool hasNormals;
    int totalTriangles;
    inline void getRawData(float *&verts, float *&normals, float *&mats, int * &matIndex, int &matCount);
    ~ObjReader()
    {
        for(int i = 0; i < model.size(); i++)
        {
            delete model[i]->mesh;
            delete model[i]->material;
            delete model[i];
        }
    }
};


/*! Parse differently formated triplets like: n0, n0/n1/n2, n0//n2, n0/n1.          */
/*! All indices are converted to C-style (from 0). Missing entries are assigned -1. */
inline Vertex ObjReader::getInt3(const char*& token)
{
    Vertex v(-1);
    v.v = fix_v(atoi(token));
    token += strcspn(token, "/ \t\r");
    if (token[0] != '/') return(v);
    token++;
    
    // it is i//n
    if (token[0] == '/') {
        token++;
        v.vn = fix_vn(atoi(token));
        token += strcspn(token, " \t\r");
        return(v);
    }
    
    // it is i/t/n or i/t
    v.vt = fix_vt(atoi(token));
    token += strcspn(token, "/ \t\r");
    if (token[0] != '/') return(v);
    token++;
    
    // it is i/t/n
    v.vn = fix_vn(atoi(token));
    token += strcspn(token, " \t\r");
    return(v);
}

/*! \brief load a OBJ material file
 *  \param mtlFilename is the full path to the material file
 */
inline void ObjReader::loadMTL(const std::string &mtlFilename)
{
    std::ifstream ifs;
    std::string trimmedFilename=mtlFilename;
    trimmedFilename=trim(trimmedFilename);
    ifs.open( trimmedFilename.c_str());
    if (!ifs.is_open()) {
        std::cerr << "can't open " << mtlFilename << std::endl;
        exit(1);
    }
    Material* mat;
    while (ifs.peek() != EOF) {
        char line[MAX_LINE_LENGTH];
        ifs.getline(line, sizeof(line), '\n');
        const char* token = line + strspn(line, " \t"); // ignore spaces and tabs
        if (token[0] == 0) continue; // ignore empty lines
        if (token[0] == '#') continue; // ignore comments

        if (!strncmp(token, "newmtl", 6)) {
            parseSep(token += 6);
            std::string name(token); //printf("Name of the material %s\n", name.c_str());
            mat = new Material;
            materials[name] = mat;
            continue;
        }

        if (!mat) 
        {
            cout<<"Bad line in mat file : "<<line<<endl;
            throw std::runtime_error("invalid material file: newmtl expected first");
        }
        
        if (!strncmp(token, "d", 1))  { parseSep(token += 1); mat->d  = getFloat(token); continue; }
        if (!strncmp(token, "Ns", 1)) { parseSep(token += 2); mat->Ns = getFloat(token); continue; }
        if (!strncmp(token, "Ns", 1)) { parseSep(token += 2); mat->Ni = getFloat(token); continue; }
        if (!strncmp(token, "Ka", 2)) { parseSep(token += 2); mat->Ka = getVec3fobj(token); continue; }
        if (!strncmp(token, "Kd", 2)) { parseSep(token += 2); mat->Kd = getVec3fobj(token); continue; }
        if (!strncmp(token, "Ks", 2)) { parseSep(token += 2); mat->Ks = getVec3fobj(token); continue; }
    }
    ifs.close();
}

/*! \brief load the geometry defined in an OBJ/Wavefront file
 *  \param filename is the path to the OJB file
 */
inline ObjReader::ObjReader(const char *filename)
{
    std::ifstream ifs;
    // extract the path from the filename (used to read the material file)
    std::string path = getFilePath(filename);
    try {
        ifs.open(filename);
        if (ifs.fail()) throw std::runtime_error("can't open file " + std::string(filename));

        // create a default material
        Material* defaultMaterial(new Material);
        curMaterial = defaultMaterial;

        char line[MAX_LINE_LENGTH]; // line buffer
        
        while (ifs.peek() != EOF) // read each line until EOF is found
        {
            ifs.getline(line, sizeof(line), '\n');
            const char* token = line + strspn(line, " \t"); // ignore space and tabs

            if (token[0] == 0) continue; // line is empty, ignore
            // read a vertex
            if (token[0] == 'v' && isSep(token[1])) { v.push_back(getVec3fobj(token += 2)); continue; }
            // read a normal
            if (!strncmp(token, "vn",  2) && isSep(token[2])) { vn.push_back(getVec3fobj(token += 3)); continue; }
            // read a texture coordinates
            if (!strncmp(token, "vt",  2) && isSep(token[2])) { vt.push_back(getVec2fobj(token += 3)); continue; }
            // read a face
            if (token[0] == 'f' && isSep(token[1])) {
                //cerr<<"f!!"<<endl;
                parseSep(token += 1);
                std::vector<Vertex> face;
                while (token[0]!=13 && token[0]!=0) {// 13= CR, \n stripped?
                    //cerr<<"w!!"<<endl;
                    //int t=(int)token[0];
                   // cerr<<t<<endl;
                    face.push_back(getInt3(token));
                    parseSepOpt(token);
                }
                curGroup.push_back(face);
                continue;
            }
            
            /*! use material */
            if (!strncmp(token, "usemtl", 6) && isSep(token[6]))
            {
                //cerr<<"usemtl!!"<<endl;
                flushFaceGroup();
                std::string name(parseSep(token += 6));
                if (materials.find(name) == materials.end()) curMaterial = defaultMaterial;
                else curMaterial = materials[name];
                continue;
            }
            
            /* load material library */
            if (!strncmp(token, "mtllib", 6) && isSep(token[6])) {
                loadMTL(path + "/" + std::string(parseSep(token += 6)));
                continue;
            }
        }
        delete defaultMaterial;
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        cerr<<"Here!!"<<endl;
    }
    
    flushFaceGroup(); // flush the last loaded object
    ifs.close();
    printStats();
}


/*! \brief utility function to keep track of the vertex already used while creating a new mesh
 *  \param vertexMap is a map used to keep track of the vertices already inserted in the position list
 *  \param position is a position list for the newly created mesh
 *  \param normals is a normal list for the newly created mesh
 *  \param texcoords is a texture coordinate list for the newly created mesh
 *  \param i is the Vertex looked for or inserted in vertexMap
 *  \return the index of this Vertex in the position vector list.
 */
inline unsigned int ObjReader::getVertex(
    std::map<Vertex, unsigned int> &vertexMap, 
    std::vector<Vec3fobj> &positions, 
    std::vector<Vec3fobj> &normals,
    std::vector<Vec2fobj> &texcoords,
    const Vertex &i)
{
    const std::map<Vertex, unsigned int>::iterator& entry = vertexMap.find(i);
    if (entry != vertexMap.end()) return(entry->second);
    
    positions.push_back(v[i.v]);
    if (i.vn >= 0) normals.push_back(vn[i.vn]);
    if (i.vt >= 0) texcoords.push_back(vt[i.vt]);
    return (vertexMap[i] = int(positions.size()) - 1);
}

/*! \brief flush the current content of currGroup and create new mesh 
 */
inline void ObjReader::flushFaceGroup()
{
    if (curGroup.empty()) return;
    
    // temporary data arrays
    std::vector<Vec3fobj> positions;
    std::vector<Vec3fobj> normals;
    std::vector<Vec2fobj> texcoords;
    std::vector<Vec3iobj> triangles;
    std::map<Vertex, unsigned int> vertexMap;
    
    // merge three indices into one
    for (size_t j = 0; j < curGroup.size(); j++)
    {
        /* iterate over all faces */
        const std::vector<Vertex>& face = curGroup[j];
        Vertex i0 = face[0], i1 = Vertex(-1), i2 = face[1];
        
        /* triangulate the face with a triangle fan */
        for (size_t k = 2; k < face.size(); k++) {
            i1 = i2; i2 = face[k];
            unsigned int v0 = getVertex(vertexMap, positions, normals, texcoords, i0);
            unsigned int v1 = getVertex(vertexMap, positions, normals, texcoords, i1);
            unsigned int v2 = getVertex(vertexMap, positions, normals, texcoords, i2);
            triangles.push_back(Vec3iobj(v0, v1, v2));
        }
    }
    curGroup.clear();

    // create new triangle mesh, allocate memory and copy data
    TriangleMesh* mesh = new TriangleMesh;
    mesh->numTriangles = triangles.size();
    mesh->triangles = new int[mesh->numTriangles * 3];
    memcpy(mesh->triangles, &triangles[0], sizeof(Vec3iobj) * mesh->numTriangles);
    mesh->positions = new Vec3fobj[positions.size()];
    memcpy(mesh->positions, &positions[0], sizeof(Vec3fobj) * positions.size());
    if (normals.size()) {
        mesh->normals = new Vec3fobj[normals.size()];
        memcpy(mesh->normals, &normals[0], sizeof(Vec3fobj) * normals.size());
    }
    if (texcoords.size()) {
        mesh->texcoords = new Vec2fobj[texcoords.size()];
        memcpy(mesh->texcoords, &texcoords[0], sizeof(Vec2fobj) * texcoords.size());
    }
    model.push_back(new OBJPrimitive(mesh, curMaterial));
}

inline  void ObjReader::printStats()
{
    int  size               =model.size();   
    int  totalTri           =0;
    //bool hasPositions       =true;
    hasNormals              =true;
    //bool hasTextureCoords   =true;
    for(int i=0; i<size;i++)
    {
        int s=model[i]->mesh->numTriangles;
        totalTri+=model[i]->mesh->numTriangles;
        //if(model[i]->mesh->positions==NULL)   hasPositions      =false;
        if(model[i]->mesh->normals  ==NULL)   hasNormals        =false;
        //if(model[i]->mesh->texcoords==NULL)   hasTextureCoords  =false;
        if(model[i]->mesh->normals  ==NULL){  
        }
        
        int count=0;
        for (int j=0;j<s*3;j++)
        {
            count=std::max(model[i]->mesh->triangles[j],count);
        }
        count++;
    }
    //cerr<<"Model Stats: "<<endl;
    //cerr<<"Meshes in model : "<<size<<endl;
    //cerr<<"Total triangles: "<<totalTri<<endl;
    //if(hasPositions)        cerr<<"Positions included"<<endl;
    //if(hasNormals)          cerr<<"Normals included"<<endl;
    //if(hasTextureCoords)    cerr<<"Texture Coords included"<<endl;
    totalTriangles=totalTri;

}

inline void ObjReader::extractNormals()
{
    
    int  size               =model.size();   

    //cerr<<"Extracting Normals..."<<endl;
    for(int i=0; i<size;i++)
    {
        int s=model[i]->mesh->numTriangles;
        int count=0;//find the number of verticies
        //cerr<<" "<<i;
        for (int j=0;j<s*3;j++)
        {
            count=std::max(model[i]->mesh->triangles[j],count);
            
        }
        if(count>0)
        {   
            //allocate space
            //cerr<<"Extracting "<<count<<endl;
            //cerr<<model[i]->mesh->normals<<endl;
            count++;//????
            Vec3fobj* norms= new Vec3fobj[count];
            model[i]->mesh->normals = &norms[0];
            //cerr<<" "<<model[i]->mesh->normals<<endl;
            //cerr<<"after Countaa "<<count<<endl;
            int* sharedVertexCount  = new int[count]; //TODO DELETE
            memset(sharedVertexCount,0,count*sizeof(int));
            Vec3fobj *norm;
            if(model[i]->mesh==NULL) cerr<<" mesh"<<model[i]->mesh<<endl;
            for(int j=0;j<s;j++)
            {
                //extract the normals
                int v0,v1,v2;
                v0=model[i]->mesh->triangles[j*3  ];
                v1=model[i]->mesh->triangles[j*3+1];
                v2=model[i]->mesh->triangles[j*3+2];
                //cerr<<"Vs "<<v0<<" "<<v1<<" "<<v2<<endl;
                sharedVertexCount[v0]++;
                sharedVertexCount[v1]++;
                sharedVertexCount[v2]++;

                Vec3fobj a;
                Vec3fobj b;
                Vec3fobj n;
                //a=v1-v0
                a.x=model[i]->mesh->positions[v1].x-model[i]->mesh->positions[v0].x;
                a.y=model[i]->mesh->positions[v1].y-model[i]->mesh->positions[v0].y;
                a.z=model[i]->mesh->positions[v1].z-model[i]->mesh->positions[v0].z;
                ///cerr<<"a "<<a<<endl;
                //b=v2-v0
                b.x=model[i]->mesh->positions[v2].x-model[i]->mesh->positions[v0].x;
                b.y=model[i]->mesh->positions[v2].y-model[i]->mesh->positions[v0].y;
                b.z=model[i]->mesh->positions[v2].z-model[i]->mesh->positions[v0].z;
                //cerr<<"b "<<b<<endl;
                //a cross b
                n.x = a.y*b.z - a.z*b.y;
                n.y = a.z*b.x - a.x*b.z;
                n.z = a.x*b.y - a.y*b.x;
                //cerr<<"n "<<n<<endl;
                //add normals
                
                model[i]->mesh->normals[v0].x+=n.x;
                //cerr<<"kKMMMMM "<< model[i]->mesh->normals[v0]<<endl;
                
                model[i]->mesh->normals[v0].y+=n.y;
                model[i]->mesh->normals[v0].z+=n.z;
                
                model[i]->mesh->normals[v1].x+=n.x;
                model[i]->mesh->normals[v1].y+=n.y;
                model[i]->mesh->normals[v1].z+=n.z;

                model[i]->mesh->normals[v2].x+=n.x;
                model[i]->mesh->normals[v2].y+=n.y;
                model[i]->mesh->normals[v2].z+=n.z;
                
            }

            //cycle back over the normals and average them, then normalize
            //cerr<<"COUNT "<<count<<endl;
            for (int j=0;j<count;j++)
            {
                norm=&model[i]->mesh->normals[j];
                norm->x/=sharedVertexCount[j];
                norm->y/=sharedVertexCount[j];
                norm->z/=sharedVertexCount[j];

                float m=sqrt(norm->x*norm->x+norm->y*norm->y+norm->z*norm->z);

                if(m==0) {cerr<<"DIVIDE BY ZER0"<<endl; continue;}
                

                norm->x*=m;
                norm->y*=m;
                norm->z*=m;
                //cerr<<"Shared "<<sharedVertexCount[j]<<endl;
                //cerr<<j<<" Norms : "<<*norm<<" "<<model[i]->mesh->normals[j]<<endl;
            }
            norm=NULL;
            delete[] sharedVertexCount;
            

        }
        

        

    }

}

inline void ObjReader::getRawData(float *&verts, float *&normals, float *&mats, int *& matIndex, int& numMats)
{
    if(totalTriangles>0) 
    {
        int     size = model.size();
        verts    = new float[totalTriangles*9];
        matIndex = new int[totalTriangles];
        mats     = new float[size*12];
        numMats=size;
        if(hasNormals) normals  = new float[totalTriangles*9];
        else normals=NULL;
        int     index= 0;
        int     matIdx=0; 
        cout<<"Size Mats "<<size<<endl;
        for(int i=0; i<size;i++) //for each model
        {
            //model[i]->material->Ka;

            //mats[i]=Material(model[i]->material->Ka,model[i]->material->Kd,model[i]->material->Ks,model[i]->material->d,model[i]->material->Ns,model[i]->material->Ni);
            //cout<<model[i]->material->Ka.x<<" "<<model[i]->material->Kd.x<<" "<<model[i]->material->Ks.x<<endl;
            //cout<<mats[i].Ka.x<<" "<<mats[i].Kd.x<<" "<<mats[i].Ks.x<<endl;
            int s=model[i]->mesh->numTriangles;

            mats[i*12   ]=model[i]->material->Ka.x;
            mats[i*12+1 ]=model[i]->material->Ka.y;
            mats[i*12+2 ]=model[i]->material->Ka.z;

            mats[i*12+3 ]=model[i]->material->Kd.x;
            mats[i*12+4 ]=model[i]->material->Kd.y;
            mats[i*12+5 ]=model[i]->material->Kd.z;

            mats[i*12+6 ]=model[i]->material->Ks.x;
            mats[i*12+7 ]=model[i]->material->Ks.y;
            mats[i*12+8 ]=model[i]->material->Ks.z;

            mats[i*12+9 ]=model[i]->material->Ns;
            mats[i*12+10]=0.f;
            mats[i*12+11]=0.f;


            for(int j=0;j<s;j++) //for each triangle
            {
                int v0,v1,v2;
                v0=model[i]->mesh->triangles[j*3  ];
                v1=model[i]->mesh->triangles[j*3+1];
                v2=model[i]->mesh->triangles[j*3+2];
                matIndex[matIdx]=i;
                matIdx++;
                //v0
                verts[index]=model[i]->mesh->positions[v0].x;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v0].x;
                index++;
                verts[index]=model[i]->mesh->positions[v0].y;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v0].y;
                index++;
                verts[index]=model[i]->mesh->positions[v0].z;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v0].z;
                index++;
                //v1
                verts[index]=model[i]->mesh->positions[v1].x;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v1].x;
                index++;
                verts[index]=model[i]->mesh->positions[v1].y;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v1].y;
                index++;
                verts[index]=model[i]->mesh->positions[v1].z;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v1].z;
                index++;
                //v2 
                verts[index]=model[i]->mesh->positions[v2].x;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v2].x;
                index++;
                verts[index]=model[i]->mesh->positions[v2].y;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v2].y;
                index++;
                verts[index]=model[i]->mesh->positions[v2].z;
                if(hasNormals) normals[index]=model[i]->mesh->normals[v2].z;
                index++;
            }

        }//for
        //cerr<<"Number of Verts output : "<<index<<" for "<<totalTriangles<<" triangles"<<endl;
    }//if
    else{
        cerr<<"No Triangles to output"<<endl;
        verts=NULL;
    }
}

#endif