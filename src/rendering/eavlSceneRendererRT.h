// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_RT_H
#define EAVL_SCENE_RENDERER_RT_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"

#define mindist 0.01

class Ray
{
  public:
    eavlPoint3 s;
    eavlVector3 v;
  public:
    Ray(eavlPoint3 eye, eavlPoint3 screenpt)
    {
        s = screenpt;
        v = (screenpt - eye).normalized();
    }
    Ray(eavlPoint3 source, eavlVector3 dir)
    {
        s = source;
        v = dir.normalized();
    }
};


class Object
{
  public:
    bool emissive;
    float reflection;
    eavlColor color;
    Object() : color(1,1,1,0)
    {
        emissive = false;
        reflection = 0;
    }
    virtual float Intersect(Ray &ray, eavlPoint3 &point, eavlVector3 &normal,
                            float &value) = 0;
};

class Triangle : public Object
{
  protected:
  public:
    eavlPoint3 p;
    eavlVector3 n;
    eavlVector3 n0, n1, n2;
    eavlVector3 e1, e2;
    float v0, v1, v2;
    float d;
  public:
    Triangle(float ax, float ay, float az,
             float bx, float by, float bz,
             float cx, float cy, float cz,
             float v0_, float v1_, float v2_,
             eavlVector3 n0_, eavlVector3 n1_, eavlVector3 n2_)
        : p(ax,ay,az), v0(v0_), v1(v1_), v2(v2_), n0(n0_), n1(n1_), n2(n2_)
    {
        eavlPoint3 b(bx,by,bz), c(cx,cy,cz);
        e1 = b-p;
        e2 = c-p;
        n = e2 % e1;
        n.normalize();
        d = -p*n;
    }
    Triangle(float ax, float ay, float az,
             float bx, float by, float bz,
             float cx, float cy, float cz)
        : p(ax,ay,az), v0(0), v1(0), v2(0)
    {
        eavlPoint3 b(bx,by,bz), c(cx,cy,cz);
        e1 = b-p;
        e2 = c-p;
        n = e2 % e1;
        n.normalize();
        d = -p*n;

        n0=n;
        n1=n;
        n2=n;
    }
    virtual float Intersect(Ray &ray, eavlPoint3 &point, eavlVector3 &normal,
                            float &value)
    {
        eavlVector3 h = ray.v % e2;
        float a = e1 * h;
        if (a == 0)
            return -1;

        float f = 1. / a;
        eavlVector3 s = ray.s - p;
        float u = f * (s*h);
        if (u < 0 || u > 1)
            return -1;

        eavlVector3 q = s % e1;
        float v = f * (ray.v*q);
        if (v < 0 || u+v > 1)
            return -1;

        float t = f * (e2*q);
        point = ray.s + t*ray.v;

        // barycentric coords are <w,u,v>
        float w = 1. - (u+v);

        // single-face normal code
        //normal = n;

        // per-node normal code
        normal = w*n0 + u*n1 + v*n2;
        normal.normalize();

        if (normal * ray.v > 0)
            normal = -normal;

        // do the value, too
        value = w*v0 + u*v1 + v*v2;

        return t;
    }
};
             
class Scene
{
  public:
    vector<Object*> objects;
  public:
    Object *Intersect(Ray &ray, float &dist, eavlPoint3 &point, eavlVector3 &normal,
                      float &value)
    {
        Object *object = NULL;
        //cerr << "objects.size="<<objects.size()<<endl;
        for (int i=0; i<objects.size(); ++i)
        {
            eavlPoint3 p;
            eavlVector3 n;
            float v;
            float d = objects[i]->Intersect(ray, p, n, v);
            //cerr << "d="<<d<<endl;
            if (d > mindist && (!object || d < dist))
            {
                //cerr << "HITHITHIT\n";
                object  = objects[i];
                dist    = d;
                point   = p;
                normal  = n;
                value   = v;
            }
        }
        return object;
    }
    Object *IntersectFarthest(Ray &ray, float &dist, eavlPoint3 &point, eavlVector3 &normal,
                              float &value)
    {
        Object *object = NULL;
        for (int i=0; i<objects.size(); ++i)
        {
            eavlPoint3 p;
            eavlVector3 n;
            float v;
            float d = objects[i]->Intersect(ray, p, n, v);
            if (d > mindist && (!object || d > dist))
            {
                object  = objects[i];
                dist    = d;
                point   = p;
                normal  = n;
                value   = v;
            }
        }
        return object;
    }
};

inline eavlColor CastRay(Ray r, Scene &scene, eavlPoint3 &lightpos,
                         float &dist, eavlPoint3 &pt,
                         eavlColorTable &ct,
                         int depth = 0)
{
    eavlColor c;
    eavlVector3 norm;
    dist = -1;
    float value;
    Object *o = scene.Intersect(r, dist, pt, norm, value);

    if (!o)
    {
        // we don't need to set dist to 0, do we?
        dist = -1;
        return c;
    }

    //cerr << "HIT\n";

#if 1 // map value to color
    eavlColor self = ct.Map(value);
#else // self-single color
    eavlColor self = o->color;
#endif


    if (o->emissive)
    {
        c = self;
    }
    else
    {
#if 0
        float bright = 1;
#else
        eavlVector3 lightvec = lightpos - pt;
        float       lightdist = lightvec.norm();
        eavlVector3 lightdir = lightvec.normalized();
                
        float bright = lightdir * norm;
        // clamp to ambient
        if (bright < .15)
            bright = .15;

        Ray lightray(pt, lightdir);
        eavlPoint3 lr_p;
        eavlVector3 lr_n;
        float lr_d;
        float lr_v;
#if 1
        Object *lr_o = scene.Intersect(lightray, lr_d, lr_p, lr_n, lr_v);
        if (lr_o && lr_d < lightdist)
            bright *= .5;
#endif

        //bright += .15;
#endif

        c.c[0] = bright * self.c[0];
        c.c[1] = bright * self.c[1];
        c.c[2] = bright * self.c[2];

        if (o->reflection > 0)
        {
            Ray ref(pt, r.v - 2 * norm * (r.v * norm));

            float ref_dist;
            eavlPoint3 refpt;
            eavlColor refcolor = CastRay(ref,scene,lightpos,ref_dist,refpt,ct,depth-1);
                    
            if (ref_dist > mindist)
            {
                c.c[0] = (1.-o->reflection)*c.c[0] + o->reflection*refcolor.c[0];
                c.c[1] = (1.-o->reflection)*c.c[1] + o->reflection*refcolor.c[1];
                c.c[2] = (1.-o->reflection)*c.c[2] + o->reflection*refcolor.c[2];
            }
        }
    }
    return c;
}



// ****************************************************************************
// Class:  eavlSceneRendererRT
//
// Purpose:
///   A very simple, though not necessarily fast, implementation of
///   a raytracing renderer.
//
// Programmer:  
// Creation:    July 14, 2014
//
// Modifications:
//
// ****************************************************************************
class eavlSceneRendererRT : public eavlSceneRenderer
{
    Scene scene;
    vector<byte> rgba;
    vector<float> depth;
  public:

    virtual void AddTriangleVnVs(double x0, double y0, double z0,
                                 double x1, double y1, double z1,
                                 double x2, double y2, double z2,
                                 double u0, double v0, double w0,
                                 double u1, double v1, double w1,
                                 double u2, double v2, double w2,
                                 double s0, double s1, double s2)
    {
        // Get points and normals into camera space
        eavlPoint3 p0(x0,y0,z0);
        eavlPoint3 p1(x1,y1,z1);
        eavlPoint3 p2(x2,y2,z2);

        eavlVector3 n0(u0,v0,w0);
        eavlVector3 n1(u1,v1,w1);
        eavlVector3 n2(u2,v2,w2);

        Triangle *tri = new Triangle(p0.x, p0.y, p0.z,
                                     p1.x, p1.y, p1.z,
                                     p2.x, p2.y, p2.z,
                                     s0, s1, s2,
                                     n0, n1, n2);
        tri->color = eavlColor::white;
        scene.objects.push_back(tri);
    }
    virtual void StartTriangles()
    {
    }

    virtual void EndTriangles()
    {
    }

    virtual void StartScene()
    {
        scene.objects.clear();
    }

    virtual void EndScene()
    {
        // nothing to do here
    }

    virtual void Render(eavlView view)
    {
        rgba.clear();
        depth.clear();
        rgba.resize(4*view.w*view.h, 0);
        depth.resize(view.w*view.h, 1.0f);

        view.SetupForWorldSpace();

        /*
        for (int i=0; i<scene.objects.size(); ++i)
        {
            if (dynamic_cast<Triangle*>(scene.objects[i]))
            {
                Triangle *t = dynamic_cast<Triangle*>(scene.objects[i]);
                t->p = view.V * t->p;
                t->n = view.V * t->n;
                t->e1 = view.V * t->e1;
                t->e2 = view.V * t->e2;
            }
        }
        */
        
        eavlPoint3 lightpos(20,40,0);
        eavlColorTable ct("default");
        int w = view.w;
        int h = view.h;

        // todo: should probably include near/far clipping planes
        double eyedist = 1./tan(view.view3d.fov/2.); // fov already radians
#if 0
        eavlPoint3 eye(0,0,0);
        eavlPoint3 screencenter(0,0,-eyedist);
        eavlVector3 screenx(view.viewportaspect,0,0);
        eavlVector3 screeny(0,1.,0);
        eavlMatrix4x4 vv = view.V;
        vv.Invert();
        eye = vv*eye;
        screencenter = vv*screencenter;
        screenx = vv*screenx;
        screeny = vv*screeny;
#else
        float lookdist = (view.view3d.at - view.view3d.from).norm();
        eavlVector3 lookdir = (view.view3d.at - view.view3d.from).normalized();
        eavlPoint3 eye = view.view3d.from;
        eavlPoint3 screencenter = view.view3d.from + lookdir*eyedist;
        eavlVector3 right = (lookdir % view.view3d.up).normalized();
        eavlVector3 up = (right % lookdir).normalized();
        eavlVector3 screenx = right * view.viewportaspect;
        eavlVector3 screeny = up;
#endif

        // need to find real z buffer values:
        float proj22=view.P(2,2);
        float proj23=view.P(2,3);
        float proj32=view.P(3,2);

        /*
        // some debug info:
        cerr << endl;
        cerr << "eyedist="<<eyedist<<endl;
        cerr << "view3d.nearplane="<<view.view3d.nearplane<<endl;
        cerr << "eye="<<eye<<endl;
        cerr << "screencenter="<<screencenter<<endl;
        cerr << "screenx="<<screenx<<endl;
        cerr << "screeny="<<screeny<<endl;

        cerr << "22 = " << view.P(2,2) << endl;
        cerr << "23 = " << view.P(2,3) << endl;
        cerr << "32 = " << view.P(3,2) << endl;
        cerr << "view.P="<<view.P<<endl;
        */
        
        float minz = FLT_MAX;
        float maxz = -FLT_MAX;
        float mind = FLT_MAX;
        float maxd = -FLT_MAX;

        const int skip=5;
#pragma omp parallel for schedule(dynamic,1) collapse(2)
        for (int y=0; y<h; y += skip)
        {
            for (int x=0; x<w; x += skip)
            {
                float xx = (float(x)/float(w-1)) * 2 - 1;
                float yy = (float(y)/float(h-1)) * 2 - 1;
                eavlPoint3 screenpt = screencenter + screenx*xx + screeny*yy;
                Ray r(eye, screenpt);

                eavlPoint3 pt;
                float dist;
                eavlColor c = CastRay(r, scene, lightpos, dist, pt, ct, 0);
                if (dist <= mindist)
                {
                    //cerr << "no intersection!\n";
                    continue;
                }

                if (pt.z < mind) mind = pt.z;
                if (pt.z > maxd) maxd = pt.z;

                if (false)
                {
                    eavlPoint3 backpt;
                    eavlVector3 backnorm;
                    float backdist = -1;
                    float backvalue;
                    Object *o = scene.IntersectFarthest(r, backdist, backpt, backnorm, backvalue);
                    if (o && backdist > mindist)
                    {
                        if (backdist < mind) mind = backdist;
                        if (backdist > maxd) maxd = backdist;
                    }
                }


                //eavlPoint3 pt = r.s + r.v * dist;
                //eavlPoint3 pt(xx,yy,-(dist + eyedist));

                // case #1 (old) where we pretransformed triangle
                // vertices by view matrix and left camera in normalized space
                //eavlPoint3 p2 = view.P * pt;
                //float projdepth = p2[2];

                // case #2: where points are still in world space:
                //eavlPoint3 p2 = view.P * view.V * pt;
                //float projdepth = p2[2];

                // same thing as previous, but much simpler (and a bit faster):
                // get the depth into the scene
                // (proj distance along ray onto distance into scene):
                float scenedepth = eyedist + (lookdir * r.v) * dist;
                // ... then use projection matrix to get projected depth
                // (but remember depth is negative in RH system)
                float projdepth = (proj22 + proj23 / (-scenedepth)) / proj32;

                if (projdepth < minz) minz = projdepth;
                if (projdepth > maxz) maxz = projdepth;
                for (int xo=0; xo<skip && x+xo<w; xo++)
                {
                    for (int yo=0; yo<skip && y+yo<h; yo++)
                    {
                        byte *pixel = &(rgba[4*((y+yo)*w+(x+xo))]);
                        pixel[0] = c.GetComponentAsByte(0);
                        pixel[1] = c.GetComponentAsByte(1);
                        pixel[2] = c.GetComponentAsByte(2);
                        float *z =  &(depth[(y+yo)*w+(x+xo)]);
                        *z = .5 * projdepth + .5;
                    }
                }
            }
        }

        /*
        cerr << "RT minz="<<minz<<endl;
        cerr << "RT maxz="<<maxz<<endl;
        cerr << "RT zrange="<<fabs(minz-maxz)<<endl;
        cerr << "RT mind="<<mind<<endl;
        cerr << "RT maxd="<<maxd<<endl;
        cerr << "RT drange="<<fabs(mind-maxd)<<endl;
        */


        glColor3f(1,1,1);
        glDisable(GL_BLEND);
        glDisable(GL_LIGHTING);
        glDisable(GL_DEPTH_TEST);

        // draw the pixel colors
        glDrawPixels(view.w, view.h, GL_RGBA, GL_UNSIGNED_BYTE, &rgba[0]);

        // drawing the Z buffer will overwrite the pixel colors
        // unless you actively prevent it....
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glDepthMask(GL_TRUE);
        // For some bizarre reason, you need GL_DEPTH_TEST enabled for
        // it to write into the Z buffer. 
        glEnable(GL_DEPTH_TEST);

        // draw the z buffer
        glDrawPixels(view.w, view.h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0]);

        // set the various masks back to "normal"
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
    }
};


#undef mindist

#endif
