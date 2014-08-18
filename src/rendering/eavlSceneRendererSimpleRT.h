// Copyright 2010-2014 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_SCENE_RENDERER_SIMPLE_RT_H
#define EAVL_SCENE_RENDERER_SIMPLE_RT_H

#include "eavlDataSet.h"
#include "eavlCellSet.h"
#include "eavlColor.h"
#include "eavlColorTable.h"
#include "eavlSceneRenderer.h"

#define mindist 0.01

// note: kmeans, noiseiso, xcoord polys, default size, plotlist open,
//       1x1, openmp, no normals, rate = 16 images in 10 seconds

//       bcc lattice, same other parameters, 23 images in 10 seconds

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
    virtual float GetFurthestDistanceFromPoint(eavlPoint3 &point) = 0;
    virtual eavlPoint3 GetCentroid() = 0;
};

class Sphere : public Object
{
  protected:
  public:
    eavlPoint3 o;
    float r;
    float v;
  public:
    Sphere(float x, float y, float z, float rad, float val)
        : Object(), o(x,y,z), r(rad), v(val)
    {
    }
    virtual float Intersect(Ray &ray, eavlPoint3 &point, eavlVector3 &normal,
                            float &value)
    {
        eavlVector3 oc = ray.s - o;
        float A = ray.v * ray.v;
        float B = 2 * oc * ray.v;
        float C = oc*oc - r*r;

        float discr = B*B - 4*A*C;

        /*
        float ocx = ray.s.x - o.x;
        float ocy = ray.s.y - o.y;
        float ocz = ray.s.z - o.z;

        float rvx = ray.v.x;
        float rvy = ray.v.y;
        float rvz = ray.v.z;

        float A = rvx*rvx + rvy*rvy + rvz*rvz;
        float B = 2 * (ocx*rvx + ocy*rvy + ocz*rvz);
        float C = (ocx*ocx + ocy*ocy + ocz*ocz) - r*r;

        float d1 = B*B;
        float d2 = 4*A*C;
        float discr = d1 - d2;
        */

        /*
        cerr << "A="<<A<<endl;
        cerr << "B="<<B<<endl;
        cerr << "C="<<C<<endl;
        cerr << "d1="<<d1<<endl;
        cerr << "d2="<<d2<<endl;
        cerr << "d1-d2="<<d1-d2<<endl;
        cerr << endl;
        */
        if (discr < 0)
            return -1;

        float q;
        if (B<0)
            q = (-B + sqrt(discr)) * 0.5;
        else
            q = (-B - sqrt(discr)) * 0.5;

        float t0 = q/A;
        float t1 = C/q;
        // choose the smaller of the two, unless it's negative, in
        // which case choose the other
        float t = (t0 < t1) ? (t0 < 0 ? t1 : t0) : (t1 < 0 ? t0 : t1);

        point = ray.s + t*ray.v;
        normal = (point - o).normalized();
        value = v;
        return t;
    }
    virtual float GetFurthestDistanceFromPoint(eavlPoint3 &point)
    {
        return (o - point).norm() + r;
    }
    virtual eavlPoint3 GetCentroid()
    {
        return o;
    }
};

class Triangle : public Object
{
  protected:
  public:
    eavlPoint3 p;
    eavlVector3 n;
    eavlVector3 e1, e2;
    float v0, v1, v2;
    eavlVector3 n0, n1, n2;
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
    virtual float GetFurthestDistanceFromPoint(eavlPoint3 &point)
    {
        float d0 = (p - point).norm();
        float d1 = ((p+e1) - point).norm();
        float d2 = ((p+e2) - point).norm();
        return (d0 > d1) ? (d0 > d2 ? d0 : d2) : (d1 > d2 ? d1 : d2);
    }
    virtual eavlPoint3 GetCentroid()
    {
        return p + (e1 + e2)/3.;
    }
};

class Container
{
  public:
    vector<Object*> objects;
    vector<Container*> containers;
  public:
    virtual Object *Intersect(Ray &ray,
                              float &dist,
                              eavlPoint3 &point,
                              eavlVector3 &normal,
                              float &value)
    {
        Object *object = NULL;
        for (int i=0; i<objects.size(); ++i)
        {
            eavlPoint3 p;
            eavlVector3 n;
            float v;
            float d = objects[i]->Intersect(ray, p, n, v);
            if (d > mindist && (!object || d < dist))
            {
                object  = objects[i];
                dist    = d;
                point   = p;
                normal  = n;
                value   = v;
            }
        }
        for (int i=0; i<containers.size(); ++i)
        {
            eavlPoint3 p;
            eavlVector3 n;
            float v;
            float d = -1;
            Object *o = containers[i]->Intersect(ray, d, p, n, v);
            if (o && d > mindist && (!object || d < dist))
            {
                object  = o;
                dist    = d;
                point   = p;
                normal  = n;
                value   = v;
            }
        }
        return object;
    }
};

class BoundingSphere : public Container
{
  public:
    Sphere *sphere;
    BoundingSphere(float x, float y, float z, float rad, float val)
        : sphere(new Sphere(x,y,z,rad,val))
    {
    }
    virtual Object *Intersect(Ray &ray,
                              float &dist,
                              eavlPoint3 &point,
                              eavlVector3 &normal,
                              float &value)
    {
        eavlPoint3 p;
        eavlVector3 n;
        float v;
        float d = sphere->Intersect(ray, p, n, v);
#if 0 // TEST BY PRETENDING CONTAINER IS ACTUALLY ITS SPHERE OBJECT
        if (d > mindist)
        {
            dist = d;
            point = p;
            normal = n;
            value = v;
            return sphere;
        }
#else // REAL CODE HERE:
        if (d > mindist)
        {
            Object *o =  Container::Intersect(ray, dist, point, normal, value);
            //value = v; // debug: override content color by bounding sphere color
            return o;
        }
#endif

        return NULL;
    }

};
             


inline eavlColor CastRay(Ray r, Container &scene, eavlVector3 &lightvec,
                         float &dist, eavlPoint3 &pt,
                         int ncolors, float *colors,
                         int depth = 0)
{
    eavlColor c;
    eavlVector3 norm;
    dist = -1;
    float value = 0;
    Object *o = scene.Intersect(r, dist, pt, norm, value);

    if (!o)
    {
        // we don't need to set dist to 0, do we?
        dist = -1;
        return c;
    }

    //cerr << "HIT\n";

#if 1 // map value to color
    int colorindex = float(ncolors-1) * value;
    eavlColor self(colors[colorindex*3+0],
                   colors[colorindex*3+1],
                   colors[colorindex*3+2]);
    //eavlColor self = ct.Map(value);
    
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
        //eavlVector3 lightvec = lightpos - pt;
        //float       lightdist = lightvec.norm();
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
        // shadow
        Object *lr_o = scene.Intersect(lightray, lr_d, lr_p, lr_n, lr_v);
        if (lr_o /*&& lr_d < lightdist*/)
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
            eavlColor refcolor = CastRay(ref,scene,lightdir,ref_dist,refpt,ncolors,colors,depth-1);
                    
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
// Class:  eavlSceneRendererSimpleRT
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
class eavlSceneRendererSimpleRT : public eavlSceneRenderer
{
    Container scene;
    vector<byte> rgba;
    vector<float> depth;

    int firstskip;
    int skip;
    eavlView lastview;
  public:

    eavlSceneRendererSimpleRT() : eavlSceneRenderer()
    {
        skip = firstskip = 4;
    }

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
        cout<<"calling start triangles"<<endl;
    }

    virtual void EndTriangles()
    {
    }

    virtual void AddPointVs(double x, double y, double z, double r, double s)
    {
        Sphere *sph = new Sphere(x,y,z,r,s);
        sph->color = eavlColor::white;
        scene.objects.push_back(sph);
    }
    virtual void AddLineVs(double x0, double y0, double z0,
                           double x1, double y1, double z1,
                           double s0, double s1)
    {
    }

    virtual void StartScene()
    {
        for (unsigned int i=0; i<scene.objects.size(); ++i)
            delete scene.objects[i];
        scene.objects.clear();
        for (unsigned int i=0; i<scene.containers.size(); ++i)
            delete scene.containers[i];
        scene.containers.clear();
        //cerr << "startscene\n";

        skip = firstskip;
        rgba.clear();
        depth.clear();
        rgba.resize(4*view.w*view.h, 0);
        depth.resize(view.w*view.h, 1.0f);
    }

    virtual void EndScene()
    {
        // Create a bounding hierarchy.
        // Either K-Means or using a fixed BCC sphere lattice.
        // (BCC seems to win in both construction and performance
        // for common vis data sets because it's designed to
        // minimize overlap.)

        int N = scene.objects.size();
#if 0 // K-Means
        const int K = 150;
        if (N < K)
        {
            cout << "Too few objects, no clusters\n";
            return;
        }

        //
        // initialize clusters to centroid of first K objects
        //
        //cout << "Creating clusters\n";
        vector<eavlPoint3> clusters(K);
        for (int k=0; k<K; ++k)
        {
            clusters[k] = scene.objects[k]->GetCentroid();
            //cout << "   k="<<k<<"  " << clusters[k] << endl;
        }

        //
        // perform K-means for a number of passes
        //
        //cout << "K-means\n";
        const int npasses = 10;
        vector<int> id(N);
        vector<int> counts(K);
        for (int pass = 0; pass < npasses; ++pass)
        {
            //cout << "   pass " << pass << endl;
            //cout << "      labeling\n";
            for (int i=0; i<N; ++i)
            {
                eavlPoint3 p = scene.objects[i]->GetCentroid();
                float d = (p - clusters[0]).norm();
                id[i] = 0;
                for (int k=1; k<K; ++k)
                {
                    float dist = (p - clusters[k]).norm();
                    if (dist < d)
                    {
                        d = dist;
                        id[i] = k;
                    }
                }
            }
            //cout << "      adjusting\n";
            counts.clear();
            counts.resize(K, 0);
            for (int k=0; k<K; ++k)
                clusters[k] = eavlPoint3(0,0,0);
            for (int i=0; i<N; ++i)
            {
                eavlPoint3 c = scene.objects[i]->GetCentroid();
                clusters[id[i]] = clusters[id[i]] + eavlVector3(c.x,c.y,c.z);
                counts[id[i]]++;
            }
            for (int k=0; k<K; ++k)
            {
                if (counts[k] > 0)
                    clusters[k] /= float(counts[k]);
            }
        }

        //
        // create a bounding sphere for each cluster
        // and put the old objects into it
        //
        vector<BoundingSphere*> spheres;
        for (int k=0; k<K; ++k)
        {
            float rad = 0;
            for (int i=0; i<N; ++i)
            {
                if (id[i] != k)
                    continue;
                float d = scene.objects[i]->GetFurthestDistanceFromPoint(clusters[k]);
                if (d > rad)
                    rad = d;
            }
            //cout << "   k="<<k<<"  " << clusters[k] << " rad="<<rad<<endl;
            BoundingSphere *b = new BoundingSphere(clusters[k].x,
                                                   clusters[k].y,
                                                   clusters[k].z,
                                                   rad * 1.01, // fuzz factor
                                                   double(random()) / double(RAND_MAX));
            for (int i=0; i<N; ++i)
            {
                if (id[i] == k)
                    b->objects.push_back(scene.objects[i]);
            }
            spheres.push_back(b);
        }

        // replace scene
        scene.objects.clear();
        for (int i=0; i<K; ++i)
            scene.containers.push_back(spheres[i]);
#else
        vector<BoundingSphere*> spheres;
        //double l = view.size / 6.5;
        double l = view.size / 11.5;
        // radius of 0.56 on both sphere locations just barely covers space;
        // we want a little overlap because we're currently looking
        // for a sphere which contains objects in their entirety,
        // so we only have to stick them in one
        double rad1 = 0.6 * l;
        double rad2 = 0.6 * l;
        for (double z = view.minextents[2]; z < view.maxextents[2]+l*.5; z += l)
        {
            for (double y = view.minextents[1]; y < view.maxextents[1]+l*.5; y += l)
            {
                for (double x = view.minextents[0]; x < view.maxextents[0]+l*.5; x += l)
                {
                    BoundingSphere *bs1 = new BoundingSphere(x,y,z,rad1,double(random())/double(RAND_MAX));
                    spheres.push_back(bs1);
                    if (x<view.maxextents[0] &&
                        y<view.maxextents[1] &&
                        z<view.maxextents[2])
                    {
                        BoundingSphere *bs2 = new BoundingSphere(x+l*.5,
                                                                 y+l*.5,
                                                                 z+l*.5,
                                                                 rad2,double(random())/double(RAND_MAX));
                        spheres.push_back(bs2);
                    }
                }
            }
        }
        int K = spheres.size();
        //cerr << "Generated "<<K<<" bounding spheres\n";

        vector<Object*> otherobjects;
        for (int i=0; i<N; ++i)
        {
            bool found = false;
            for (int k=0; k<K; ++k)
            {
                float d = scene.objects[i]->GetFurthestDistanceFromPoint(spheres[k]->sphere->o);
                if (d <= spheres[k]->sphere->r)
                {
                    found = true;
                    spheres[k]->objects.push_back(scene.objects[i]);
                    break;
                }
            }
            if (!found)
            {
                //cerr << "couldn't find a complete container for some object\n";
                otherobjects.push_back(scene.objects[i]);
            }
        }

        // replace scene
        scene.objects.clear();
        for (int i=0; i<K; ++i)
        {
            if (spheres[i]->objects.size() > 0)
                scene.containers.push_back(spheres[i]);
            else
                delete spheres[i];
        }
        //cerr << "Final bounding spheres: " << scene.containers.size() << endl;
        scene.objects.insert(scene.objects.end(),
                             otherobjects.begin(),
                             otherobjects.end());
        
#endif
        //cout << "endscene N="<<N<<"\n";
    }

    virtual bool ShouldRenderAgain()
    {
        return false;
    }

    virtual unsigned char *GetRGBAPixels()
    {
        return &rgba[0];
    }

    virtual float *GetDepthPixels()
    {    
        return &depth[0];
    }

    virtual void Render()
    {
        if (view != lastview)
        {
            rgba.clear();
            depth.clear();
            rgba.resize(4*view.w*view.h, 0);
            depth.resize(view.w*view.h, 1.0f);
            skip = firstskip;
            lastview = view;
        }
        else if (skip > 0)
        {
            rgba.clear();
            depth.clear();
            rgba.resize(4*view.w*view.h, 0);
            depth.resize(view.w*view.h, 1.0f);
        }

        if (skip == 0)
        {
            return;
        }

        /*
          // we used to transform geometry into view space;
          // not a very efficient method.....
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
        
        eavlVector3 lightdir(Lx,Ly,Lz);
        if (eyeLight)
        {
            eavlMatrix4x4 IV = view.V;
            IV.Invert();
            lightdir = IV * lightdir;
        }
        int w = view.w;
        int h = view.h;

        // todo: should probably include near/far clipping planes
        double eyedist = 1./tan(view.view3d.fov/2.); // fov already radians

#if 0
        // "simple" way of calculating eye/screen positions in world space
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
        eavlVector3 lookdir = (view.view3d.at - view.view3d.from).normalized();
#else
        // "direct" way of calculating eye/screen positions in world space
        eavlVector3 lookdir = (view.view3d.at - view.view3d.from).normalized();
        eavlPoint3 eye = view.view3d.from;
        eavlPoint3 screencenter = view.view3d.from + lookdir*eyedist;
        eavlVector3 right = (lookdir % view.view3d.up).normalized();
        eavlVector3 up = (right % lookdir).normalized();
        eavlVector3 screenx = right * view.viewportaspect;
        eavlVector3 screeny = up;
#endif

        screenx /= view.view3d.zoom;
        screeny /= view.view3d.zoom;

        screencenter -= view.view3d.xpan * screenx;
        screencenter -= view.view3d.ypan * screeny;


        // need to find real z buffer values:
        float proj22=view.P(2,2);
        float proj23=view.P(2,3);
        float proj32=view.P(3,2);

        //cerr << "rendering\n";
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

#pragma omp parallel for schedule(dynamic,1) collapse(2)
        for (int y=0; y<h; y += skip)
        {
            for (int x=0; x<w; x += skip)
            {
                float xx = (float(x+skip/2)/float(w-1)) * 2 - 1;
                float yy = (float(y+skip/2)/float(h-1)) * 2 - 1;
                eavlPoint3 screenpt = screencenter + screenx*xx + screeny*yy;
                Ray r(eye, screenpt);

                eavlPoint3 pt;
                float dist;
                eavlColor c = CastRay(r, scene, lightdir, dist, pt, ncolors,colors, 0);
                if (dist <= mindist)
                {
                    //cerr << "no intersection!\n";
                    continue;
                }

                if (pt.z < mind) mind = pt.z;
                if (pt.z > maxd) maxd = pt.z;


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

        skip /= 2;
    }
};


#undef mindist

#endif
