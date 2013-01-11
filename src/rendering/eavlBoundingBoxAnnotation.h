// Copyright 2010-2013 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef EAVL_BOUNDING_BOX_ANNOTATION_H
#define EAVL_BOUNDING_BOX_ANNOTATION_H

// ****************************************************************************
// Class:  eavlBoundingBoxAnnotation
//
// Purpose:
///   A 3D bounding box.
//
// Programmer:  Jeremy Meredith
// Creation:    January 11, 2013
//
// Modifications:
// ****************************************************************************
class eavlBoundingBoxAnnotation : public eavlWorldSpaceAnnotation
{
  protected:
    double dmin[3], dmax[3];
  public:
    eavlBoundingBoxAnnotation(eavlWindow *win) :
        eavlWorldSpaceAnnotation(win)
    {
        dmin[0] = dmin[1] = dmin[2] = -1;
        dmax[0] = dmax[1] = dmax[2] = +1;
    }
    void SetExtents(double xmin, double xmax,
                    double ymin, double ymax,
                    double zmin, double zmax)
    {
        dmin[0] = xmin;
        dmax[0] = xmax;
        dmin[1] = ymin;
        dmax[1] = ymax;
        dmin[2] = zmin;
        dmax[2] = zmax;
    }
    virtual void Render()
    {
        glDisable(GL_LIGHTING);
        glLineWidth(1);
        glColor3f(.4,.4,.4);
        glBegin(GL_LINES);
        glVertex3d(dmin[0],dmin[1],dmin[2]); glVertex3d(dmin[0],dmin[1],dmax[2]);
        glVertex3d(dmin[0],dmax[1],dmin[2]); glVertex3d(dmin[0],dmax[1],dmax[2]);
        glVertex3d(dmax[0],dmin[1],dmin[2]); glVertex3d(dmax[0],dmin[1],dmax[2]);
        glVertex3d(dmax[0],dmax[1],dmin[2]); glVertex3d(dmax[0],dmax[1],dmax[2]);

        glVertex3d(dmin[0],dmin[1],dmin[2]); glVertex3d(dmin[0],dmax[1],dmin[2]);
        glVertex3d(dmin[0],dmin[1],dmax[2]); glVertex3d(dmin[0],dmax[1],dmax[2]);
        glVertex3d(dmax[0],dmin[1],dmin[2]); glVertex3d(dmax[0],dmax[1],dmin[2]);
        glVertex3d(dmax[0],dmin[1],dmax[2]); glVertex3d(dmax[0],dmax[1],dmax[2]);

        glVertex3d(dmin[0],dmin[1],dmin[2]); glVertex3d(dmax[0],dmin[1],dmin[2]);
        glVertex3d(dmin[0],dmin[1],dmax[2]); glVertex3d(dmax[0],dmin[1],dmax[2]);
        glVertex3d(dmin[0],dmax[1],dmin[2]); glVertex3d(dmax[0],dmax[1],dmin[2]);
        glVertex3d(dmin[0],dmax[1],dmax[2]); glVertex3d(dmax[0],dmax[1],dmax[2]);
        glEnd();
    }    
};


#endif
