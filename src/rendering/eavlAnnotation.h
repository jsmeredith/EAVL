#ifndef EAVL_ANNOTATION_H
#define EAVL_ANNOTATION_H

class eavlWindow;

class eavlAnnotation
{
  protected:
    eavlWindow *win;
  public:
    eavlAnnotation(eavlWindow *w)
        : win(w)
    {
    }
    
};

/*class eavl3DAnnotation
{
  protected:
  public:
    void Prepare(
};*/

#endif
