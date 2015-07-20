#ifndef SplitBVH_H
#define SplitBVH_H
#include "bvh/BVHNode.h"
#include "bvh/Util.h"
#include "bvh/SplitBVHBuilder.h"

#include <sstream>
#include <string>
#include <vector>
#include <queue>

using namespace std;
class SplitBVH
{


public: 
BVHNode * root;
SplitBVHBuilder *builder;


SplitBVH(float * verts, int numPrimitives, int primitveType)
{
	
	BuildParams buildParams;
	buildParams.splitAlpha = 1e-5f;

	builder = new SplitBVHBuilder(verts,numPrimitives, buildParams, primitveType);
	root=builder->run();
	
}

void getFlatArray(int &size, int &leafSize, float *& inner, int *&leafs)
{
	return builder->bvhToFlatArray(root,size, leafSize, inner, leafs);
}
~SplitBVH()
{
	root->deleteSubtree();
  delete builder;

}



void bvhToObj(const char * filename,const int maxDepth)
{
  ostringstream os; //string building
  ofstream ofs;
  vector<string> lines;
  int nodeCount=0;
  ofs.open(filename);
  if(!ofs.is_open()) {cout<<"Couldn't open file "<<filename<<endl; exit(1);}
  int idx=1-8;
  ofs<<"g bvh"<<endl<<endl;
  ofs<<"mltlib mat.mtl"<<endl;
  queue<BVHNode*> tree;
  tree.push(root);
  long long int maxNodes=pow(2,maxDepth)-1; //root = level 1
  cout<<"Max Nodes "<<maxNodes;
  BVHNode * current;
   while(!tree.empty())
  {
    nodeCount++; //flag for do them all
    current=tree.front();
    tree.pop();
    idx+=8;

    ofs<<"v "<<current->m_bounds.min().x<<" "<<current->m_bounds.min().y<<" "<<current->m_bounds.min().z<<endl;  // A
    ofs<<"v "<<current->m_bounds.max().x<<" "<<current->m_bounds.min().y<<" "<<current->m_bounds.min().z<<endl;  // B
    ofs<<"v "<<current->m_bounds.min().x<<" "<<current->m_bounds.max().y<<" "<<current->m_bounds.min().z<<endl;  // C
    ofs<<"v "<<current->m_bounds.max().x<<" "<<current->m_bounds.max().y<<" "<<current->m_bounds.min().z<<endl;  // D

    ofs<<"v "<<current->m_bounds.min().x<<" "<<current->m_bounds.min().y<<" "<<current->m_bounds.max().z<<endl;  // E
    ofs<<"v "<<current->m_bounds.max().x<<" "<<current->m_bounds.min().y<<" "<<current->m_bounds.max().z<<endl;  // F
    ofs<<"v "<<current->m_bounds.min().x<<" "<<current->m_bounds.max().y<<" "<<current->m_bounds.max().z<<endl;  // G
    ofs<<"v "<<current->m_bounds.max().x<<" "<<current->m_bounds.max().y<<" "<<current->m_bounds.max().z<<endl;  // H
    
    if(current->isLeaf()) {os<<"usemtl red"<<endl;}
    else{os<<"usemtl blue"<<endl;}
    os<<"l "<<idx  <<" "<<idx+1<<endl;
    os<<"l "<<idx  <<" "<<idx+2<<endl;
    os<<"l "<<idx+2<<" "<<idx+3<<endl;
    os<<"l "<<idx+1<<" "<<idx+3<<endl;

    os<<"l "<<idx+1<<" "<<idx+5<<endl;
    os<<"l "<<idx+5<<" "<<idx+4<<endl;
    os<<"l "<<idx  <<" "<<idx+4<<endl;
    os<<"l "<<idx+5<<" "<<idx+7<<endl;

    os<<"l "<<idx+7<<" "<<idx+3<<endl;
    os<<"l "<<idx+7<<" "<<idx+6<<endl;
    os<<"l "<<idx+6<<" "<<idx+2<<endl;
    os<<"l "<<idx+4<<" "<<idx+6<<endl;
    lines.push_back(os.str());

    os.str("");
    os.clear();

    if(!current->isLeaf())
    {
        for(int i=0;i<current->getNumChildNodes();i++)
        tree.push(current->getChildNode(i));

    }
    
  }
  
  //now write all the lines segments to the file
  ofs<<endl;
  for(int i=0; i<lines.size();i++)
  {
    ofs<<lines.at(i);

  }
  //clean up
  ofs<<os.str();
  ofs.close();


}

};

#endif