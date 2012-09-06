#include "InnerProductScratchPad.h"

double pi = 2.0*acos(0.0);

Teuchos::RCP<Mesh> BuildHemkerMesh(BFPtr confusionBF, int nseg, bool CircleMesh, bool TriangulateMesh, int H1Order, int pToAdd)
{
    // Generate Mesh
    vector< FieldContainer<double> > vertices;
    FieldContainer<double> pt(2);
    vector< vector<int> > elementIndices;
    vector<int> el(4);
    vector<int> el1(3);
    vector<int> el2(3);
    // Inner Square
    double S;
    if (CircleMesh)
      S = 1.5;
    else
      S = 1 + 1./nseg;
    double angle;
    // Bottom edge
    for (int i=0; i < nseg; i++)
    {
      if (CircleMesh)
      {
        angle = -3.*pi/4. + pi/2.*double(i)/nseg;
        pt(0) = S*cos(angle);
        pt(1) = S*sin(angle);
      }
      else
      {
        pt(0) = -S + double(i)/nseg*2*S;
        pt(1) = -S;
      }
      vertices.push_back(pt);
      el[0] = i; 
      el[1] = i + 1;
      el[2] = 4*nseg + i + 1;
      el[3] = 4*nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // Right edge
    for (int i=0; i < nseg; i++)
    {
      if (CircleMesh)
      {
        angle = -pi/4. + pi/2.*double(i)/nseg;
        pt(0) = S*cos(angle);
        pt(1) = S*sin(angle);
      }
      else
      {
        pt(0) = S;
        pt(1) = -S + double(i)/nseg*2*S;
      }
      vertices.push_back(pt);
      el[0] = nseg + i; 
      el[1] = nseg + i + 1;
      el[2] = 5*nseg + i + 1;
      el[3] = 5*nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // Top edge
    for (int i=0; i < nseg; i++)
    {
      if (CircleMesh)
      {
        angle = pi/4. + pi/2.*double(i)/nseg;
        pt(0) = S*cos(angle);
        pt(1) = S*sin(angle);
      }
      else
      {
        pt(0) = S - double(i)/nseg*2*S;
        pt(1) = S;
      }
      vertices.push_back(pt);
      el[0] = 2*nseg + i; 
      el[1] = 2*nseg + i + 1;
      el[2] = 6*nseg + i + 1;
      el[3] = 6*nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // Left edge
    for (int i=0; i < nseg; i++)
    {
      if (CircleMesh)
      {
        angle = 3.*pi/4. + pi/2.*double(i)/nseg;
        pt(0) = S*cos(angle);
        pt(1) = S*sin(angle);
      }
      else
      {
        pt(0) = -S;
        pt(1) = S - double(i)/nseg*2*S;
      }
      vertices.push_back(pt);
      el[0] = 3*nseg + i; 
      el[1] = 3*nseg + i + 1;
      el[2] = 7*nseg + i + 1;
      el[3] = 7*nseg + i;
      if (i == nseg-1)
      {
        el[1] = 0;
        el[2] = 4*nseg;
      }
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // if (TriangulateMesh)
    // {
    //   elementIndices[elementIndices.size()-2][1] = 0;
    //   elementIndices[elementIndices.size()-2][2] = 4*nseg-1;
    //   cout << "Left Element 1: " 
    //     << elementIndices[elementIndices.size()-2][0]
    //     << elementIndices[elementIndices.size()-2][1]
    //     << elementIndices[elementIndices.size()-2][2] << endl;
    //   cout << "Left Element 2: " 
    //     << elementIndices[elementIndices.size()-1][0]
    //     << elementIndices[elementIndices.size()-1][1]
    //     << elementIndices[elementIndices.size()-1][2] << endl;
    // }
    // else
    // {
    //   elementIndices.back()[1] = 0;
    //   elementIndices.back()[2] = 4*nseg;
    // }
    // elementIndices[4*nseg-1][1] = 0;
    // elementIndices[4*nseg-1][2] = 4*nseg;
    // Circle
    for (int i=0; i < 4*nseg; i++)
    {
      angle = 5./4.*pi + 2.*pi*i/(4*nseg);
      pt(0) = cos(angle);
      pt(1) = sin(angle);
      vertices.push_back(pt);
    }
    // Outer Rectangle
    int N = vertices.size();
    // Below square
    for (int i=0; i < nseg; i++)
    {
      pt(0) = -S + double(i)/nseg*2*S;
      pt(1) = -3.0;
      vertices.push_back(pt);
      el[0] = N + i;
      el[1] = N + i + 1;
      el[2] = i + 1;
      el[3] = i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    pt(0) = S;
    pt(1) = -3.0;
    vertices.push_back(pt);
    // Right of square
    for (int i=0; i < nseg; i++)
    {
      pt(0) = 9.0;
      pt(1) = -S + double(i)/nseg*2*S;
      vertices.push_back(pt);
      el[0] = N + nseg+1 + i;
      el[1] = N + nseg+1 + i + 1;
      el[2] = nseg + i + 1;
      el[3] = nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    pt(0) = 9.0;
    pt(1) = S;
    vertices.push_back(pt);
    // Above square
    for (int i=0; i < nseg; i++)
    {
      pt(0) = S - double(i)/nseg*2*S;
      pt(1) = 3.0;
      vertices.push_back(pt);
      el[0] = N + 2*(nseg+1) + i;
      el[1] = N + 2*(nseg+1) + i + 1;
      el[2] = 2*nseg + i + 1;
      el[3] = 2*nseg + i;
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    pt(0) = -S;
    pt(1) = 3.0;
    vertices.push_back(pt);
    // Left of square
    for (int i=0; i < nseg; i++)
    {
      pt(0) = -3.0;
      pt(1) = S - double(i)/nseg*2*S;
      vertices.push_back(pt);
      el[0] = N + 3*(nseg+1) + i;
      el[1] = N + 3*(nseg+1) + i + 1;
      el[2] = 3*nseg + i + 1;
      el[3] = 3*nseg + i;
      if (i == nseg-1)
      {
        el[2] = 0;
      }
      if (TriangulateMesh)
      {
        el1[0] = el[0];
        el1[1] = el[1];
        el1[2] = el[2];
        el2[0] = el[0];
        el2[1] = el[2];
        el2[2] = el[3];
        elementIndices.push_back(el1);
        elementIndices.push_back(el2);
      }
      else
        elementIndices.push_back(el);
    }
    // if (TriangulateMesh)
    // {
    //   elementIndices[elementIndices.size()-2][2] = 0;
    //   elementIndices[elementIndices.size()-1][1] = 0;
    // }
    // else
    //   elementIndices.back()[2] = 0;
    pt(0) = -3.0;
    pt(1) = -S;
    vertices.push_back(pt);
    // Bottom left corner
    pt(0) = -3.0;
    pt(1) = -3.0;
    vertices.push_back(pt);
    el[0] = N + 4*(nseg+1);
    el[1] = N;
    el[2] = 0;
    el[3] = N + 4*(nseg+1) - 1;
    if (TriangulateMesh)
    {
      el1[0] = el[0];
      el1[1] = el[1];
      el1[2] = el[2];
      el2[0] = el[0];
      el2[1] = el[2];
      el2[2] = el[3];
      elementIndices.push_back(el1);
      elementIndices.push_back(el2);
    }
    else
      elementIndices.push_back(el);
    // Bottom right corner
    pt(0) = 9.0;
    pt(1) = -3.0;
    vertices.push_back(pt);
    el[0] = N + nseg;
    el[1] = N + 4*(nseg+1) + 1;
    el[2] = N + nseg+1;
    el[3] = nseg;
    if (TriangulateMesh)
    {
      el1[0] = el[0];
      el1[1] = el[1];
      el1[2] = el[2];
      el2[0] = el[0];
      el2[1] = el[2];
      el2[2] = el[3];
      elementIndices.push_back(el1);
      elementIndices.push_back(el2);
    }
    else
      elementIndices.push_back(el);
    // Top right corner
    pt(0) = 9.0;
    pt(1) = 3.0;
    vertices.push_back(pt);
    el[0] = 2*nseg;
    el[1] = N + 2*(nseg+1)-1;
    el[2] = N + 4*(nseg+1) + 2;
    el[3] = N + 2*(nseg+1);
    if (TriangulateMesh)
    {
      el1[0] = el[0];
      el1[1] = el[1];
      el1[2] = el[2];
      el2[0] = el[0];
      el2[1] = el[2];
      el2[2] = el[3];
      elementIndices.push_back(el1);
      elementIndices.push_back(el2);
    }
    else
      elementIndices.push_back(el);
    // Top Left corner
    pt(0) = -3.0;
    pt(1) = 3.0;
    vertices.push_back(pt);
    el[0] = N + 3*(nseg+1);
    el[1] = 3*nseg;
    el[2] = N + 3*(nseg+1)-1;
    el[3] = N + 4*(nseg+1) + 3;
    if (TriangulateMesh)
    {
      el1[0] = el[0];
      el1[1] = el[1];
      el1[2] = el[2];
      el2[0] = el[0];
      el2[1] = el[2];
      el2[2] = el[3];
      elementIndices.push_back(el1);
      elementIndices.push_back(el2);
    }
    else
      elementIndices.push_back(el);

    Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, confusionBF, H1Order, pToAdd) );  
    return mesh;
}
