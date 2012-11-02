#ifndef MESHREFINEMENTTESTS_H
#define MESHREFINEMENTTESTS_H

#include "gtest/gtest.h"
#include "TestUtil.h"

#include "Mesh.h"
#include "Solution.h"
#include "ExactSolution.h"
#include "BilinearForm.h"
#include "TestBilinearFormFlux.h"
#include "BilinearFormUtility.h"

typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
typedef Teuchos::RCP<Element> ElementPtr;

class MeshRefinementTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    // in what follows:
    /*
     Mesh B is a 2x1 mesh (uniform)
     Mesh A is like B, but with the West element refined (one hanging node)
     Mesh C is like B, but with both elements refined (uniform)
     
     For easy reference, we number the elements as shown below (happens to line up with
     cellIDs at the moment).
     
     A:
     ---------------------------------
     |       |       |               |
     |   5   |   4   |               |
     |       |       |               |
     |-------0-------|       1       |
     |       |       |               |
     |   2   |   3   |               |
     |       |       |               |
     ---------------------------------

     B:
     ---------------------------------
     |               |               |
     |               |               |
     |               |               |
     |       0       |       1       |
     |               |               |
     |               |               |
     |               |               |
     ---------------------------------

     C:
     ---------------------------------
     |       |       |       |       |
     |   5   |   4   |   9   |   8   |
     |       |       |       |       |
     |-------0-------|-------1-------|
     |       |       |       |       |
     |   2   |   3   |   6   |   7   |
     |       |       |       |       |
     ---------------------------------
     */
    
    // MultiBasis meshes:
    Teuchos::RCP<Mesh> _multiA, _multiB, _multiC;
    ElementPtr _A1multi, _A3multi, _A4multi, _A5multi;
    ElementPtr _B1multi;
    ElementPtr _C4multi, _C5multi;  
    
    // PatchBasis meshes:
    Teuchos::RCP<Mesh> _patchA, _patchB, _patchC;
    ElementPtr _A1patch, _A3patch, _A4patch, _A5patch;
    ElementPtr _B1patch;
    ElementPtr _C4patch, _C5patch;
    
    double _h, _h_small;
    
    Teuchos::RCP< TestBilinearFormFlux > _fluxBilinearForm;
    
    void checkMultiElementStiffness(Teuchos::RCP<Mesh> mesh);
    void checkMultiElementStiffness(Teuchos::RCP<Mesh> mesh, int cellID);
    
    void checkPatchElementStiffness(Teuchos::RCP<Mesh> mesh);
    void checkPatchElementStiffness(Teuchos::RCP<Mesh> mesh, int cellID);
    
    void multiBrokenSides(set<int> &brokenSideSet, ElementPtr elem);
    void patchParentSideIndices(map<int,int> &parentSideIndices, Teuchos::RCP<Mesh> mesh, ElementPtr elem);
    
    void preStiffnessExpectedUniform(FieldContainer<double> &preStiff, double h, ElementTypePtr elemType,
                                     FieldContainer<double> &sideParities);
    void preStiffnessExpectedPatch(FieldContainer<double> &preStiff, double h, 
                                   const map<int,int> &sidesWithBiggerNeighborToIndexInParentSide,
                                   ElementTypePtr elemType, FieldContainer<double> &sideParities);
    void preStiffnessExpectedMulti(FieldContainer<double> &preStiff, double h, const set<int> &brokenSides, ElementTypePtr elemType,
                                   FieldContainer<double> &sideParities);
};

#endif /* end of include guard: MESHREFINEMENTTESTS_H */
