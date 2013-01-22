//
//  PreviousSolutionFunction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/18/13.
//
//

#include "PreviousSolutionFunction.h"

#include "Function.h"
#include "Element.h"
#include "Solution.h"
#include "InnerProductScratchPad.h"

PreviousSolutionFunction::PreviousSolutionFunction(SolutionPtr soln, LinearTermPtr solnExpression, bool multiplyFluxesByCellParity) : Function(solnExpression->rank()) {
  _soln = soln;
  _solnExpression = solnExpression;
  _overrideMeshCheck = false;
  if ((solnExpression->termType() == FLUX) && multiplyFluxesByCellParity) {
    FunctionPtr parity = Teuchos::rcp( new SideParityFunction );
    _solnExpression = parity * solnExpression;
  }
}
PreviousSolutionFunction::PreviousSolutionFunction(SolutionPtr soln, VarPtr var, bool multiplyFluxesByCellParity) : Function(var->rank()) {
  _soln = soln;
  _solnExpression = 1.0 * var;
  _overrideMeshCheck = false;
  if ((var->varType() == FLUX) && multiplyFluxesByCellParity) {
    FunctionPtr parity = Teuchos::rcp( new SideParityFunction );
    _solnExpression = parity * var;
  }
}
bool PreviousSolutionFunction::boundaryValueOnly() { // fluxes and traces are only defined on element boundaries
  return (_solnExpression->termType() == FLUX) || (_solnExpression->termType() == TRACE);
}
void PreviousSolutionFunction::setOverrideMeshCheck(bool value) {
  _overrideMeshCheck = value;
}
void PreviousSolutionFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  if (_overrideMeshCheck) {
    _solnExpression->evaluate(values, _soln, basisCache);
    return;
  }
  if (!basisCache.get()) cout << "basisCache is nil!\n";
  if (!_soln.get()) cout << "_soln is nil!\n";
  // TODO: get the mesh-checking thing working, along with an override that lets you
  //       say these two meshes are the same...
  // values are stored in (C,P,D) order
  if (basisCache->mesh().get() == _soln->mesh().get()) {
    _solnExpression->evaluate(values, _soln, basisCache);
  } else {
    static bool warningIssued = false;
    if (!warningIssued) {
      cout << "NOTE: In PreviousSolutionFunction, basisCache's mesh doesn't match solution's.  If this is not what you intended, would be a good idea to make sure that the mesh is passed in on BasisCache construction; the evaluation will be a lot slower without it...\n";
      warningIssued = true;
    }
    // get the physicalPoints, and make a basisCache for each...
    FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
    FieldContainer<double> value(1,1); // assumes scalar-valued solution function.
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    physicalPoints.resize(numCells*numPoints,spaceDim);
    vector< ElementPtr > elements = _soln->mesh()->elementsForPoints(physicalPoints);
    FieldContainer<double> point(1,spaceDim);
    FieldContainer<double> refPoint(1,spaceDim);
    int combinedIndex = 0;
    vector<int> cellID;
    cellID.push_back(-1);
    BasisCachePtr basisCacheOnePoint;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++, combinedIndex++) {
        ElementTypePtr elemType = elements[combinedIndex]->elementType();
        for (int d=0; d<spaceDim; d++) {
          point(0,d) = physicalPoints(combinedIndex,d);
        }
        if (elements[combinedIndex]->cellID() != cellID[0]) {
          cellID[0] = elements[combinedIndex]->cellID();
          basisCacheOnePoint = Teuchos::rcp( new BasisCache(elemType, _soln->mesh()) );
          basisCacheOnePoint->setPhysicalCellNodes(_soln->mesh()->physicalCellNodesForCell(cellID[0]),cellID,false); // false: don't createSideCacheToo
        }
        // compute the refPoint:
        typedef CellTools<double>  CellTools;
        int whichCell = 0;
        CellTools::mapToReferenceFrame(refPoint,point,_soln->mesh()->physicalCellNodesForCell(cellID[0]),
                                       *(elemType->cellTopoPtr),whichCell);
        basisCacheOnePoint->setRefCellPoints(refPoint);
        //          cout << "refCellPoints:\n " << refPoint;
        //          cout << "physicalCubaturePoints:\n " << basisCacheOnePoint->getPhysicalCubaturePoints();
        _solnExpression->evaluate(value, _soln, basisCacheOnePoint);
        //          cout << "value at point (" << point(0,0) << ", " << point(0,1) << ") = " << value(0,0) << endl;
        values(cellIndex,ptIndex) = value(0,0);
      }
    }
  }
}
map<int, FunctionPtr > PreviousSolutionFunction::functionMap( vector< VarPtr > varPtrs, SolutionPtr soln) {
  map<int, FunctionPtr > functionMap;
  for (vector< VarPtr >::iterator varIt = varPtrs.begin(); varIt != varPtrs.end(); varIt++) {
    VarPtr var = *varIt;
    functionMap[var->ID()] = Teuchos::rcp( new PreviousSolutionFunction(soln, var));
  }
  return functionMap;
}
string PreviousSolutionFunction::displayString() {
  ostringstream str;
  str << "\\overline{" << _solnExpression->displayString() << "} ";
  return str.str();
}
