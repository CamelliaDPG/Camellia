#include "SimpleSolutionFunction.h"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "GlobalDofAssignment.h"
#include "Solution.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

SimpleSolutionFunction::SimpleSolutionFunction(VarPtr var, SolutionPtr soln) : Function(var->rank()) {
  _var = var;
  _soln = soln;
}

bool SimpleSolutionFunction::boundaryValueOnly() {
  return (_var->varType() == FLUX) || (_var->varType() == TRACE);
}

string SimpleSolutionFunction::displayString() {
  ostringstream str;
  str << "\\overline{" << _var->displayString() << "} ";
  return str.str();
}

void SimpleSolutionFunction::importCellData(std::vector<GlobalIndexType> cells) {
  int rank = Teuchos::GlobalMPISession::getRank();
  set<GlobalIndexType> offRankCells;
  const set<GlobalIndexType>* rankLocalCells = &_soln->mesh()->globalDofAssignment()->cellsInPartition(rank);
  for (int cellOrdinal=0; cellOrdinal < cells.size(); cellOrdinal++) {
    if (rankLocalCells->find(cells[cellOrdinal]) == rankLocalCells->end()) {
      offRankCells.insert(cells[cellOrdinal]);
    }
  }
  _soln->importSolutionForOffRankCells(offRankCells);
}

void SimpleSolutionFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  bool dontWeightForCubature = false;
  if (basisCache->mesh().get() != NULL) { // then we assume that the BasisCache is appropriate for solution's mesh...
    _soln->solutionValues(values, _var->ID(), basisCache, dontWeightForCubature, _var->op());
  } else {
    // the following adapted from PreviousSolutionFunction.  Probably would do well to consolidate
    // that class with this one at some point...
    LinearTermPtr solnExpression = 1.0 * _var;
    // get the physicalPoints, and make a basisCache for each...
    Intrepid::FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
    Intrepid::FieldContainer<double> value(1,1); // assumes scalar-valued solution function.
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    physicalPoints.resize(numCells*numPoints,spaceDim);
    vector< ElementPtr > elements = _soln->mesh()->elementsForPoints(physicalPoints, false); // false: don't make elements null just because they're off-rank.
    Intrepid::FieldContainer<double> point(1,1,spaceDim);
    Intrepid::FieldContainer<double> refPoint(1,spaceDim);
    int combinedIndex = 0;
    vector<GlobalIndexType> cellID;
    cellID.push_back(-1);
    BasisCachePtr basisCacheOnePoint;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++, combinedIndex++) {
        if (elements[combinedIndex].get()==NULL) continue; // no element found for point; skip it…
        ElementTypePtr elemType = elements[combinedIndex]->elementType();
        for (int d=0; d<spaceDim; d++) {
          point(0,0,d) = physicalPoints(combinedIndex,d);
        }
        if (elements[combinedIndex]->cellID() != cellID[0]) {
          cellID[0] = elements[combinedIndex]->cellID();
          basisCacheOnePoint = Teuchos::rcp( new BasisCache(elemType, _soln->mesh()) );
          basisCacheOnePoint->setPhysicalCellNodes(_soln->mesh()->physicalCellNodesForCell(cellID[0]),cellID,false); // false: don't createSideCacheToo
        }
        refPoint.resize(1,1,spaceDim); // CamelliaCellTools::mapToReferenceFrame wants a numCells dimension...  (perhaps it shouldn't, though!)
        // compute the refPoint:
        CamelliaCellTools::mapToReferenceFrame(refPoint,point,_soln->mesh()->getTopology(), cellID[0],
                                               _soln->mesh()->globalDofAssignment()->getCubatureDegree(cellID[0]));
        refPoint.resize(1,spaceDim);
        basisCacheOnePoint->setRefCellPoints(refPoint);
        //          cout << "refCellPoints:\n " << refPoint;
        //          cout << "physicalCubaturePoints:\n " << basisCacheOnePoint->getPhysicalCubaturePoints();
        solnExpression->evaluate(value, _soln, basisCacheOnePoint);
        //          cout << "value at point (" << point(0,0) << ", " << point(0,1) << ") = " << value(0,0) << endl;
        values(cellIndex,ptIndex) = value(0,0);
      }
    }
  }
  if (_var->varType()==FLUX) { // weight by sideParity
    sideParity()->scalarMultiplyFunctionValues(values, basisCache);
  }
}

FunctionPtr SimpleSolutionFunction::dx() {
  if (_var->op() != Camellia::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->dx(), _soln);
  }
}

FunctionPtr SimpleSolutionFunction::dy() {
  if (_var->op() != Camellia::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->dy(), _soln);
  }
}

FunctionPtr SimpleSolutionFunction::dz() {
  if (_var->op() != Camellia::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->dz(), _soln);
  }
}

FunctionPtr SimpleSolutionFunction::x() {
  if (_var->op() != Camellia::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->x(), _soln);
  }
}

FunctionPtr SimpleSolutionFunction::y() {
  if (_var->op() != Camellia::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->y(), _soln);
  }
}

FunctionPtr SimpleSolutionFunction::z() {
  if (_var->op() != Camellia::OP_VALUE) {
    return Function::null();
  } else {
    return Function::solution(_var->z(), _soln);
  }
}
