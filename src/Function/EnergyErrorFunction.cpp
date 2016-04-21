//
//  EnergyErrorFunction.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/12/15.
//
//

#include "EnergyErrorFunction.h"

#include "BasisCache.h"
#include "Function.h"
#include "RieszRep.h"
#include "Solution.h"
#include "TypeDefs.h"

using namespace Camellia;
using namespace std;

EnergyErrorFunction::EnergyErrorFunction(SolutionPtr soln)
{
  _soln = soln;
}

EnergyErrorFunction::EnergyErrorFunction(RieszRepPtr rieszRep)
{
  _rieszRep = rieszRep;
}

void EnergyErrorFunction::values(Intrepid::FieldContainer<double> &values,
                                 BasisCachePtr basisCache)
{
  // values should have shape: (C,P)
  int numPoints = values.dimension(1);
  
  const map<GlobalIndexType,double>* cellError;
  bool takeSqrt;
  if (_soln != Teuchos::null)
  {
    cellError = &_soln->rankLocalEnergyError();
    takeSqrt = false;
  }
  else
  {
    takeSqrt = true;
    cellError = &_rieszRep->getNormsSquared();
  }
  
  int cellOrdinal = 0;
  for (GlobalIndexType cellID : basisCache->cellIDs())
  {
    if (cellError->find(cellID) == cellError->end())
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found");
    
    double error = cellError->find(cellID)->second;
    if (takeSqrt)
    {
      error = sqrt(error);
    }
    for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
    {
      values(cellOrdinal,ptOrdinal) = error;
    }
    
    cellOrdinal++;
  }
}

FunctionPtr EnergyErrorFunction::energyErrorFunction(SolutionPtr soln)
{
  return Teuchos::rcp( new EnergyErrorFunction(soln) );
}
