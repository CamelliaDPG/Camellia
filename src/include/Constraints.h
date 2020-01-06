#ifndef DPG_CONSTRAINTS
#define DPG_CONSTRAINTS

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

/*
 *  Constraints.h
 *
 */

// abstract class

#include "Camellia_FieldContainer.hpp"

using namespace std;

namespace Camellia
{
class Constraints
{
public:
  //given trialID, field container for constraint matrix gives one coeff per point
  virtual void getConstraints(Camellia::FieldContainer<double> &physicalPoints,
                              Camellia::FieldContainer<double> &unitNormals,
                              vector<map<int,Camellia::FieldContainer<double > > > &constraintCoeffs,
                              vector<Camellia::FieldContainer<double > > &constraintValues) {};
  // TODO - figure out some way to skip over edges, points, or variables we don't use
};
}

#endif
