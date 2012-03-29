#ifndef DPG_BC
#define DPG_BC

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 

/*
 *  BC.h
 *
 */

// abstract class

#include "Intrepid_FieldContainer.hpp"
#include "BasisCache.h"

using namespace Intrepid;

class BC {
public:
  virtual bool bcsImposed(int varID) = 0; // returns true if there are any BCs anywhere imposed on varID
  virtual void imposeBC(FieldContainer<double> &dirichletValues, FieldContainer<bool> &imposeHere, 
                        int varID, FieldContainer<double> &unitNormals, BasisCachePtr basisCache) {
    // by default, call legacy version:
    // (basisCache->getPhysicalCubaturePoints() doesn't really return *cubature* points, but the boundary points
    //  that we're interested in)
    FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
    imposeBC(varID,physicalPoints,unitNormals,dirichletValues,imposeHere);
  }
  
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
                        FieldContainer<bool> &imposeHere) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "BC::imposeBC unimplemented.");
  }
  
//  virtual void imposeBC(int varID, Teuchos::RCP<BasisCache> sideBasisCache, 
//                        FieldContainer<double> &unitNormals,
//                        FieldContainer<double> &dirichletValues,
//                        FieldContainer<bool> &imposeHere) {
//    imposeBC(varID,sideBasisCache->getPhysicalCubaturePoints(),unitNormals,dirichletValues,imposeHere);
//  }
  
  virtual bool singlePointBC(int varID) {
    return false; 
  } 
  
  virtual bool imposeZeroMeanConstraint(int varID) {
    return false;
  }
  // override if you want to implement a BC at a single, arbitrary point (and nowhere else).
};

#endif