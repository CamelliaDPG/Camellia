// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef CAMELLIA_DEBUG_UTILITY
#define CAMELLIA_DEBUG_UTILITY

#include "TypeDefs.h"

#include <vector>
#include <set>
#include <map>
#include <ostream>
#include <string>

#include "Epetra_Map.h"
#include "Teuchos_RCP.hpp"
#include "Camellia_FieldContainer.hpp"
#include "Teuchos_FancyOStream.hpp"

namespace Camellia
{
  void print(const std::string &name, const std::map<unsigned, unsigned> &data);
  void print(const std::string &name, const std::map<unsigned, double> &data);
  void print(const std::string &name, const std::map<int, double> &data);
  
  void print(const std::string &name, const std::vector<int> &data);
  void print(const std::string &name, const std::vector<unsigned> &data);
  void print(const std::string &name, const std::vector<long long> &data);
  void print(const std::string &name, const std::vector<double> &data);
  
  void print(const std::string &name, const std::set<unsigned> &data);
  void print(const std::string &name, const std::set<int> &data);
  void print(const std::string &name, const std::set<long long> &data);
  void print(const std::string &name, const std::set<double> &data);
  
  void print(std::ostream &out, const std::string &name, const std::map<unsigned, unsigned> &data);
  void print(std::ostream &out, const std::string &name, const std::map<unsigned, double> &data);
  void print(std::ostream &out, const std::string &name, const std::map<int, double> &data);
  
  void print(std::ostream &out, const std::string &name, const std::vector<int> &data);
  void print(std::ostream &out, const std::string &name, const std::vector<unsigned> &data);
  void print(std::ostream &out, const std::string &name, const std::vector<long long> &data);
  void print(std::ostream &out, const std::string &name, const std::vector<double> &data);
  
  void print(std::ostream &out, const std::string &name, const std::set<unsigned> &data);
  void print(std::ostream &out, const std::string &name, const std::set<int> &data);
  void print(std::ostream &out, const std::string &name, const std::set<long long> &data);
  void print(std::ostream &out, const std::string &name, const std::set<double> &data);
  
  template<class Rank1Functor>
  void print1(std::ostream &out, const std::string &name, const Rank1Functor &functor)
  {
    int dim0 = functor.extent_int(0);
    out << name << ":\n";
    for (int i0=0; i0<dim0; i0++)
    {
      out << "(" << i0 << "): " << functor(i0) << std::endl;
    }
  }

  template<class Rank2Functor>
  void print2(std::ostream &out, const std::string &name, const Rank2Functor &functor)
  {
    const int dim0 = functor.extent_int(0);
    const int dim1 = functor.extent_int(1);
    out << name << ":\n";
    for (int i0=0; i0<dim0; i0++)
    {
      for (int i1=0; i1<dim1; i1++)
      {
        out << "(" << i0 << "," << i1 << "): " << functor(i0,i1) << std::endl;
      }
    }
  }

  template<class Rank3Functor>
  void print3(std::ostream &out, const std::string &name, const Rank3Functor &functor)
  {
    const int dim0 = functor.extent_int(0);
    const int dim1 = functor.extent_int(1);
    const int dim2 = functor.extent_int(2);
    out << name << ":\n";
    for (int i0=0; i0<dim0; i0++)
    {
      for (int i1=0; i1<dim1; i1++)
      {
        for (int i2=0; i2<dim2; i2++)
        {
          out << "(" << i0 << "," << i1 << "," << i2 << "): " << functor(i0,i1,i2) << std::endl;
        }
      }
    }
  }

  // ! prints out the coefficients for each variable, labelled using the names from the VarFactory.
  // ! If the trialSpaceDofs boolean is set to true, interprets the variable IDs as trial space variables;
  // ! otherwise, interprets them as test space variables.
  void printLabeledDofCoefficients(VarFactoryPtr vf, DofOrderingPtr dofOrdering,
                                   const Camellia::FieldContainer<double> &dofCoefficients,
                                   bool trialSpaceDofs = true);
  
  // ! prints out the coefficients for each variable, labelled using the names from the VarFactory.
  // ! If the trialSpaceDofs boolean is set to true, interprets the variable IDs as trial space variables;
  // ! otherwise, interprets them as test space variables.
  void printLabeledDofCoefficients(std::ostream &out, VarFactoryPtr vf, DofOrderingPtr dofOrdering,
                                   const Camellia::FieldContainer<double> &dofCoefficients,
                                   bool trialSpaceDofs = true);
  
  // ! Prints out a summary of the rank-local map information.  Not an MPI collective operation.
  void printMapSummary(const Epetra_BlockMap &map, std::string mapName);
}

#endif
