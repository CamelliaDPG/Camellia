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
#include "Intrepid_FieldContainer.hpp"
#include "Teuchos_FancyOStream.hpp"

namespace Camellia
{
  template<typename data_type>
  void print(std::string name, std::set<data_type> &data);
  
  template<typename key_type, typename value_type>
  void print(std::string name, std::map<key_type, value_type> &data);
  
  void print(std::string name, std::map<unsigned, unsigned> data);
  void print(std::string name, std::map<unsigned, double> data);
  void print(std::string name, std::map<int, double> data);
  
  void print(std::string name, std::vector<int> data);
  void print(std::string name, std::vector<unsigned> data);
  void print(std::string name, std::vector<long long> data);
  void print(std::string name, std::vector<double> data);
  
  void print(std::string name, std::set<unsigned> data);
  void print(std::string name, std::set<int> data);
  void print(std::string name, std::set<long long> data);
  void print(std::string name, std::set<double> data);
  
  // ! prints out the coefficients for each variable, labelled using the names from the VarFactory.
  // ! If the trialSpaceDofs boolean is set to true, interprets the variable IDs as trial space variables;
  // ! otherwise, interprets them as test space variables.
  void printLabeledDofCoefficients(VarFactoryPtr vf, DofOrderingPtr dofOrdering,
                                   const Intrepid::FieldContainer<double> &dofCoefficients,
                                   bool trialSpaceDofs = true);
  
  // ! prints out the coefficients for each variable, labelled using the names from the VarFactory.
  // ! If the trialSpaceDofs boolean is set to true, interprets the variable IDs as trial space variables;
  // ! otherwise, interprets them as test space variables.
  void printLabeledDofCoefficients(std::ostream &out, VarFactoryPtr vf, DofOrderingPtr dofOrdering,
                                   const Intrepid::FieldContainer<double> &dofCoefficients,
                                   bool trialSpaceDofs = true);
  
  // ! Prints out a summary of the rank-local map information.  Not an MPI collective operation.
  void printMapSummary(const Epetra_Map &map, std::string mapName);
}

#endif
