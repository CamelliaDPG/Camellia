//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CamelliaDebugUtility.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 3/18/14.
//
//

#include "CamelliaDebugUtility.h"

#include "DofOrdering.h"
#include "MPIWrapper.h"
#include "VarFactory.h"

#include <iostream>

namespace Camellia
{
  template<typename container_type>
  void print(ostream &out, const std::string &name, const container_type &data)
  {
    out << name << ": ";
    for (const auto & entry : data)
    {
      out << entry << " ";
    }
    out << endl;
  }
  
  template<typename key_type, typename value_type>
  void print(ostream &out, const std::string &name, const map<key_type, value_type> &data)
  {
    out << name << ": ";
    for (const auto & entry : data)
    {
      out << "(" << entry.first << " => " << entry.second << ") ";
    }
    out << endl;
  }

  template<typename data_type>
  void print(const std::string &name, const std::set<data_type> &data);

  template<typename key_type, typename value_type>
  void print(const std::string &name, const std::map<key_type, value_type> &data);
  
  template<typename data_type>
  void print(ostream &out, const std::string &name, const FieldContainer<data_type> &data)
  {
    out << name << ": ";
    int numEntries = data.size();
    for (int i=0; i<numEntries; i++)
    {
      out << data[i] << " ";
    }
    out << std::endl;
  }

  template<typename data_type>
  void print(const std::string &name, const FieldContainer<data_type> &data)
  {
    print<data_type>(std::cout, name, data);
  }

  void print(std::ostream &out, const std::string &name, const FieldContainer<double> &data)
  {
    print<double>(out, name, data);
  }

  void print(const std::string &name, const FieldContainer<double> &data)
  {
    print<double>(name, data);
  }

//  template<typename data_type>
//  void print(const std::string &name, const vector<data_type> &data)
//  {
//    cout << name << ": ";
//    for (int i=0; i<data.size(); i++)
//    {
//      cout << data[i] << " ";
//    }
//    cout << endl;
//  }
//  
//  template<typename data_type>
//  void print(const std::string &name, const set<data_type> &data)
//  {
//    cout << name << ": ";
//    for (typename set<data_type>::iterator dataIt=data.begin(); dataIt != data.end(); dataIt++)
//    {
//      cout << *dataIt << " ";
//    }
//    cout << endl;
//  }
  
  template<typename container_type>
  void print(const std::string &name, const container_type &data)
  {
    print<container_type>(cout,name,data);
  }
  
  template<typename key_type, typename value_type>
  void print(const std::string &name, const map<key_type, value_type> &data)
  {
    print<key_type,value_type>(cout,name,data);
  }
  
  template void print<int,double>(const std::string &name, const std::map<int, double> &data);
  template void print<int,int>(const std::string &name, const std::map<int, int> &data);
  template void print<string,double>(const std::string &name, const std::map<string, double> &data);
  
  void print(const std::string &name, const std::map<int, double> &data)
  {
    print<int, double>(name, data);
  }

  void print(const std::string &name, const std::map<int, int> &data)
  {
    print<int, int>(name, data);
  }
  
  void print(const std::string &name, const std::map<unsigned, double> &data)
  {
    print<unsigned, double>(name, data);
  }
  
  void print(const std::string &name, const std::map<unsigned, unsigned> &data)
  {
    print<unsigned, unsigned>(name, data);
  }
  
  void print(const std::string &name, const vector<long long> &data)
  {
    print<vector<long long>>(name, data);
  }
  
  void print(const std::string &name, const vector<int> &data)
  {
    print<vector<int>>(name, data);
  }
  void print(const std::string &name, const vector<unsigned> &data)
  {
    print<vector<unsigned>>(name, data);
  }
  void print(const std::string &name, const vector<double> &data)
  {
    print<vector<double>>(name,data);
  }
  void print(const std::string &name, const set<unsigned> &data)
  {
    print<set<unsigned>>(name, data);
  }
  void print(const std::string &name, const set<int> &data)
  {
    print<set<int>>(name, data);
  }
  void print(const std::string &name, const set<long long> &data)
  {
    print<set<long long>>(name, data);
  }
  void print(const std::string &name, const set<double> &data)
  {
    print<set<double>>(name, data);
  }
  
  void print(ostream &out, const std::string &name, const map<int, double> &data)
  {
    print<int, double>(out, name, data);
  }

  void print(ostream &out, const std::string &name, const map<int, int> &data)
  {
    print<int, int>(out, name, data);
  }
  
  void print(ostream &out, const std::string &name, const map<unsigned, double> &data)
  {
    print<unsigned, double>(out, name, data);
  }
  
  void print(ostream &out, const std::string &name, const map<unsigned, unsigned> &data)
  {
    print<unsigned, unsigned>(out, name, data);
  }
  
  void print(ostream &out, const std::string &name, const vector<long long> &data)
  {
    print<vector<long long>>(out, name, data);
  }
  
  void print(ostream &out, const std::string &name, const vector<int> &data)
  {
    print<vector<int>>(out, name, data);
  }
  void print(ostream &out, const std::string &name, const vector<unsigned> &data)
  {
    print<vector<unsigned>>(out, name, data);
  }
  void print(ostream &out, const std::string &name, const vector<double> &data)
  {
    print<vector<double>>(out, name, data);
  }
  void print(ostream &out, const std::string &name, const set<unsigned> &data)
  {
    print<set<unsigned>>(out, name, data);
  }
  void print(ostream &out, const std::string &name, const set<int> &data)
  {
    print<set<int>>(out, name, data);
  }
  void print(ostream &out, const std::string &name, const set<long long> &data)
  {
    print<set<long long>>(out, name, data);
  }

  void print(ostream &out, const std::string &name, const set<double> &data)
  {
    print<set<double>>(out, name, data);
  }

  void print2DArray(std::ostream &out, std::string &label, const Camellia::FieldContainer<double> &array, int precision, double nonzeroThreshold)
  {
    Teuchos::oblackholestream oldFormatState;
    oldFormatState.copyfmt(out);
    
    out << label << ": ";
    
    out << setprecision(precision);
    
    out.setf(std::ios_base::scientific, std::ios_base::floatfield);
    
    for (int i=0; i<array.dimension(0); i++)
    {
      out << "[";
      for (int j=0; j<array.dimension(1); j++)
      {
        const double value = array(i,j);
        if (std::abs(value) < nonzeroThreshold)
          out << 0 << " ";
        else
          out << value << " ";
      }
      out << "]";
    }
    out << std::endl;
    
    out.copyfmt(oldFormatState);
  }
  
  void printLabeledDofCoefficients(VarFactoryPtr vf, DofOrderingPtr dofOrdering,
                                   const Camellia::FieldContainer<double> &dofCoefficients,
                                   bool trialSpaceDofs, double nonzeroThreshold)
  {
    printLabeledDofCoefficients(std::cout, vf, dofOrdering, dofCoefficients, trialSpaceDofs);
  }
  
  void printLabeledDofCoefficients(ostream &out, VarFactoryPtr vf, DofOrderingPtr dofOrdering,
                                   const Camellia::FieldContainer<double> &dofCoefficients,
                                   bool trialSpaceDofs, double nonzeroThreshold)
  {
    if (dofOrdering->totalDofs() != dofCoefficients.size())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"dofOrdering size does not match dofCoefficients container size");
    }
    
    if (!trialSpaceDofs)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"printLabeledCoefficients does not yet support test variables");
    }
    
    Teuchos::oblackholestream oldFormatState;
    oldFormatState.copyfmt(out);
    
    int myprec = out.precision();
    out.setf(std::ios_base::scientific, std::ios_base::floatfield);
    out.setf(std::ios_base::right);
    
    vector< VarPtr > fieldVars = vf->fieldVars();
    vector< VarPtr > traceVars = vf->traceVars();
    vector< VarPtr > fluxVars = vf->fluxVars();
    
    bool printNonZerosOnly = true;
    if (printNonZerosOnly) out << "*** (printing only nonzeros) *** \n";
    
    if (fieldVars.size() > 0)
    {
      out << "\n\n **************   FIELD coefficients   **************\n";
      for (vector<VarPtr>::iterator varIt = fieldVars.begin(); varIt != fieldVars.end(); varIt++)
      {
        VarPtr var = *varIt;
        out << var->displayString() << ":\n";
        const std::vector<int>* dofIndices = &dofOrdering->getDofIndices(var->ID());
        int basisDofOrdinal = 0;
        for (std::vector<int>::const_iterator dofIndexIt = dofIndices->begin(); dofIndexIt != dofIndices->end(); dofIndexIt++, basisDofOrdinal++)
        {
          if (printNonZerosOnly && (std::abs(dofCoefficients[*dofIndexIt])>nonzeroThreshold)) continue;
          out.setf(std::ios::right, std::ios::adjustfield);
          out << std::setw(3) << basisDofOrdinal;
          out << "             ";
          out.setf(std::ios::right, std::ios::adjustfield);
          out << std::setw(myprec+8) << dofCoefficients[*dofIndexIt] << "\n";
        }
      }
    }
    
    if (traceVars.size() > 0)
    {
      out << "\n\n ***************   TRACE coefficients   ***************\n";
      for (vector<VarPtr>::iterator varIt = traceVars.begin(); varIt != traceVars.end(); varIt++)
      {
        VarPtr var = *varIt;
        const vector<int>* sides = &dofOrdering->getSidesForVarID(var->ID());
        for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
        {
          int sideOrdinal = *sideIt;
          out << var->displayString() << ", side " << sideOrdinal << ":\n";
          const std::vector<int>* dofIndices = &dofOrdering->getDofIndices(var->ID(), sideOrdinal);
          int basisDofOrdinal = 0;
          for (std::vector<int>::const_iterator dofIndexIt = dofIndices->begin(); dofIndexIt != dofIndices->end(); dofIndexIt++, basisDofOrdinal++)
          {
            if (printNonZerosOnly && (std::abs(dofCoefficients[*dofIndexIt])>nonzeroThreshold)) continue;
            out.setf(std::ios::right, std::ios::adjustfield);
            out << std::setw(3) << basisDofOrdinal;
            out << "             ";
            out.setf(std::ios::right, std::ios::adjustfield);
            out << std::setw(myprec+8) << dofCoefficients[*dofIndexIt] << "\n";
          }
        }
      }
    }
    
    if (fluxVars.size() > 0)
    {
      out << "\n\n ***************   FLUX coefficients   ***************\n";
      for (vector<VarPtr>::iterator varIt = fluxVars.begin(); varIt != fluxVars.end(); varIt++)
      {
        VarPtr var = *varIt;
        const vector<int>* sides = &dofOrdering->getSidesForVarID(var->ID());
        for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
        {
          int sideOrdinal = *sideIt;
          out << var->displayString() << ", side " << sideOrdinal << ":\n";
          const std::vector<int>* dofIndices = &dofOrdering->getDofIndices(var->ID(), sideOrdinal);
          int basisDofOrdinal = 0;
          for (std::vector<int>::const_iterator dofIndexIt = dofIndices->begin(); dofIndexIt != dofIndices->end(); dofIndexIt++, basisDofOrdinal++)
          {
            if (printNonZerosOnly && (std::abs(dofCoefficients[*dofIndexIt])>nonzeroThreshold)) continue;
            out.setf(std::ios::right, std::ios::adjustfield);
            out << std::setw(3) << basisDofOrdinal;
            out << "             ";
            out.setf(std::ios::right, std::ios::adjustfield);
            out << std::setw(myprec+8) << dofCoefficients[*dofIndexIt] << "\n";
          }
        }
      }
    }
    
    out.copyfmt(oldFormatState);
  }
  
  void printMapSummary(const Epetra_BlockMap &map, string mapName = "map")
  {
    int rank = MPIWrapper::rank();
    cout << "On rank " << rank << ", " << mapName << " has " << map.NumMyElements() << " of " << map.NumGlobalElements() << " global elements: ";
    for (int elementOrdinal=0; elementOrdinal<map.NumMyElements(); elementOrdinal++)
    {
      cout << "{ " << elementOrdinal << " -> " << map.GID(elementOrdinal) << " }";
      if (elementOrdinal != map.NumMyElements() - 1)
      {
        cout << ", ";
      }
    }
    cout << endl;
  }
}
