//
//  CamelliaDebugUtility.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 3/18/14.
//
//

#include "CamelliaDebugUtility.h"

#include "VarFactory.h"
#include "DofOrdering.h"

#include <iostream>

using namespace std;

using namespace Intrepid;
namespace Camellia {

  template<typename data_type>
  void print(string name, vector<data_type> &data) {
    cout << name << ": ";
    for (int i=0; i<data.size(); i++) {
      cout << data[i] << " ";
    }
    cout << endl;
  }

  template<typename data_type>
  void print(string name, set<data_type> &data) {
    cout << name << ": ";
    for (typename set<data_type>::iterator dataIt=data.begin(); dataIt != data.end(); dataIt++) {
      cout << *dataIt << " ";
    }
    cout << endl;
  }

  template<typename key_type, typename value_type>
  void print(string name, map<key_type, value_type> &data) {
    cout << name << ": ";
    for (typename map<key_type, value_type>::iterator dataIt=data.begin(); dataIt != data.end(); dataIt++) {
      cout << "(" << dataIt->first << " => " << dataIt->second << ") ";
    }
    cout << endl;
  }

  void print(string name, map<int, double> data) {
    print<int, double>(name, data);
  }

  void print(string name, map<unsigned, double> data) {
    print<unsigned, double>(name, data);
  }

  void print(string name, map<unsigned, unsigned> data) {
    print<unsigned, unsigned>(name, data);
  }

  void print(string name, vector<long long> data) {
    print<long long>(name, data);
  }

  void print(string name, vector<int> data) {
    print<int>(name, data);
  }
  void print(string name, vector<unsigned> data) {
    print<unsigned>(name, data);
  }
  void print(string name, vector<double> data) {
    print<double>(name,data);
  }
  void print(string name, set<unsigned> data) {
    print<unsigned>(name, data);
  }
  void print(string name, set<int> data) {
    print<int>(name, data);
  }
  void print(string name, set<long long> data) {
    print<long long>(name, data);
  }
  void print(string name, set<double> data) {
    print<double>(name, data);
  }

  void printLabeledDofCoefficients(const VarFactoryPtr vf, DofOrderingPtr dofOrdering,
                                   const Intrepid::FieldContainer<double> &dofCoefficients,
                                   bool trialSpaceDofs) {
    if (dofOrdering->totalDofs() != dofCoefficients.size()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"dofOrdering size does not match dofCoefficients container size");
    }

    if (!trialSpaceDofs) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"printLabeledCoefficients does not yet support test variables");
    }

    Teuchos::oblackholestream oldFormatState;
    oldFormatState.copyfmt(cout);

    int myprec = cout.precision();
    cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
    cout.setf(std::ios_base::right);

    vector< VarPtr > fieldVars = vf->fieldVars();
    vector< VarPtr > traceVars = vf->traceVars();
    vector< VarPtr > fluxVars = vf->fluxVars();

    if (fieldVars.size() > 0) {
      cout << "\n\n **************   FIELD coefficients   **************\n";
      for (vector<VarPtr>::iterator varIt = fieldVars.begin(); varIt != fieldVars.end(); varIt++) {
        VarPtr var = *varIt;
        cout << var->displayString() << ":\n";
        const std::vector<int>* dofIndices = &dofOrdering->getDofIndices(var->ID());
        int basisDofOrdinal = 0;
        for (std::vector<int>::const_iterator dofIndexIt = dofIndices->begin(); dofIndexIt != dofIndices->end(); dofIndexIt++, basisDofOrdinal++) {
          cout.setf(std::ios::right, std::ios::adjustfield);
          cout << std::setw(3) << basisDofOrdinal;
          cout << "             ";
          cout.setf(std::ios::right, std::ios::adjustfield);
          cout << std::setw(myprec+8) << dofCoefficients[*dofIndexIt] << "\n";
        }
      }
    }

    if (traceVars.size() > 0) {
      cout << "\n\n ***************   TRACE coefficients   ***************\n";
      for (vector<VarPtr>::iterator varIt = traceVars.begin(); varIt != traceVars.end(); varIt++) {
        VarPtr var = *varIt;
        const vector<int>* sides = &dofOrdering->getSidesForVarID(var->ID());
        for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++) {
          int sideOrdinal = *sideIt;
          cout << var->displayString() << ", side " << sideOrdinal << ":\n";
          const std::vector<int>* dofIndices = &dofOrdering->getDofIndices(var->ID(), sideOrdinal);
          int basisDofOrdinal = 0;
          for (std::vector<int>::const_iterator dofIndexIt = dofIndices->begin(); dofIndexIt != dofIndices->end(); dofIndexIt++, basisDofOrdinal++) {
            cout.setf(std::ios::right, std::ios::adjustfield);
            cout << std::setw(3) << basisDofOrdinal;
            cout << "             ";
            cout.setf(std::ios::right, std::ios::adjustfield);
            cout << std::setw(myprec+8) << dofCoefficients[*dofIndexIt] << "\n";
          }
        }
      }
    }

    if (fluxVars.size() > 0) {
      cout << "\n\n ***************   FLUX coefficients   ***************\n";
      for (vector<VarPtr>::iterator varIt = fluxVars.begin(); varIt != fluxVars.end(); varIt++) {
        VarPtr var = *varIt;
        const vector<int>* sides = &dofOrdering->getSidesForVarID(var->ID());
        for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++) {
          int sideOrdinal = *sideIt;
          cout << var->displayString() << ", side " << sideOrdinal << ":\n";
          const std::vector<int>* dofIndices = &dofOrdering->getDofIndices(var->ID(), sideOrdinal);
          int basisDofOrdinal = 0;
          for (std::vector<int>::const_iterator dofIndexIt = dofIndices->begin(); dofIndexIt != dofIndices->end(); dofIndexIt++, basisDofOrdinal++) {
            cout.setf(std::ios::right, std::ios::adjustfield);
            cout << std::setw(3) << basisDofOrdinal;
            cout << "             ";
            cout.setf(std::ios::right, std::ios::adjustfield);
            cout << std::setw(myprec+8) << dofCoefficients[*dofIndexIt] << "\n";
          }
        }
      }
    }

    cout.copyfmt(oldFormatState);
  }
}
