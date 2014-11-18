//
//  CamelliaDebugUtility.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 3/18/14.
//
//

#include "CamelliaDebugUtility.h"

#include <iostream>

using namespace std;

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
}
