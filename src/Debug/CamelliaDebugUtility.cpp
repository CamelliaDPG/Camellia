//
//  CamelliaDebugUtility.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 3/18/14.
//
//

#include "CamelliaDebugUtility.h"

#include <iostream>

namespace Camellia {
  
  void print(string name, vector<int> data) {
    cout << name << ": ";
    for (int i=0; i<data.size(); i++) {
      cout << data[i] << " ";
    }
    cout << endl;
  }
  void print(string name, vector<unsigned> data) {
    cout << name << ": ";
    for (int i=0; i<data.size(); i++) {
      cout << data[i] << " ";
    }
    cout << endl;
  }
  void print(string name, vector<double> data) {
    cout << name << ": ";
    for (int i=0; i<data.size(); i++) {
      cout << data[i] << " ";
    }
    cout << endl;
  }
  void print(string name, set<unsigned> data) {
    cout << name << ": ";
    for (set<unsigned>::iterator dataIt=data.begin(); dataIt != data.end(); dataIt++) {
      cout << *dataIt << " ";
    }
    cout << endl;
  }
  void print(string name, set<int> data) {
    cout << name << ": ";
    for (set<int>::iterator dataIt=data.begin(); dataIt != data.end(); dataIt++) {
      cout << *dataIt << " ";
    }
    cout << endl;
  }
  void print(string name, set<double> data) {
    cout << name << ": ";
    for (set<double>::iterator dataIt=data.begin(); dataIt != data.end(); dataIt++) {
      cout << *dataIt << " ";
    }
    cout << endl;
  }
}
