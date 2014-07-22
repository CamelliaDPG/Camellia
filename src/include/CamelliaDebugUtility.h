#ifndef CAMELLIA_DEBUG_UTILITY
#define CAMELLIA_DEBUG_UTILITY

#include <vector>
#include <set>
#include <string>

using namespace std;

namespace Camellia {
  void print(string name, vector<int> data);
  void print(string name, vector<unsigned> data);
  void print(string name, vector<double> data);
  void print(string name, set<unsigned> data);
  void print(string name, set<int> data);
  void print(string name, set<double> data);
}

#endif
