#ifndef CAMELLIA_DEBUG_UTILITY
#define CAMELLIA_DEBUG_UTILITY

#include <vector>
#include <set>
#include <string>

namespace Camellia {
  void print(std::string name, std::vector<int> data);
  void print(std::string name, std::vector<unsigned> data);
  void print(std::string name, std::vector<long long> data);
  void print(std::string name, std::vector<double> data);
  void print(std::string name, std::set<unsigned> data);
  void print(std::string name, std::set<int> data);
  void print(std::string name, std::set<long long> data);
  void print(std::string name, std::set<double> data);
}

#endif
