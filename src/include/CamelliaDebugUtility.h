#ifndef CAMELLIA_DEBUG_UTILITY
#define CAMELLIA_DEBUG_UTILITY

#include <vector>
#include <set>

using namespace std;

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

#endif