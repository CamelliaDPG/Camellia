#include <iostream>

#include <map>
#include <vector>

using namespace std;

int Sort_ints_( int *vals_sort,     //  values to be sorted
               int *vals_other,    // other array to be reordered with sort
               int  nvals)         // length of these two arrays
{
  // a somewhat less efficient, but easy to write, reimplementation, which I hope will
  // work around a bug in the bgclang compiler
  if (nvals <= 1) return 0;
  
  map<int, vector<int> > sorter;
  for (int i=0; i<nvals; i++) {
    sorter[vals_sort[i]].push_back(vals_other[i]);
  }
  
  int i=0;
  for (map<int, vector<int> >::iterator sortIt = sorter.begin(); sortIt != sorter.end(); sortIt++) {
    for (vector<int>::iterator otherValIt = sortIt->second.begin(); otherValIt != sortIt->second.end(); otherValIt++) {
      vals_sort[i] = sortIt->first;
      vals_other[i] = *otherValIt;
      i++;
    }
  }
  
  return 0;
}

int Old_Sort_ints_( int *vals_sort,     //  values to be sorted
                int *vals_other,    // other array to be reordered with sort
                int  nvals)         // length of these two arrays
{
  // It is primarily used to sort messages to improve communication flow.
  // This routine will also insure that the ordering produced by the invert_map
  // routines is deterministic.  This should make bugs more reproducible.  This
  // is accomplished by sorting the message lists by processor ID.
  // This is a distribution count sort algorithm (see Knuth)
  //  This version assumes non negative integers.
  
  if (nvals <= 1) return 0;
  
  int i;                        // loop counter
  
  // find largest int, n, to size sorting array, then allocate and clear it
  int n = 0;
  for (i = 0; i < nvals; i++)
    if (n < vals_sort[i]) n = vals_sort[i];
  int *pos = new int [n+2];
  for (i = 0; i < n+2; i++) pos[i] = 0;
  
  // copy input arrays into temporary copies to allow sorting original arrays
  int *copy_sort  = new int [nvals];
  int *copy_other = new int [nvals];
  for (i = 0; i < nvals; i++)
  {
    copy_sort[i]  = vals_sort[i];
    copy_other[i] = vals_other[i];
  }
  
  // count the occurances of integers ("distribution count")
  int *p = pos+1;
  for (i = 0; i < nvals; i++) p[copy_sort[i]]++;
  
  // create the partial sum of distribution counts
  for (i = 1; i < n; i++) p[i] += p[i-1];
  
  // the shifted partitial sum is the index to store the data  in sort order
  p = pos;
  for (i = 0; i < nvals; i++)
  {
    vals_sort  [p[copy_sort [i]]]   = copy_sort[i];
    vals_other [p[copy_sort [i]]++] = copy_other[i];
  }
  
  delete [] copy_sort;
  delete [] copy_other;
  delete [] pos; 
  
  return 0;
}

int main(int argc, char *argv[]) {
    int nvals = 48;
    int *procs_from_ = new int[nvals];
    int *lengths_from_ = new int[nvals];
    procs_from_[0] = 21; procs_from_[1] = 23; procs_from_[2] = 29; procs_from_[3] = 17; procs_from_[4] = 20; procs_from_[5] = 22; procs_from_[6] = 19; procs_from_[7] = 28; procs_from_[8] = 16; procs_from_[9] = 25; procs_from_[10] = 18; procs_from_[11] = 30; procs_from_[12] = 27; procs_from_[13] = 24; procs_from_[14] = 26; procs_from_[15] = 5; procs_from_[16] = 7; procs_from_[17] = 4; procs_from_[18] = 13; procs_from_[19] = 1; procs_from_[20] = 15; procs_from_[21] = 3; procs_from_[22] = 6; procs_from_[23] = 12; procs_from_[24] = 2; procs_from_[25] = 0; procs_from_[26] = 11; procs_from_[27] = 9; procs_from_[28] = 14; procs_from_[29] = 8; procs_from_[30] = 10; procs_from_[31] = 43; procs_from_[32] = 40; procs_from_[33] = 42; procs_from_[34] = 34; procs_from_[35] = 41; procs_from_[36] = 46; procs_from_[37] = 35; procs_from_[38] = 38; procs_from_[39] = 44; procs_from_[40] = 39; procs_from_[41] = 32; procs_from_[42] = 45; procs_from_[43] = 33; procs_from_[44] = 36; procs_from_[45] = 37; procs_from_[46] = 47; procs_from_[47] = 31;
    lengths_from_[0] = 683; lengths_from_[1] = 683; lengths_from_[2] = 683; lengths_from_[3] = 683; lengths_from_[4] = 683; lengths_from_[5] = 683; lengths_from_[6] = 683; lengths_from_[7] = 683; lengths_from_[8] = 683; lengths_from_[9] = 683; lengths_from_[10] = 683; lengths_from_[11] = 518; lengths_from_[12] = 683; lengths_from_[13] = 683; lengths_from_[14] = 683; lengths_from_[15] = 683; lengths_from_[16] = 683; lengths_from_[17] = 683; lengths_from_[18] = 683; lengths_from_[19] = 683; lengths_from_[20] = 683; lengths_from_[21] = 683; lengths_from_[22] = 683; lengths_from_[23] = 683; lengths_from_[24] = 683; lengths_from_[25] = 683; lengths_from_[26] = 683; lengths_from_[27] = 683; lengths_from_[28] = 683; lengths_from_[29] = 683; lengths_from_[30] = 683; lengths_from_[31] = 683; lengths_from_[32] = 683; lengths_from_[33] = 683; lengths_from_[34] = 683; lengths_from_[35] = 683; lengths_from_[36] = 683; lengths_from_[37] = 683; lengths_from_[38] = 683; lengths_from_[39] = 683; lengths_from_[40] = 683; lengths_from_[41] = 683; lengths_from_[42] = 683; lengths_from_[43] = 683; lengths_from_[44] = 683; lengths_from_[45] = 683; lengths_from_[46] = 683; lengths_from_[47] = 165;

    Sort_ints_(procs_from_, lengths_from_, nvals);
    cout << "After sorting";
    for (int i=0; i<nvals; i++) {
      cout << ", procs_from["<< i << "] = " << procs_from_[i];
    }
    for (int i=0; i<nvals; i++) {
      cout << ", lengths_from["<< i << "] = " << lengths_from_[i];
    }
    cout << endl;
    delete [] procs_from_;
    delete [] lengths_from_;
    return 0;
}
