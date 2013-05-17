//
//  DataIO.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/3/13.
//
//

#ifndef Camellia_debug_DataIO_h
#define Camellia_debug_DataIO_h

#include "Teuchos_TestForException.hpp"

class DataIO {
public:
  static void outputTableToFile(std::vector<string> &tableHeaders, vector< vector< double > > &data, string filePath) {
    ofstream fout(filePath.c_str());
    // check dimensions:
    int numCols = tableHeaders.size();
    int numRows = data[0].size();
    
    if (numCols != data.size()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "first dimension of data must match size of tableHeaders.");
    }
    for (int col=0; col<numCols; col++) {
      if (data[col].size() != numRows) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "all columns must have the same number of rows");
      }
    }
    
    for (int col=0; col<numCols-1; col++) {
      fout << tableHeaders[col] << "\t";
    }
    fout << tableHeaders[numCols-1] << "\n";
    
    for (int row=0; row<numRows; row++) {
      for (int col=0; col<numCols-1; col++) {
        fout << data[col][row] << "\t";
      }
      fout << data[numCols-1][row] << "\n";
    }
    fout.close();
  }
  
  static void readMatrixFromSparseDataFile(FieldContainer<double> &matrix, string filename) {
    ifstream fin(filename.c_str());
    
    long rows, cols;
    if (fin.good()) {
      // get first line, which says how big the matrix is
      string line;
      do {
        std::getline(fin, line, '\n');
      } while (line.c_str()[0] == '%'); // skip over comment lines
      
      std::istringstream firstlinestream(line);
      firstlinestream >> rows;
      firstlinestream >> cols;
      
      if (rows * cols > 35000 * 35000) { // about 10 GB in memory--that's too big...
        cout << "Warning: can't form dense matrix from sparse data file: memory limit would be exceeded.  Exiting...\n";
        matrix.resize(1,1);
        matrix(0,0) = -1;
        return;
      }
      //    cout << "resizing matrix to " << rows << " x " << cols << endl;
      
      matrix.resize(rows,cols);
      
      while (fin.good()) {
        int row, col;
        double value;
        std::getline(fin, line, '\n');
        std::istringstream linestream(line);
        linestream >> row >> col >> value;
        matrix(row-1,col-1) = value;
      }
    } else {
      // better design would be to return with an error code
      cout << "Warning: readMatrix failed.\n";
    }
    fin.close();
  }
  
  static void writeMatrixToSparseDataFile(const FieldContainer<double> &matrix, string filename) {
    // matlab-friendly format (use spconvert)
    int rows = matrix.dimension(0);
    int cols = matrix.dimension(1);
    ofstream fout(filename.c_str());
    // specify dimensions:
    fout << rows << "\t" << cols << "\t"  << 0 << endl;
    double tol = 1e-15;
    for (int i=0; i<rows; i++) {
      for (int j=0; j<cols; j++) {
        if (abs(matrix(i,j)) > tol) { // nonzero
          fout << i+1 << "\t" << j+1 << "\t" << matrix(i,j) << endl;
        }
      }
    }
    fout.close();
  }

};

#endif
