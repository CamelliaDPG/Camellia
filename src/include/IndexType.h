//
//  IndexType.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/30/14.
//
//

#ifndef Camellia_debug_IndexType_h
#define Camellia_debug_IndexType_h

typedef unsigned IndexType;

typedef unsigned GlobalIndexType;

typedef unsigned PartitionIndexType; // for partition numbering

typedef unsigned CellIDType;

typedef int GlobalIndexTypeToCast; // for constructing Epetra_Maps, etc.  (these like either int or long long)

#endif