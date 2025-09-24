//
//  Camellia_HDF5.cpp
//  dec
//
//  Created by Roberts, Nathan V on 8/21/25.
//
// Most code copied from EpetraExt_HDF5.cpp, via KinCat_HDF5.cpp.

#include "Camellia_HDF5.hpp"
#include "MPIWrapper.h"

#include <Tpetra_MultiVector.hpp>

#include "mpi.h"

using namespace Camellia;

#define CHECK_ERROR(err, msg)                                                                                   \
  if (err)                                                                                                             \
    throw std::logic_error(msg);

#define CHECK_STATUS(status) \
  { if (status < 0) \
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, \
                    "HDF5 returned a negative value"); }

namespace {
// ==========================================================================
// data container and iterators to find a dataset with a given name
struct FindDataset_t {
  std::string name;
  bool found;
};

static herr_t FindDataset(hid_t loc_id, const char *name, void *opdata) {
  std::string &token = ((FindDataset_t *)opdata)->name;
  if (token == name)
    ((FindDataset_t *)opdata)->found = true;

  return (0);
}
} // namespace

HDF5::HDF5(Teuchos_CommPtr comm) : IsOpen_(false) {
  _comm = comm;
  auto mpiCommPtr = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(comm);
  if (mpiCommPtr != Teuchos::null)
  {
    _mpiComm = *Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(comm)->getRawMpiComm();
  }
  else
  {
    _mpiComm = MPI_COMM_SELF;
  }
}

void HDF5::Create(const std::string FileName) {
  CHECK_ERROR(
      IsOpen(),
      "an HDF5 is already open; first close the current one using method Close(), then open/create a new one");

  FileName_ = FileName;

  // Set up file access property list with parallel I/O access
  plist_id_ = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id_, _mpiComm, MPI_INFO_NULL);
  
  // create the file collectively and release property list identifier.
  file_id_ = H5Fcreate(FileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_);
  H5Pclose(plist_id_);
  
  IsOpen_ = true;
}

// ==========================================================================
bool HDF5::IsContained(std::string Name, std::string GroupName) {
  CHECK_ERROR(!IsOpen(), "no file open yet");

  FindDataset_t data;
  data.name = Name;
  data.found = false;

  // recursively look for groups
  size_t pos = Name.find("/");
  if (pos != std::string::npos) {
    std::string NewGroupName = Name.substr(0, pos);
    if (GroupName != "")
      NewGroupName = GroupName + "/" + NewGroupName;
    std::string NewName = Name.substr(pos + 1);
    return IsContained(NewName, NewGroupName);
  }

  GroupName = "/" + GroupName;

  // int idx_f =
  H5Giterate(file_id_, GroupName.c_str(), NULL, FindDataset, (void *)&data);

  return (data.found);
}

void HDF5::Open(const std::string FileName, int AccessType) {
  CHECK_ERROR(
      IsOpen(), "an HDF5 is already open; first close the current one using method Close(), then open/create a new one")

  FileName_ = FileName;

  // Set up file access property list with parallel I/O access
  plist_id_ = H5Pcreate(H5P_FILE_ACCESS);

  // create the file collectively and release property list identifier.
  file_id_ = H5Fopen(FileName.c_str(), AccessType, plist_id_);
  H5Pclose(plist_id_);

  IsOpen_ = true;
}

// ==========================================================================
void HDF5::Write(const std::string &GroupName, const std::string &DataSetName, int what) {
  if (!IsContained(GroupName))
    CreateGroup(GroupName);

  hid_t filespace_id = H5Screate(H5S_SCALAR);
  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);
  hid_t dset_id =
      H5Dcreate(group_id, DataSetName.c_str(), H5T_NATIVE_INT, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status = H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, filespace_id, H5P_DEFAULT, &what);
  CHECK_ERROR((status < 0), "function H5Dwrite returned a negative value");

  // Close/release resources.
  H5Dclose(dset_id);
  H5Gclose(group_id);
  H5Sclose(filespace_id);
}

// ==========================================================================
void HDF5::Write(const std::string &GroupName, const std::string &DataSetName, double what) {
  if (!IsContained(GroupName))
    CreateGroup(GroupName);

  hid_t filespace_id = H5Screate(H5S_SCALAR);
  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);
  hid_t dset_id =
      H5Dcreate(group_id, DataSetName.c_str(), H5T_NATIVE_DOUBLE, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, filespace_id, H5P_DEFAULT, &what);
  CHECK_ERROR((status < 0), "function H5Giterater returned a negative value");

  // Close/release resources.
  H5Dclose(dset_id);
  H5Gclose(group_id);
  H5Sclose(filespace_id);
}

// ==========================================================================
void HDF5::Read(const std::string &GroupName, const std::string &DataSetName, int &data) {
  CHECK_ERROR(!IsContained(GroupName), "requested group not found");

  // Create group in the root group using absolute name.
  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  hid_t filespace_id = H5Screate(H5S_SCALAR);
  hid_t dset_id = H5Dopen(group_id, DataSetName.c_str(), H5P_DEFAULT);

  herr_t status = H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, filespace_id, H5P_DEFAULT, &data);
  CHECK_ERROR((status < 0), "function H5Giterater returned a negative value");

  H5Sclose(filespace_id);
  H5Dclose(dset_id);
  H5Gclose(group_id);
}

// ==========================================================================
void HDF5::Read(const std::string &GroupName, const std::string &DataSetName, double &data) {
  CHECK_ERROR(!IsContained(GroupName), "requested group not found");

  // Create group in the root group using absolute name.
  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  hid_t filespace_id = H5Screate(H5S_SCALAR);
  hid_t dset_id = H5Dopen(group_id, DataSetName.c_str(), H5P_DEFAULT);

  herr_t status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, filespace_id, H5P_DEFAULT, &data);
  CHECK_ERROR((status < 0), "function H5Giterater returned a negative value");

  H5Sclose(filespace_id);
  H5Dclose(dset_id);
  H5Gclose(group_id);
}

// ==========================================================================
void HDF5::Write(const std::string &GroupName, const std::string &DataSetName, const std::string &data) {
  if (!IsContained(GroupName))
    CreateGroup(GroupName);

  hsize_t len = 1;

  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  hid_t dataspace_id = H5Screate_simple(1, &len, NULL);

  hid_t atype = H5Tcopy(H5T_C_S1);
  H5Tset_size(atype, data.size() + 1);

  hid_t dataset_id =
      H5Dcreate(group_id, DataSetName.c_str(), atype, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status = H5Dwrite(dataset_id, atype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.c_str());
  CHECK_ERROR((status < 0), "function H5Dwrite returned a negative value");

  status = H5Tclose(atype);
  CHECK_ERROR((status < 0), "function H5Tclose returned a negative value");
  status = H5Dclose(dataset_id);
  CHECK_ERROR((status < 0), "function H5Dclose returned a negative value");
  status = H5Sclose(dataspace_id);
  CHECK_ERROR((status < 0), "function H5Sclose returned a negative value");
  status = H5Gclose(group_id);
  CHECK_ERROR((status < 0), "function H5Gclose returned a negative value");
}

// ==========================================================================
void HDF5::Read(const std::string &GroupName, const std::string &DataSetName, std::string &data) {
  CHECK_ERROR(!IsContained(GroupName), "requested group not found");

  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  hid_t dataset_id = H5Dopen(group_id, DataSetName.c_str(), H5P_DEFAULT);

  hid_t datatype_id = H5Dget_type(dataset_id);
  //  size_t typesize_id = H5Tget_size(datatype_id);
  H5T_class_t typeclass_id = H5Tget_class(datatype_id);

  CHECK_ERROR(typeclass_id != H5T_STRING, "requested group is not a std::string");

  char data2[160];
  herr_t status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2);
  CHECK_ERROR((status < 0), "function H5Dread returned a negative value");

  data = data2;

  H5Dclose(dataset_id);
  H5Gclose(group_id);
}

// ============= //
// serial arrays //
// ============= //

// ==========================================================================
void HDF5::Write(const std::string &GroupName, const std::string &DataSetName, const hid_t type, const int Length,
                 const void *data) {
  if (!IsContained(GroupName))
    CreateGroup(GroupName);

  hsize_t dimsf = Length;

  hid_t filespace_id = H5Screate_simple(1, &dimsf, NULL);

  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  hid_t dset_id = H5Dcreate(group_id, DataSetName.c_str(), type, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  herr_t status = H5Dwrite(dset_id, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  CHECK_ERROR((status < 0), "function H5Dwrite returned a negative value");

  // Close/release resources.
  H5Dclose(dset_id);
  H5Gclose(group_id);
  H5Sclose(filespace_id);
}

// ==========================================================================
void HDF5::Read(const std::string &GroupName, const std::string &DataSetName, const hid_t type, const int Length,
                void *data) {
  CHECK_ERROR(!IsContained(GroupName), "requested group not found")

  hsize_t dimsf = Length;

  hid_t filespace_id = H5Screate_simple(1, &dimsf, NULL);

  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  hid_t dset_id = H5Dopen(group_id, DataSetName.c_str(), H5P_DEFAULT);

  herr_t status = H5Dread(dset_id, type, H5S_ALL, filespace_id, H5P_DEFAULT, data);
  CHECK_ERROR((status < 0), "function H5Dread returned a negative value");

  // Close/release resources.
  H5Dclose(dset_id);
  H5Gclose(group_id);
  H5Sclose(filespace_id);
}

// ================== //
// distributed arrays //
// ================== //

// ==========================================================================
void HDF5::Write(const std::string &GroupName, const std::string &DataSetName, int MySize, int GlobalSize, hid_t type,
                 const void *data) {
  int Offset = MPIWrapper::scanSum(_comm, MySize);
  Offset -= MySize;
  
  hsize_t MySize_t = MySize;
  hsize_t GlobalSize_t = GlobalSize;
  hsize_t Offset_t = Offset;

  hid_t filespace_id = H5Screate_simple(1, &GlobalSize_t, NULL);

  // Create the dataset with default properties and close filespace.
  if (!IsContained(GroupName))
    CreateGroup(GroupName);

  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);
  hid_t dset_id = H5Dcreate(group_id, DataSetName.c_str(), type, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Sclose(filespace_id);

  // Each process defines dataset in memory and writes it to the hyperslab
  // in the file.

  hid_t memspace_id = H5Screate_simple(1, &MySize_t, NULL);

  // Select hyperslab in the file.
  filespace_id = H5Dget_space(dset_id);
  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, &Offset_t, NULL, &MySize_t, NULL);

  plist_id_ = H5Pcreate(H5P_DATASET_XFER);

  status = H5Dwrite(dset_id, type, memspace_id, filespace_id, plist_id_, data);
  CHECK_ERROR((status < 0), "function H5Dwrite returned a negative value");

  // Close/release resources.
  H5Dclose(dset_id);
  H5Gclose(group_id);
  H5Sclose(filespace_id);
  H5Sclose(memspace_id);
  H5Pclose(plist_id_);
}

// ==========================================================================
void HDF5::Read(const std::string &GroupName, const std::string &DataSetName, int MySize, int GlobalSize,
                const hid_t type, void *data) {
  CHECK_ERROR(!IsOpen(), "no file open yet");

  int Offset = MPIWrapper::scanSum(_comm, MySize);
  Offset -= MySize;
  
  hsize_t MySize_t = MySize;

  // offset
  hsize_t Offset_t = Offset;

  hid_t group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);
  hid_t dataset_id = H5Dopen(group_id, DataSetName.c_str(), H5P_DEFAULT);
  // hid_t space_id = H5Screate_simple(1, &Offset_t, 0);

  // Select hyperslab in the file.
  hid_t filespace_id = H5Dget_space(dataset_id);
  H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, &Offset_t, NULL, &MySize_t, NULL);

  hid_t mem_dataspace = H5Screate_simple(1, &MySize_t, NULL);

  herr_t status = H5Dread(dataset_id, type, mem_dataspace, filespace_id, H5P_DEFAULT, data);
  CHECK_ERROR((status < 0), "function H5Dread returned a negative value");

  H5Sclose(mem_dataspace);
  H5Gclose(group_id);
  // H5Sclose(space_id);
  H5Dclose(dataset_id);
  //  H5Dclose(filespace_id);
}

TVectorPtr<double> HDF5::ReadMultiVector(const std::string& GroupName, TMapPtr map, bool readTranspose, const int indexBase)
{
  // adapted from EpetraExt::HDF5::Read(const std::string& GroupName, Epetra_MultiVector*& LinearX, bool readTranspose, const int& indexBase).
  
  using MultiVector    = TVector<double>;
  using MultiVectorPtr = Teuchos::RCP<MultiVector>;
  using   LO = MultiVector::local_ordinal_type;
  using   GO = MultiVector::global_ordinal_type;
  using Node = MultiVector::node_type;
  
  int globalNumRows, numVectors;

  if (!IsContained(GroupName))
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested group " + GroupName + " not found");

  std::string Label;
  Read(GroupName, "__type__", Label);

  if (Label != "Epetra_MultiVector")
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested group " + GroupName + " is not an Epetra_MultiVector");

  Read(GroupName, "GlobalLength", globalNumRows);
  Read(GroupName, "NumVectors", numVectors);
  
  hid_t group_id;
  hid_t memspace_id;

  // Whether or not we do readTranspose or not is
  // handled by one of the components of q_dimsf, offset and count.
  // They are determined by indexT
  int indexT(0);
  if (readTranspose) indexT = 1;

  hsize_t q_dimsf[] = {static_cast<hsize_t>(globalNumRows), static_cast<hsize_t>(globalNumRows)};
  q_dimsf[indexT] = numVectors;

  hid_t filespace_id = H5Screate_simple(2, q_dimsf, NULL);

  if (!IsContained(GroupName))
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested group " + GroupName + " not found");

  group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  // Create the dataset with default properties and close filespace_id.
  hid_t dset_id = H5Dopen(group_id, "Values", H5P_DEFAULT);

  // Create property list for collective dataset write.
  plist_id_ = H5Pcreate(H5P_DATASET_XFER);
#ifdef HAVE_MPI
  H5Pset_dxpl_mpio(plist_id_, H5FD_MPIO_COLLECTIVE);
#endif
  H5Pclose(plist_id_);

  auto linearMap = Tpetra::createUniformContigMap<LO, GO>(globalNumRows, _comm);
  
  auto numLocalRows = linearMap->getLocalNumElements();
  
  auto linearX = Teuchos::rcp(new TVector<double>(linearMap, numVectors));
  
  hsize_t offsetValue = (numLocalRows > 0) ? static_cast<hsize_t>(linearMap->getGlobalElement(0)-indexBase)
                                           : 0;
  
  // Select hyperslab in the file.
  hsize_t offset[] = {offsetValue, offsetValue};
  hsize_t stride[] = {1, 1};

  {
    // following same logic as in Write()
    const hsize_t totalEntryCount = numVectors * numLocalRows;
    std::vector<double> entriesVector(totalEntryCount);

    // Select hyperslab in the file.
    hsize_t count[] = {static_cast<hsize_t>(numLocalRows), static_cast<hsize_t>(numLocalRows)};
    hsize_t block[] = {1, 1};

    // read vectors one by one
    for (int n(0); n < numVectors; ++n)
    {
      // Select hyperslab in the file.
      offset[indexT] = n;
      count [indexT] = 1;

      filespace_id = H5Dget_space(dset_id);
      H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, stride, count, block);

      // Each process defines dataset in memory and writes it to the hyperslab in the file.
      hsize_t dimsm[] = {static_cast<hsize_t>(numLocalRows)};
      memspace_id = H5Screate_simple(1, dimsm, NULL);

      // Read hyperslab
      double *data = (numLocalRows > 0) ? &entriesVector[n*numLocalRows] : NULL;
      CHECK_STATUS(H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace_id, filespace_id,
                           H5P_DEFAULT, data));
    }
    
    auto contiguousValuesViewHost = linearX->getLocalViewHost(Tpetra::Access::ReadWrite);
    
    Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>> hostPolicy(
        {0, 0},
        {static_cast<long long>(numLocalRows), static_cast<long long>(numVectors)}
    );

    Kokkos::parallel_for("fill host view", hostPolicy,
    [&](const int rowOrdinal, const int vectorOrdinal)
    // note the pass-by-reference [&].
    // This is OK here because we are on host, and it is guaranteed that each
    // thread modifies a distinct entry in contiguousValuesViewHost.
    {
      const int & tpetraIndex0 = rowOrdinal;
      const int & tpetraIndex1 = vectorOrdinal;
      const size_t vectorIndex = vectorOrdinal * numLocalRows + rowOrdinal; // indexed so that each vector is contiguous in memory
      contiguousValuesViewHost(tpetraIndex0, tpetraIndex1) = entriesVector[vectorIndex];
    });
    
    auto contiguousValuesView = linearX->getLocalViewDevice(Tpetra::Access::ReadWrite);
    Kokkos::deep_copy(contiguousValuesView, contiguousValuesViewHost);
  }

  CHECK_STATUS(H5Gclose(group_id));
  CHECK_STATUS(H5Sclose(memspace_id));
  CHECK_STATUS(H5Sclose(filespace_id));
  CHECK_STATUS(H5Dclose(dset_id));
  
  Tpetra::Import<LO,GO,Node> importer (linearX->getMap(), map); // source, destination
  auto X = Teuchos::rcp(new MultiVector(map, numVectors) );
  X->doImport(*linearX, importer, Tpetra::INSERT);
  return X;
}


void HDF5::Write(const std::string& GroupName, TVectorPtr<double> values, bool writeTranspose)
{
  const size_t numVectors = values->getNumVectors();
  // to use HDF5 hyperslabs, we need contiguous data.  So if we don't already have it, we create a new object that does using Tpetra::Import.
  using MultiVector    = std::remove_reference_t<decltype(*values.get())>;
  using MultiVectorPtr = Teuchos::RCP<MultiVector>;
  MultiVectorPtr contiguousValues;
  if ( values->getMap()->isContiguous() )
  {
    contiguousValues = values;
  }
  else
  {
    using LO = MultiVector::local_ordinal_type;
    using GO = MultiVector::global_ordinal_type;
    
    auto contiguousMap = Tpetra::createUniformContigMap<LO, GO>(values->getMap()->getGlobalNumElements(), values->getMap()->getComm());
    
    contiguousValues = Teuchos::rcp( new MultiVector(contiguousMap, values->getNumVectors()) );
    
    Tpetra::Import<LO, GO> importer(values->getMap(), contiguousMap);

    contiguousValues->doImport( *values, importer, Tpetra::INSERT);
  }
  
  auto map = contiguousValues->getMap();
  const hsize_t numGlobalRows = static_cast<hsize_t>(map->getGlobalNumElements());
  const hsize_t numLocalRows = static_cast<hsize_t>(map->getLocalNumElements());
  
  auto contiguousValuesViewHost = contiguousValues->getLocalViewHost(Tpetra::Access::ReadOnly);
  
  const hsize_t totalEntryCount = numVectors * numLocalRows;
  std::vector<double> entriesVector(totalEntryCount);
    
  Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>> hostPolicy(
      {0, 0},
      {static_cast<long long>(numLocalRows), static_cast<long long>(numVectors)}
  );

  // Here, we explicitly copy out the values from the BlockMultiVector.
  // This is safer in the general case than handing HDF5 a pointer to the View, but
  // the latter would probably work for the use cases we are likely to encounter.
  // Ideally, we would do some check of the Kokkos Layout and only do the copy if
  // the Layout is not compatible with pointer passing.
  Kokkos::parallel_for("fill HDF5 1D entriesVector", hostPolicy,
  [&](const int rowOrdinal, const int vectorOrdinal)
  // note the pass-by-reference [&] to allow entriesVector[] as lvalue.
  // This is OK here because we are on host, and it is guaranteed that each
  // thread modifies a distinct entry in entriesVector.
  {
    const int & tpetraIndex0 = rowOrdinal;
    const int & tpetraIndex1 = vectorOrdinal;
    const size_t vectorIndex = vectorOrdinal * numLocalRows + rowOrdinal; // indexed so that each vector is contiguous in memory
    entriesVector[vectorIndex] = contiguousValuesViewHost(tpetraIndex0, tpetraIndex1);
  });
  
  if (!IsOpen())
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "no file open yet");

  hid_t       group_id, dset_id;
  hid_t       filespace_id, memspace_id;
  herr_t      status;

  // Whether or not we do writeTranspose or not is
  // handled by one of the components of q_dimsf, offset and count.
  // They are determined by indexT
  int indexT(0);
  if (writeTranspose) indexT = 1;

  hsize_t q_dimsf[] = {static_cast<hsize_t>(numGlobalRows), static_cast<hsize_t>(numGlobalRows)};
  q_dimsf[indexT] = numVectors;

  filespace_id = H5Screate_simple(2, q_dimsf, NULL);

  if (!IsContained(GroupName))
    CreateGroup(GroupName);

  group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);

  // Create the dataset with default properties and close filespace_id.
  dset_id = H5Dcreate(group_id, "Values", H5T_NATIVE_DOUBLE, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Create property list for collective dataset write.
  plist_id_ = H5Pcreate(H5P_DATASET_XFER);
#ifdef HAVE_MPI
  H5Pset_dxpl_mpio(plist_id_, H5FD_MPIO_COLLECTIVE);
#endif

  // Select hyperslab in the file.
  hsize_t offsetValue = (numLocalRows > 0) ? static_cast<hsize_t>(contiguousValues->getMap()->getGlobalElement(0)-values->getMap()->getIndexBase())
                                           : 0;
  hsize_t offset[] =  {offsetValue, offsetValue};
  hsize_t stride[] = {1, 1};
  hsize_t count[] = {numLocalRows, numLocalRows};
  hsize_t block[] = {1, 1};

  // write vectors one by one
  for (int n(0); n < numVectors; ++n)
  {
    // Select hyperslab in the file.
    offset[indexT] = n;
    count [indexT] = 1;

    filespace_id = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, offset, stride,
                        count, block);

    // Each process defines dataset in memory and writes it to the hyperslab in the file.
    hsize_t dimsm[] = { numLocalRows };
    memspace_id = H5Screate_simple(1, dimsm, NULL);

    // Write hyperslab
    status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace_id, filespace_id,
                      plist_id_, &entriesVector[n*numLocalRows]);
    if (status < 0)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "H5Dwrite returned a negative value");
    }
    H5Sclose(memspace_id);
  }
  H5Gclose(group_id);
  H5Sclose(filespace_id);
  H5Dclose(dset_id);
  H5Pclose(plist_id_);

  // matching what EpetraExt did
  Write(GroupName, "GlobalLength", static_cast<int>(numGlobalRows));
  Write(GroupName, "NumVectors", static_cast<int>(numVectors));
  Write(GroupName, "__type__", "Epetra_MultiVector");
}

//void HDF5::Write(const std::string& GroupName, Teuchos::RCP<Statistic> stat, int recordNumber)
//{
//  Kokkos::View<double**> values = stat->values();
//  
//  const int numDofs = stat->numDofs();
//  
//  TEUCHOS_TEST_FOR_EXCEPTION(values.extent_int(1) != numDofs, std::invalid_argument, "Statistic has numDofs that differs from the number of vectors in its values() storage");
//  
//  const hsize_t numRows = values.extent_int(0);
//  
//  // all statistics should be available on every MPI rank; we write on rank 0
//  /// NOTE: we assume MPI_COMM_WORLD here; we could modify constructor to accept an MPI communicator.
//  int myRank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
//  
//  std::pair<std::string,std::string> key {GroupName, stat->name()};
//  
//  if (_dataSetIDs.find(key) == _dataSetIDs.end())
//  {
//    hid_t group_id;
//    htri_t group_exists = H5Lexists(file_id_, GroupName.c_str(), H5P_DEFAULT);
//    if (group_exists > 0)
//    {
//      group_id = H5Gopen(file_id_, GroupName.c_str(), H5P_DEFAULT);
//      if (group_id < 0)
//      {
//        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "error opening group");
//      }
//    }
//    else if (group_exists == 0)
//    {
//      group_id = H5Gcreate(file_id_, GroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//      if (group_id < 0)
//      {
//        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "error creating group");
//      }
//    }
//    else
//    {
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "error checking for group's existence");
//    }
//    
//    std::vector<hsize_t>    dims   {            0, numRows};
//    std::vector<hsize_t> maxDims   {H5S_UNLIMITED, numRows};
//    std::vector<hsize_t> chunkDims {            1, numRows}; // for each recordNumber, we output numGlobalRows x numDofs data
//    if (numDofs != 1)
//    {
//      dims.push_back(numDofs);
//      maxDims.push_back(numDofs);
//      chunkDims.push_back(numDofs);
//    }
//    hid_t dataspace = H5Screate_simple(static_cast<int>(dims.size()), &dims[0], &maxDims[0]);
//    if (dataspace < 0) {
//      TEUCHOS_TEST_FOR_EXCEPTION(dataspace < 0, std::invalid_argument, "error creating dataspace.")
//    }
//    
//    // Set up dataset creation property list with chunking.
//    hid_t dataSetCreationPropertyList = H5Pcreate(H5P_DATASET_CREATE);
//    H5Pset_chunk(dataSetCreationPropertyList, static_cast<int>(chunkDims.size()), &chunkDims[0]);
//    
//    hid_t dataSetID = H5Dcreate(group_id, stat->name().c_str(), H5T_NATIVE_DOUBLE,
//                                dataspace, H5P_DEFAULT, dataSetCreationPropertyList, H5P_DEFAULT);
//    _dataSetIDs[key] = dataSetID;
//    
//    H5Pclose(dataSetCreationPropertyList);
//    H5Sclose(dataspace);
//    H5Gclose(group_id);
//  }
//  auto dataSetID = _dataSetIDs[key];
//  
//  std::vector<hsize_t> newDims {static_cast<hsize_t>(recordNumber+1), numRows};
//  if (numDofs > 1)
//  {
//    newDims.push_back(numDofs);
//  }
//  
//  herr_t status = H5Dextend(dataSetID, &newDims[0]);
//  if (status < 0)
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error extending data set for new record");
//  }
//
//  hid_t dataSpaceExtended = H5Dget_space(dataSetID);
//  if (dataSpaceExtended < 0)
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "error obtaining dataSpaceExtended");
//  }
//
//  // Define the hyperslab.
//  auto myRowOffset = 0;
//  std::vector<hsize_t> offset {static_cast<hsize_t>(recordNumber), static_cast<hsize_t>(myRowOffset)};
//  if (numDofs > 1)
//  {
//    offset.push_back(0);
//  }
//  
//  hsize_t myElements = (myRank == 0) ? numRows : 0;
//  std::vector<hsize_t> count { 1, myElements };
//  if (numDofs > 1)
//  {
//    count.push_back(numDofs);
//  }
//  
//  status = H5Sselect_hyperslab(dataSpaceExtended, H5S_SELECT_SET, &offset[0],
//                               /*stride=*/nullptr, &count[0], /*block=*/nullptr);
//  if (status < 0)
//  {
//    H5Sclose(dataSpaceExtended);
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error selecting hyperslab in dataSpaceExtended");
//  }
//  
//  hid_t memspace = H5Screate_simple(static_cast<int>(count.size()), &count[0], nullptr);
//  if (memspace < 0)
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error creating memory space");
//  }
//  
//  const hsize_t totalEntryCount = myElements * numDofs;
//  std::vector<double> entriesVector(totalEntryCount);
//    
//  Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>> hostPolicy(
//      {0, 0},
//      {static_cast<int>(myElements), numDofs}
//  );
//
//  // Here, we explicitly copy out the values from the BlockMultiVector.
//  // This is safer in the general case than handing HDF5 a pointer to the View, but
//  // the latter would probably work for the use cases we are likely to encounter.
//  // Ideally, we would do some check of the Kokkos Layout and only do the copy if
//  // the Layout is not compatible with pointer passing.
//  Kokkos::parallel_for("fill HDF5 1D entriesVector", hostPolicy,
//  [&](const int entityOrdinal, const int dofOrdinal)
//  // note the pass-by-reference [&] to allow entriesVector[] as lvalue.
//  // This is OK here because we are on host, and it is guaranteed that each
//  // thread modifies a distinct entry in entriesVector.
//  {
//    const int vectorIndex = entityOrdinal * numDofs + dofOrdinal;
//    entriesVector[vectorIndex] = values(entityOrdinal, dofOrdinal);
//  });
//
//  hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
//  status = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
//  if (status < 0) {
//    H5Pclose(xfer_plist);
//    H5Sclose(memspace);
//    H5Sclose(dataSpaceExtended);
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error setting transfer property list for collective write");
//  }
//
//  status = H5Dwrite(dataSetID, H5T_NATIVE_DOUBLE,
//                    memspace, dataSpaceExtended, xfer_plist,
//                    static_cast<const void*>(&entriesVector[0]));
//  if (status < 0) {
//    std::cerr << "Error writing data for record number " << recordNumber << std::endl;
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error writing data for record");
//  }
//
//  H5Pclose(xfer_plist);
//  H5Sclose(memspace);
//  H5Sclose(dataSpaceExtended);
//}
