#ifndef SPACE_TIME_MIXED_VLASOV_SETUP_HPP
#define SPACE_TIME_MIXED_VLASOV_SETUP_HPP

#include <set>

#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_FEVector.h>

#include "Camellia.h"
#include "BOBA/boba.hpp"

std::tuple<Camellia::SolutionPtr, Camellia::MeshPtr, Camellia::VarFactoryPtr,
           Camellia::VarPtr, Camellia::VarPtr, Camellia::VarPtr>
setupVlasovSystem(const int &numElementX, const int &numElementT, const int &numElementV,
                  const int &polyOrder, const int &polyOrderEnrich,
                  const std::vector<std::vector<double>> &domainDim,
                  const double &beta,
                  Teuchos::GlobalMPISession &mpiSession,
                  bool solveSystem = true);

// TODO rename!
struct subvectors
{
  static constexpr size_t epsw = 0;
  static constexpr size_t tnx = 1;
  static constexpr size_t tnt = 2;
  static constexpr size_t tnv = 3;
  static constexpr size_t f = 4;
  static constexpr size_t size = 5;
};

boba::Array<std::set<Camellia::GlobalIndexType>, subvectors::size>
identifyVlasovBlockPartition(Camellia::SolutionPtr soln, Camellia::MeshPtr mesh, Camellia::VarFactoryPtr vf,
                             Camellia::VarPtr tn_hat, Camellia::VarPtr f, Camellia::VarPtr eps_w);

#endif
