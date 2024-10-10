//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MimeticVlasovSpaceTimeFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 12/16/22.
//
//

#include "MimeticVlasovSpaceTimeFormulation.h"

#include "AbsFunction.h"
#include "MeshTools.h"

using namespace Camellia;
using std::vector;
using std::string;

const string MimeticVlasovSpaceTimeFormulation::S_GA_X  = "g_a_x";
const string MimeticVlasovSpaceTimeFormulation::S_GA_Y  = "g_a_y";
const string MimeticVlasovSpaceTimeFormulation::S_GA_Z  = "g_a_z";
const string MimeticVlasovSpaceTimeFormulation::S_GC    = "g_c";
const string MimeticVlasovSpaceTimeFormulation::S_GT    = "g_t";
const string MimeticVlasovSpaceTimeFormulation::S_GV_VX = "g_v_x";
const string MimeticVlasovSpaceTimeFormulation::S_GV_VY = "g_v_y";
const string MimeticVlasovSpaceTimeFormulation::S_GV_VZ = "g_v_z";

const string MimeticVlasovSpaceTimeFormulation::S_WA_X  = "w_a_x";
const string MimeticVlasovSpaceTimeFormulation::S_WA_Y  = "w_a_y";
const string MimeticVlasovSpaceTimeFormulation::S_WA_Z  = "w_a_z";
const string MimeticVlasovSpaceTimeFormulation::S_WC    = "w_c";
const string MimeticVlasovSpaceTimeFormulation::S_WT    = "w_t";
const string MimeticVlasovSpaceTimeFormulation::S_WV_VX = "w_v_x";
const string MimeticVlasovSpaceTimeFormulation::S_WV_VY = "w_v_y";
const string MimeticVlasovSpaceTimeFormulation::S_WV_VZ = "w_v_z";

const vector<string> MimeticVlasovSpaceTimeFormulation::S_GV {S_GV_VX, S_GV_VY, S_GV_VZ};
const vector<string> MimeticVlasovSpaceTimeFormulation::S_GA {S_GA_X,  S_GA_Y,  S_GA_Z };

const vector<string> MimeticVlasovSpaceTimeFormulation::S_WV {S_WV_VX, S_WV_VY, S_WV_VZ};
const vector<string> MimeticVlasovSpaceTimeFormulation::S_WA {S_WA_X,  S_WA_Y,  S_WA_Z };

MimeticVlasovSpaceTimeFormulation::MimeticVlasovSpaceTimeFormulation(double q, double m,
                                                                     bool useSerendipityBasis,
                                                                     Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                                     MeshTopologyPtr spatialMeshTopology,
                                                                     std::vector<double> temporalVertices,
                                                                     std::vector<double> velocitySpaceVertices_x,
                                                                     std::vector<double> velocitySpaceVertices_y,
                                                                     std::vector<double> velocitySpaceVertices_z,
                                                                     const int H1Order, const int delta_k,
                                                                     const int numUniformRefinements,
                                                                     const int maxMomentNumberForDriver)
:
VlasovSpaceTimeFormulation(q, m, useSerendipityBasis, configurationSpaceFormulation,
                           spatialMeshTopology, temporalVertices,
                           velocitySpaceVertices_x,
                           velocitySpaceVertices_y,
                           velocitySpaceVertices_z,
                           H1Order, delta_k,
                           numUniformRefinements,
                           maxMomentNumberForDriver,
                           true) // true: defer initialization; we call it below.
{
  this->_useSumFactorizedAssembly = false; // set base class's flag: we don't (yet) support sum-factorized assembly here.
  this->initialize();
}

IPPtr MimeticVlasovSpaceTimeFormulation::constructVlasovInnerProduct(const bool sumFactorizedAssembly)
{
  TEUCHOS_TEST_FOR_EXCEPTION(sumFactorizedAssembly, std::invalid_argument, "MimeticVlasovSpaceTimeFormulation does not support sum-factorized Gram assembly.");
  
  const int spaceDim    = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
  
  // tests
  VarPtr w = this->w(0); // 0: first species (the only one we have)
  vector<VarPtr> w_a(velocityDim);
  vector<VarPtr> w_v(spaceDim);
  VarPtr w_c = this->w_c();
  VarPtr w_t = this->w_t();
  
  for (int vd=0; vd<velocityDim; vd++)
  {
    w_a[vd] = this->w_a(vd);
  }
  
  for (int d=0; d<spaceDim; d++)
  {
    w_v[d] = this->w_v(d);
  }
  
  vector<FunctionPtr> a = this->a();
  
  IPPtr ip = IP::ip();
  
  LinearTermPtr f_term = - w->di(indexOffsetTimeOneBased); // -dw/dt
  for (int d=0; d<spaceDim; d++)
  {
    auto  v_d    = v_i(d);
    auto dw_dx_d = w->di(indexOffsetSpaceOneBased + d);
    f_term = f_term - v_d * dw_dx_d; // - v \cdot dw/dx
  }
  for (int vd=0; vd<velocityDim; vd++)
  {
    auto dw_dv_vd = w->di(indexOffsetVelocityOneBased + vd);
    f_term = f_term - a[vd] * dw_dv_vd;
  }
  ip->addTerm(f_term);
  
  LinearTermPtr new_f_term = 1.0 * w_t;
  for (int d=0; d<spaceDim; d++)
  {
    auto  v_d    = v_i(d);
    new_f_term = new_f_term + v_d * w_v[d]; // - v \cdot w_v
  }
  for (int vd=0; vd<velocityDim; vd++)
  {
    new_f_term = new_f_term + a[vd] * w_a[vd];
  }
  ip->addTerm(new_f_term);
  
  // use L^2 term for w_v:
  for (int d=0; d<spaceDim; d++)
  {
    ip->addTerm(w_v[d]);
  }
  
  // use L^2 term for w_a:
  for (int vd=0; vd<velocityDim; vd++)
  {
    ip->addTerm(w_a[vd]);
  }
  
  // use L^2 term for w_c:
  ip->addTerm(w_c);
  
  // use L^2 term for w_t:
  ip->addTerm(w_t);
  
  LinearTermPtr stabilization_term = 1.0 * w;
  ip->addTerm(stabilization_term);

//  std::cout << "test inner product:\n";
//  std::cout << ip->displayString() << std::endl;
  
  return ip;
}

VarFactoryPtr MimeticVlasovSpaceTimeFormulation::constructVlasovVarFactory()
{
  VarFactoryPtr vlasovVF = VarFactory::varFactory();
  
  BasisFamilyPtr basisFamily;
  if (this->useSerendipityBasis())
  {
    basisFamily = BasisFamily::serendipityBasisFamily();
  }
  else
  {
    basisFamily = BasisFamily::hierarchicalBasisFamily();
  }
  
  VarPtr f = vlasovVF->fieldVar(VlasovSpaceTimeFormulation::S_F);            // distribution function (field variable)
  VarPtr w = vlasovVF->testVar(VlasovSpaceTimeFormulation::S_W, HGRAD);      // test function w; lies in HGRAD of the whole space-time phase space
  VarPtr t_n_hat = vlasovVF->fluxVar(VlasovSpaceTimeFormulation::S_T_N_HAT); // t_n_hat: L^2 (non-conforming) trace
  
  const int spaceDim    = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  // g_a, w_a:
  vector<VarPtr> g_a(spaceDim), w_a(spaceDim);
  for (int vd=0; vd<velocityDim; vd++)
  {
    g_a[vd] = vlasovVF->fieldVar(S_GA[vd]);
    w_a[vd] = vlasovVF->testVar(S_WA[vd], L2);
  }
  
  // g_v, w_v:
  vector<VarPtr> g_v(spaceDim), w_v(spaceDim);
  for (int d=0; d<spaceDim; d++)
  {
    g_v[d] = vlasovVF->fieldVar(S_GV[d]);
    w_v[d] = vlasovVF->testVar(S_WV[d], L2);
  }

  // g_t, w_t:
  auto g_t = vlasovVF->fieldVar(S_GT);
  auto w_t = vlasovVF->testVar(S_WT, L2);

  // g_c, w_c:
  auto g_c = vlasovVF->fieldVar(S_GC, L2);
  auto w_c = vlasovVF->testVar(S_WC, HGRAD_DISC);
  
  vlasovVF->setTrialBasisFamily(basisFamily);
  vlasovVF->setTestBasisFamily(basisFamily);
  
  // sample a spatial element topology:
  CellTopoPtr spatialTopo = this->spatialMeshTopology()->getCell(0)->topology();
  CellTopoPtr cellTopo    = CellTopology::cellTopology(spatialTopo, this->velocityDim() + 1);
  
  {
    // likely this logic will need to be revised once we're actually running with velocityDim > 1.
    int p = this->H1Order();
    
    auto L2 = FUNCTION_SPACE_HVOL;
    // w_t, g_t get H1 * L2 * H1
    std::vector<EFunctionSpace> fs {L2, L2, L2}; // need the basis to be isotropic in FS because otherwise the pullback computation throws an exception (the function space gets labeled as UNKNOWN, so it doesn't know how to pull it back).  Presumably this will be resolved once we actually convert all the basis evaluation logic to use the newer Intrepid2 sum-factorization-friendly approach.
    auto basis_t = basisFamily->getBasis(cellTopo, fs, {p+1,p,p+1});
    auto basisFamily_t = BasisFamily::singletonBasisFamily(basis_t);
    
    w_t->setBasisFamily(basisFamily_t);
    g_t->setBasisFamily(basisFamily_t);
    
    // w_a, g_a get H1 * H1 * L2
    auto basis_a = basisFamily->getBasis(cellTopo, fs, {p+1,p+1,p});
    auto basisFamily_a = BasisFamily::singletonBasisFamily(basis_a);
    
    w_a[0]->setBasisFamily(basisFamily_a);
    g_a[0]->setBasisFamily(basisFamily_a);
    
    // w_v, g_v get L2 * H1 * H1
    auto basis_v = basisFamily->getBasis(cellTopo, fs, {p,p+1,p+1});
    auto basisFamily_v = BasisFamily::singletonBasisFamily(basis_v);
    
    w_v[0]->setBasisFamily(basisFamily_v);
    g_v[0]->setBasisFamily(basisFamily_v);
  }
  
  return vlasovVF;
}

VarPtr MimeticVlasovSpaceTimeFormulation::g_a(const int zeroBasedSpatialComponent)
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->trialVar(S_GA[zeroBasedSpatialComponent]);
}

VarPtr MimeticVlasovSpaceTimeFormulation::g_c()
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->trialVar(S_GC);
}

VarPtr MimeticVlasovSpaceTimeFormulation::g_t()
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->trialVar(S_GT);
}

VarPtr MimeticVlasovSpaceTimeFormulation::g_v(const int zeroBasedVelocityComponent)
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->trialVar(S_GV[zeroBasedVelocityComponent]);
}

void MimeticVlasovSpaceTimeFormulation::initializeVlasovBilinearForm(BFPtr vlasovBF)
{
  const int spaceDim = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  // trace
  VarPtr t_n_hat = this->t_n_hat(0); // 0: first species (the only one we have)
  
  // fields
  VarPtr f = this->f(0); // 0: first species (the only one we have)
  vector<VarPtr> g_a(velocityDim);
  vector<VarPtr> g_v(spaceDim);
  VarPtr g_c = this->g_c();
  VarPtr g_t = this->g_t();
  
  // tests
  VarPtr w = this->w(0); // 0: first species (the only one we have)
  vector<VarPtr> w_a(velocityDim);
  vector<VarPtr> w_v(spaceDim);
  VarPtr w_c = this->w_c();
  VarPtr w_t = this->w_t();
  
  for (int vd=0; vd<velocityDim; vd++)
  {
    w_a[vd] = this->w_a(vd);
    g_a[vd] = this->g_a(vd);
  }
  
  for (int d=0; d<spaceDim; d++)
  {
    w_v[d] = this->w_v(d);
    g_v[d] = this->g_v(d);
  }
  
  Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation = this->configurationSpaceFormulation();
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
    
  // The below assumes that spaceDim <= velocityDim; enforce that here:
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim > velocityDim, std::invalid_argument, "velocityDim must be at least as large as spaceDim");
  std::vector<FunctionPtr> v;
  for (int i=0; i<velocityDim; i++)
  {
    v.push_back(this->v_i(i));
  }

  // w equation:
  vlasovBF->addTerm(t_n_hat, w);
  vlasovBF->addTerm( -g_t, w->di(indexOffsetTimeOneBased)); // dt
  for (int d=0; d<spaceDim; d++)
  {
    vlasovBF->addTerm( -g_v[d], w->di(indexOffsetSpaceOneBased + d));  // dx_d
  }
  for (int vd=0; vd<velocityDim; vd++)
  {
    vlasovBF->addTerm( -g_a[vd], w->di(indexOffsetVelocityOneBased + vd));  // dvx_d
  }
  vlasovBF->addTerm( -g_c, w);
  
  // w_v equation:
  for (int d=0; d<spaceDim; d++)
  {
    vlasovBF->addTerm( g_v[d] - v[d] * f, w_v[d]);
  }
  
  // w_a equation:
  vector<FunctionPtr> a = this->a();
  for (int vd=0; vd<velocityDim; vd++)
  {
    vlasovBF->addTerm( g_a[vd] - a[vd] * f, w_a[vd]);
  }
  
  // w_c equation:
  vlasovBF->addTerm(g_c, w_c);
  
  // w_t equation:
  vlasovBF->addTerm(g_t - f, w_t);
}

Teuchos::RCP<VlasovSpaceTimeFormulation> MimeticVlasovSpaceTimeFormulation::nextTimeSlab(double tWidth, int numTemporalElements)
{
  auto temporalVertices = MeshTools::verticesFor1DIntervalMesh(t1(), tWidth, numTemporalElements);
  
  // create a new spatial MeshTopology based on the root of the current _spatialMeshTopology
  auto rootSpatialMeshTopology = MeshFactory::rootMeshTopologyCopy(spatialMeshTopology());
  
  auto nextSlabForm = Teuchos::rcp(new MimeticVlasovSpaceTimeFormulation(q(), m(), useSerendipityBasis(), configurationSpaceFormulation(), rootSpatialMeshTopology,
                                                                         temporalVertices, velocitySpaceVertices_x(), velocitySpaceVertices_y(), velocitySpaceVertices_z(),
                                                                         H1Order(), delta_k(), numUniformRefinements()));
  
  configureNextSlab(nextSlabForm);
  
  return nextSlabForm;
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForMassConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(i != 0, std::invalid_argument, "MimeticVlasovSpaceTimeFormulation only supports a single species");
  VarPtr w   = this->w(i);
  VarPtr w_c = this->w_c();
  map<int,FunctionPtr> testFunction;
  testFunction[w->ID()  ] = Function::constant(1.0);
  testFunction[w_c->ID()] = Function::constant(1.0);
  
  return testFunction;
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForEnergyConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForXMomentumConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForYMomentumConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForZMomentumConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,int> MimeticVlasovSpaceTimeFormulation::testOrderEnhancements()
{
  map<int,int> orderEnhancements;
  // for all test IDs other than w1, we don't enrich
  auto w1 = this->w(0); // 0: first species (the only one we have)
  // want to exclude all test IDs other than w1
  std::vector<int> testIDs = this->vlasovBF()->varFactory()->testIDs();
  std::set<int> otherTestIDs(testIDs.begin(),testIDs.end());
  otherTestIDs.erase(w1->ID());
  for (auto otherTestID : otherTestIDs)
  {
    orderEnhancements[otherTestID] = -delta_k(); // match the order of the trial space
  }
  return orderEnhancements;
}

map<int,int> MimeticVlasovSpaceTimeFormulation::trialOrderEnhancements()
{
  map<int,int> orderEnhancements;
  orderEnhancements[t_n_hat(0)->ID()] = 1; // +1 for non-conforming; 0: first species (the only one we have)
  orderEnhancements[g_c()->ID()]      = 1; // H^1 order
  return orderEnhancements;
}

VarPtr MimeticVlasovSpaceTimeFormulation::w_a(const int zeroBasedSpatialComponent)
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->testVar(S_WA[zeroBasedSpatialComponent]);
}

VarPtr MimeticVlasovSpaceTimeFormulation::w_c()
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->testVar(S_WC);
}

VarPtr MimeticVlasovSpaceTimeFormulation::w_t()
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->testVar(S_WT);
}

VarPtr MimeticVlasovSpaceTimeFormulation::w_v(const int zeroBasedVelocityComponent)
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->testVar(S_WV[zeroBasedVelocityComponent]);
}
