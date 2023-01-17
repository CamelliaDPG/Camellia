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

#include "MeshTools.h"

using namespace Camellia;
using std::vector;
using std::string;

const string MimeticVlasovSpaceTimeFormulation::S_GA_X  = "g_a_x";
const string MimeticVlasovSpaceTimeFormulation::S_GA_Y  = "g_a_y";
const string MimeticVlasovSpaceTimeFormulation::S_GA_Z  = "g_a_z";
const string MimeticVlasovSpaceTimeFormulation::S_GC    = "g_c";
const string MimeticVlasovSpaceTimeFormulation::S_GV_VX = "g_v_x";
const string MimeticVlasovSpaceTimeFormulation::S_GV_VY = "g_v_y";
const string MimeticVlasovSpaceTimeFormulation::S_GV_VZ = "g_v_z";

const string MimeticVlasovSpaceTimeFormulation::S_WA_X  = "w_a_x";
const string MimeticVlasovSpaceTimeFormulation::S_WA_Y  = "w_a_y";
const string MimeticVlasovSpaceTimeFormulation::S_WA_Z  = "w_a_z";
const string MimeticVlasovSpaceTimeFormulation::S_WC    = "w_c";
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
                                                                     const int numUniformRefinements)
:
VlasovSpaceTimeFormulation(q, m, useSerendipityBasis, configurationSpaceFormulation,
                           spatialMeshTopology, temporalVertices,
                           velocitySpaceVertices_x,
                           velocitySpaceVertices_y,
                           velocitySpaceVertices_z,
                           H1Order, delta_k,
                           numUniformRefinements,
                           true) // true: defer initialization; we call it below.
{
  this->initialize();
}

IPPtr MimeticVlasovSpaceTimeFormulation::constructVlasovInnerProduct()
{
  const int spaceDim    = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
  
  // tests
  VarPtr w = this->w();
  vector<VarPtr> w_a(velocityDim);
  vector<VarPtr> w_v(spaceDim);
  VarPtr w_c = this->w_c();
  
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
    auto  v_d = v_i(d);
    f_term = f_term - v_d * w_v[d]; // - v \cdot w_v
  }
  for (int vd=0; vd<velocityDim; vd++)
  {
    f_term = f_term - a[vd] * w_a[vd];
  }
  ip->addTerm(f_term);
  
  LinearTermPtr gv_term;
  for (int d=0; d<spaceDim; d++)
  {
    gv_term = gv_term + w_v[d] - w->di(indexOffsetSpaceOneBased + d); // - grad_x w + w_v
  }
  ip->addTerm(gv_term);
  
  LinearTermPtr ga_term;
  for (int vd=0; vd<velocityDim; vd++)
  {
    ga_term = ga_term + w_a[vd] - w->di(indexOffsetVelocityOneBased + vd); // - grad_v w + w_a
  }
  ip->addTerm(ga_term);
  
  LinearTermPtr gc_term = -w + w_c;
  ip->addTerm(gc_term);
  
  LinearTermPtr stabilization_term = 1.0 * w_c;
  ip->addTerm(stabilization_term);

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
  for (int vd=0; vd<velocityDim; vd++)
  {
    vlasovVF->fieldVar(S_GA[vd]);
    vlasovVF->testVar(S_WA[vd], L2);
  }
  
  // g_v, w_v:
  for (int d=0; d<spaceDim; d++)
  {
    vlasovVF->fieldVar(S_GV[d]);
    vlasovVF->testVar(S_WV[d], L2);
  }

  // g_c, w_c:
  auto g_c = vlasovVF->fieldVar(S_GC);
  auto w_c = vlasovVF->testVar(S_WC, L2);
  
  vlasovVF->setTrialBasisFamily(basisFamily);
  vlasovVF->setTestBasisFamily(basisFamily);
  
  // force g_c to be a constant:
  // sample a spatial element topology:
  CellTopoPtr spatialTopo = this->spatialMeshTopology()->getCell(0)->topology();
  CellTopoPtr cellTopo    = CellTopology::cellTopology(spatialTopo, this->velocityDim() + 1);
  
  BasisPtr constantBasis   = BasisFactory::basisFactory()->getConformingBasis(1, cellTopo, EFunctionSpace::FUNCTION_SPACE_HVOL);
  auto constantBasisFamily = BasisFamily::singletonBasisFamily(constantBasis);
  
  g_c->setBasisFamily(constantBasisFamily);
  w_c->setBasisFamily(constantBasisFamily);
  
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
  VarPtr t_n_hat = this->t_n_hat();
  
  // fields
  VarPtr f = this->f();
  vector<VarPtr> g_a(velocityDim);
  vector<VarPtr> g_v(spaceDim);
  VarPtr g_c = this->g_c();
  
  // tests
  VarPtr w = this->w();
  vector<VarPtr> w_a(velocityDim);
  vector<VarPtr> w_v(spaceDim);
  VarPtr w_c = this->w_c();
  
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
  vlasovBF->addTerm( -f, w->di(indexOffsetTimeOneBased)); // dt
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

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForMassConservation()
{
  VarPtr w   = this->w();
  VarPtr w_c = this->w_c();
  map<int,FunctionPtr> testFunction;
  testFunction[w->ID()  ] = Function::constant(1.0);
  testFunction[w_c->ID()] = Function::constant(1.0);
  
  return testFunction;
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForEnergyConservation()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForXMomentumConservation()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForYMomentumConservation()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> MimeticVlasovSpaceTimeFormulation::testForZMomentumConservation()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
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

VarPtr MimeticVlasovSpaceTimeFormulation::w_v(const int zeroBasedVelocityComponent)
{
  VarFactoryPtr vf = vlasovBF()->varFactory();
  return vf->testVar(S_WV[zeroBasedVelocityComponent]);
}
