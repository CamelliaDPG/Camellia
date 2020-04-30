//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "AbsFunction.h"
#include "Boltzmann1D1VFormulation.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "HDF5Exporter.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

using namespace Camellia;

class Maxwellian1V : public TFunction<double>
{
  double _u;
  double _sigma;
  double _n;
public:
  Maxwellian1V(double u, double sigma, double n)
  :
  _u(u),
  _sigma(sigma),
  _n(n)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & phaseSpacePoints = basisCache->getPhysicalCubaturePoints();
    const int numCells  = phaseSpacePoints.dimension(0);
    const int numPoints = phaseSpacePoints.dimension(1);
    
    const double weight = _n / (sqrt(2. * M_PI) * _sigma);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,1); // vx is the second dimension for 1D1V
        const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
        values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
      }
    }
  }
};

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  mpiSession.barrier();
  
  bool implicit = false;
  double k_x    = 0.2; // wavelength
  double u0     = 3.0; // initial drift velocity
  double gamma  = 1e-3; // initial perturbation
  double sigma  = 1.0; // temperature
  double n0     = 1.0; // density
  
  double q      =  1.0;
  double eps0   =  1.0;
  double m      =  1.0;
  double x0     =  0.0;
  double xWidth =  2 * M_PI / k_x;
  double v0     = -10.0;
  double vWidth =  20.0;
  
  int xElements = 30;
  int vElements = 40;
  int delta_k   = 2;
  double dt_max = 1e-3;
  int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  int numTimeSteps = 100;
  int outputFrequency = 10; // number of time steps to take before outputting viz.
  double timeStepC0 = -1.0;
  bool useL2Trace = true;
  
  cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
  cmdp.setOption("implicit", "explicit", &implicit);
  cmdp.setOption("xElements", &xElements);
  cmdp.setOption("vElements", &vElements);
  cmdp.setOption("H1Order", &H1Order);
  cmdp.setOption("delta_k", &delta_k);
  cmdp.setOption("dt", &dt_max );
  cmdp.setOption("u0", &u0);
  cmdp.setOption("numTimeSteps", &numTimeSteps);
  cmdp.setOption("outputFrequency", &outputFrequency);
  cmdp.setOption("timeStepC0", &timeStepC0);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  const int rank = mpiSession.getRank();
  
  Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice formulationChoice;
  using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
  if (implicit)
  {
    if (useL2Trace)
    {
      formulationChoice = FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE;
    }
    else
    {
      formulationChoice = FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE;
    }
  }
  else
  {
    if (useL2Trace)
    {
      formulationChoice = FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE;
    }
    else
    {
      formulationChoice = FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE;
    }
  }
  
  ERKTableauPtr erkTableau;
  if (!implicit)
  {
    erkTableau = ERKTableau::ssprk3();
  }
  
  double dt = dt_max;
  Boltzmann1D1VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, erkTableau);
  
  double n = 0.5 * n0;
  FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1V( u0, sigma, n) );
  FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1V(-u0, sigma, n) );
  FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
  FunctionPtr f0           = (fM_u0 + fM_minus_u0) * (1. + gamma * cos_kx);
  
  form.setInitialConditions(f0);
  
  std::ostringstream suffix;
  if (implicit)
  {
    suffix << "Implicit";
  }
  if (useL2Trace)
  {
    suffix << "_L2Trace";
  }
  else
  {
    suffix << "_H1Trace";
  }
  suffix << "_u0" << u0;
  suffix << "_dt" << dt_max;
  suffix << "_H1Order" << H1Order;
  suffix << "_" << xElements << "x" << vElements;
  
  std::ostringstream boltzmannSS;
  boltzmannSS << "TwoStreamBoltzmann" << suffix.str();
  
  std::ostringstream gaussSS;
  gaussSS << "TwoStreamGauss" << suffix.str();
  
//  std::cout << "boltzmann name: " << boltzmannSS.str() << std::endl;
//  std::cout << "gauss name: " << gaussSS.str() << std::endl;
  
  const std::string directoryPath = "/tmp";
  
  HDF5Exporter boltzmannViz(form.boltzmannMesh(), boltzmannSS.str(), directoryPath);
  HDF5Exporter gaussViz(form.gaussMesh(), gaussSS.str(), directoryPath);
  
  std::ostringstream E_magnitudeFilePath;
  E_magnitudeFilePath << directoryPath << "/" << "E_magnitude" << suffix.str() << ".dat";
  std::fstream E_magnitudeOut;
  if (rank == 0)
  {
    E_magnitudeOut.open(E_magnitudeFilePath.str(), std::ios::out);
    if (!E_magnitudeOut.is_open())
    {
      std::cout << "Failed to open file " << E_magnitudeFilePath.str() << " on rank 0.\n";
      return -1;
    }
    else
    {
      std::cout << "Will write integral of |E|^2 to " << E_magnitudeFilePath.str() << "\n";
    }
  }
  
  double t = 0.0;
  gaussViz.exportSolution(form.gaussSolution(),t);
  boltzmannViz.exportSolution(form.boltzmannSolution(),t);
  
  FunctionPtr E_soln = Function::solution(form.E(), form.gaussSolution());
  FunctionPtr E_squared = E_soln * E_soln;
  FunctionPtr abs_E_soln = Teuchos::rcp( new AbsFunction(E_soln) );
  
  for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
  {
    if (timeStepC0 > 0)
    {
      // adapt time step
      double E_max = abs_E_soln->maximumValue(form.gaussMesh());
      double v_max = std::max(std::abs(v0), std::abs(v0+vWidth));
      double h_x   = xWidth / xElements;
      double h_v   = vWidth / vElements;
      double cfl_dt_max = 1.0 / (v_max / h_x + (q/m) * E_max / h_v) * timeStepC0;
      double dt_new = std::min(dt_max, cfl_dt_max);
      if (dt != dt_new)
      {
        std::cout << "Set dt = " << dt_new << std::endl;
        dt = dt_new;
        form.setTimeStep(dt);
      }
    }
    
    form.takeTimeStep();
    t += dt;
    const bool computeMinimum_bn = true; // average value on a side
    if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
    {
      double min_bn = 1000;
      if (computeMinimum_bn)
      {
        // v * n->x() + (q/m) * E * n->y()
        auto E = form.E_extruded();
        auto v = Function::yn(1);
        
        const double xElemWidth = xWidth / xElements;
        const double vElemWidth = vWidth / vElements;
        
        auto myCellIDs = form.boltzmannMesh()->cellIDsInPartition();
        double localMin_bn = min_bn;
        for (auto cellID : myCellIDs)
        {
          auto basisCache = BasisCache::basisCacheForCell(form.boltzmannMesh(), cellID);
          for (int sideOrdinal=0; sideOrdinal<4; sideOrdinal++)
          {
            const bool sideParallelToXAxis = (sideOrdinal % 2 == 0);
            FunctionPtr f = sideParallelToXAxis ? (q/m) * E : v;
            const double integralLength = sideParallelToXAxis ? xElemWidth : vElemWidth;
            
            auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
            double integral = std::abs(f->integrate(sideCache));
            double averageValue = integral / integralLength;
            
            localMin_bn = std::min(localMin_bn, averageValue);
          }
        }
        MPIWrapper::CommWorld()->MinAll(&localMin_bn, &min_bn, 1);
      }
      
      double E_mag = E_squared->integrate(form.gaussMesh());
      boltzmannViz.exportSolution(form.boltzmannSolution(),t);
      gaussViz.exportSolution(form.gaussSolution(),t);
      if (rank == 0)
      {
        std::cout << "Completed time step " << timeStepNumber << " (t = " << t << ")" << std::endl;
        std::cout << "|E|^2 integral = " << E_mag << std::endl;
        E_magnitudeOut << t << "\t" << E_mag << std::endl << std::flush;
        if (computeMinimum_bn)
        {
          std::cout << "Minimum average value for b_n on a side: " << min_bn << std::endl;
        }
      }
    }
  }
  if (rank == 0)
  {
    E_magnitudeOut.close();
  }
  
  return 0;
}
