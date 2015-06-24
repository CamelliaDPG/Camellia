
#include "Function.h"
#include "SpatialFilter.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PenaltyConstraints.h"
#include "SpaceTimeIncompressibleFormulation.h"

namespace Camellia
{
class IncompressibleProblem
{
  protected:
    Teuchos::RCP<PenaltyConstraints> _pc = Teuchos::null;
    FunctionPtr _u1_exact;
    FunctionPtr _u2_exact;
    FunctionPtr _sigma1_exact;
    FunctionPtr _sigma2_exact;
    double _tInit;
    double _tFinal;
    int _numSlabs = 1;
    int _currentStep = 0;
    bool _steady;
    bool _pureVelocityBCs = true;
  public:
    LinearTermPtr forcingTerm = Teuchos::null;
    FunctionPtr u1_exact() { return _u1_exact; }
    FunctionPtr u2_exact() { return _u2_exact; }
    virtual MeshTopologyPtr meshTopology(int temporalDivisions=1) = 0;
    virtual MeshGeometryPtr meshGeometry() { return Teuchos::null; }
    virtual void preprocessMesh(MeshPtr proxyMesh) {};
    virtual void setBCs(SpaceTimeIncompressibleFormulationPtr form) = 0;
    Teuchos::RCP<PenaltyConstraints> pc() { return _pc; }
    bool imposeZeroMeanPressure() { return _pureVelocityBCs; }
    virtual double computeL2Error(SpaceTimeIncompressibleFormulationPtr form, SolutionPtr solutionBackground) { return 0; }
    int numSlabs() { return _numSlabs; }
    int currentStep() { return _currentStep; }
    void advanceStep() { _currentStep++; }
    double stepSize() { return (_tFinal-_tInit)/_numSlabs; }
    double currentT0() { return stepSize()*_currentStep; }
    double currentT1() { return stepSize()*(_currentStep+1); }
};

class AnalyticalIncompressibleProblem : public IncompressibleProblem
{
  protected:
    vector<double> _x0;
    vector<double> _dimensions;
    vector<int> _elementCounts;
    map<int, FunctionPtr> _exactMap;
  public:
    virtual MeshTopologyPtr meshTopology(int temporalDivisions=1)
    {
      MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(_dimensions, _elementCounts, _x0);
      MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, currentT0(), currentT1(), temporalDivisions);
      if (_steady)
        return spatialMeshTopo;
      else
        return spaceTimeMeshTopo;
    }

    void initializeExactMap(SpaceTimeIncompressibleFormulationPtr form)
    {
      _exactMap[form->u(1)->ID()] = _u1_exact;
      _exactMap[form->u(2)->ID()] = _u2_exact;
      _exactMap[form->sigma(1,1)->ID()] = _sigma1_exact->x();
      _exactMap[form->sigma(1,2)->ID()] = _sigma1_exact->y();
      _exactMap[form->sigma(2,1)->ID()] = _sigma2_exact->x();
      _exactMap[form->sigma(2,2)->ID()] = _sigma2_exact->y();
      _exactMap[form->uhat(1)->ID()] = form->uhat(1)->termTraced()->evaluate(_exactMap);
      _exactMap[form->uhat(2)->ID()] = form->uhat(2)->termTraced()->evaluate(_exactMap);
    }

    void projectExactSolution(SolutionPtr solution)
    {
      solution->projectOntoMesh(_exactMap);
    }

    virtual void setBCs(SpaceTimeIncompressibleFormulationPtr form)
    {
      initializeExactMap(form);

      BCPtr bc = form->solutionUpdate()->bc();
      SpatialFilterPtr initTime = SpatialFilter::matchingT(_tInit);
      SpatialFilterPtr leftX  = SpatialFilter::matchingX(_x0[0]);
      SpatialFilterPtr rightX = SpatialFilter::matchingX(_x0[0]+_dimensions[0]);
      SpatialFilterPtr leftY  = SpatialFilter::matchingY(_x0[1]);
      SpatialFilterPtr rightY = SpatialFilter::matchingY(_x0[1]+_dimensions[1]);
      bc->addDirichlet(form->uhat(1), leftX,    _exactMap[form->uhat(1)->ID()]);
      bc->addDirichlet(form->uhat(2), leftX,    _exactMap[form->uhat(2)->ID()]);
      bc->addDirichlet(form->uhat(1), rightX,   _exactMap[form->uhat(1)->ID()]);
      bc->addDirichlet(form->uhat(2), rightX,   _exactMap[form->uhat(2)->ID()]);
      bc->addDirichlet(form->uhat(1), leftY,    _exactMap[form->uhat(1)->ID()]);
      bc->addDirichlet(form->uhat(2), leftY,    _exactMap[form->uhat(2)->ID()]);
      bc->addDirichlet(form->uhat(1), rightY,   _exactMap[form->uhat(1)->ID()]);
      bc->addDirichlet(form->uhat(2), rightY,   _exactMap[form->uhat(2)->ID()]);
      if (!_steady)
      {
        bc->addDirichlet(form->tmhat(1),initTime,-_exactMap[form->uhat(1)->ID()]);
        bc->addDirichlet(form->tmhat(2),initTime,-_exactMap[form->uhat(2)->ID()]);
      }
    }
    double computeL2Error(SpaceTimeIncompressibleFormulationPtr form, SolutionPtr solutionBackground)
    {
      FunctionPtr u1_soln, u2_soln, sigma11_soln, sigma12_soln, sigma21_soln, sigma22_soln,
                  u1_diff, u2_diff, sigma11_diff, sigma12_diff, sigma21_diff, sigma22_diff,
                  u1_sqr, u2_sqr, sigma11_sqr, sigma12_sqr, sigma21_sqr, sigma22_sqr;
      u1_soln = Function::solution(form->u(1), solutionBackground);
      u2_soln = Function::solution(form->u(2), solutionBackground);
      sigma11_soln = Function::solution(form->sigma(1,1), solutionBackground);
      sigma12_soln = Function::solution(form->sigma(1,2), solutionBackground);
      sigma21_soln = Function::solution(form->sigma(2,1), solutionBackground);
      sigma22_soln = Function::solution(form->sigma(2,2), solutionBackground);
      u1_diff = u1_soln - _u1_exact;
      u2_diff = u2_soln - _u2_exact;
      sigma11_diff = sigma11_soln - _sigma1_exact->x();
      sigma12_diff = sigma12_soln - _sigma1_exact->y();
      sigma21_diff = sigma21_soln - _sigma2_exact->x();
      sigma22_diff = sigma22_soln - _sigma2_exact->y();
      u1_sqr = u1_diff*u1_diff;
      u2_sqr = u2_diff*u2_diff;
      sigma11_sqr = sigma11_diff*sigma11_diff;
      sigma12_sqr = sigma12_diff*sigma12_diff;
      sigma21_sqr = sigma21_diff*sigma21_diff;
      sigma22_sqr = sigma22_diff*sigma22_diff;
      double u1_l2, u2_l2, sigma11_l2, sigma12_l2, sigma21_l2, sigma22_l2;
      u1_l2 = u1_sqr->integrate(solutionBackground->mesh(), 5);
      u2_l2 = u2_sqr->integrate(solutionBackground->mesh(), 5);
      sigma11_l2 = sigma11_sqr->integrate(solutionBackground->mesh(), 5);
      sigma12_l2 = sigma12_sqr->integrate(solutionBackground->mesh(), 5);
      sigma21_l2 = sigma21_sqr->integrate(solutionBackground->mesh(), 5);
      sigma22_l2 = sigma22_sqr->integrate(solutionBackground->mesh(), 5);
      double l2Error = sqrt(u1_l2+u2_l2+sigma11_l2+sigma12_l2+sigma21_l2+sigma22_l2);
      return l2Error;
    }
};

class KovasznayProblem : public AnalyticalIncompressibleProblem
{
  private:
  public:
    KovasznayProblem(bool steady, double Re)
    {
      _steady = steady;
      // problemName = "Kovasznay";
      double pi = atan(1)*4;
      double lambda = Re/2-sqrt(Re*Re/4+4*pi*pi);
      FunctionPtr explambdaX = Teuchos::rcp(new Exp_ax(lambda));
      FunctionPtr cos2piY = Teuchos::rcp(new Cos_ay(2*pi));
      FunctionPtr sin2piY = Teuchos::rcp(new Sin_ay(2*pi));
      _u1_exact = 1 - explambdaX*cos2piY;
      _u2_exact = lambda/(2*pi)*explambdaX*sin2piY;
      _sigma1_exact = 1./Re*_u1_exact->grad();
      _sigma2_exact = 1./Re*_u2_exact->grad();

      _x0.push_back(-.5);
      _x0.push_back(-.5);
      _dimensions.push_back(1.5);
      _dimensions.push_back(2.0);
      _elementCounts.push_back(3);
      _elementCounts.push_back(4);
      _tInit = 0.0;
      _tFinal = 0.25;
    }
};

class TaylorGreenProblem : public AnalyticalIncompressibleProblem
{
  private:
  public:
    TaylorGreenProblem(bool steady, double Re, int numXElems=2, int numSlabs=1)
    {
      _steady = steady;
      // problemName = "Kovasznay";
      double pi = atan(1)*4;
      FunctionPtr temporalDecay = Teuchos::rcp(new Exp_at(-2./Re));
      FunctionPtr sinX = Teuchos::rcp(new Sin_x());
      FunctionPtr cosX = Teuchos::rcp(new Cos_x());
      FunctionPtr sinY = Teuchos::rcp(new Sin_y());
      FunctionPtr cosY = Teuchos::rcp(new Cos_y());
      _u1_exact = sinX*cosY*temporalDecay;
      _u2_exact = -cosX*sinY*temporalDecay;
      _sigma1_exact = 1./Re*_u1_exact->grad();
      _sigma2_exact = 1./Re*_u2_exact->grad();

      _x0.push_back(0);
      _x0.push_back(0);
      _dimensions.push_back(2*pi);
      _dimensions.push_back(2*pi);
      _elementCounts.push_back(numXElems);
      _elementCounts.push_back(numXElems);
      _tInit = 0.0;
      _tFinal = 1.0;
      _numSlabs = numSlabs;
    }
};

class NearCylinder : public SpatialFilter
{
  double _enlarged_radius;
public:
  NearCylinder(double radius)
  {
    double enlargement_factor = 1.2;
    _enlarged_radius = radius * enlargement_factor;
  }
  bool matchesPoint(double x, double y)
  {
    if (x*x + y*y < _enlarged_radius * _enlarged_radius)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y, double z)
  {
    if (x*x + y*y < _enlarged_radius * _enlarged_radius)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

class CylinderProblem : public IncompressibleProblem
{
  private:
    double _radius = 0.5;
    double _xLeft = -3;
    double _xRight = 9;
    double _meshHeight = 6;
    // double _xLeft = -60;
    // double _xRight = 180;
    // double _meshHeight = 120;
    double _yBottom = -_meshHeight/2;
    double _yTop = _meshHeight/2;
  public:
    CylinderProblem(bool steady, double Re, int numSlabs=1)
    {
      _steady = steady;
      _u1_exact = Function::min(Function::tn(1),Function::constant(1));
      _u2_exact = Function::constant(0);
      // _sigma1_exact = 1./Re*_u1_exact->grad();
      // _sigma2_exact = 1./Re*_u2_exact->grad();
      _sigma1_exact = Function::zero();
      _sigma2_exact = Function::zero();

      _tInit = 0.0;
      _tFinal = 12.0;
      _numSlabs = numSlabs;
      _pureVelocityBCs = false;
    }
    virtual MeshGeometryPtr meshGeometry()
    {
      double embeddedSideLength = 3 * _radius;
      // double embeddedSideLength = 60;
      return MeshFactory::shiftedHemkerGeometry(_xLeft, _xRight, _yBottom, _yTop, _radius, embeddedSideLength);
    }
    virtual MeshTopologyPtr meshTopology(int temporalDivisions=1)
    {
      MeshGeometryPtr geometry = meshGeometry();
      MeshTopologyPtr spatialMeshTopo = Teuchos::rcp(new MeshTopology(geometry));
      MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, currentT0(), currentT1(), temporalDivisions);
      if (_steady)
        return spatialMeshTopo;
      else
        return spaceTimeMeshTopo;
    }

    virtual void preprocessMesh(MeshPtr hemkerMeshNoCurves)
    {
      // cout << "Beginning mesh preprocessing" << endl;
      double radius = _radius;
      bool enforceOneIrregularity = true;

      // start by identifying the various elements: there are 10 of interest to us
      // to find the thin banded elements, note that radius * 3 will be outside the bounding square
      // and that radius / 2 will be inside the band
      // Intrepid::FieldContainer<double> elementPoints(10,2);
      // // ESE band
      // elementPoints(0,0) =   radius * 3;
      // elementPoints(0,1) = - radius / 2;
      // // ENE band
      // elementPoints(1,0) = radius * 3;
      // elementPoints(1,1) = radius / 2;
      // // WSW band
      // elementPoints(2,0) = - radius * 3;
      // elementPoints(2,1) = - radius / 2;
      // // WNW band
      // elementPoints(3,0) = - radius * 3;
      // elementPoints(3,1) =   radius / 2;
      // // NNE band
      // elementPoints(4,0) =   radius / 2;
      // elementPoints(4,1) =   radius * 3;
      // // NNW band
      // elementPoints(5,0) = - radius / 2;
      // elementPoints(5,1) =   radius * 3;
      // // SSE band
      // elementPoints(6,0) =   radius / 2;
      // elementPoints(6,1) = - radius * 3;
      // // SSE band
      // elementPoints(7,0) = - radius / 2;
      // elementPoints(7,1) = - radius * 3;
      // // NE big element
      // elementPoints(8,0) = radius * 3;
      // elementPoints(8,1) = radius * 3;
      // // SE big element
      // elementPoints(9,0) =   radius * 3;
      // elementPoints(9,1) = - radius * 3;

      Intrepid::FieldContainer<double> horizontalBandPoints(6,2);
      if (!_steady)
        horizontalBandPoints.resize(6,3);
      // ESE band
      horizontalBandPoints(0,0) =   radius * 3;
      horizontalBandPoints(0,1) = - radius / 2;
      // ENE band
      horizontalBandPoints(1,0) = radius * 3;
      horizontalBandPoints(1,1) = radius / 2;
      // WSW band
      horizontalBandPoints(2,0) = - radius * 3;
      horizontalBandPoints(2,1) = - radius / 2;
      // WNW band
      horizontalBandPoints(3,0) = - radius * 3;
      horizontalBandPoints(3,1) =   radius / 2;
      // the bigger, fatter guys in the corners count as horizontal bands (because that's the direction of their anisotropy)
      // NE big element
      horizontalBandPoints(4,0) = radius * 3;
      horizontalBandPoints(4,1) = radius * 3;
      // SE big element
      horizontalBandPoints(5,0) =   radius * 3;
      horizontalBandPoints(5,1) = - radius * 3;

      Intrepid::FieldContainer<double> verticalBandPoints(4,2);
      if (!_steady)
        verticalBandPoints.resize(6,3);
      // NNE band
      verticalBandPoints(0,0) =   radius / 2;
      verticalBandPoints(0,1) =   radius * 3;
      // NNW band
      verticalBandPoints(1,0) = - radius / 2;
      verticalBandPoints(1,1) =   radius * 3;
      // SSE band
      verticalBandPoints(2,0) =   radius / 2;
      verticalBandPoints(2,1) = - radius * 3;
      // SSE band
      verticalBandPoints(3,0) = - radius / 2;
      verticalBandPoints(3,1) = - radius * 3;

      // vector< ElementPtr > elements = hemkerMeshNoCurves->elementsForPoints(elementPoints, false);

      // vector<int> horizontalBandCellIDs;
      // horizontalBandCellIDs.push_back(elements[0]->cellID());
      // horizontalBandCellIDs.push_back(elements[1]->cellID());
      // horizontalBandCellIDs.push_back(elements[2]->cellID());
      // horizontalBandCellIDs.push_back(elements[3]->cellID());

      // vector<int> verticalBandCellIDs;
      // verticalBandCellIDs.push_back(elements[4]->cellID());
      // verticalBandCellIDs.push_back(elements[5]->cellID());
      // verticalBandCellIDs.push_back(elements[6]->cellID());
      // verticalBandCellIDs.push_back(elements[7]->cellID());

      // // the bigger, fatter guys in the corners count as horizontal bands (because that's the direction of their anisotropy)
      // horizontalBandCellIDs.push_back(elements[8]->cellID());
      // horizontalBandCellIDs.push_back(elements[9]->cellID());

      vector< GlobalIndexType > horizontalBandCellIDs = hemkerMeshNoCurves->cellIDsForPoints(horizontalBandPoints, false);
      vector< GlobalIndexType > verticalBandCellIDs = hemkerMeshNoCurves->cellIDsForPoints(verticalBandPoints, false);

      Teuchos::RCP<RefinementPattern> verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuad();
      Teuchos::RCP<RefinementPattern> horizontalCut = RefinementPattern::yAnisotropicRefinementPatternQuad();

      Intrepid::FieldContainer<double> vertices(4,2);

      // horizontal bands want vertical cuts, and vice versa
      for (vector<GlobalIndexType>::iterator cellIDIt = horizontalBandCellIDs.begin();
          cellIDIt != horizontalBandCellIDs.end(); cellIDIt++)
      {
        int cellID = *cellIDIt;
        // cout << "Refining cell " << cellID << endl;
        //    cout << "Identified cell " << cellID << " as a horizontal band.\n";
        // work out what the current aspect ratio is
        hemkerMeshNoCurves->verticesForCell(vertices, cellID);
        //    cout << "vertices for cell " << cellID << ":\n" << vertices;
        // here, we use knowledge of the implementation of the hemker mesh generation:
        // we know that the first edges are always horizontal...
        double xDiff = abs(vertices(1,0)-vertices(0,0));
        double yDiff = abs(vertices(2,1)-vertices(1,1));

        //    cout << "xDiff: " << xDiff << endl;
        //    cout << "yDiff: " << yDiff << endl;

        set<GlobalIndexType> cellIDsToRefine;
        cellIDsToRefine.insert(cellID);
        double aspect = xDiff / yDiff;
        while (aspect > 2.0)
        {
          //      cout << "aspect ratio: " << aspect << endl;
          hemkerMeshNoCurves->hRefine(cellIDsToRefine, verticalCut);

          // the next set of cellIDsToRefine are the children of the ones just refined
          set<GlobalIndexType> childCellIDs;
          for (set<GlobalIndexType>::iterator refinedCellIDIt = cellIDsToRefine.begin();
              refinedCellIDIt != cellIDsToRefine.end(); refinedCellIDIt++)
          {
            int refinedCellID = *refinedCellIDIt;
            set<GlobalIndexType> refinedCellChildren = hemkerMeshNoCurves->getTopology()->getCell(refinedCellID)->getDescendants();
            childCellIDs.insert(refinedCellChildren.begin(),refinedCellChildren.end());
          }

          cellIDsToRefine = childCellIDs;
          aspect /= 2;
        }
      }

      // horizontal bands want vertical cuts, and vice versa
      for (vector<GlobalIndexType>::iterator cellIDIt = verticalBandCellIDs.begin();
          cellIDIt != verticalBandCellIDs.end(); cellIDIt++)
      {
        int cellID = *cellIDIt;
        // cout << "Refining cell " << cellID << endl;
        //    cout << "Identified cell " << cellID << " as a vertical band.\n";
        // work out what the current aspect ratio is
        hemkerMeshNoCurves->verticesForCell(vertices, cellID);
        // here, we use knowledge of the implementation of the hemker mesh generation:
        // we know that the first edges are always horizontal...
        double xDiff = abs(vertices(1,0)-vertices(0,0));
        double yDiff = abs(vertices(2,1)-vertices(1,1));

        set<GlobalIndexType> cellIDsToRefine;
        cellIDsToRefine.insert(cellID);
        double aspect = yDiff / xDiff;
        while (aspect > 2.0)
        {
          hemkerMeshNoCurves->hRefine(cellIDsToRefine, horizontalCut);

          // the next set of cellIDsToRefine are the children of the ones just refined
          set<GlobalIndexType> childCellIDs;
          for (set<GlobalIndexType>::iterator refinedCellIDIt = cellIDsToRefine.begin();
              refinedCellIDIt != cellIDsToRefine.end(); refinedCellIDIt++)
          {
            int refinedCellID = *refinedCellIDIt;
            set<GlobalIndexType> refinedCellChildren = hemkerMeshNoCurves->getTopology()->getCell(refinedCellID)->getDescendants();
            childCellIDs.insert(refinedCellChildren.begin(),refinedCellChildren.end());
          }

          cellIDsToRefine = childCellIDs;
          aspect /= 2;
        }
      }
      if (enforceOneIrregularity)
        hemkerMeshNoCurves->enforceOneIrregularity();
      // cout << "Done with mesh preprocessing" << endl;
    }

    virtual void setBCs(SpaceTimeIncompressibleFormulationPtr form)
    {
      FunctionPtr zero = Function::zero();

      BCPtr bc = form->solutionUpdate()->bc();
      SpatialFilterPtr initTime = SpatialFilter::matchingT(_tInit);
      SpatialFilterPtr leftX  = SpatialFilter::matchingX(_xLeft);
      SpatialFilterPtr rightX = SpatialFilter::matchingX(_xRight);
      SpatialFilterPtr leftY  = SpatialFilter::matchingY(_yBottom);
      SpatialFilterPtr rightY = SpatialFilter::matchingY(_yTop);
      SpatialFilterPtr nearCylinder = Teuchos::rcp( new NearCylinder(_radius) );
      bc->addDirichlet(form->uhat(1), leftX,    _u1_exact);
      bc->addDirichlet(form->uhat(2), leftX,    _u2_exact);
      // bc->addDirichlet(form->uhat(1), rightX,   _u1_exact);
      // bc->addDirichlet(form->uhat(2), rightX,   _u2_exact);
      // bc->addDirichlet(form->uhat(1), leftY,    _u1_exact);
      // bc->addDirichlet(form->uhat(2), leftY,    _u2_exact);
      // bc->addDirichlet(form->uhat(1), rightY,   _u1_exact);
      // bc->addDirichlet(form->uhat(2), rightY,   _u2_exact);
      bc->addDirichlet(form->uhat(1), nearCylinder, zero);
      bc->addDirichlet(form->uhat(2), nearCylinder, zero);
      if (!_steady)
      {
        bc->addDirichlet(form->tmhat(1),initTime,-_u1_exact);
        bc->addDirichlet(form->tmhat(2),initTime,-_u2_exact);
      }

      // define traction components in terms of field variables
      FunctionPtr n = Function::normal();
      VarPtr sigma11 = form->sigma(1,1);
      VarPtr sigma12 = form->sigma(1,2);
      VarPtr sigma21 = form->sigma(2,1);
      VarPtr sigma22 = form->sigma(2,2);
      VarPtr p = form->p();
      LinearTermPtr t1 = n->x() * (2 * sigma11 - p) + n->y() * (sigma12 + sigma21);
      LinearTermPtr t2 = n->x() * (sigma12 + sigma21) + n->y() * (2 * sigma22 - p);

      _pc = Teuchos::rcp(new PenaltyConstraints);
      _pc->addConstraint(t1==zero, rightX);
      _pc->addConstraint(t2==zero, rightX);
      _pc->addConstraint(t1==zero, leftY);
      _pc->addConstraint(t2==zero, leftY);
      _pc->addConstraint(t1==zero, rightY);
      _pc->addConstraint(t2==zero, rightY);

      form->solutionUpdate()->setFilter(_pc);
    }
};
}
