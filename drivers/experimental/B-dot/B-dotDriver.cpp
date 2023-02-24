//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "ConfigurationSpaceFormulation.h"
#include "ErrorIndicator.h"
#include "ExpFunction.h"
#include "FixedCountRefinementStrategy.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"
#include "VlasovBackwardEulerFormulation.h"

#include <sys/stat.h>
#include <sys/types.h>

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
    auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = phaseSpacePoints.dimension(0);
    const int numPoints = phaseSpacePoints.dimension(1);
    
    const double weight = _n / (sqrt(2. * M_PI) * _sigma);
//    std::cout << "weight: " << weight << std::endl;
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

//! The class we use for interpolating the amplitude of the current in time, and the spectrum in velocity space.
class PointwiseDataFunction1D : public TFunction<double>
{
  int _dimensionOfVariation; // 0 for x, 1 for v, -1 for t (that's for time-marching; for 1D1V space-time, this would be 0 for x, 1 for t, 2 for v)
  FieldContainer<double> _points; // physical points at which the function is known (shape P,D) -- D=1
  FieldContainer<double> _values; // values at those points (shape P)
  
  double _currentCoord = 0.0; // only used when _dimensionOfVariation is negative (i.e., that dimension is not present in the mesh)
public:
  PointwiseDataFunction1D(const vector< double > &coords1D,
                          const vector< double > &functionValues,
                          int dimensionOfVariation)
  :
  _dimensionOfVariation(dimensionOfVariation)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(coords1D.size() != functionValues.size(), std::invalid_argument, "coords length must match values length");
    const int numPoints = coords1D.size();
    const int spaceDim  = 1;
    
    std::vector<double> points;
    std::vector<double> values;
    
    // data files are allowed to have duplicate coords with differing values; we take the average for any duplicated points
    int pointOrdinal=0;
    while (pointOrdinal < numPoints)
    {
      int numIdenticalPoints = 1;
      double valueSum = functionValues[pointOrdinal];
      const double &point = coords1D[pointOrdinal];
      while ((pointOrdinal + 1 < numPoints) && (coords1D[pointOrdinal+1] == point))
      {
        pointOrdinal++;
        numIdenticalPoints++;
        valueSum += functionValues[pointOrdinal];
      }
      points.push_back(point);
      values.push_back(valueSum / numIdenticalPoints);
      pointOrdinal++;
    }
    
    int numUniquePoints = points.size();
    _points = FieldContainer<double>(numUniquePoints,spaceDim);
    _values = FieldContainer<double>(numUniquePoints);
    
    double prevPoint = std::numeric_limits<double>::lowest();
    for (int pointOrdinal=0; pointOrdinal<numUniquePoints; pointOrdinal++)
    {
      const double &point = points[pointOrdinal];
      _points(pointOrdinal,0) = point;
      _values(pointOrdinal)   = values[pointOrdinal];
      TEUCHOS_TEST_FOR_EXCEPTION(point < prevPoint, std::invalid_argument, "points must be in ascending order");
      prevPoint = point;
    }
  }
  
  double interpolatedPointValue(const double coordInDimensionOfVariation)
  {
    const int numPoints = _points.dimension(0);
    if ((coordInDimensionOfVariation < _points(0,0)) || (coordInDimensionOfVariation > _points(numPoints-1,0)))
    {
      // point is outside the bounds of the data provided
      return 0;
    }
    for (int pointOrdinal=numPoints-1; pointOrdinal>=0; pointOrdinal--)
    {
      if (coordInDimensionOfVariation >= _points(pointOrdinal,0))
      {
        // point lies between pointOrdinal and pointOrdinal+1, or (if this is the last point) matches pointOrdinal precisely
        if (pointOrdinal == numPoints-1)
        {
          return _values(pointOrdinal);
        }
        
        const double & leftPoint  = _points(pointOrdinal,0);
        const double & rightPoint = _points(pointOrdinal+1,0);
        
        const double &  leftValue = _values(pointOrdinal);
        const double & rightValue = _values(pointOrdinal+1);
        
        const double binWidth    = rightPoint - leftPoint;
        const double leftWeight  = 1.0 - (coordInDimensionOfVariation - leftPoint)  / binWidth;
        const double rightWeight = 1.0 - (rightPoint - coordInDimensionOfVariation) / binWidth;
        
        return leftWeight * leftValue + rightWeight * rightValue;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: matching bin not found for point");
  }
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    const int numCells  = values.dimension(0);
    const int numPoints = values.dimension(1);
    
    auto physicalPointsFC = basisCache->getPhysicalCubaturePointsFC(); // (C,P,D)
    const int spaceDim = physicalPointsFC.dimension(2);
    
    if (_dimensionOfVariation < 0)
    {
      const double constantValue = interpolatedPointValue(_currentCoord);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          values(cellOrdinal,pointOrdinal) = constantValue;
        }
      }
    }
    else
    {
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          const double &coord = physicalPointsFC(cellOrdinal,pointOrdinal,_dimensionOfVariation);
          
          const double value = interpolatedPointValue(coord);
          values(cellOrdinal,pointOrdinal) = value;
        }
      }
    }
  }
  
  static pair< vector< double >, vector< double > > readFile(const string &filePath)
  {
    using std::ifstream;
    ifstream file;
    
    file.open(filePath);
    
    vector<double> coords;
    vector<double> values;
    
    using std::getline;
    std::string line;
    while ( getline(file, line) )
    {
      vector<double> lineData;
      string valueString;
      std::stringstream lineStream;
      lineStream.str(line);
      while (getline(lineStream, valueString, '\t'))
      {
        lineData.push_back(std::stod(valueString));
      }
      
      TEUCHOS_TEST_FOR_EXCEPTION((lineData.size() == 1) || (lineData.size() > 2), std::invalid_argument, "Each line that has any data should have exactly two tab-delimited numbers: coord and value");
      
      coords.push_back(lineData[0]);
      values.push_back(lineData[1]);
    }
    
    return {coords,values};
  }
  
  //! only supported when dimensionOfVariation < 0.
  void setCurrentCoord(const double &coord)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_dimensionOfVariation >= 0, std::invalid_argument, "setCurrentCoord is only supported when dimension of variation is negative (i.e., mesh-independent)");
    _currentCoord = coord;
  }
};

//! \note This class is more general than our actual data requires.  We leave it here in case it is useful later (in higher-dimensional geometries, for example).
class PointwiseDataFunction : public TFunction<double>
{
  const int _numPointDimensions;
  const vector< vector<double> > _dataTable;
  const vector<int> _columnDimension;
  const set<int> _dimensionsToDrop;
  
  FieldContainer<double> _times;
  
  FieldContainer<double> _points; // physical points at which the function is known (shape P,D)
  FieldContainer<double> _values; // corresponding values at a fixed time (shape (P))
  FieldContainer<double> _spaceTimeValues; // values across all time (shape (T,P))
  FieldContainer<double> _minCoordinateValues; // minimum coordinate values in each dimension (shape D)
  FieldContainer<double> _maxCoordinateValues; // maximum coordinate values in each dimension (shape D)
public:
  FieldContainer<double> valuesForTime(int timeOrdinal)
  {
    const int numPoints = _spaceTimeValues.dimension(1);
    FieldContainer<double> values(numPoints);
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      values(pointOrdinal) = _spaceTimeValues(timeOrdinal, pointOrdinal);
    }
    return values;
  }
  
  void setTime(double t) override // TODO: see if we can eliminate the corresponding method in TFunction (it's not really supported/used)
  {
    double timeBelow = std::numeric_limits<double>::min();
    double timeAbove = std::numeric_limits<double>::max();
    int timeOrdinalBelow = -1;
    int timeOrdinalAbove = -1;
    for (int timeOrdinal=0; timeOrdinal<_times.dimension(0); timeOrdinal++)
    {
      const double & t_entry = _times(timeOrdinal);
      if ( t_entry <= t )
      {
        timeBelow = std::max(t_entry, timeBelow);
        if (timeBelow == t_entry)
        {
          timeOrdinalBelow = timeOrdinal;
        }
      }
      if ( t <= t_entry)
      {
        timeAbove = std::min(t_entry, timeAbove);
        if (timeAbove == t_entry)
        {
          timeOrdinalAbove = timeOrdinal;
        }
      }
    }
    if (timeOrdinalBelow == timeOrdinalAbove)
    {
      _values = valuesForTime(timeOrdinalBelow);
    }
    else if ((timeOrdinalAbove > 0) && (timeOrdinalBelow > 0))
    {
      const double dt = _times(timeOrdinalAbove) - _times(timeOrdinalBelow);
      double upper_weight = (t - _times(timeOrdinalBelow)) / dt;
      double lower_weight = 1.0 - upper_weight;
      
      auto values_lower = valuesForTime(timeOrdinalBelow);
      auto values_upper = valuesForTime(timeOrdinalAbove);
      
      const int numPoints = values_lower.dimension(0);
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        _values(pointOrdinal) = lower_weight * values_lower(pointOrdinal) + upper_weight * values_upper(pointOrdinal);
      }
    }
    else if (timeOrdinalBelow > 0)
    {
      _values = valuesForTime(timeOrdinalBelow);
    }
    else // timeOrdinalAbove > 0
    {
      _values = valuesForTime(timeOrdinalAbove);
    }
  }
  
  //! first dimension in dataTable is the time.
  //! columnDimension: which spatial dimension the column corresponds to (zero-based; -1 for a non-mesh dimension, as time for a time-marching scheme, and -2 for a value dimension)
  //! dimensionsToDrop: during Function::values, ignore the listed dimensions — e.g., if {0} is provided here, then the first point dimension will be ignored, and the "y" coordinates map to the "x" coordinates in the provided table.
  PointwiseDataFunction(const vector< vector<double> > &dataTable)
  :
  _numPointDimensions(1), // 1V is our present use case
  _dataTable(dataTable),
  _dimensionsToDrop(set<int>{0}) // drop the x dimension
  {
    // extract unique times and points:
    set<double> times;
    set< vector<double> > points;
    for (const auto & entry : dataTable)
    {
      times.insert(entry[0]);
      vector<double> point(_numPointDimensions);
      for (int d=0; d<_numPointDimensions; d++)
      {
        point[d] = entry[d+1];
      }
      points.insert(point);
    }
    
    map<double,int> timeOrdinalLookup;
    map<vector<double>, int> pointOrdinalLookup;
    
    const int numPoints = points.size();
    _points = FieldContainer<double>(numPoints, _numPointDimensions);
    
    int pointOrdinal = 0;
    for (const auto &point : points)
    {
      for (int d=0; d<point.size(); d++)
      {
        _points(pointOrdinal,d) = point[d];
      }
      pointOrdinalLookup[point] = pointOrdinal;
      pointOrdinal++;
    }
    
    const int numTimes = times.size();
    _times = FieldContainer<double>(numTimes);
    int timeOrdinal = 0;
    for (const auto &time : times)
    {
      _times(timeOrdinal) = time;
      timeOrdinalLookup[time] = timeOrdinal;
      timeOrdinal++;
    }
    
    _values          = FieldContainer<double>(numPoints); // unset until setTime() is called
    _spaceTimeValues = FieldContainer<double>(numTimes,numPoints);
    
    _minCoordinateValues = FieldContainer<double>(_numPointDimensions);
    _maxCoordinateValues = FieldContainer<double>(_numPointDimensions);
    
    _minCoordinateValues.initialize(std::numeric_limits<double>::max());
    _maxCoordinateValues.initialize(std::numeric_limits<double>::min());
    
    for (const auto & entry : dataTable)
    {
      const double time = entry[0];
      vector<double> point(_numPointDimensions);
      for (int d=0; d<_numPointDimensions; d++)
      {
        point[d] = entry[d+1];
        
        _minCoordinateValues(d) = std::min(_minCoordinateValues(d), point[d]);
        _maxCoordinateValues(d) = std::max(_maxCoordinateValues(d), point[d]);
      }
      
      const int timeOrdinal = timeOrdinalLookup[time];
      const int pointOrdinal = pointOrdinalLookup[point];
      
      const double value = entry[_numPointDimensions + 1]; // value is the final column
      _spaceTimeValues(timeOrdinal, pointOrdinal) = value;
    }
  }
  
  double interpolatedPointValue(vector<double> point)
  {
    // we assume our points are in a grid -- i.e. if (x1,y1) and (x2,y2) are among our points, so are (x1,y2) and (x2,y1)
    // we linearly interpolate, weighting point values in the bounding box by the distance to the intended point.
    // if the point is outside the convex hull of points whose values we know, we return 0.
    const int numPoints = _points.dimension(0);
    const int spaceDim = _minCoordinateValues.dimension(0);
    vector<double> leastUpperBounds(spaceDim, std::numeric_limits<double>::max());
    vector<double> greatestLowerBounds(spaceDim, std::numeric_limits<double>::min());
    
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        if (_points(pointOrdinal,d) <= point[d])
        {
          greatestLowerBounds[d] = std::max(greatestLowerBounds[d], _points(pointOrdinal,d));
        }
        if (_points(pointOrdinal,d) >= point[d])
        {
          leastUpperBounds[d] = std::min(leastUpperBounds[d], _points(pointOrdinal,d));
        }
      }
    }
    
    // check if the point requested is outside the convex hull, and if so, return 0:
    for (int d=0; d<spaceDim; d++)
    {
      if (leastUpperBounds[d] == std::numeric_limits<double>::max())
      {
        return 0;
      }
      if (greatestLowerBounds[d] == std::numeric_limits<double>::min())
      {
        return 0;
      }
    }
    
    // find the bounding box
    vector<int> boundingBox;
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      bool pointMatches = true;
      for (int d=0; d<spaceDim; d++)
      {
        const double & pd = _points(pointOrdinal,d);
        if ((pd != leastUpperBounds[d]) && (pd != greatestLowerBounds[d]))
        {
          pointMatches = false;
          break;
        }
      }
      if (pointMatches)
      {
        boundingBox.push_back(pointOrdinal);
      }
    }
    
    // boundingBox may have as few points as 1 (if the point requested is among our points), but it should not have more than 2^d points.
    const int maxPointsExpected = 1 << spaceDim;
    TEUCHOS_TEST_FOR_EXCEPTION(boundingBox.size() > maxPointsExpected, std::invalid_argument, "boundingBox has more points than we expect!");
    TEUCHOS_TEST_FOR_EXCEPTION(boundingBox.size() == 0, std::invalid_argument, "boundingBox does not have any points!");
    
    double value = 0;
    vector<double> weights_lower(spaceDim);
    vector<double> weights_upper(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      double scale = leastUpperBounds[d] - greatestLowerBounds[d];
      double d_lower = point[d] - greatestLowerBounds[d];
      double d_upper = leastUpperBounds[d] - point[d];
      weights_lower[d] = d_lower / scale;
      weights_upper[d] = d_upper / scale;
    }
    
    for (int boundingPointOrdinal=0; boundingPointOrdinal<boundingBox.size(); boundingPointOrdinal++)
    {
      double weight = 1.0;
      for (int d=0; d<spaceDim; d++)
      {
        const bool isLower = greatestLowerBounds[d] == _points(boundingBox[boundingPointOrdinal],d);
        double weight_d = isLower ? weights_lower[d] : weights_upper[d];
        weight *= weight_d;
      }
      value += _values(boundingBox[boundingPointOrdinal]) * weight;
    }
    
    return value;
  }
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    const int numCells  = values.dimension(0);
    const int numPoints = values.dimension(1);
    
    auto physicalPointsFC = basisCache->getPhysicalCubaturePointsFC(); // (C,P,D)
    const int spaceDim = physicalPointsFC.dimension(2);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        vector<double> point;
        for (int d=0; d<spaceDim; d++)
        {
          if (_dimensionsToDrop.find(d) == _dimensionsToDrop.end())
          {
            point.push_back(physicalPointsFC(cellOrdinal,pointOrdinal,d));
          }
        }
        const double value = interpolatedPointValue(point);
        values(cellOrdinal,pointOrdinal) = value;
      }
    }
  }
  
  static vector< vector<double> > readFile(const string &filePath)
  {
    using std::ifstream;
    ifstream file;
    
    file.open(filePath);
    
    vector<vector<double> > dataTable;
    
    using std::getline;
    std::string line;
    while ( getline(file, line) )
    {
      vector<double> lineData;
      string valueString;
      std::stringstream lineStream;
      lineStream.str(line);
      while (getline(lineStream, valueString, '\t'))
      {
        lineData.push_back(std::stod(valueString));
      }
      
      dataTable.push_back(lineData);
    }
    
    return dataTable;
  }
};

static const std::string S_B_DOT = "B-dot";

enum ProblemChoice
{
  B_DOT
};

RefinementStrategyPtr constructRefStrategy(VlasovBackwardEulerFormulation &form, double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator)
{
  auto vlasovSoln  = form.vlasovSolution();
  auto refStrategy = RefinementStrategy::energyErrorRefinementStrategy(vlasovSoln, refinementThreshold);

  refStrategy->setMaxElements(maxElements);
  refStrategy->setMaximumRefinementLevel(maxRefinementLevel);
  refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
  const int H1Order = vlasovSoln->mesh()->globalDofAssignment()->getInitialH1Order()[0];
  refStrategy->setMaxP(H1Order); // disallow p refinements (these would require revision of VelocitySpaceExtrusion)
  return refStrategy;
}

void performRefinements(double & lastRefinementMaxElementError, VlasovBackwardEulerFormulation &form, RefinementStrategyPtr refStrategy, RefinementStrategyPtr unrefinementStrategy,
                        int maxElements)
{
  if (refStrategy == Teuchos::null) return;
  
  const bool doUnrefinements = (maxElements < std::numeric_limits<int>::max());
  if (doUnrefinements)
  {
    // for now, print refinement info to console on rank 0.
    const bool printToConsole = form.vlasovMesh()->Comm()->MyPID() == 0;
    unrefinementStrategy->setRelativeErrorThreshold(1.1); // ensure that we only do unrefinements
    int maxUnrefinementSteps = 15; // In case we end up unable to unrefine to maxElements for some reason, cap the number of unrefinements
    int numUnrefinementStepsTaken = 0;
    int unrefinementTarget = maxElements * .90;
    unrefinementStrategy->setMaxElements(unrefinementTarget);
    while ((form.vlasovMesh()->numActiveElements() > unrefinementTarget) && (numUnrefinementStepsTaken < maxUnrefinementSteps))
    {
      // update solution to ensure that the errors measured are up to date post-refinement
      if (form.gmgSolver() != Teuchos::null)
      {
        form.vlasovSolution()->solve(form.gmgSolver());
      }
      else
      {
        form.vlasovSolution()->solve();
      }
      unrefinementStrategy->refineAndUnrefine(doUnrefinements, printToConsole);
      form.reconstructGMGSolver();
      numUnrefinementStepsTaken++;
    }
        
    if (numUnrefinementStepsTaken > 0)
    {
      // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
      form.configurationSpaceMesh()->repartitionAndRebuild();
      
      // update solution to ensure that the errors measured are up to date post-refinement
      if (form.gmgSolver() != Teuchos::null)
      {
        form.vlasovSolution()->solve(form.gmgSolver());
      }
      else
      {
        form.vlasovSolution()->solve();
      }
    }
  }
  
  // for now, print refinement info to console on rank 0.
  const bool printToConsole = form.vlasovMesh()->Comm()->MyPID() == 0;
  refStrategy->refineAndUnrefine(false, printToConsole); // false: don't unrefine
  
  // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
  form.configurationSpaceMesh()->repartitionAndRebuild();
  
  // also reconstruct GMG solver, if one is in use:
  form.reconstructGMGSolver();
}

//! Sets a homogeneous BC at the x0 point, and parameter-based BC at the x1 point.  Returns a pointer to the parameter function; the caller may then set it to an appropriate value before (any/each) Poisson solve.
ParameterFunctionPtr setPoissonBC(VlasovBackwardEulerFormulation &form, const double &x0, const double &x1)
{
  // we have periodic BCs; we just need to pin the potential at a point -- we choose the x0 (left) point.
  const int poissonTagLeft  = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x0));
  const int poissonTagRight = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x1));
  
  auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
  // homogeneous on left:
  form.addConfigurationSpaceBC(poissonTagLeft, VE_hat, Function::constant(0.0));
  
  // parameter on right:
  ParameterFunctionPtr rightBCFunction = ParameterFunction::parameterFunction(0.0);
  form.addConfigurationSpaceBC(poissonTagRight, VE_hat, rightBCFunction);
  
  return rightBCFunction;
}

class BackwardEulerCircuitSolver {
  double _dt;
  double _R, _L, _C;
  double _A; // area
  double _Ekl, _Ekp1l; // left values for E^k, E^{k+1}
  double _Ekr, _Ekp1r; // right values for E^k, E^{k+1}
  double _Ik;
  double _Iem;
  double _Vk;
  bool   _printToConsole;
public:
  BackwardEulerCircuitSolver(double R, double L, double C, double dt, double A, bool printToConsole)
  :
  _dt(dt),
  _R(R),
  _L(L),
  _C(C),
  _A(A),
  _printToConsole(printToConsole)
  {}
  
  //! set prior-time electric field values
  void setEk(double valueLeft, double valueRight)
  {
    _Ekl = valueLeft;
    _Ekr = valueRight;
  }

  //! set current-time electric field values
  void setEkp1(double valueLeft, double valueRight)
  {
    _Ekp1l = valueLeft;
    _Ekp1r = valueRight;
  }
  
  //! set prior-time current value
  void setIk(double value)
  {
    _Ik = value;
  }
  
  //! set the emission current
  void setIem(double value)
  {
    _Iem = value;
  }
  
  void setVk(double value)
  {
    _Vk = value;
  }
  
  double voltage()
  {
    // returns V_{k+1}
    
    // our backward-Euler circuit solve has formula:
    // V^{k+1} = (CL V^k - dt I^k + dt I_drive^{k+1}) / (CL + RC dt + dt^2)
    // where:
    // - V^k is the previous voltage value
    // - I^k is the previous current value
    // - I_drive is defined as I_em(x1) + dE/dt(x1)
    // - I_em is the emitted current -- the one computed as a moment of f
    // - dE/dt is the time derivative of the electric field, which we approximate by (E^{k+1} - E^k) / dt
    
    const double permittivity = 8.85418782e-12; // permittivity of free space
    
    const double dE_dt       = (_Ekp1r - _Ekr) / _dt - (_Ekp1l - _Ekl) / _dt;
    const double I_drive     = _Iem + _A * permittivity * dE_dt;
    const double numerator   = _C * _L * _Vk - _dt * _Ik + _dt * I_drive;
    const double denominator = _C * _L + _R * _C * _dt + _dt * _dt / _L;
    
    if (_printToConsole)
    {
      std::cout << "Circuit solve, inputs:\n";
      std::cout << "E_{k+1}_l                = " << _Ekp1l << std::endl;
      std::cout << "E_{k}_l                  = " << _Ekl   << std::endl;
      std::cout << "E_{k+1}_r                = " << _Ekp1r << std::endl;
      std::cout << "E_{k}_r                  = " << _Ekr   << std::endl;
      std::cout << "A * permittivity * dE_dt = " << _A * permittivity * dE_dt << std::endl;
      std::cout << "I_{em}                   = " << _Iem  << std::endl;
      std::cout << "I_k                      = " << _Ik   << std::endl;
      std::cout << "V_k                      = " << _Vk   << std::endl;
      
      std::cout << "Solution: V_{k+1} = " << numerator / denominator << std::endl;
    }
    
    return numerator / denominator;
  }
};

//! MPI-collective method for 1D meshes.  Evaluates f_soln on whatever rank owns the point x, and communicates the result to every rank.  Returns the value.
double getSolutionFunctionValue(MeshPtr solnMesh, FunctionPtr f_soln, const double x)
{
  FieldContainer<double> points(1,1);
  points(0,0) = x;
  
  auto cellIDs = solnMesh->cellIDsForPoints(points);
  
  double value = 0.0;
  if (cellIDs.size() > 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != 1, std::invalid_argument, "Found more than one cellID for point x.");
    const auto cellID = cellIDs[0];
    
    if (solnMesh->myCellsInclude(cellID))
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(solnMesh, cellID);
      
      if (f_soln->boundaryValueOnly())
      {
        // i.e., a trace
        // 1D trace: sides are just points.  Does x match one of the sides?
        const auto & nodes = basisCache->getPhysicalCellNodes();
        double leftNode  = nodes(0,0,0); // (C,N,D)
        double rightNode = nodes(0,1,0);
        BasisCachePtr sideCache;
        if (std::abs(x-leftNode) < 1e-14)
        {
          sideCache = basisCache->getSideBasisCache(0);
        }
        else if (std::abs(x-rightNode) < 1e-14)
        {
          sideCache = basisCache->getSideBasisCache(1);
        }
        else
        {
          std::cout << "ERROR: x is not a mesh vertex, but f_soln is a trace solution.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found");
        }
        
        FieldContainer<double> fValue(1,1); // (C,P)
        
        f_soln->values(fValue, sideCache);
        
        value = fValue(0,0);
      }
      else
      {
        // volume variable
        FieldContainer<double> refPoint(1,1,1);
        FieldContainer<double> physPoint(1,1,1);
        physPoint(0,0,0) = x;
        
        int cubatureDegree = 1; // not curvilinear…
        
        CamelliaCellTools::mapToReferenceFrame( refPoint, physPoint, solnMesh->getTopology(), cellID, cubatureDegree);
        
        refPoint.resize(1,1);
        
        basisCache->setRefCellPointsFC(refPoint);
        
        FieldContainer<double> fValue(1,1); // (C,P)
        
        f_soln->values(fValue, basisCache);
        
        value = fValue(0,0);
      }
    }
  }
  
  // all but the MPI-owner have a zero value, so we can sum to communicate the value to all ranks.
  value = MPIWrapper::sum(value);
  return value;
}

double inflowErrorOnBoundary(VlasovBackwardEulerFormulation &form, FunctionPtr f_in, const int bcTag)
{
  FunctionPtr f_soln = Function::solution(form.f(), form.vlasovSolution());
  
  const std::string & tagName = DIRICHLET_SET_TAG_NAME;
  FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
  
  FunctionPtr bc_err = (f_in - f_soln) * bcRestriction;
  int quadratureEnrichment = 4;
  
  double bc_norm = (f_in * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
  double relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
  return relErr;
}

void setUseGMG(VlasovBackwardEulerFormulation &form, int maxDofsForDirectSolve, bool &previousSolveUsedGeometricMultigrid, bool &previousSolveUsedDirect, bool logChangeToConsole)
{
  auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
  if (numVlasovDofs > maxDofsForDirectSolve)
  {
    if (logChangeToConsole && (!previousSolveUsedGeometricMultigrid))
    {
      std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
    }
    form.setUseGeometricMultigrid(true);
    previousSolveUsedGeometricMultigrid = true;
    previousSolveUsedDirect = false;
  }
  else
  {
    if (logChangeToConsole && (previousSolveUsedGeometricMultigrid))
    {
      std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
    }
    form.setUseGeometricMultigrid(false);
    previousSolveUsedGeometricMultigrid = false;
    previousSolveUsedDirect = true;
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  const int rank = mpiSession.getRank();
  
  // block off variables so that everything gets destructed before the finalize() call at the end.
  {
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    
    mpiSession.barrier();
    
    double q      = -1.0;
    double eps0   =  1.0; // overridden below
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  1.0; // non-dimensionalized in some appropriate way
    
    double R = 0.8;      // Omega
    double L = 6.88e-9;  // 6.88 nH
    double C = 24.0e-12; // pF
    
    // v can range from 0 to 1; we pad the domain to extend from -0.5 to 1.5.
    const double DEFAULT_V0     = -0.50;
    const double DEFAULT_VWIDTH =  2.00;
    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    bool enforceQuasiNeutrality = false;
    
    bool useAdaptivity = false;
    int maxElements = 64 * 64; // do unrefinements to keep total mesh size smaller than this
    
    int xElements = 16;
    int vElements = 20;
    int numUniformRefinements = 2;
    int delta_k   = 2;
    double dt_max = -1.0;
    int H1Order = 3; // this gives f, E (which are in L^2) a polynomial order of 2
    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    int numTimeSteps = 80;
    int outputFrequency = 1; // number of time steps to take before outputting viz.
    const ProblemChoice problemChoice = B_DOT;
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool useEnergyForUnrefinements = false;
    bool saveViz = true;
    std::string outputDir = "/private/viz";
    int solutionOutputFrequency = 1;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    int maxDofsForDirectSolve = 20000;
    bool useSerendipityBasis = false;
    bool disableCircuitSolve = false; // a debugging tool
    double overrideEps0 = -1.; // if set from CLI, will override the computed eps0 value.
    double quantumEfficiency = 0.01; // value is between 0.01 and 0.05; still need to find out the correct value from the experiment.
    double scalingFactorForInflow = 1.0; // for debugging experiments
    
    double physicalGapWidth = .01; // for now, use a 10 [mm] = 0.01 [m] gap.  Likely due to other errors (currentScaling?), right now we are not able to run to completion with a 1 [mm] gap.  (For even small time steps, we end up with E and V in the circuit solve (the RHS of gap) changing sign in every time step, and growing in magnitude; after ~25 steps, we devolve into numerical noise because the E field has overwhelmed the flow.  But when we run with the 10 [mm] gap, the circuit solve behaves more reasonably, with values in one time step close to the values in the next.)
    
    std::string resultsFilePrefix = "";
    
    std::string resultsFile = "";
    std::string inflowVelocityDistributionFile = "/Users/nvrober/fluid-kinetic/data/bdot/rescaled/v-f.dat";   // tab-delimited, columns: velocity, f
    std::string inflowTemporalDistributionFile = "/Users/nvrober/fluid-kinetic/data/bdot/rescaled/t-amp.dat"; // tab-delimited, columns: time, amplitude
    std::string circuitResultsFile = ""; // file with voltage, electric field, current for each time step
    
    cmdp.setOption("resultsFile", &resultsFile);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("dt", &dt_max );
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("maximumRefinementLevel", &maximumRefinementLevel);
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("overrideEps0", &overrideEps0);
    cmdp.setOption("quantumEfficiency", &quantumEfficiency);
    cmdp.setOption("physicalGapWidth", &physicalGapWidth);
    cmdp.setOption("disableCircuitSolve", "enableCircuitSolve", &disableCircuitSolve);
    cmdp.setOption("scalingFactorForInflow", &scalingFactorForInflow);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
  #ifdef HAVE_MPI
      MPI_Finalize();
  #endif
      return -1;
    }
    
    double velocityScaling = 114012000.0; // consider: change to c
    // scale so that our domain width is 1.0
    double spatialScaling  = physicalGapWidth;
    double timeScaling     = spatialScaling / velocityScaling; // 8.77e-12
    
    // finalTime is used in scaling the time ramp function's input time (the rescaled, shifted data file goes from time 0 to time 1;
    // it omits the first 1.5e-9 [s] of data, which is all 0's; so instead of going for 8.65e-9 [s] it goes for 7.15e-9 [s] in physical
    // units).
    const double finalTime = 7.15e-09 / timeScaling; // ~ 815
    
    double EScaling        = 1.;
    double n0              = 1.;
    double potentialScaling = 1.;
    double currentScaling   = 1.;
    
    double A;
    
    // determine scalings
    {
      const double e   = 1.602176634e-19; // electron's electric charge in Coulombs
      const double m_e =  9.10938356e-31;  // m_e, mass of electron in kg
      
      const double q0  = e;
      const double m0  = m_e;
      
      EScaling         = m0 * velocityScaling / (q0 * timeScaling);
      potentialScaling = EScaling * spatialScaling;
      
      xWidth = physicalGapWidth / spatialScaling; // 1 [mm] = .001 [m] gap
      
      const double permittivity = 8.85418782e-12; // permittivity of free space
      
      // (the J-based determination is what we did in cold diode problem)
//      const double J  = ??; // current density [A/m^2]
//      nScaling        = J / (e * velocityScaling); // number density of plasma
      
      const double sourcePhotons = 2.0299157e+19;
      // f at inflow is given in terms of an amplitude scaled so that the integral over the whole time domain (in physical units) is 1.
      // (this is therefore scaled by 1/timeScaling)
      const double r_e = .02;  // emission radius (2cm)
      const double r_f = .3262; // fielding distance (326.2 mm)
      
      A = M_PI * r_e * r_e; // area at emission surface
      const double surfacePhotons = sourcePhotons / 4 * std::log(1. + r_e * r_e / (r_f * r_f));
      n0 = surfacePhotons * quantumEfficiency;
      
      currentScaling = n0 * e / timeScaling; // TODO: confirm this scaling is correct
      
      const double omega_p = q0 * sqrt( n0 / (m0 * permittivity));
//      std::cout << "omega_p = sqrt(n0) * " << q0 * sqrt( 1.0 / (m0 * permittivity)) << std::endl;
      
      // eps0 is a weight that enters PoissonFormulation; we use this to scale the problem:
      eps0 = 1.0 / (omega_p * omega_p * timeScaling * timeScaling);
      
      if (overrideEps0 != -1.0)
      {
        if (rank == 0)
        {
          std::cout << "NOTE: overriding eps0 (which was " << eps0 << ") with value " << overrideEps0 << std::endl;
        }
        eps0 = overrideEps0;
      }
      
      
      if (rank == 0)
      {
        std::cout << "Characteristic E-field strength E_0: " << EScaling << std::endl;
        std::cout << "physical gap width:                  " << physicalGapWidth << std::endl;
        std::cout << "quantum efficiency:                  " << quantumEfficiency << std::endl;
        std::cout << "spatial scaling:                     " << spatialScaling << std::endl;
        std::cout << "temporal scaling:                    " << timeScaling << std::endl;
        std::cout << "scaled final time:                   " << finalTime << std::endl;
        std::cout << "domain width:                        " << xWidth << std::endl;
        std::cout << "n0:                                  " << n0 << std::endl;
        std::cout << "current scaling:                     " << currentScaling << std::endl;
        std::cout << "eps0:                                " << eps0 << std::endl;
      }
      enforceQuasiNeutrality = false;
    }
    
    if (vWidth == std::numeric_limits<double>::infinity())
    {
      vWidth = DEFAULT_VWIDTH;
    }
    if (v0 == std::numeric_limits<double>::infinity())
    {
      v0 = DEFAULT_V0;
    }
    
    bool usePeriodicBCs = false;
    
    if (dt_max < 0)
    {
      dt_max = .005;
    }
    
    double x1 = x0 + xWidth;
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x1, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    
    double dt = dt_max;
    int maxMomentNumber = 1; // we want to compute current for circuit solve.
    
    VlasovBackwardEulerFormulation form(q, m, useSerendipityBasis, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k, maxMomentNumber);
//    poissonConfig->configurationSpaceMesh()->bilinearForm()->printTrialTestInteractions();
    form.refineUniformly(numUniformRefinements);
    form.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    
    auto voltageParameter = setPoissonBC(form, x0 + xWidth, x0);
    
    auto moments_k   = form.solutionMomentsPriorTime(); // we need to call update() on this after each time step
    auto moments_kp1 = form.solutionMoments();
    
    BackwardEulerCircuitSolver circuitSolver(R,L,C,dt*timeScaling,A,(rank==0)); // print to console on rank 0
    
    FunctionPtr I_drive_kp1; // set up below
    ParameterFunctionPtr I_drive_kp1_at_x1 = ParameterFunction::parameterFunction(0.0);
    
    FunctionPtr E_k, E_kp1;
    FunctionPtr I_k, I_kp1;
    
    // voltageParameter is a "ParameterFunction" -- we need fill it in with an appropriate value
    {
      VarPtr E = poissonConfig->E_hat(0); // 0: x component
      
      bool weightFluxesBySideParity = false;
      E_k   = Function::solution(E, poissonConfig->solutionPriorTime(), weightFluxesBySideParity) * EScaling;
      E_kp1 = Function::solution(E, poissonConfig->solution(),          weightFluxesBySideParity) * EScaling;
      I_k   = moments_k->solutionMomentFunction(1) * currentScaling;
      I_kp1 = moments_kp1->solutionMomentFunction(1) * currentScaling;
    }
    
    // prior-time voltage: start at 0.
    double V_k = 0.0;
    auto configMesh = poissonConfig->configurationSpaceMesh();
    
    auto timeRampTable      = PointwiseDataFunction1D::readFile(inflowTemporalDistributionFile);
    auto vDistributionTable = PointwiseDataFunction1D::readFile(inflowVelocityDistributionFile);
    
    Teuchos::RCP<PointwiseDataFunction1D> timeRampFunction      = Teuchos::rcp( new PointwiseDataFunction1D(timeRampTable.first, timeRampTable.second, -1) );          // -1: not a mesh coordinate
    Teuchos::RCP<PointwiseDataFunction1D> vDistributionFunction = Teuchos::rcp( new PointwiseDataFunction1D(vDistributionTable.first, vDistributionTable.second, 1) ); //  1: the second coordinate dimension ("y")
    
    const double vDistributionIntegral = vDistributionFunction->integrate(form.vlasovMesh()) / xWidth;
    if (rank == 0)
    {
      std::cout << "vDistributionIntegral = " << vDistributionIntegral << std::endl;
    }
    
    FunctionPtr f_in = scalingFactorForInflow * FunctionPtr(timeRampFunction) * FunctionPtr(vDistributionFunction); // note that we need to update timeRampFunction with the current time in each time step
        
    ErrorIndicatorPtr unrefinementErrorIndicator;
    if (!useEnergyForUnrefinements)
    {
      unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
    }
    double refinementThreshold = 1.1; // > 1 ==> no refinements will take place
    RefinementStrategyPtr unrefinementStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
    
    ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
    const int numElementsToRefine = std::max(int(.05 / 3. * maxElements), 1);
    auto fixedCountRefStrategy = Teuchos::rcp( new FixedCountRefinementStrategy(errorIndicator, numElementsToRefine) );
    
    form.setFixedPointMaxSubsteps(maxSubsteps);
    form.setFixedPointMinSubsteps(minSubsteps);
    form.setFixedPointTolerance(fixedPointTol);
    
    form.setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
    form.setIterativeSolveTolerance(iterativeSolveTolerance);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if (maxSubsteps > 3)
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 100.0;
      cgTols[1] = iterativeSolveTolerance * 10.0;
      form.setIterativeSolveTolerances(cgTols);
    }
    form.setIterativeSolveMaxIterations(iterativeSolveMaxIters);
    form.setFixedPointTolerance(fixedPointTol);
    
    FunctionPtr f0 = Function::zero();  // vacuum initial conditions
    
    using std::vector;
    using std::pair;
    
    form.setInitialAndInflowConditions(f0, f_in);
    
    std::ostringstream suffix;
    if (useAdaptivity)
    {
      suffix << "_budget" << maxElements;
    }
    suffix << "_dt" << dt_max;
    suffix << "_H1Order" << H1Order;
    suffix << "_delta_k" << delta_k;
    suffix << "_qeff" << quantumEfficiency;
    suffix << "_gap" << physicalGapWidth;
    
    if (scalingFactorForInflow != 1.0)
    {
      suffix << "_scaled" << scalingFactorForInflow;
    }

    // adjust x and v element counts according to the number of uniform refinements requested.
    xElements <<= numUniformRefinements;
    vElements <<= numUniformRefinements;

    suffix << "_" << xElements << "x" << vElements;

    std::string problemPrefix;
    if (problemChoice == B_DOT)
    {
      problemPrefix = "B-dot";
    }

    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix << "Vlasov" << suffix.str();

    std::ostringstream poissonSS;
    poissonSS << problemPrefix << "Poisson" << suffix.str();
    
    if (resultsFile == "")
    {
      std::ostringstream resultsFileStream;
      resultsFileStream << outputDir << "/";
      if (resultsFilePrefix == "")
      {
        resultsFileStream << "results_" << problemPrefix << "_" << suffix.str();
      }
      else
      {
        resultsFileStream << resultsFilePrefix;
      }
      
      resultsFileStream << "_" << numTimeSteps << "_timeSteps";
      resultsFileStream << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    if (circuitResultsFile == "")
    {
      std::ostringstream circuitResultsFileNameStream;
      circuitResultsFileNameStream << outputDir << "/";
      if (resultsFilePrefix == "")
      {
        circuitResultsFileNameStream << "circuit_" << problemPrefix << "_" << suffix.str();
      }
      else
      {
        circuitResultsFileNameStream << resultsFilePrefix;
      }
      
      circuitResultsFileNameStream << "_" << numTimeSteps << "_timeSteps";
      circuitResultsFileNameStream << ".dat";
      circuitResultsFile = circuitResultsFileNameStream.str();
    }
    
    auto getSpatialMeshFilePath = [&]() {
      std::ostringstream spatialMeshInfoPathStringStream;
      spatialMeshInfoPathStringStream << outputDir << "/";
      spatialMeshInfoPathStringStream << "mesh1D_" << problemPrefix << suffix.str();
      
      spatialMeshInfoPathStringStream << numTimeSteps << "_timeSteps";
      spatialMeshInfoPathStringStream << ".dat";
      
      return spatialMeshInfoPathStringStream.str();
    };
    
    auto getTimeStepReportFilePath = [&]() {
      std::ostringstream timeStepReportPathStringStream;
      timeStepReportPathStringStream << outputDir << "/";
      timeStepReportPathStringStream << "timeStepReport_" << problemPrefix << suffix.str();
      
      timeStepReportPathStringStream << numTimeSteps << "_timeSteps";
      timeStepReportPathStringStream << ".dat";
      
      return timeStepReportPathStringStream.str();
    };
    
    auto refReportFilePath = getTimeStepReportFilePath();
    const std::string directoryPath = outputDir;
    
    std::ofstream refReportFileStream(refReportFilePath, std::ios::out);
    
    refReportFileStream << "Vlasov Elements\tVlasov Dofs\tPoisson Elements\tPoisson Dofs\tEnergy Error\n";
    refReportFileStream.close();
    
    auto writeTimeStepReport = [&](const int numElementsVlasov, const int numDofsVlasov, const int numElementsPoisson, const int numDofsPoisson, const double energyError) {
      auto refReportFilePath = getTimeStepReportFilePath();
      std::ofstream refReportFileStream(refReportFilePath, std::ios_base::app);
      refReportFileStream << numElementsVlasov << "\t" << numDofsVlasov << "\t" << numElementsPoisson << "\t" << numDofsPoisson << "\t" << energyError << std::endl;
      refReportFileStream.close();
    };
    
    auto reportTimeStepComplete = [&](const int &timeStepNumber, const double &energyError) {
      auto numVlasovDofs       = form.vlasovMesh()->numGlobalDofs();
      auto vlasovElementCount  = form.vlasovMesh()->numActiveElements();
      auto poissonElementCount = form.configurationSpaceMesh()->numActiveElements();
      auto poissonDofCount     = form.configurationSpaceMesh()->numGlobalDofs();
      
      if (rank == 0)
      {
        writeTimeStepReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, energyError);
        std::cout << "Time step " << timeStepNumber << " complete.\n";
      }
    };
    
    if (resultsFile == "")
    {
      std::ostringstream resultsFileStream;
      resultsFileStream << directoryPath << "/" << problemPrefix << "_results" << suffix.str() << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
    if (saveViz)
    {
      vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovSS.str(), directoryPath));
      poissonViz = Teuchos::rcp(new HDF5Exporter (form.configurationSpaceMesh(), poissonSS.str(), directoryPath) );
    }
    
    double t = 0.0;
    timeRampFunction->setCurrentCoord(t / finalTime); // time ramp table goes from t=0 to t=1
    if (saveViz)
    {
      const int num1DPointsPoisson = 10; // default is 4
      poissonViz->exportSolution(form.configurationSpaceSolution(),t,num1DPointsPoisson);
      vlasovViz->exportSolution(form.vlasovSolution(),t);
    }
    
    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(0), form.configurationSpaceSolution());
            
    auto solutionMoments = form.solutionMoments();
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1); // aka current
    auto velocity = momentum / density;
            
    bool previousSolveUsedGeometricMultigrid = false;
    bool previousSolveUsedDirect = false;
    
    double lastRefinementMaxElementError = 0.0;
    
    std::ofstream circuitResultsFileStream(circuitResultsFile, std::ios::out);
    circuitResultsFileStream << "t\tV\tE\tI\n"; // time, voltage, E-field, current
    
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
      
      {
        const double amplitude = timeRampFunction->interpolatedPointValue(t / finalTime); // time ramp table goes from t=0 to t=1
        if (rank == 0)
        {
          std::cout << "for time " << t << ", amplitude of f_em is " << amplitude << std::endl;
        }
      }
      
      form.startTimeStep();
      
      if (useAdaptivity)
      {
        int maxRefinementSteps = 1; // just do 1 pass on every time step after the first
        int minElements = 0;   // don't gate on the element count except for the first time step
        if (timeStepNumber == 1)
        {
          maxRefinementSteps = 15; // a somewhat arbitary number; aim for this to be enough to reach minElements
          minElements = maxElements * .90; // in first time step, refine until we get within 10% of the maximum
          if (rank == 0)
          {
            std::cout << "\n\nAdapting initial mesh to meet budget of " << maxElements << " elements.\n";
          }
        }
        for (int refNumber=0; refNumber<maxRefinementSteps; refNumber++)
        {
          auto gmgSolver = form.gmgSolver();
          if (gmgSolver != Teuchos::null)
          {
            form.vlasovSolution()->solve(gmgSolver);
          }
          else
          {
            form.vlasovSolution()->solve();
          }
          form.configurationSpaceSolve();
                    
          performRefinements(lastRefinementMaxElementError, form, fixedCountRefStrategy, unrefinementStrategy, maxElements);
          
          setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
          
          if (form.vlasovMesh()->numActiveElements() >= minElements)
          {
            break;
          }
        }
      }
      form.finishTimeStep();
      
      t += dt;
      
      {
        // compute, report BC error for inflow:
        const double relErr = inflowErrorOnBoundary(form, f_in, form.tagVlasovSpatialInflow());
        if (rank == 0)
        {
          std::cout << "Relative L^2 error in f compared to f_in at inflow: " << relErr << std::endl;
        }
      }
      
      timeRampFunction->setCurrentCoord(t / finalTime); // time ramp table goes from t=0 to t=1
      
      moments_kp1->update();
      if (rank == 0) std::cout << "moments_kp1 updated.\n";
      moments_k->update();
      if (rank == 0) std::cout << "moments_k   updated.\n";
      
      if (!disableCircuitSolve)
      {
        const double Ek_value_l   = getSolutionFunctionValue(configMesh, E_k,   x0);
        const double Ekp1_value_l = getSolutionFunctionValue(configMesh, E_kp1, x0);
        const double Ek_value_r   = getSolutionFunctionValue(configMesh, E_k,   x1);
        const double Ekp1_value_r = getSolutionFunctionValue(configMesh, E_kp1, x1);
        const double Ik_value     = getSolutionFunctionValue(configMesh, I_k,   x1);
        const double Ikp1_value   = getSolutionFunctionValue(configMesh, I_kp1, x1);
        
        circuitSolver.setEk  ( Ek_value_l,   Ek_value_r   );
        circuitSolver.setEkp1( Ekp1_value_l, Ekp1_value_r );
        circuitSolver.setIk  ( Ik_value   );
        circuitSolver.setVk  ( V_k        );
        circuitSolver.setIem ( Ikp1_value );
        
        V_k = circuitSolver.voltage();
        voltageParameter->setValue(V_k / potentialScaling); // sets the BC for the next time step; divide by the reference-to-physical scaling to get a non-dimensional value.
        
        // record the k+1 values (V_k is actually V_kp1 now)
        // time, voltage, E-field, current -- note the circuit solver uses physical units
        circuitResultsFileStream << t * timeScaling << "\t";
        circuitResultsFileStream << V_k << "\t";
        circuitResultsFileStream << Ekp1_value_r << "\t";
        circuitResultsFileStream << Ikp1_value << "\n";
        circuitResultsFileStream.flush();
      }
      
      double energyError = -1.;
      if ((fixedCountRefStrategy != Teuchos::null) && (fixedCountRefStrategy->getNumRefinements() > 0))
      {
        energyError = fixedCountRefStrategy->getEnergyError(fixedCountRefStrategy->getNumRefinements() - 1);
      }
      
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        if (saveViz)
        {
          solutionMoments->update(); // update to get latest density, velocity
          vlasovViz->exportSolution(form.vlasovSolution(),t);
          const int num1DPointsPoisson = 10; // default is 4
          poissonViz->exportSolution(form.configurationSpaceSolution(), {density,velocity}, {"n_e","v"},t,num1DPointsPoisson);
        }
        
        reportTimeStepComplete(timeStepNumber,energyError);
      }
    } // end time step loop
    
    circuitResultsFileStream.close();
    
    FunctionPtr potential = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), form.configurationSpaceSolution());
        
    // output to results file:
    {
      // output (physical/dimensional) results
      // ultimately, we want E-field, velocity, and some other things
      // to begin with, let's output the E-field values computed at quadrature points at final time
            
      auto allActivePhysicalCells = form.configurationSpaceMesh()->getActiveCellIDsGlobal();
      
      form.configurationSpaceSolution()->importSolutionForOffRankCells(allActivePhysicalCells);
      
      // update solutionMoments once more and gather data to rank 0, so we can access it in the block below
      solutionMoments->update();
      solutionMoments->gatherOnRankZero();
      
      if (rank == 0)
      {
        std::map<double,double> E_values; // map from x coordinate to final E field value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> ne_values; // map from x coordinate to final density value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> velocity_values; // map from x coordinate to final velocity value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> potential_values; // map from x coordinate to final potential value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> pointWeight_values; // map from x coordinate to cell measure-weighted quadrature weight
        GlobalIndexType firstPhysicalCellID = *allActivePhysicalCells.begin();
        
        const int cubatureEnrichmentDegree = 5; // over-integrate to make up for undermeasurement on coarse initial meshes
        auto basisCache = BasisCache::basisCacheForCell(form.configurationSpaceMesh(), firstPhysicalCellID, false, cubatureEnrichmentDegree); // false: not test versus test
        std::vector<GlobalIndexType> physicalCellIDsVector(allActivePhysicalCells.begin(),allActivePhysicalCells.end());
        
        const int numPoints = basisCache->getRefCellPoints().extent_int(0);
        auto & cubWeights = basisCache->getCubatureWeightsFC();
        
        auto physicalCellNodes = form.configurationSpaceMesh()->physicalCellNodes(physicalCellIDsVector);
        basisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, false);
        
        FieldContainer<double> E_values_FC(physicalCellIDsVector.size(), numPoints);
        E_soln->values(E_values_FC, basisCache);
        
        FieldContainer<double> potential_values_FC(physicalCellIDsVector.size(), numPoints);
        potential->values(potential_values_FC, basisCache);
        
        FieldContainer<double> density_values_FC(physicalCellIDsVector.size(), numPoints);
        density->values(density_values_FC, basisCache);
        
        FieldContainer<double> velocity_values_FC(physicalCellIDsVector.size(), numPoints);
        velocity->values(velocity_values_FC, basisCache);
        
        const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
        
        const int numCells = physicalCellIDsVector.size();
        
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
          {
            const double & x = physicalQuadraturePoints(cellOrdinal,pointOrdinal,0);
            const double pointWeight = cubWeights(pointOrdinal) * cellWidth / 2.0; // 2.0: width of the reference line.
            
            E_values[x]           = E_values_FC(cellOrdinal,pointOrdinal);
            potential_values[x]   = potential_values_FC(cellOrdinal,pointOrdinal);
            ne_values[x]          = density_values_FC(cellOrdinal,pointOrdinal);
            velocity_values[x]    = velocity_values_FC(cellOrdinal,pointOrdinal);
            pointWeight_values[x] = pointWeight;
          }
        }
        
        std::ofstream resultsFileStream(resultsFile, std::ios::out);
        
        resultsFileStream << "x\tquadrature weight\tE\tphi\tne\tvx\n";
        resultsFileStream << setprecision(15);
        
        for (const auto & pointEntry : pointWeight_values)
        {
          const auto x           = pointEntry.first * spatialScaling;
          const auto pointWeight = pointEntry.second * spatialScaling;
          const auto E           = E_values[pointEntry.first]         * EScaling;
          const auto potential   = potential_values[pointEntry.first] * potentialScaling;
          const auto ne          = ne_values[pointEntry.first]        * n0; // TODO: determine if this scaling is correct; we probably need to divide by spatialScaling^3.
          const auto velocity    = velocity_values[pointEntry.first]  * velocityScaling;
          
          resultsFileStream << x           << "\t";
          resultsFileStream << pointWeight << "\t";
          resultsFileStream << E           << "\t";
          resultsFileStream << potential   << "\t";
          resultsFileStream << ne          << "\t";
          resultsFileStream << velocity    << "\n";
        }
        
        resultsFileStream.close();
        
        auto spatialMeshFile = getSpatialMeshFilePath();
        std::ofstream spatialMeshFileStream(spatialMeshFile, std::ios::out);
        // let's collect the spatial nodes and sort them
        std::set<double> spatialNodes;
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          spatialNodes.insert(physicalCellNodes(cellOrdinal,0,0)); // (min x, min t)
          spatialNodes.insert(physicalCellNodes(cellOrdinal,1,0)); // (max x, min t)
        }
        resultsFileStream << "x\n";
        for (const auto & node : spatialNodes)
        {
          spatialMeshFileStream << node << "\n";
        }
        spatialMeshFileStream.close();
      }
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
