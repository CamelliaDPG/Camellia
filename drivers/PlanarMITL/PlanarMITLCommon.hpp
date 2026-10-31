#pragma once

#include "Teuchos_TestForException.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace PlanarMITL
{
/// Proper velocity u = gamma*v; all arguments use the same scaled units.
inline double inverseGamma(double ux, double uz, double c)
{
  return 1.0 / std::sqrt(1.0 + (ux * ux + uz * uz) / (c * c));
}

inline double properVelocityFromBeta(double beta, double c)
{
  if (!(c > 0.0) || !(beta > -1.0 && beta < 1.0))
  {
    throw std::invalid_argument("Expected c > 0 and -1 < beta < 1");
  }
  return c * beta / std::sqrt(1.0 - beta * beta);
}

/// Coordinate order (x,t,ux,uz); the reference driver's uy represents uz.
inline std::array<double, 4> transport(double ux, double uz, double c,
                                       double qOverM, double Ex, double By)
{
  const double h = inverseGamma(ux, uz, c);
  return {ux * h, 1.0, qOverM * (Ex - uz * h * By), qOverM * ux * h * By};
}

//! Vertices on [left, right] that resolve a feature at center: a uniform core of elements of width coreElementWidth
//! covering [center - coreHalfWidth, center + coreHalfWidth] (clipped to the domain), and element widths growing
//! geometrically by growthRatio from the core out to each boundary.  A final element narrower than half its
//! predecessor is merged into it.
inline std::vector<double> gradedVertices(const double left, const double right, const double center,
                                   const double coreElementWidth, const double coreHalfWidth, const double growthRatio)
{
  TEUCHOS_TEST_FOR_EXCEPTION((coreElementWidth <= 0) || (growthRatio < 1) || (right <= left), std::invalid_argument,
                             "gradedVertices(): invalid arguments");
  const double coreLeft  = std::max(left,  center - coreHalfWidth);
  const double coreRight = std::min(right, center + coreHalfWidth);
  // geometrically graded vertices from start (exclusive) to end (inclusive), first width growthRatio * coreElementWidth
  auto gradedSide = [&](const double start, const double end) {
    std::vector<double> sideVertices;
    const double length = std::abs(end - start);
    const double direction = (end > start) ? 1.0 : -1.0;
    double position = 0, width = coreElementWidth * growthRatio;
    while (length - position > 0)
    {
      const double remaining = length - position;
      if (remaining <= width * 1.5)
      {
        sideVertices.push_back(end); // last element absorbs the remainder (at most 1.5x its nominal width)
        break;
      }
      position += width;
      sideVertices.push_back(start + direction * position);
      width *= growthRatio;
    }
    return sideVertices;
  };
  std::vector<double> vertices;
  auto leftSide = gradedSide(coreLeft, left);
  for (auto it = leftSide.rbegin(); it != leftSide.rend(); it++) vertices.push_back(*it);
  const int numCoreElements = std::max(1, int(std::ceil((coreRight - coreLeft) / coreElementWidth - 1e-10)));
  for (int i=0; i<=numCoreElements; i++)
  {
    vertices.push_back(coreLeft + (coreRight - coreLeft) * double(i) / double(numCoreElements));
  }
  for (double vertex : gradedSide(coreRight, right)) vertices.push_back(vertex);
  return vertices;
}

//! Ensures that value is a vertex, if it lies strictly inside the vertex range: moves the nearer vertex of the interval
//! containing it if that vertex is within a quarter of the interval width, and otherwise inserts value.  (The inflow
//! boundary condition is imposed on whole mesh sides, so a side straddling u_x = 0 on the cathode or anode would receive
//! it on its outflow part too.)
inline void ensureVertexAt(std::vector<double> &vertices, const double value)
{
  for (size_t i=0; i+1<vertices.size(); i++)
  {
    const double left = vertices[i], right = vertices[i+1];
    if ((value <= left) || (value >= right)) continue;
    const double width = right - left;
    if      ((value - left) < 0.25 * width) vertices[i]   = value;
    else if ((right - value) < 0.25 * width) vertices[i+1] = value;
    else    vertices.insert(vertices.begin() + i + 1, value);
    return;
  }
}

inline std::vector<double> uniformVertices(const double left, const double width, const int numElements)
{
  std::vector<double> vertices(numElements+1);
  for (int i=0; i<=numElements; i++) vertices[i] = left + double(i) / double(numElements) * width;
  return vertices;
}

} // namespace PlanarMITL
