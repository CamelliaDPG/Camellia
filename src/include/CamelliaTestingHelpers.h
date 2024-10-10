// @HEADER
// ***********************************************************************
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-Teuchos in the licenses directory.
//
// Some variations on Teuchos_TestingHelpers.hpp and Teuchos_LocalTestingHelpers.hpp, which are included with Trilinos.
//
// Questions? Contact Nathan V. Roberts (nate@nateroberts.com)
//
// ***********************************************************************
// @HEADER

#ifndef CAMELLIA_TESTING_HELPERS_HPP
#define CAMELLIA_TESTING_HELPERS_HPP

/*! \file CamelliaTestingHelpers.hpp
    \brief Utilities to make writing tests easier.
*/

#include <Intrepid2_TestUtils.hpp>

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TypeNameTraits.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_TestingHelpers.hpp"


#include "MeshFactory.h"
#include "TypeDefs.h"

namespace Camellia
{

  MeshPtr makeSingleElementExtrudedTestMesh( int baseSpaceDim, int numTensorExtrusions, std::vector<int> H1Orders );

/** \brief Compare if two array objects are the same or not up to a relative
 * floating point precision.
 *
 * This function works with any two array objects are the same size and have
 * the same element value types.  The funtion is templated on the container
 * types and therefore can compare any two objects that have size() and
 * operator[](i) defined.
 *
 * Compared to the method of the same name provided by Teuchos_TestingHelpers,
 * this method uses a tolerance relative to the maximum absolute value in the two arrays,
 * rather than a relative to the maximum absolute value of the elements being compared.
 *
 * \returns Returns <tt>true</tt> if the arrays match and <tt>false</tt> otherwise.
 *
 * \ingroup teuchos_testing_grp
 */
template<class Array1, class Array2, class ScalarMag>
bool compareFloatingArraysCamellia(const Array1 &a1, const std::string &a1_name,
                                   const Array2 &a2, const std::string &a2_name,
                                   const ScalarMag &relTol,
                                   Teuchos::FancyOStream &out
                                  );
  
  /** \brief Compare if two array objects are the same or not up to a relative
   * floating point precision.
   *
   * This function works with any two array objects are the same size and have
   * the same element value types.  The funtion is templated on the container
   * types and therefore can compare any two objects that have size() and
   * operator[](i) defined.
   *
   * Compared to the method of the same name provided by Teuchos_TestingHelpers,
   * this method uses a tolerance relative to the maximum absolute value in the two arrays,
   * rather than a relative to the maximum absolute value of the elements being compared.
   *
   * \returns Returns <tt>true</tt> if the arrays match and <tt>false</tt> otherwise.
   *
   * \ingroup teuchos_testing_grp
   */
  template<class Array1, class Array2, class ScalarMag>
  bool compareFloatingArraysCamellia(const Array1 &a1, const std::string &a1_name,
                                     const Array2 &a2, const std::string &a2_name,
                                     const ScalarMag &relTol, const ScalarMag &absTol,
                                     Teuchos::FancyOStream &out
                                     );
  
  
  template <class Scalar1, class Scalar2>
  KOKKOS_INLINE_FUNCTION
  bool
  errMeetsAbsAndRelTol( const Scalar1 &s1, const Scalar2 &s2, const double &relTol, const double &absTol )
  {
    return fabs( s1 - s2 ) <= absTol + fabs(s1) * relTol;
  }

//! Wrap Intrepid2::testFloatingEquality to convert compile-time rank-mismatch errors to runtime ones.
//! this method assumes both functors are accessible on device.
template <int rank, class FunctorType1, class FunctorType2, typename Scalar=typename FunctorType1::value_type, class ExecutionSpace = typename FunctorType1::execution_space>
typename std::enable_if< (Intrepid2::supports_rank<FunctorType1,rank>::value && Intrepid2::supports_rank<FunctorType2,rank>::value), void >::type
testFloatingEquality(const FunctorType1 &functor1, const FunctorType2 &functor2, double relTol, double absTol, Teuchos::FancyOStream &out, bool &success,
                     std::string functor1Name = "Functor 1", std::string functor2Name = "Functor 2")
{
  static_assert( Intrepid2::supports_rank<FunctorType1,rank>::value, "Functor1 must support the specified rank through operator().");
  static_assert( Intrepid2::supports_rank<FunctorType2,rank>::value, "Functor2 must support the specified rank through operator().");
  
  Intrepid2::testFloatingEquality<FunctorType1, FunctorType2, rank, Scalar, ExecutionSpace> (functor1,functor2,relTol,absTol,out,success,functor1Name,functor2Name);
}

//! Wrap Intrepid2::testFloatingEquality to convert compile-time rank-mismatch errors to runtime ones.
//! this method assumes both functors are accessible on device.
template <int rank, class FunctorType1, class FunctorType2, typename Scalar=typename FunctorType1::value_type, class ExecutionSpace = typename FunctorType1::execution_space>
typename std::enable_if< !(Intrepid2::supports_rank<FunctorType1,rank>::value && Intrepid2::supports_rank<FunctorType2,rank>::value), void >::type
testFloatingEquality(const FunctorType1 &functor1, const FunctorType2 &functor2, double relTol, double absTol, Teuchos::FancyOStream &out, bool &success,
                     std::string functor1Name = "Functor 1", std::string functor2Name = "Functor 2")
{
  bool functor1SupportsRank = Intrepid2::supports_rank<FunctorType1,rank>::value;
  if (!functor1SupportsRank)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Functor 1 does not support the specified rank");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Functor 2 does not support the specified rank");
  }
}

class D1
{
public:
  static const int spaceDim = 1;
};
class D2
{
public:
  static const int spaceDim = 2;
};
class D3
{
public:
  static const int spaceDim = 3;
};
class D4
{
public:
  static const int spaceDim = 4;
};
class D5
{
public:
  static const int spaceDim = 5;
};
class D6
{
public:
  static const int spaceDim = 6;
};
class D7
{
public:
  static const int spaceDim = 7;
};

class R0
{
public:
  static const int rank = 0;
};
class R1
{
public:
  static const int rank = 1;
};
class R2
{
public:
  static const int rank = 2;
};

} // namespace Camellia

/** \brief Assert that a1.size()==a2.size() and rel_error(a[i],b[i]) <= tol, i=0...., where the error is relative to the maximum entry in the arrays.
 *
 * Works for any object types that support a1[i], a1.size(), a2[j], and
 * a2.size() and types a1 and a2 can be different types!
 *
 */
#define TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA( a1, a2, tol ) \
{ \
const bool result = Camellia::compareFloatingArraysCamellia(a1,#a1,a2,#a2,tol,out); \
if (!result) success = false; \
}

/** \brief Assert that a1.size()==a2.size() and err(a[i],b[i]) <= absTol OR rel_error(a[i],b[i]) <= relTol, i=0...., where the rel_error is relative to the maximum entry in the arrays.
 *
 * Works for any object types that support a1[i], a1.size(), a2[j], and
 * a2.size() and types a1 and a2 can be different types!
 *
 */
#define TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA_ABSTOLTOO( a1, a2, relTol, absTol ) \
{ \
const bool result = Camellia::compareFloatingArraysCamellia(a1,#a1,a2,#a2,relTol,absTol,out); \
if (!result) success = false; \
}

//
// Implementations
//

template<class Array1, class Array2, class ScalarMag>
bool Camellia::compareFloatingArraysCamellia(
                                             const Array1 &a1, const std::string &a1_name,
                                             const Array2 &a2, const std::string &a2_name,
                                             const ScalarMag &relTol,
                                             Teuchos::FancyOStream &out
                                             )
{
  double absTol = 0.0;
  return Camellia::compareFloatingArraysCamellia(a1,a1_name,a2,a2_name,relTol,absTol,out);
}

template<class Array1, class Array2, class ScalarMag>
bool Camellia::compareFloatingArraysCamellia(
  const Array1 &a1, const std::string &a1_name,
  const Array2 &a2, const std::string &a2_name,
  const ScalarMag &relTol,
  const ScalarMag &absTol,
  Teuchos::FancyOStream &out
)
{
  using Teuchos::as;
  bool success = true;

  out << "Comparing " << a1_name << " == " << a2_name << " ... ";

  const int n = a1.size();

  // Compare sizes
  if (as<int>(a2.size()) != n)
  {
    out << "\nError, "<<a1_name<<".size() = "<<a1.size()<<" == "
        << a2_name<<".size() = "<<a2.size()<<" : failed!\n";
    return false;
  }

  // Get maximum value in element arrays (or tol, if that's bigger)
  ScalarMag maxValue = relTol;
  for( int i = 0; i < n; ++i )
  {
    maxValue = std::max(abs(a1[i]),maxValue);
    maxValue = std::max(abs(a2[i]),maxValue);
  }

  // Compare elements
  for( int i = 0; i < n; ++i )
  {
    const ScalarMag absErr = abs(a1[i]-a2[i]);
    if (absErr <= absTol) continue;
    
    const ScalarMag relErr = abs(a1[i]-a2[i]) / maxValue;
    if ( !(relErr <= relTol) )
    {
      out
          <<"\nError, relErr("<<a1_name<<"["<<i<<"],"
          <<a2_name<<"["<<i<<"]) = relErr("<<a1[i]<<","<<a2[i]<<") = "
          <<relErr<<" <= relTol = "<<relTol<<": failed!\n";
      success = false;
    }
  }
  if (success)
  {
    out << "passed\n";
  }

  return success;
}

#endif  // CAMELLIA_TESTING_HELPERS_HPP
