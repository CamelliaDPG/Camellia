// @HEADER
// ***********************************************************************
//
// Some variations on Teuchos_TestingHelpers.hpp and Teuchos_LocalTestingHelpers.hpp, which are included with Trilinos.
//
// Questions? Contact Nathan V. Roberts (nvroberts@anl.gov)
//
// ***********************************************************************
// @HEADER

#ifndef CAMELLIA_TESTING_HELPERS_HPP
#define CAMELLIA_TESTING_HELPERS_HPP

/*! \file CamelliaTestingHelpers.hpp
    \brief Utilities to make writing tests easier.
*/

#include "Teuchos_ConfigDefs.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_TypeNameTraits.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_TestingHelpers.hpp"


namespace Camellia {
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
                                   const ScalarMag &tol, Teuchos::FancyOStream &out
  );

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

//
// Implementations
//

template<class Array1, class Array2, class ScalarMag>
bool Camellia::compareFloatingArraysCamellia(
  const Array1 &a1, const std::string &a1_name,
  const Array2 &a2, const std::string &a2_name,
  const ScalarMag &tol,
  Teuchos::FancyOStream &out
  )
{
  using Teuchos::as;
  bool success = true;

  out << "Comparing " << a1_name << " == " << a2_name << " ... ";

  const int n = a1.size();

  // Compare sizes
  if (as<int>(a2.size()) != n) {
    out << "\nError, "<<a1_name<<".size() = "<<a1.size()<<" == "
        << a2_name<<".size() = "<<a2.size()<<" : failed!\n";
    return false;
  }

  // Get maximum value in element arrays (or tol, if that's bigger)
  ScalarMag maxValue = tol;
  for( int i = 0; i < n; ++i ) {
    maxValue = std::max(abs(a1[i]),maxValue);
    maxValue = std::max(abs(a2[i]),maxValue);
  }
  
  // Compare elements
  for( int i = 0; i < n; ++i ) {
    const ScalarMag err = abs(a1[i]-a2[i]) / maxValue;
    if ( !(err <= tol) ) {
      out
        <<"\nError, relErr("<<a1_name<<"["<<i<<"],"
        <<a2_name<<"["<<i<<"]) = relErr("<<a1[i]<<","<<a2[i]<<") = "
        <<err<<" <= tol = "<<tol<<": failed!\n";
      success = false;
    }
  }
  if (success) {
    out << "passed\n";
  }

  return success;
}

#endif  // CAMELLIA_TESTING_HELPERS_HPP
