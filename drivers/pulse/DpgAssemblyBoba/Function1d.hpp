#ifndef FUNCTION_1D_HPP
#define FUNCTION_1D_HPP

#include <functional>

#include <Teuchos_RCP.hpp>

class Function1d;
using Function1dPtr = Teuchos::RCP<const Function1d>;

// univariate function
class Function1d {
public:
  Function1d(const std::function<double(const double)> &func) : func_(func) {}

  double
  operator()(double input) const {
    return func_(input);
  }

private:
  std::function<double(double)> func_;
};

inline Function1dPtr
newFunction1d(const std::function<double(const double)> &func) {
  return Teuchos::rcp(new Function1d(func));
}

#endif
