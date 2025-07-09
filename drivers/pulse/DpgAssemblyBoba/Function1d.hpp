#ifndef FUNCTION_1D_HPP
#define FUNCTION_1D_HPP

#include <cmath>

#include <Teuchos_RCP.hpp>

// univariate function
class Function1d {
public:
  virtual ~Function1d() = default;

  virtual double
  operator()(double input) const = 0;
};

using Function1dPtr = Teuchos::RCP<const Function1d>;

class Constant1d : public Function1d {
public:
  Constant1d(double constant) : constant_(constant) {}

  double
  operator()(double input) const {
    return constant_;
  }

private:
  double constant_;
};

inline Function1dPtr
newConstant1d(double constant) {
  return Teuchos::rcp(new Constant1d(constant));
}

class Monomial1d : public Function1d {
public:
  Monomial1d(std::size_t degree) : degree_(degree) {}

  double
  operator()(double input) const {
    return std::pow(input, degree_);
  }

private:
  std::size_t degree_;
};

inline Function1dPtr
newMonomial1d(std::size_t degree) {
  return Teuchos::rcp(new Monomial1d(degree));
}

class FunctionSum1d : public Function1d {
public:
  FunctionSum1d(Function1dPtr f, Function1dPtr g) : f_(f), g_(g) {}

  double
  operator()(double input) const {
    return (*f_)(input) + (*g_)(input);
  }

private:
  Function1dPtr f_;
  Function1dPtr g_;
};

inline Function1dPtr
operator+(Function1dPtr f, Function1dPtr g) {
  return Teuchos::rcp(new FunctionSum1d(f, g));
}

class FunctionDifference1d : public Function1d {
public:
  FunctionDifference1d(Function1dPtr f, Function1dPtr g) : f_(f), g_(g) {}

  double
  operator()(double input) const {
    return (*f_)(input) - (*g_)(input);
  }

private:
  Function1dPtr f_;
  Function1dPtr g_;
};

inline Function1dPtr
operator-(Function1dPtr f, Function1dPtr g) {
  return Teuchos::rcp(new FunctionDifference1d(f, g));
}

class FunctionProduct1d : public Function1d {
public:
  FunctionProduct1d(Function1dPtr f, Function1dPtr g) : f_(f), g_(g) {}

  double
  operator()(double input) const {
    return (*f_)(input) * (*g_)(input);
  }

private:
  Function1dPtr f_;
  Function1dPtr g_;
};

inline Function1dPtr
operator*(Function1dPtr f, Function1dPtr g) {
  return Teuchos::rcp(new FunctionProduct1d(f, g));
}

class FunctionScaled1d : public Function1d {
public:
  FunctionScaled1d(Function1dPtr f, double alpha) : f_(f), alpha_(alpha) {}

  double
  operator()(double input) const {
    return alpha_ * (*f_)(input);
  }

private:
  Function1dPtr f_;
  double alpha_;
};

inline Function1dPtr
operator*(double alpha, Function1dPtr f) {
  return Teuchos::rcp(new FunctionScaled1d(f, alpha));
}

inline Function1dPtr
operator/(Function1dPtr f, double alpha) {
  return Teuchos::rcp(new FunctionScaled1d(f, 1.0 / alpha));
}

class FunctionDivision1d : public Function1d {
public:
  FunctionDivision1d(Function1dPtr f, Function1dPtr g) : f_(f), g_(g) {}

  double
  operator()(double input) const {
    return (*f_)(input) / (*g_)(input);
  }

private:
  Function1dPtr f_;
  Function1dPtr g_;
};

inline Function1dPtr
operator/(Function1dPtr f, Function1dPtr g) {
  return Teuchos::rcp(new FunctionDivision1d(f, g));
}

#endif
