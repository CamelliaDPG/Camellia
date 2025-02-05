
//
// TODO: move wrapper infrastructure into boba
//

template<typename vector_t, size_t size>
struct vector_wrapper
{
  using data_type = typename vector_t::data_type;
  using index_type = typename vector_t::index_type;

  boba::Array<vector_t, size> wrapped_vector;

  /**
    *  Access an element of the vector
    */
  vector_t& operator()(size_t id)
  {
    return wrapped_vector[id];
  }

  vector_t const& operator()(size_t id) const
  {
    return wrapped_vector[id];
  }

  vector_wrapper operator-(const vector_wrapper& rhs) const
  {
    vector_wrapper output;
    for(size_t element = 0; element < size; element++)
    {
      output(element) = wrapped_vector[element];
      output(element) -= rhs(element);
    }
    return output;
  }

  vector_wrapper operator+(const vector_wrapper& rhs) const
  {
    vector_wrapper output;
    for(size_t element = 0; element < size; element++)
    {
      output(element) = wrapped_vector[element];
      output(element) += rhs(element);
    }
    return output;
  }

  vector_wrapper& operator-=(const vector_wrapper& rhs)
  {
    for(size_t element = 0; element < size; element++)
    {
      this->operator()(element) -= rhs(element);
    }
    return *this;
  }

  vector_wrapper& operator+=(const vector_wrapper& rhs)
  {
    for(size_t element = 0; element < size; element++)
    {
      this->operator()(element) += rhs(element);
    }
    return *this;
  }

  vector_wrapper& operator*=(data_type const x)
  {
    for(size_t element = 0; element < size; element++)
    {      
      this->wrapped_vector[element] *= x;
    }
    return *this;
  }

  vector_wrapper operator*(data_type const x) const
  {
    vector_wrapper output;
    for(size_t element = 0; element < size; element++)
    {
      output(element) = this->wrapped_vector[element];
    }
    output *= x;
    return output;
  }

  double norm_frobenius() const
  {
    double output = 0.0;
    for(size_t element = 0; element < size; element++)
    {
      auto norm_segment = wrapped_vector[element].norm_frobenius();
      output += norm_segment*norm_segment;
    }
    return ::boba::sqrt(output);
  }

  double compression_rate() const
  {
    size_t numerator = 0;
    size_t denominator = 0;
    for(size_t element = 0; element < size; element++)
    {
      numerator += wrapped_vector[element].get_full_size();
      denominator += wrapped_vector[element].get_number_elements();
    }
    auto cr = static_cast<double>(numerator) / static_cast<double>(denominator);
    return cr;
  }

  void round()
  {
    for(size_t element = 0; element < size; element++)
    {
      //
      //boba_print(wrapped_vector[element].compression_rate());
      wrapped_vector[element].round();
      //boba_print(wrapped_vector[element].compression_rate());
      //boba_print(wrapped_vector[element].ranks_string());
    }
  }

  void fill_with(data_type x)
  {
    for(size_t element = 0; element < size; element++)
    {
      wrapped_vector[element].fill_with(x);
    }
  }

  void fill_with_zeros()
  {
    for(size_t element = 0; element < size; element++)
    {
      wrapped_vector[element].fill_with_zeros();
    }
  }

  void rename(std::string name)
  {
    for(size_t element = 0; element < size; element++)
    {
      wrapped_vector[element].rename(name + "_" + std::to_string(element));
    }
  }

  void savetxt(std::string name="vector") const
  {
    for (size_t element = 0; element < size; element++)
    {
      wrapped_vector[element].savetxt(name + "_" + std::to_string(element) + ".dat");
    }
  }
};

template<typename vector_t, size_t size, typename data_type>
vector_wrapper<vector_t, size> operator*(const data_type& x, const vector_wrapper<vector_t, size>& vec)
{
  return vec * x;
}

template<typename vector_t, size_t size>
double inner_product(vector_wrapper<vector_t, size> a, vector_wrapper<vector_t, size> b)
{
  double ip = 0.0;
  for(size_t element = 0; element < size; element++)
  {
    ip += boba::inner_product(a(element), b(element));
  }
  return ip;
}
namespace boba
{
  template<typename vector_t, size_t size, typename data_type, typename index_type>
  vector_wrapper<vector_t, size> add_reduce(const std::vector<vector_wrapper<vector_t, size>>& a, vector<execution_space::CPU, data_type, index_type>& coeff)
  {
    auto coeff_view = coeff.const_view();
    vector_wrapper<vector_t, size> output;
    for(size_t element = 0; element < size; element++)
    {
      std::vector<vector_t> element_sequence;
      for(auto& item : a)
      {
        element_sequence.push_back(item(element));
      }
      output(element) = ::boba::add_reduce(element_sequence, coeff);
    }
    return output;
  }
}

template<typename operator_t, typename vector_t, size_t rows, size_t cols>
struct operator_wrapper
{
  using data_type = typename vector_t::data_type;
  using index_type = typename vector_t::index_type;

  boba::Array<operator_t, rows*cols> wrapped_matrix;

  /**
    *  Access an element of the wrapped_matrix
    *  Example:  example_matrix({2, 3})
    */

  operator_t& operator()(::boba::Array<size_t, 2> rc)
  { 
    auto row = rc[0];
    auto col = rc[1];
    return wrapped_matrix[row + rows*col];
  }

  operator_t const& operator()(::boba::Array<size_t, 2> rc) const
  { 
    auto row = rc[0];
    auto col = rc[1];
    return wrapped_matrix[row + rows*col];
  }

  /**
    *  Action of the wrapped_matrix on a wrapped_vector
    */

  vector_wrapper<vector_t, rows> operator*(const vector_wrapper<vector_t, cols>& wrapped_vector)
  { 
    vector_wrapper<vector_t, rows> output_vector;
    for(size_t row = 0; row < rows; row++)
    {
      output_vector(row) = this->operator()({row, 0})*wrapped_vector(0);
      for(size_t col = 1; col < cols; col++)
      {
        output_vector(row) += this->operator()({row, col})*wrapped_vector(col);
      }
    }
    return output_vector;
  }

  void savetxt(std::string name="matrix") const
  {
    for (size_t row = 0; row < rows; row++)
    {
      for (size_t col = 0; col < cols; col++)
      {
        wrapped_matrix[row + rows * col].savetxt(name + "_" + std::to_string(row) + "_" + std::to_string(col) + ".dat");
      }
    }
  }

  double compression_rate() const
  {
    size_t numerator = 0;
    size_t denominator = 0;
    for (const auto &matrix : wrapped_matrix)
    {
      numerator += matrix.get_full_size();
      denominator += matrix.get_number_elements();
    }
    auto cr = static_cast<double>(numerator) / static_cast<double>(denominator);
    return cr;
  }
};
