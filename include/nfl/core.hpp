#ifndef NFL_CORE_HPP
#define NFL_CORE_HPP

#include <type_traits>
#include <vector>
#include <numeric>
#include <algorithm>

#include "nfl/poly.hpp"
#include "nfl/ops.hpp"
#include "nfl/algos.hpp"
#include "nfl/permut.hpp"
#include <type_traits>

namespace nfl {

/* poly methods implementation
 */
template<class T, size_t Degree, size_t NbModuli>
template<class Op, class... Args> inline poly<T, Degree, NbModuli>::poly(ops::expr<Op, Args...> const& expr) {
  *this = expr;
}

template<class T, size_t Degree, size_t NbModuli>
template<class Op, class... Args> inline poly<T, Degree, NbModuli>& poly<T, Degree, NbModuli>::operator=(ops::expr<Op, Args...> const& expr) {
  using E = ops::expr<Op, Args...>;
  constexpr size_t vector_size = E::simd_mode::template elt_count<T>::value;
  constexpr size_t vector_bound = degree / vector_size * vector_size;
  static_assert(vector_bound == degree, "no need for a footer");

  for(size_t cm = 0; cm < nmoduli; ++cm) {
    for(size_t j = 0; j < vector_bound; j+= vector_size) {
      E::simd_mode::store(&(*this)(cm,j), expr.template load<typename E::simd_mode>(cm, j));
    }
  }
  return *this;
}

template<class T, size_t Degree, size_t NbModuli>
inline poly<T, Degree, NbModuli>::operator bool() const {
  return std::find_if(begin(), end(),
                      [](value_type v) { return v != 0; }) != end();
}

// *********************************************************
//    Constructor
// *********************************************************

template<class T, size_t Degree, size_t NbModuli>
poly<T, Degree, NbModuli>::poly() : poly(std::integral_constant<T, 0>::value) {
}

template<class T, size_t Degree, size_t NbModuli>
poly<T, Degree, NbModuli>::poly(value_type v, bool reduce_coeffs) {
  set(v, reduce_coeffs);
}

template<class T, size_t Degree, size_t NbModuli>
poly<T, Degree, NbModuli>::poly(std::initializer_list<value_type> values, bool reduce_coeffs) {
  set(values, reduce_coeffs);
}

template<class T, size_t Degree, size_t NbModuli>
template<class It>
poly<T, Degree, NbModuli>::poly(It first, It last, bool reduce_coeffs) {
  set(first, last, reduce_coeffs);
}

template<class T, size_t Degree, size_t NbModuli>
void poly<T, Degree, NbModuli>::set(value_type v, bool reduce_coeffs) {
  if (v == 0) {
    // CRITICAL: the object must be 32-bytes aligned to avoid vectorization issues
    assert((unsigned long)(this->_data) % 32 == 0);
    std::fill(begin(), end(), 0);
  }
  else {
    set({v}, reduce_coeffs);
  }
}

template<class T, size_t Degree, size_t NbModuli>
void poly<T, Degree, NbModuli>::set(std::initializer_list<value_type> values, bool reduce_coeffs) {
  set(values.begin(), values.end(), reduce_coeffs);
}

template <class T, size_t Degree, size_t NbModuli>
template <class It>
void poly<T, Degree, NbModuli>::set(It first, It last, bool reduce_coeffs) {
  // CRITICAL: the object must be 32-bytes aligned to avoid vectorization issues
  assert((unsigned long)(this->_data) % 32 == 0);

  // The initializer needs to have either less values than the polynomial degree
  // (and the remaining coefficients are set to 0), or be fully defined (i.e.
  // the degree*nmoduli coefficients needs to be provided)
  size_t size = std::distance(first, last);
  if (size > degree && size != degree * nmoduli) {
    throw std::runtime_error(
        "core: CRITICAL, initializer of size above degree but not equal "
        "to nmoduli*degree");
  }

  auto* iter = begin();
  auto viter = first;

  for (size_t cm = 0; cm < nmoduli; cm++) {
    auto const p = get_modulus(cm);

    if (size != degree * nmoduli) viter = first;

    // set the coefficients
    size_t i = 0;
    for (; i < degree && viter < last; ++i, ++viter, ++iter) {
      *iter = reduce_coeffs ? (*viter) % p : *viter;
    }

    // pad with zeroes if needed
    for (; i < degree; ++i, ++iter) {
      *iter = 0;
    }
  }
}

// ****************************************************
// Random polynomial generation functions
// ****************************************************

// Sets a pre-allocated random polynomial in FFT form
// uniformly random, else the coefficients are uniform below the bound

template<class T, size_t Degree, size_t NbModuli>
poly<T, Degree, NbModuli>::poly(uniform const& u) {
  set(u);
}

template<class T, size_t Degree, size_t NbModuli>
void poly<T, Degree, NbModuli>::set(uniform const &) {
  // CRITICAL: the object must be 32-bytes aligned to avoid vectorization issues
  assert((unsigned long)(this->_data) % 32 == 0);

  // In uniform mode we need randomness for all the polynomials in the CRT
  fastrandombytes((unsigned char *)data(), sizeof(poly));

  for (unsigned int cm = 0; cm < nmoduli; cm++) {
    // In the uniform case, instead of getting a big random (within the general
    // moduli), We rather prefer, for performance issues, to get smaller
    // randoms for each module The mask should be the same for all moduli
    // (because they are the same size) But for generality we prefer to compute
    // it for each moduli so that we could have moduli of different bitsize

    value_type mask =
        (1ULL << (int)(floor(log2(get_modulus(cm))) + 1)) - 1;

    for (size_t i = 0; i < degree; i++) {
      // First remove the heavy weight bits we dont need
      value_type tmp = _data[i + degree * cm] & mask;

      // When the random is still too large, reduce it
      if (tmp >= get_modulus(cm)) {
        tmp -= get_modulus(cm);
      }
      _data[i + degree * cm] = tmp;
    }
  }
#ifdef CHECK_STRICTMOD
  for (size_t cm = 0; cm < nmoduli; cm++) {
    for (size_t i = 0; i < degree; i++) {
      assert(_data[i + degree * cm] < get_modulus(cm));
    }
  }
#endif
}

template<class T, size_t Degree, size_t NbModuli>
poly<T, Degree, NbModuli>::poly(non_uniform const& mode) {
  set(mode);
}

template<class T, size_t Degree, size_t NbModuli>
void poly<T, Degree, NbModuli>::set(non_uniform const& mode) {
  // CRITICAL: the object must be 32-bytes aligned to avoid vectorization issues
  assert((unsigned long)(this->_data) % 32 == 0);

  uint64_t const upper_bound = mode.upper_bound;
  uint64_t const amplifier = mode.amplifier;
  // In bounded mode upper_bound must be below the smaller of the moduli
  for (unsigned int cm = 0; cm < nmoduli; cm++) {
    if (upper_bound >= get_modulus(cm)) {
      throw std::runtime_error(
          "core: upper_bound is larger than the modulus");
    }
  }

  // We play with the rnd pointer (in the uniform case), and thus
  // we need to remember the allocated pointer to free it at the end
  value_type rnd[degree];

  // Get some randomness from the PRNG
  fastrandombytes((unsigned char *)rnd, sizeof(rnd));

  // upper_bound is below the moduli so we create the same mask for all the
  // moduli
  value_type mask =
      (1ULL << (int)(floor(log2(2*upper_bound-1)) + 1)) - 1;

  if(amplifier == 1){
    for (unsigned int i = 0; i < degree; i++) {

      // First remove the heavy weight bits we dont need
      value_type tmp = rnd[i] & mask;

      // When the random is still too large, reduce it
      // In order to follow strictly a uniform distribution we should
      // get another rnd but in order to follow the proofs of security
      // strictly we should also take noise from a gaussian ...
      if (tmp >= 2*upper_bound-1) {
        tmp -= 2*upper_bound-1;
      }
      // Center the noise
      if (tmp >= upper_bound) {
        for (unsigned int cm = 0; cm < nmoduli; cm++) {
          _data[degree * cm + i] = get_modulus(cm) + tmp - (2*upper_bound-1);
        }
      }
      else
      {
        for (unsigned int cm = 0; cm < nmoduli; cm++) {
          _data[degree * cm + i] = tmp;
        }
      }
    }
  } 
  else
  {
    for (unsigned int i = 0; i < degree; i++) {

      // First remove the heavy weight bits we dont need
      value_type tmp = rnd[i] & mask;

      // When the random is still too large, reduce it
      // In order to follow strictly a uniform distribution we should
      // get another rnd but in order to follow the proofs of security
      // strictly we should also take noise from a gaussian ...
      if (tmp >= 2*upper_bound-1) {
        tmp -= 2*upper_bound-1;
      }
      // Center the noise
      if (tmp >= upper_bound) {
        for (unsigned int cm = 0; cm < nmoduli; cm++) {
          _data[degree * cm + i] = get_modulus(cm) + tmp*amplifier - (2*upper_bound-1)*amplifier;
        }
      }
      else
      {
        for (unsigned int cm = 0; cm < nmoduli; cm++) {
          _data[degree * cm + i] = tmp*amplifier;
        }
      }
    }
  }
#ifdef CHECK_STRICTMOD
  for (size_t cm = 0; cm < nmoduli; cm++) {
    for (size_t i = 0; i < degree; i++) {
      assert(_data[i + degree * cm] < get_modulus(cm));
    }
  }
#endif
}

template<class T, size_t Degree, size_t NbModuli>
template<class in_class, unsigned _lu_depth>
poly<T, Degree, NbModuli>::poly(gaussian<in_class, T, _lu_depth> const& mode) {
  set(mode);
}

template<class T, size_t Degree, size_t NbModuli>
template<class in_class, unsigned _lu_depth>
void poly<T, Degree, NbModuli>::set(gaussian<in_class, T, _lu_depth> const& mode) {
  // CRITICAL: the object must be 32-bytes aligned to avoid vectorization issues
  assert((unsigned long)(this->_data) % 32 == 0);

  uint64_t const amplifier = mode.amplifier;

  // We play with the rnd pointer (in the uniform case), and thus
  // we need to remember the allocated pointer to free it at the end
  signed_value_type rnd[degree];

  // Get some randomness from the PRNG
  mode.fg_prng->getNoise((value_type *)rnd, degree);

  if (amplifier != 1) for (unsigned int i = 0; i < degree; i++) rnd[i]*= amplifier;
  for (size_t cm = 0; cm < nmoduli; cm++) 
  {
    for (size_t i = 0 ; i < degree; i++)
    { 
      if(rnd[i]<0)
        _data[degree*cm+i] = get_modulus(cm) + rnd[i];
      else
        _data[degree*cm+i] = rnd[i];
    }
    //memcpy(_data+degree*cm, rnd, degree*sizeof(value_type));
  }


#ifdef CHECK_STRICTMOD
  for (size_t cm = 0; cm < nmoduli; cm++) {
    for (size_t i = 0; i < degree; i++) {
      assert(_data[i + degree * cm] < get_modulus(cm));
    }
  }
#endif
}

template<class T, size_t Degree, size_t NbModuli>
poly<T, Degree, NbModuli>::poly(ZO_dist const& mode) {
  set(mode);
}

template<class T, size_t Degree, size_t NbModuli>
void poly<T, Degree, NbModuli>::set(ZO_dist const& mode) {
  uint8_t rnd[Degree];
  fastrandombytes(rnd, sizeof(rnd));
  value_type *ptr = &_data[0];
  for (size_t cm = 0; cm < NbModuli; ++cm) {
    const T p = params<T>::P[cm] - 1U;
    for (size_t i = 0; i < Degree; ++i) {
      *ptr++ = rnd[i] <= mode.rho ? p + (rnd[i] & 2) : 0U;
    }
  }
}

template<class T, size_t Degree, size_t NbModuli>
poly<T, Degree, NbModuli>::poly(hwt_dist const& mode) {
  set(mode);
}

template<class T, size_t Degree, size_t NbModuli>
void poly<T, Degree, NbModuli>::set(hwt_dist const& mode) {
  assert(mode.hwt > 0 && mode.hwt <= Degree);
  std::vector<size_t> hitted(mode.hwt);
  std::iota(hitted.begin(), hitted.end(), 0U); // select the first hwt positions.
  std::vector<size_t> rnd(hitted.size());
  auto rnd_end = rnd.end();
  auto rnd_ptr = rnd_end;
  /* Reservoir Sampling: uniformly select hwt coefficients. */
  for (size_t k = mode.hwt; k < degree; ++k) 
  {
    size_t pos = 0;
    size_t reject_sample = std::numeric_limits<size_t>::max() / k;
    /* sample uniformly from [0, k) using reject sampling. */
    for (;;) {
      if (rnd_ptr == rnd_end)
      {
        fastrandombytes((unsigned char *)rnd.data(), rnd.size() * sizeof(size_t));
        rnd_ptr = rnd.begin();
      }
      pos = *rnd_ptr++;
      if (pos <= reject_sample * k) {
        pos %= k;
        break;
      }
    }
    if (pos < mode.hwt)
      hitted[pos] = k;
  }

  std::sort(hitted.begin(), hitted.end()); // for better locality ?
  std::memset(_data, 0x0, N * sizeof(value_type)); // clear up all
  fastrandombytes((unsigned char *)rnd.data(), rnd.size() * sizeof(size_t));
  for (size_t cm = 0, offset = 0; cm < nmoduli; ++cm) {
    rnd_ptr = rnd.begin();
    const T p = params<T>::P[cm] - 1u;
    for (size_t pos : hitted)
      _data[pos + offset] = p + ((*rnd_ptr++) & 2); // {-1, 1}
    offset += degree;
  }
  std::memset(hitted.data(), 0x0, hitted.size() * sizeof(size_t)); // erase from memory
}

// *********************************************************
// Helper functions
// *********************************************************


template<class T, size_t Degree, size_t NbModuli>
std::ostream& operator<<(std::ostream& outs, poly<T, Degree, NbModuli> const& p)
{
  bool first = true;
  std::string term;
  if (typeid(T) == typeid(uint64_t)) term = "ULL"; 
  else if (typeid(T) == typeid(uint32_t)) term = "UL"; 
  else term = "U"; 

  outs << "{ ";
  for(auto v : p)
  {
    if (first)
    {
      first = false;
      outs << v ;
    }
    else
    {
      outs << term << ", " << v;
    }
  }
  return outs << term << " }";
}
}
#endif
