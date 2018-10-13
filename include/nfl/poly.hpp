/* Copyright (C) 2015  Carlos Aguilar, Tancrède Lepoint, Adrien Guinet and Serge Guelton
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#ifndef NFL_POLY_HPP
#define NFL_POLY_HPP

/***
 * Polynomial class for NFL
 *
 * The poly class is used to manipulate blah blah
 */

#include "nfl/debug.hpp"
#include "nfl/meta.hpp"
#include "nfl/params.hpp"
#include "nfl/ops.hpp"
#include "nfl/arch.hpp"
#include "nfl/prng/fastrandombytes.h"
#include "nfl/prng/FastGaussianNoise.hpp"

#include <iostream>
#include <algorithm>
#include <gmpxx.h>
#include <array>


namespace nfl {

/***
 * Generators to initialize random polynomials
 */
struct uniform {};

struct non_uniform {
  uint64_t upper_bound;
  uint64_t amplifier;
  non_uniform(uint64_t ub) : upper_bound{ub}, amplifier{1} {}
  non_uniform(uint64_t ub, uint64_t amp) : upper_bound{ub}, amplifier{amp} {}
};

struct hwt_dist { // hamming weight distribution.
  uint32_t hwt;
  hwt_dist(uint32_t hwt_) : hwt(hwt_) {}
};

struct ZO_dist { // zero distribution.
  uint8_t rho; // P(1) = P(-1) = (rho/0xFF)/2, P(0) = 1 - P(1) - P(-1)
  ZO_dist(uint8_t rho_ = 0x7F) : rho(rho_) {}
};

template<class in_class, class out_class, unsigned _lu_depth>
struct gaussian {
  FastGaussianNoise<in_class, out_class, _lu_depth> *fg_prng;
  uint64_t amplifier;
  gaussian(FastGaussianNoise<in_class, out_class, _lu_depth> *prng) : fg_prng{prng}, amplifier{1} {}
  gaussian(FastGaussianNoise<in_class, out_class, _lu_depth> *prng, uint64_t amp) : fg_prng{prng}, amplifier{amp} {}
};

// Forward declaration for proxy class used in tests to access poly
// protected/private function members
namespace tests {

template <class P>
class poly_tests_proxy;

} // tests

/* Core polynomial class an array of value_types
 * No indication of whether these are coefficients or values (in NTT form)
 * The developer must keep track of which representation each object is under
 */
template<class T, size_t Degree, size_t NbModuli>
class poly {

  template <class P> friend class tests::poly_tests_proxy;

  static constexpr size_t N = Degree * NbModuli;
  T _data[N] __attribute__((aligned(32)));

public:
  using value_type = typename params<T>::value_type;
  using greater_value_type = typename params<T>::greater_value_type;
  using signed_value_type = typename params<T>::signed_value_type;
  using pointer_type = T*;
  using const_pointer_type = T const*;

  using iterator = pointer_type;
  using const_iterator = const_pointer_type;

  using simd_mode = CC_SIMD;
  static constexpr size_t degree = Degree;
  static constexpr size_t nmoduli = NbModuli;
  static constexpr size_t nbits = params<T>::kModulusBitsize;
  static constexpr size_t aggregated_modulus_bit_size = NbModuli * nbits;

public:
  /* constructors
   */
  poly();
  poly(uniform const& mode);
  poly(non_uniform const& mode);
  poly(hwt_dist const& mode);
  poly(ZO_dist const& mode);
  poly(value_type v, bool reduce_coeffs = true);
  poly(std::initializer_list<value_type> values, bool reduce_coeffs = true);
  template <class It> poly(It first, It last, bool reduce_coeffs = true);
  template <class Op, class... Args> poly(ops::expr<Op, Args...> const& expr);
  template <class in_class, unsigned _lu_depth> poly(gaussian<in_class, T, _lu_depth> const& mode);

  void set(uniform const& mode);
  void set(non_uniform const& mode);
  void set(hwt_dist const& mode);
  void set(ZO_dist const& mode);
  void set(value_type v, bool reduce_coeffs = true);
  void set(std::initializer_list<value_type> values, bool reduce_coeffs = true);
  template <class It> void set(It first, It last, bool reduce_coeffs = true);
  template <class in_class, unsigned _lu_depth> void set(gaussian<in_class, T, _lu_depth> const& mode);
  
  /* assignment
   */
  poly& operator=(value_type v) { set(v); return *this; }
  poly& operator=(uniform const& mode) { set(mode); return *this; }
  poly& operator=(non_uniform const& mode) { set(mode); return *this; }
  poly& operator=(hwt_dist const& mode) { set(mode); return *this; }
  poly& operator=(ZO_dist const& mode) { set(mode); return *this; }
  poly& operator=(std::initializer_list<value_type> values) { set(values); return *this; }
  template <class in_class, unsigned _lu_depth> poly& operator=(gaussian<in_class, T, _lu_depth> const& mode) { set(mode); return *this; }
  template <class Op, class... Args> poly& operator=(ops::expr<Op, Args...> const& expr);

  /* conversion operators
   */
  explicit operator bool() const;

  /* iterators
   */
  iterator begin() { return std::begin(_data); }
  iterator end() { return std::end(_data); }
  const_iterator begin() const { return std::begin(_data); }
  const_iterator end() const { return std::end(_data); }
  const_iterator cbegin() const { return std::begin(_data); }
  const_iterator cend() const { return std::end(_data); }

  /* polynomial indexing
   */
  value_type const& operator()(size_t cm, size_t i) const { return _data[cm * degree + i]; }
  value_type& operator()(size_t cm, size_t i) { return _data[cm * degree + i]; }
  template<class M> auto load(size_t cm, size_t i) const -> decltype(M::load(&(this->operator()(cm, i)))) { return M::load(&(*this)(cm, i)); }

  /* misc
   */
  pointer_type data() { return _data; }
  static constexpr value_type get_modulus(size_t n) { return params<T>::P[n]; }

  // Serialization API
  //
  // Serialization of polynomials is not portable in terms of "cross
  // architecture portability".
  // Therefore, serialization should not be used in the rare cases a big-endian
  // machine will use NFLlib and communicate serialized data to a little-endian
  // machine.
  
  /* manual serializers
  */
  void serialize_manually(std::ostream& outputstream) {
    outputstream.write(reinterpret_cast<char*>(_data), N * sizeof(T));
  }
  void deserialize_manually(std::istream& inputstream) {
    inputstream.read(reinterpret_cast<char*>(_data), N * sizeof(T));
  }

  /* serializer (cereal)
  */
  template<class Archive> void serialize(Archive & archive) { 
    archive( _data ); // serialize coefficients by passing them to the archive
  }
}  __attribute__((aligned(32)));

/* misc type adaptor
 */
template<class T, size_t Degree, size_t AggregatedModulusBitSize>
using poly_from_modulus = poly<T, Degree, AggregatedModulusBitSize / params<T>::kModulusBitsize>;

/* stream operator
 */
template<class T, size_t Degree, size_t NbModuli>
std::ostream& operator<<(std::ostream& os, nfl::poly<T, Degree, NbModuli> const& p);
}

#include "nfl/core.hpp"

#endif
