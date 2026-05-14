// dual.rs — scalar dual number for forward-mode automatic differentiation
//
// Dual = a + b·ε   where ε² = 0
//
// To compute column j of the Jacobian ∂f/∂x:
//   set xd[j] = Dual::new(x[j], 1.0),  all others = Dual::from_re(x[i])
//   call ode::<Dual>(xd, t)
//   column j = output.map(|d| d.eps)
//
// `Dual` satisfies the `Scalar` trait (Float + Copy + AddAssign) so every
// generic physics function can be instantiated with T = Dual without changes.

use std::num::FpCategory;
use std::ops::*;
use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub re:  f64,
    pub eps: f64,
}

impl Dual {
    #[inline] pub fn new(re: f64, eps: f64) -> Self { Self { re, eps } }
    #[inline] pub fn from_re(re: f64) -> Self { Self { re, eps: 0.0 } }
}

// ── Arithmetic operators ──────────────────────────────────────────────────────

impl Add for Dual {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self::new(self.re + r.re, self.eps + r.eps) }
}
impl Sub for Dual {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self::new(self.re - r.re, self.eps - r.eps) }
}
impl Mul for Dual {
    type Output = Self;
    // (a + bε)(c + dε) = ac + (ad + bc)ε
    fn mul(self, r: Self) -> Self {
        Self::new(self.re * r.re, self.re * r.eps + self.eps * r.re)
    }
}
impl Div for Dual {
    type Output = Self;
    // (a + bε)/(c + dε) = a/c + (bc − ad)/c²·ε
    fn div(self, r: Self) -> Self {
        let re  = self.re / r.re;
        let eps = (self.eps * r.re - self.re * r.eps) / (r.re * r.re);
        Self::new(re, eps)
    }
}
impl Rem for Dual {
    type Output = Self;
    fn rem(self, r: Self) -> Self { Self::new(self.re % r.re, self.eps) }
}
impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self { Self::new(-self.re, -self.eps) }
}

impl AddAssign for Dual {
    fn add_assign(&mut self, r: Self) { self.re += r.re; self.eps += r.eps; }
}
impl SubAssign for Dual {
    fn sub_assign(&mut self, r: Self) { self.re -= r.re; self.eps -= r.eps; }
}
impl MulAssign for Dual {
    fn mul_assign(&mut self, r: Self) {
        let new_eps = self.re * r.eps + self.eps * r.re;
        self.re  *= r.re;
        self.eps  = new_eps;
    }
}
impl DivAssign for Dual {
    fn div_assign(&mut self, r: Self) {
        let new_eps = (self.eps * r.re - self.re * r.eps) / (r.re * r.re);
        self.re  /= r.re;
        self.eps  = new_eps;
    }
}
impl RemAssign for Dual {
    fn rem_assign(&mut self, r: Self) { self.re %= r.re; }
}

// ── Ordering (by real part) ───────────────────────────────────────────────────

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

// ── Zero / One ────────────────────────────────────────────────────────────────

impl Zero for Dual {
    fn zero() -> Self { Self::from_re(0.0) }
    fn is_zero(&self) -> bool { self.re == 0.0 && self.eps == 0.0 }
}
impl One for Dual {
    fn one() -> Self { Self::from_re(1.0) }
}

// ── Num ───────────────────────────────────────────────────────────────────────

impl Num for Dual {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f64::from_str_radix(s, radix).map(Self::from_re)
    }
}

// ── NumCast / ToPrimitive / FromPrimitive ─────────────────────────────────────

impl ToPrimitive for Dual {
    fn to_i64(&self) -> Option<i64> { self.re.to_i64() }
    fn to_u64(&self) -> Option<u64> { self.re.to_u64() }
    fn to_f64(&self) -> Option<f64> { Some(self.re) }
    fn to_f32(&self) -> Option<f32> { Some(self.re as f32) }
}
impl FromPrimitive for Dual {
    fn from_i64(n: i64) -> Option<Self> { Some(Self::from_re(n as f64)) }
    fn from_u64(n: u64) -> Option<Self> { Some(Self::from_re(n as f64)) }
    fn from_f64(v: f64) -> Option<Self> { Some(Self::from_re(v)) }
    fn from_f32(v: f32) -> Option<Self> { Some(Self::from_re(v as f64)) }
}
impl NumCast for Dual {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> { n.to_f64().map(Self::from_re) }
}

// ── Float ─────────────────────────────────────────────────────────────────────

impl Float for Dual {
    // constants — eps = 0 (not differentiable)
    fn nan()               -> Self { Self::from_re(f64::NAN) }
    fn infinity()          -> Self { Self::from_re(f64::INFINITY) }
    fn neg_infinity()      -> Self { Self::from_re(f64::NEG_INFINITY) }
    fn neg_zero()          -> Self { Self::from_re(-0.0) }
    fn min_value()         -> Self { Self::from_re(f64::MIN) }
    fn min_positive_value()-> Self { Self::from_re(f64::MIN_POSITIVE) }
    fn max_value()         -> Self { Self::from_re(f64::MAX) }
    fn epsilon()           -> Self { Self::from_re(f64::EPSILON) }

    // classification — based on real part
    fn is_nan(self)        -> bool { self.re.is_nan() }
    fn is_infinite(self)   -> bool { self.re.is_infinite() }
    fn is_finite(self)     -> bool { self.re.is_finite() }
    fn is_normal(self)     -> bool { self.re.is_normal() }
    fn is_sign_positive(self) -> bool { self.re.is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.re.is_sign_negative() }
    fn classify(self) -> FpCategory { self.re.classify() }
    fn integer_decode(self) -> (u64, i16, i8) { self.re.integer_decode() }

    // rounding — discontinuous, eps = 0
    fn floor(self) -> Self { Self::from_re(self.re.floor()) }
    fn ceil(self)  -> Self { Self::from_re(self.re.ceil()) }
    fn round(self) -> Self { Self::from_re(self.re.round()) }
    fn trunc(self) -> Self { Self::from_re(self.re.trunc()) }
    fn fract(self) -> Self { Self::from_re(self.re.fract()) }

    // sign
    fn abs(self)    -> Self { Self::new(self.re.abs(), self.eps * self.re.signum()) }
    fn signum(self) -> Self { Self::from_re(self.re.signum()) }
    fn abs_sub(self, other: Self) -> Self {
        if self.re > other.re { self - other } else { Self::zero() }
    }

    // basic algebra
    fn recip(self) -> Self {
        let re  = self.re.recip();
        let eps = -self.eps / (self.re * self.re);
        Self::new(re, eps)
    }
    fn mul_add(self, a: Self, b: Self) -> Self { self * a + b }
    fn max(self, other: Self) -> Self { if self.re >= other.re { self } else { other } }
    fn min(self, other: Self) -> Self { if self.re <= other.re { self } else { other } }

    // powers
    fn powi(self, n: i32) -> Self {
        // d[x^0]/dx = 0 always; avoid 0 * inf = NaN when re = 0
        if n == 0 { return Self::one(); }
        let re  = self.re.powi(n);
        let eps = n as f64 * self.re.powi(n - 1) * self.eps;
        Self::new(re, eps)
    }
    fn powf(self, n: Self) -> Self {
        let re  = self.re.powf(n.re);
        let eps = re * (n.re * self.eps / self.re + self.re.ln() * n.eps);
        Self::new(re, eps)
    }
    fn sqrt(self) -> Self {
        let re = self.re.sqrt();
        // avoid 0/0 = NaN; d[sqrt(x)]/dx at x=0 is +inf (non-smooth), map to 0
        let eps = if re == 0.0 { 0.0 } else { self.eps / (2.0 * re) };
        Self::new(re, eps)
    }
    fn cbrt(self) -> Self {
        let re  = self.re.cbrt();
        let eps = self.eps / (3.0 * re * re);
        Self::new(re, eps)
    }
    fn hypot(self, other: Self) -> Self {
        let h   = self.re.hypot(other.re);
        let eps = (self.re * self.eps + other.re * other.eps) / h;
        Self::new(h, eps)
    }

    // exponential / logarithm
    fn exp(self)    -> Self { let re = self.re.exp();  Self::new(re, self.eps * re) }
    fn exp2(self)   -> Self {
        let re = self.re.exp2();
        Self::new(re, self.eps * re * 2.0_f64.ln())
    }
    fn exp_m1(self) -> Self {
        Self::new(self.re.exp_m1(), self.eps * self.re.exp())
    }
    fn ln(self)     -> Self { Self::new(self.re.ln(),    self.eps / self.re) }
    fn log2(self)   -> Self { Self::new(self.re.log2(),  self.eps / (self.re * 2.0_f64.ln())) }
    fn log10(self)  -> Self { Self::new(self.re.log10(), self.eps / (self.re * 10.0_f64.ln())) }
    fn log(self, b: Self) -> Self { self.ln() / b.ln() }
    fn ln_1p(self)  -> Self { Self::new(self.re.ln_1p(), self.eps / (1.0 + self.re)) }

    // trig
    fn sin(self) -> Self { Self::new(self.re.sin(),  self.eps * self.re.cos()) }
    fn cos(self) -> Self { Self::new(self.re.cos(), -self.eps * self.re.sin()) }
    fn tan(self) -> Self {
        let (s, c) = self.re.sin_cos();
        Self::new(s / c, self.eps / (c * c))
    }
    fn sin_cos(self) -> (Self, Self) { (self.sin(), self.cos()) }
    fn asin(self) -> Self {
        Self::new(self.re.asin(), self.eps / (1.0 - self.re * self.re).sqrt())
    }
    fn acos(self) -> Self {
        Self::new(self.re.acos(), -self.eps / (1.0 - self.re * self.re).sqrt())
    }
    fn atan(self) -> Self {
        Self::new(self.re.atan(), self.eps / (1.0 + self.re * self.re))
    }
    fn atan2(self, x: Self) -> Self {
        // ∂atan2(y,x)/∂y = x/(x²+y²),  ∂/∂x = -y/(x²+y²)
        let d   = self.re * self.re + x.re * x.re;
        let eps = (self.eps * x.re - x.eps * self.re) / d;
        Self::new(self.re.atan2(x.re), eps)
    }

    // hyperbolic
    fn sinh(self)  -> Self { Self::new(self.re.sinh(),  self.eps * self.re.cosh()) }
    fn cosh(self)  -> Self { Self::new(self.re.cosh(),  self.eps * self.re.sinh()) }
    fn tanh(self)  -> Self {
        let t = self.re.tanh();
        Self::new(t, self.eps * (1.0 - t * t))
    }
    fn asinh(self) -> Self {
        Self::new(self.re.asinh(), self.eps / (self.re * self.re + 1.0).sqrt())
    }
    fn acosh(self) -> Self {
        Self::new(self.re.acosh(), self.eps / (self.re * self.re - 1.0).sqrt())
    }
    fn atanh(self) -> Self {
        Self::new(self.re.atanh(), self.eps / (1.0 - self.re * self.re))
    }

    fn to_degrees(self) -> Self {
        const D: f64 = 180.0 / std::f64::consts::PI;
        Self::new(self.re.to_degrees(), self.eps * D)
    }
    fn to_radians(self) -> Self {
        const R: f64 = std::f64::consts::PI / 180.0;
        Self::new(self.re.to_radians(), self.eps * R)
    }
}