use itertools::{Itertools, MultiProduct};
use numerica::{
    domains::{
        finite_field::FiniteFieldWorkspace,
        integer::{Integer, Z},
        rational::{Q, Rational},
    },
    tensors::matrix::{Matrix, Vector},
};

/// Let A^T =
/// [
///   (1,a,a^2,a^3,...),
///   (0, c, 0, 0, ...),
///   (0, 0, c, 0, ...),
///   ...
/// ]
/// Let b =
/// [
///   0,
///   b,
///   ab + b,
///   a^2 b + ab + b,
///   ...
/// ]
/// Then, for some integer LCG measurement vectors c, d, we want to solve
/// c <= Ax + b <= d
/// for integer vector solutions x.
/// We define u = c - b and v = d - b to simplify the problem to:
/// u <= Ax <= v
///
/// because the ranges of u and v are too big, if we were to consider coordinates in base A and
/// create a bounding box that contains the now distorted box formed by u and v, then trying all
/// points in this bounding box to check if they satisfy the equation would take too much time.
/// This is because the u-v box is now too distorted because our basis A is a terrible lattice basis.
/// For this, we find a more orthogonal basis with shorter vectors using LLL lattice reduction to obtain
/// a new basis A'. We now make the bounding box again and define it by it's min and max vectors
/// (not integers anymore):
/// min <= z <= max
///
/// They are defined by:
/// min = min { A'^-1 v | v is corner of u-v box }
/// max = max { A'^-1 v | v is corner of u-v box }
///
/// We want to test all integer solutions z to check if they also satisfy the original statement.
/// Any integer point z we test now, when multiplied by A' will produce a point which is also
/// reachable by Ax with an integer x. We just must ensure that this point is between u and v.
///
/// Once we find such a z, because we only care about the value of Ax and not of x, we only need to
/// compute A'z to have our solution. The first element will contain the seed.

#[derive(Debug, Clone)]
pub struct LCGCracker {
    b_vector: Vector<Z>,
    to_pos: Matrix<Z>,
    to_cords: Matrix<Q>,
    measurements: usize,
}

impl LCGCracker {
    pub fn new(a: Integer, b: Integer, c: Integer, measurements: usize) -> Self {
        let (to_pos, to_cords) = generate_basis(&a, &c, measurements);
        let b_vector = generate_b_vector(&a, &b, measurements);

        Self {
            to_pos,
            to_cords,
            b_vector,
            measurements,
        }
    }

    pub fn crack(&self, mut min: Vector<Z>, mut max: Vector<Z>) -> LCGSolutionIterator<'_> {
        min -= &self.b_vector;
        max -= &self.b_vector;

        let r_min = min.map(|x| Rational::from(x), Q);
        let r_max = max.map(|x| Rational::from(x), Q);

        // Now we must find the correct min and max corners for the bounding box.
        // We do this entry by entry by just picking the best combination of min/max entries to
        // minimize the selected entry. We don't need to bruteforce through all 2^n possibilities.
        //
        // For
        // z = Ax
        // if we only care about z_i, then we only need to look at the ith row of A, transposed and
        // scalar multiplied with x. One notices that if you want the smallest possible value, then
        // if A_i_j is positive, you want to pick the small value for x_j. If it's negative, you
        // want to pick the big value for x_j. This is assuming that x_j is positive. For negative
        // x_j, one must adjust.
        //
        // I believe there might be a better way, but for now I find this more intuitive and good
        // enough.
        //
        // We will also round them appropriately.

        let mut c_min = Vec::with_capacity(self.measurements);
        let mut c_max = Vec::with_capacity(self.measurements);

        for i in 0..self.measurements {
            // Finding entry i of c_min and c_max
            let mut entry_min = Rational::zero();
            let mut entry_max = Rational::zero();

            for j in 0..self.measurements {
                let val = &self.to_cords[(i as u32, j as u32)];
                if val.is_negative() {
                    entry_min += r_max[j].clone() * val;
                    entry_max += r_min[j].clone() * val;
                } else {
                    entry_min += r_min[j].clone() * val;
                    entry_max += r_max[j].clone() * val;
                }
            }

            c_min.push(entry_min.ceil());
            c_max.push(entry_max.floor());
        }

        let c_min = Vector::new(c_min, Z);
        let c_max = Vector::new(c_max, Z);

        LCGSolutionIterator::new(self, c_min, c_max, min, max)
    }
}

/// Iterates over the valid LCG solutions. Computation happens each time, it is not
/// pre-computed
/// If no solutions left or no solutions at all, returns none.
pub struct LCGSolutionIterator<'a> {
    cracker: &'a LCGCracker,
    inner: MultiProduct<IntegerIteratorInclusive>,
    min_pos: Vector<Z>,
    max_pos: Vector<Z>,
    start: Vector<Z>,
    end: Vector<Z>,
}

impl<'a> LCGSolutionIterator<'a> {
    fn new(
        cracker: &'a LCGCracker,
        start: Vector<Z>,
        end: Vector<Z>,
        min_pos: Vector<Z>,
        max_pos: Vector<Z>,
    ) -> Self {
        Self {
            cracker,
            inner: start
                .clone()
                .into_vec()
                .into_iter()
                .zip(end.clone().into_vec().into_iter())
                .map(|(a, b)| IntegerIteratorInclusive::new(a, b))
                .multi_cartesian_product(),
            min_pos,
            max_pos,
            start,
            end,
        }
    }

    pub fn print_space_information(&self) {
        println!("Space information:");
        let mut size = Integer::one();
        for (i, (min, max)) in self
            .start
            .clone()
            .into_vec()
            .into_iter()
            .zip(self.end.clone().into_vec())
            .enumerate()
        {
            let diff = (max.clone() - &min) + 1;
            println!("- Dimension {}: {}", i, diff);
            if diff < 0 {
                size = Integer::zero();
            } else {
                size *= diff;
            }
        }
        println!("- Size: {}", size);
    }
}

impl<'a> Iterator for LCGSolutionIterator<'a> {
    type Item = Vector<Z>;

    fn next(&mut self) -> Option<Self::Item> {
        'outer: while let Some(x) = self.inner.next() {
            // We must now check if min_pos <= to_pos * x <= max_pos
            // If it is, we return to_pos * x + b_vector

            let mut vec = Vec::new();

            for i in 0..self.cracker.measurements {
                let mut sum = Integer::zero();
                for j in 0..self.cracker.measurements {
                    sum += x[j].clone() * &self.cracker.to_pos[(i as u32, j as u32)];
                }
                if sum < self.min_pos[i] || sum > self.max_pos[i] {
                    continue 'outer;
                }
                vec.push(sum);
            }
            let mut vec = Vector::new(vec, Z);
            vec += &self.cracker.b_vector;

            return Some(vec);
        }

        None
    }
}

#[derive(Debug, Clone)]
pub struct IntegerIteratorInclusive {
    current: Integer,
    end: Integer,
}

impl IntegerIteratorInclusive {
    pub fn new(start: Integer, end: Integer) -> Self {
        Self {
            current: start - 1,
            end,
        }
    }
}

// TODO: Can we do this without cloning the integer?
impl Iterator for IntegerIteratorInclusive {
    type Item = Integer;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            None
        } else {
            self.current += 1;
            Some(self.current.clone())
        }
    }
}

/// Generate a vector b =
/// [
///   0,
///   b,
///   ab + b,
///   a^2 b + ab + b,
///   ...
/// ]
fn generate_b_vector(a: &Integer, b: &Integer, measurements: usize) -> Vector<Z> {
    let mut current = Integer::zero();
    let mut vec = Vec::with_capacity(measurements);

    vec.push(Integer::zero());

    for _ in 1..measurements {
        current *= a;
        current += b;

        vec.push(current.clone());
    }

    Vector::new(vec, Z)
}

/// Run LLL lattice reduction on the basis
/// [
///   (1,a,a^2,a^3,...),
///   (0, c, 0, 0, ...),
///   (0, 0, c, 0, ...),
///   ...
/// ]
/// to obtain a better, more orthogonal basis with shorter vectors.
///
/// It returns a tuple containing 2 matrices:
/// - A matrix to go from lattice coordinates in the good basis to a position
/// - The inverse of the above (opposite direction)
fn generate_basis(a: &Integer, c: &Integer, measurements: usize) -> (Matrix<Z>, Matrix<Q>) {
    let mut bad_basis: Vec<Vector<Z>> = Vec::with_capacity(measurements);

    let mut a_power = Integer::one();
    let mut first = Vec::with_capacity(measurements);
    first.push(a_power.clone());
    for _ in 1..measurements {
        a_power *= a;
        first.push(a_power.clone());
    }
    bad_basis.push(Vector::new(first, Z));

    for i in 1..measurements {
        let mut v = vec![Integer::zero(); measurements];
        v[i] = c.clone();
        bad_basis.push(Vector::new(v, Z));
    }

    let reduced = Vector::basis_reduction(&bad_basis, (99, 100).into());

    let good_basis_inv: Matrix<Q> = Matrix::from_nested_vec(
        reduced
            .clone()
            .into_iter()
            .map(|v| v.into_vec().into_iter().map(|x| x.into()).collect())
            .collect(),
        Q,
    )
    .expect("Bad basis matrix")
    .into_transposed()
    .inv()
    .expect("Not a basis!");

    let good_basis: Matrix<Z> =
        Matrix::from_nested_vec(reduced.into_iter().map(|v| v.into_vec()).collect(), Z)
            .expect("Bad basis matrix")
            .into_transposed();

    (good_basis, good_basis_inv)
}

// https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/Random.java
const JAVA_MULTIPLIER: u64 = 0x5DEECE66D;
const JAVA_ADDEND: u64 = 0xB;
const JAVA_MASK: u64 = (1u64 << 48) - 1;

pub struct JavaNextDoubleLCG(u64);

impl JavaNextDoubleLCG {
    pub fn new(seed: u64) -> Self {
        assert!(seed < (1u64 << 48));
        Self(seed)
    }

    const DOUBLE_UNIT: f64 = 1.0f64 / ((1u64 << 53) as f64);

    fn next_seed(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(JAVA_MULTIPLIER)
            .wrapping_add(JAVA_ADDEND)
            & JAVA_MASK;
        self.0
    }

    fn next_bits(&mut self, bits: u32) -> u64 {
        self.next_seed() >> (48 - bits)
    }

    /// Implemented as
    /// https://github.com/openjdk/jdk/blob/6ed82b3b15804a9e6b1d93513936db0d10e40a7e/src/java.base/share/classes/java/util/Random.java#L691
    pub fn next(&mut self) -> f64 {
        // Ensure correct order when calling next_bits.
        let first = self.next_bits(f64::MANTISSA_DIGITS - 27) << 27;
        ((first + self.next_bits(27)) as f64) * Self::DOUBLE_UNIT
    }

    pub fn get_seed(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct JavaNextDoubleCracker(LCGCracker);

impl JavaNextDoubleCracker {
    /// JAVA_MULTIPLIER^2 mod 2^48
    const A: u64 = 205749139540585;
    /// JAVA_MULTIPLIER * JAVA_ADDEND + JAVA_ADDEND
    const B: u64 = 277363943098;

    /// JAVA_MULTIPLIER^-1 mod 2^48
    const INVERSE_A: u64 = 246154705703781;
    /// (-JAVA_MULTIPLIER^-1 * JAVA_ADDEND) mod 2^48
    const INVERSE_B: u64 = 107048004364969;

    /// Note: Computation heavy.
    pub fn new(measurements: usize) -> Self {
        Self(LCGCracker::new(
            Self::A.into(),
            Self::B.into(),
            (1u64 << 48).into(),
            measurements,
        ))
    }

    /// Extract the first of the 2 next() calls
    fn reverse_calculation(val: f64, min: bool) -> u64 {
        const VAL: f64 = (1u64 << 53) as f64;
        let int = if min {
            (val * VAL).floor()
        } else {
            (val * VAL).ceil()
        } as u64;
        const MISSING: i32 = 48 - (53 - 27);
        let mut first = (int >> 27) << MISSING;
        if !min {
            first |= (1u64 << MISSING) - 1;
        }
        assert!(first < (1u64 << 48));
        first
    }

    /// Takes a list of measurements (min, max) where 0 <= min, max < 1.
    pub fn crack(
        &self,
        measurements: &[(f64, f64)],
        print: bool,
    ) -> impl Iterator<Item = JavaNextDoubleLCG> {
        assert_eq!(measurements.len(), self.0.measurements);
        let (min, max): (Vec<_>, Vec<_>) = measurements
            .into_iter()
            .map(|(a, b)| {
                (
                    Integer::from(Self::reverse_calculation(*a, true)),
                    Integer::from(Self::reverse_calculation(*b, false)),
                )
            })
            .unzip();

        let result_iter = self.0.crack(Vector::new(min, Z), Vector::new(max, Z));

        if print {
            result_iter.print_space_information();
        }

        result_iter.map(|x| {
            let next_seed = x
                .into_vec()
                .into_iter()
                .next()
                .expect("X is empty.")
                .to_u64()
                .expect("Solution doesn't fit bounds.");

            // Go back 1 step before the first seed.
            let seed = next_seed
                .wrapping_mul(Self::INVERSE_A)
                .wrapping_add(Self::INVERSE_B)
                & JAVA_MASK;

            JavaNextDoubleLCG(seed)
        })
    }
}
