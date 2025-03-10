// Brainfuck bytecode compiler by ekstdo
#![feature(trait_alias)]
#![allow(non_camel_case_types)]
use std::env;
use std::fmt::Debug;
use std::fs;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::num::Wrapping;
use std::cmp::Ord;
use std::path::Path;
mod ludecomp;
mod utils;
use ludecomp::{addi_vec, Matrix};

use crate::utils::{shift_bmap, multinv, w8, Zero, One};

// while thinking about a useful representation of a lot of 
// brainfuck instructions, we may come to the conclusion of 
// a matrix for affine linear transformation with homogenous 
// coordinates:
//
// simple additions are just affine transformation
//
// setting variables to zero is just setting the current row
// in the matrix to zero and is common in brainfuck as
// [+] or [-] (or in theoretical terms a matrix multiplication
// just a multiplication with the row being zero)
//
// adding a multiplication to a cell, commonly done by: 
// [->+++<] is just setting an element within the matrix to
// change the linear transformation 
//
// we can also extend that method for [--->+++++<] by using 
// multiplicative inverses within $\mathbb F_{128}$
// with the extended euclidian algorithm
//
//
// By closure of the affine linear transformations we can
// represent any finite combination of them with one 
// affine linear matrix! Therefore being able to potentially
// minimize the amount of computation or at least bring it to
// a more manageable form to do so
//
// Caveats: This is limited by exponentiation or higher order 
// operations and is separated by IO Operations
//
// general trait to represent the affine linear transformations
pub trait BFAffineT<T: Copy + Ord>: Sized {
    // generates the identity representation of 
    fn new_ident() -> Self; 
    // ++++ or ----- or any combination of that
    fn add_const(&mut self, i: T, v: w8);
    // sets the row to zero, equivalent to setting the variable
    // after the operation to 0, for [-] and [+] and [+++] etc.
    fn set_zero(&mut self, i: T);
    fn rm_row(&mut self, i: T);
    fn rm_column(&mut self, i: T);
    fn set_const(&mut self, i: T, v: w8) {
        self.set_zero(i);
        self.add_const(i, v);
    }
    // multiplies an affine matrix to a variable to get a affine linear
    // matrix
    fn mul_var(&mut self, i: isize, v: w8);
    // just adds the value to the matrix, disregarding the fact
    // that the src may have been transformed beforehand
    fn add_mul_raw(&mut self, dest: T, src: T, v: w8);
    // can be thought of as a merge operation
    // note however, that it goes from right to left
    fn matmul(&self, other: &Self) -> Self;
    // actually adds the multiplied value to the matrix
    //
    // default implementation is a pretty slow approach
    fn add_mul(&mut self, dest: T, src: T, v: w8) {
        if v.0 == 0 { return; }
        let mut tmp = Self::new_ident();
        tmp.add_mul_raw(dest, src, v);
        self.matmul(&mut tmp);
    }
    fn set_mul(&mut self, dest: T, src: T, v: w8) {
        self.set_zero(dest);
        self.add_mul(dest, src, v);
    }

    fn is_ident(&self) -> bool;
    fn is_affine(&self) -> bool;
    fn is_affine_at0(&self) -> bool;
    fn affine_zero_lins(&self) -> Vec<T>;
    fn is_affine_zero_lin(&self, i: T) -> bool;
    fn zero_dep(&self) -> Vec<isize>; // indices, that only depend on 0 or a constant
    fn is_sure_ident(&self, i: T) -> bool;
    fn is_sure0(&self, i: T) -> bool;

    fn get_affine_raw(&self, i: T) -> Option<w8>;
    fn get_affine(&self, i: T) -> w8 {
        self.get_affine_raw(i).unwrap_or(w8::ZERO)
    }
    fn get_mat(&self, i:T, j:T) -> w8 {
        self.get_mat_raw(i, j).unwrap_or(if i == j { w8::ONE } else { w8::ZERO })
    }
    fn get_mat_raw(&self, i:T, j:T) -> Option<w8>;
    fn get_involved(&self) -> BTreeSet<T> {
        self.get_involved_lin().union(&self.get_involved_aff()).copied().collect()
    }
    fn get_involved_lin(&self) -> BTreeSet<T>;
    fn get_involved_aff(&self) -> BTreeSet<T>;

    fn shift_keys(&mut self, by: T);
    fn unset_linear(&mut self);

    fn cleanup(&mut self);
    fn to_opcode(self) -> Vec<Opcode>;
}


// simple and naive approach to represent the arbitrary dimensional matrix
#[derive(Clone)]
pub struct BFAddMap { 
    // default is the zero vector
    affine: BTreeMap<isize, w8>,

    // default is identity matrix
    matrix: Matrix,
}

impl std::fmt::Debug for BFAddMap {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        let involved_indices = self.get_involved();
        for i in &involved_indices {
            write!(f, "{}: │ ", i)?;
            for j in &involved_indices {
                match self.get_mat_raw(*i, *j) {
                    Some(x) => write!(f, "{} ", x),
                    None => write!(f, "  ")
                }?;
            }
            match self.get_affine_raw(*i) {
                Some(x) => writeln!(f, "║ {} │", x),
                None => writeln!(f, "║   │")
            }?;
        }

        Ok(())
    }
}

impl BFAffineT<isize> for BFAddMap {
    fn new_ident() -> Self {
        Self {
            affine: BTreeMap::new(),
            matrix: Matrix::new(),
        }
    }

    fn add_const(&mut self, i: isize, v: w8) {
        self.affine.entry(i).and_modify(|e| *e += v).or_insert(v);
        if self.affine[&i] == w8::ZERO {
            self.affine.remove(&i);
        }
    }

    fn set_zero(&mut self, i:isize) {
        self.affine.remove(&i);
        self.matrix.zero_row(i);
    }

    fn add_mul(&mut self, dest: isize, src: isize, v: w8) {
        self.matrix.add_mul_row(dest, src, v);
    }

    fn add_mul_raw(&mut self, dest: isize, src: isize, v: w8) {
        self.matrix.set(dest, src, v);
    }

    fn get_affine_raw(&self, i: isize) -> Option<w8> {
        self.affine.get(&i).copied()
    }

    fn get_mat_raw(&self, i: isize, j: isize) -> Option<w8> {
        self.matrix.get_raw(i, j)
    }

    fn is_ident(&self) -> bool {
        self.affine.is_empty() && self.matrix.inner.is_empty()
    }

    fn is_affine(&self) -> bool {
        self.matrix.inner.is_empty()
    }

    fn is_affine_at0(&self) -> bool {
        !self.matrix.inner.contains_key(&0)
    }

    fn is_affine_zero_lin(&self, i: isize) -> bool {
        self.matrix.get(i, i) == w8::ZERO && self.matrix.inner.get(&i).map_or(false, |x| x.values().all(|x| *x == w8::ZERO))
    }

    fn affine_zero_lins(&self) -> Vec<isize> {
        self.matrix.rows().filter(|x| self.is_affine_zero_lin(**x)).copied().collect()
    }

    fn zero_dep(&self) -> Vec<isize> {
        let mut result = Vec::new();
        for (k, row) in &self.matrix.inner {
            if row.contains_key(&0) && row.len() == 1 {
                result.push(*k);
            }
        }
        result
    }

    fn rm_row(&mut self, i: isize) {
        self.affine.remove(&i);
        self.matrix.inner.remove(&i);
    }

    fn rm_column(&mut self, i: isize) {
        self.matrix.rm_column(i);
    }

    fn is_sure_ident(&self, i: isize) -> bool {
        !self.affine.contains_key(&i) && !self.matrix.inner.contains_key(&i)
    }

    fn is_sure0(&self, i: isize) -> bool {
        !self.affine.contains_key(&i) && self.is_affine_zero_lin(i)
    }

    // might cache this instead
    fn shift_keys(&mut self, by: isize) {
        shift_bmap(&mut self.affine, by);
        self.matrix.shift_key(by);
    }

    // very inefficient, O(n^3 log n) with a rel. high constant, should be optimized in the future!
    //
    // ( A b ) ( A' b' ) = ( A A'      A b' + b )
    // ( 0 1 ) ( 0  1  ) = ( 0         1        )
    fn matmul(&self, other: &Self) -> Self {
        // transposing the other matrix 
        let mut aa_ = self.matrix.matmul(&other.matrix);
        aa_.optimize_all();
        let mut ab_ = self.matrix.vecmul(&other.affine);
        addi_vec(&mut ab_, &self.affine);

        Self { matrix: aa_, affine: ab_ }
    }

    // turns affine transform into linear transform, i.e. x3 += 3 -> x3 += 3 * x1
    // note that behaviour about the variable itself is undefined behaviour, as
    // mul_var should be followed by a set0
    fn mul_var(&mut self, ind: isize, x: w8) {
        for (i, v) in &self.affine {
            self.matrix.set(*i, ind, v * x);
        }
        self.affine.clear();
    }
    
    fn get_involved_lin(&self) -> BTreeSet<isize> {
        let mut result = self.matrix.columns();
        result.extend(self.matrix.rows());
        result
    }

    fn get_involved_aff(&self) -> BTreeSet<isize> {
        self.affine.keys().cloned().collect()
    }

    fn unset_linear(&mut self) {
        self.matrix = Matrix::new();
    }

    fn cleanup(&mut self) {
        // adding 0 as a constant doesn't do anything
        self.affine.retain(|_, x| *x != w8::ZERO);
        self.matrix.optimize_all();
    }

    fn to_opcode(self) -> Vec<Opcode> {
        let (perm, mat_opcode) = self.matrix.to_opcode();
        let mut out_opcode = mat_opcode.into_iter().map(|x| match x {
            ludecomp::MatOpCode::Add(to, from, val) =>
            if val == w8::ONE {
                Opcode::AddCell((from - to) as i16, to as i32)
            } else if val == Wrapping(255) {
                Opcode::SubCell((from - to) as i16, to as i32)
            } else {
                Opcode::AddMul(val, (from - to) as i16, to as i32)
            },
            ludecomp::MatOpCode::Mul(to, val) => 
                if val == w8::ZERO {
                    Opcode::SetConst(w8::ZERO, to as i32)
                } else {
                    Opcode::MulConst(val, to as i32)
                }
            
        }).collect::<Vec<_>>();

        let rperm = perm.reverse();
        let mut affine = self.affine;
        rperm.apply(&mut affine);
        for (k, v) in affine {
            let pos = out_opcode.iter().position(|x| *x == Opcode::SetConst(w8::ZERO, k as i32));
            match pos {
                Some(pos) => {
                    let mut swapping_op = Opcode::SetConst(v, k as i32);
                    std::mem::swap(&mut out_opcode[pos], &mut swapping_op);
                },
                None => out_opcode.push(Opcode::AddConst(v, k as i32))
            }
        }
        let mut selected = BTreeSet::new();
        for (k, mut v) in &perm.inner {
            out_opcode.push(Opcode::Load(*k as i32));
            while !selected.contains(v) {
                selected.insert(v);
                out_opcode.push(Opcode::LoadSwap(*v as i32));
                v = perm.inner.get(v).unwrap();
            }
        }

        out_opcode
    }
}

#[derive(Debug)]
pub enum Optree<T: BFAffineT<isize>> {
    OffsetMap(T),
    Branch(Vec<Optree<T>>, isize, isize),
    Input(isize),
    Output(isize),
    DebugBreakpoint,
}

fn gen_ccode_op<T: BFAffineT<isize> + Clone + Debug>(optree: &Optree<T>, indent_level: usize) -> String {
    let indentation = "    ".repeat(indent_level);

    match optree {
        Optree::Input(i) => format!("{}t[{}] = getchar();\n", indentation, i),
        Optree::Output(i) => format!("{}putchar(t[{}]);\n", indentation, i),
        Optree::OffsetMap(m) => {
            let offset_map_ops = m.clone().to_opcode();
            let mut result = String::new();
            for op in offset_map_ops {
                result.push_str(&match op {
                    Opcode::AddCell(offset, to) => format!("{}t[{}] += t[{}];\n", indentation, to, to + offset as i32),
                    Opcode::SubCell(offset, to) => format!("{}t[{}] -= t[{}];\n", indentation, to, to + offset as i32),
                    Opcode::AddMul(val, offset, to) => format!("{}t[{}] += {} * t[{}];\n", indentation, to, val, to + offset as i32),
                    Opcode::SetConst(val, to) => format!("{}t[{}] = {};\n", indentation, to, val),
                    Opcode::MulConst(val, to) => format!("{}t[{}] *= {};\n", indentation, to, val),
                    Opcode::AddConst(val, to) => format!("{}t[{}] += {};\n", indentation, to, val),
                    Opcode::Load(from) => format!("{}reg = t[{}];\n", indentation, from),
                    Opcode::LoadSwap(from) => format!("{}tmp = reg; reg = t[{}]; t[{}] = tmp;\n", indentation, from, from),
                    _ => String::new()
                });
            }
            result
        },
        Optree::Branch(uo, preshift, itershift) => {
            let mut result = String::new();
            result.push_str(&format!("{}t += {};\n", indentation, preshift));
            result.push_str(&format!("{}while (*t) {{\n", indentation));
            for o in uo {
                result.push_str(&gen_ccode_op(o, indent_level + 1));
            }
            result.push_str(&format!("{}    t += {};\n{}}}\n", indentation, itershift, indentation));
            result

        },
        Optree::DebugBreakpoint => todo!(),
    }
}

fn gen_ccode<T: BFAffineT<isize> + Clone + Debug>(optrees: &Vec<Optree<T>>) -> String {
    let mut result = format!("#include<stdio.h>
unsigned char tape[{}];
int main() {{
    unsigned char* t = tape + {};
    unsigned char reg;
    unsigned char tmp;\n", u16::MAX, i16::MAX);

    for ot in optrees {
        result.push_str(&gen_ccode_op(ot, 1));
    }

    result.push_str("    return 0;\n}");
    result
}

fn gen_opcode<T: BFAffineT<isize> + Clone + Debug>(optree: &Vec<Optree<T>>) -> Vec<Opcode> {
    let mut result = Vec::new();

    let mut stack = optree.iter().rev().collect::<Vec<_>>();
    let mut jmp_stack = Vec::new();
    let mut label_counter: u32 = 0;
    let mut i16_offset: usize = 0;
    let mut min_i16: isize = i16::MIN as isize;
    let mut max_i16: isize = i16::MAX as isize;
    let check_in_range = |x: isize| min_i16 <= x && x <= max_i16;
    loop {
        // println!("remaining jmp stack: {:?}", jmp_stack);
        // println!("stack len: {:?}", stack.len());
        while let Some((when_len, cur_label_counter, shift)) = jmp_stack.last() {
            if *when_len == stack.len() {
                // println!("{when_len} {cur_label_counter} {shift} matched!");
                if *shift != 0 {
                    result.push(Opcode::Shift(*shift));
                }
                result.push(Opcode::Jnez(*cur_label_counter << 1));
                result.push(Opcode::Label((*cur_label_counter << 1) + 1u32));
                jmp_stack.pop();
            } else {
                break;
            }
        }

        let Some(cur_el) = stack.pop() else {break;};
        // println!("cur el: {:?} \n\n", &cur_el);

        match cur_el {
            // todo: Check for i32 precision loss from isize
            Optree::Input(i) => result.push(Opcode::Read(*i as i32)),
            Optree::DebugBreakpoint => result.push(Opcode::DebugBreakpoint),
            Optree::Output(i) => result.push(Opcode::Write(*i as i32)),
            Optree::OffsetMap(m) => result.extend(m.clone().to_opcode().into_iter()),
            Optree::Branch(v, pre_shift, post_shift) => {
                if *pre_shift != 0 {
                    result.push(Opcode::Shift(*pre_shift as i32));
                }
                result.push(Opcode::Jez((label_counter << 1) + 1));
                result.push(Opcode::Label(label_counter << 1));
                // when to insert the jump, where to jump to and how much to shift at each
                // iteration
                jmp_stack.push((stack.len(), label_counter, *post_shift as i32));
                stack.extend(v.iter().rev());
                label_counter += 1;
            }
        }
    }
    result
}

impl<T: BFAffineT<isize> + std::fmt::Debug> std::fmt::Display for Optree<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<T: BFAffineT<isize> + std::fmt::Debug> Optree<T> {
    fn shift(&mut self, by: isize) -> bool { // returns whether the shift should propagate
        match self {
            Optree::OffsetMap(bfaddmap) => { bfaddmap.shift_keys(by); true },
            Optree::Input(x) | Optree::Output(x) => { *x += by; true },
            Optree::Branch(_, x, _) => { *x += by; false },
            Optree::DebugBreakpoint => true
        }
    }

    fn size(&self) -> usize {
        match self {
            Optree::OffsetMap(_) | Optree::Input(_) | Optree::Output(_) => 1,
            Optree::Branch(a, _, _) => 1 + a.iter().map(|x| x.size()).sum::<usize>(),
            Optree::DebugBreakpoint => 0
        }
    }
}


pub fn compile<T: BFAffineT<isize>>(s: String) -> Vec<Optree<T>> {
    let mut inst = vec![vec![]];
    let mut current_inst = &mut inst[0];
    let mut current_offset_map = T::new_ident();
    let mut current_offset: isize = 0;
    let mut preshift = vec![0];
    for i in s.chars() {
        if i != '+' && i != '-' && i != '>' && i != '<' && !current_offset_map.is_ident() {
            current_inst.push(Optree::OffsetMap(current_offset_map));
            current_offset_map = T::new_ident();
        }
        match i {
            '+' => {
                current_offset_map.add_const(current_offset, w8::ONE);
            },
            '-' => {
                current_offset_map.add_const(current_offset, Wrapping(u8::MAX));
            },
            '<' => {
                current_offset -= 1;
            },
            '>' => {
                current_offset += 1;
            },
            '.' => {
                current_inst.push(Optree::Output(current_offset));
            },
            '#' => {
                current_inst.push(Optree::DebugBreakpoint);
            },
            ',' => {
                current_inst.push(Optree::Input(current_offset));
            },
            '[' => {
                preshift.push(current_offset);
                current_offset = 0;
                inst.push(vec![]);
                current_inst = inst.last_mut().unwrap();
            },
            ']' => {
                let postshift = current_offset;
                current_offset = 0;
                let a = inst.pop().unwrap();
                current_inst = inst.last_mut().unwrap();
                current_inst.push(Optree::Branch(a, preshift.pop().unwrap(), postshift));
            }
            _ => {}
        }
    }
    inst.pop().unwrap()
}

// Optimizations:
//
// 1. Branch([OffsetMap o], x, 0) 
//
//    i.e. a matrix multiplication gets repeated and the net shift within the loop is 0
//
//    a) o.is_affine() && o.affine[0] % 2 == 1 && o contains nothing else
//
//      e.g. [-], [+], [---], [+++] 
//
//      sets the variable to 0
//
//    b) o.is_affine() && o.affine[0] % 2 == 1
//
//      e.g. [->++<], [--->-<-], [--->->--<<]
//
//      sets the current pointer to 0 and multiplies other values by mult_inv(affine[0]) *
//      current[0]
//
//  => a) is a special case of b)
//
//    c) o contains affine only lines (can be applied to Branch([OffsetMap o, ...]) if ... doesn't
//    rely on t[b])
//
//      while t[0] {
//          t[a] = ... + a * t[b];
//          t[b] = q;
//      } 
//
//      We don't have to add t[b] at every iteration, if it's gonna be q at every iteration anyways
//
//      first do the if do while transform
//
//      if t[0] {
//          do {
//              t[a] = ... + a * t[b];
//              t[b] = q;
//          } while (t[0]);
//      }
//
//      to
//
//      if t[0] {
//          t[a] = ... + a * t[b];
//          t[b] = q;
//          while(t[0]) {
//              t[a] = ... + a * t[b];
//              t[b] = q;
//          }
//      }
//
//      to
//
//      if t[0] {
//          t[a] = ... + a * t[b];
//          t[b] = q;
//          while(t[0]) {
//              t[a] = ... + a * t[b];
//          }
//      }
//
//      to
//
//      if t[0] {
//          t[a] = ... + a * t[b];
//          t[b] = q;
//          while(t[0]) {
//              t[a] = ... + a * q;
//          }
//      }
//
//    d) o.is_affine_at0() && o.affine[0] % 2 == 1 && ...
//
//      this is a repeated matrix multiplication, which can indicate different things
//
//      e.g. the pattern x[temp0+x-]temp0[-[temp1+x++temp0-]x+temp1[temp0+temp1-]temp0] for x := x^2
//
//      becomes:
//
//      temp0 = x; x = 0;
//      while temp0 > 0 {
//          temp0 -= 1;
//          x += 2 * temp0;
//          x += 1;
//         temp1 = 0;
//      }
//      => linear then affine representation =>
//      while temp0 > 0 {
//          x += 2 * temp0;
//          x -= 1;
//          temp0 -= 1;
//          temp1 = 0;
//      }
//
//      which is just a repeated matrix multiplication and represents a square
//
//      we can try to generalize this to any case, where the counter variable k (at offset "0") is only
//      transformed affine by r and every other variable either depends on that counter (with factor a) or a constant (c)
//      (and not each other)
//
//      the value of the counter at each iteration i is then:
//
//      k + i * r
//
//      and the end value of the other variable is:
//
//      sum_(i = 0)^(n - 1) (k + i * r) * a + c       (with n being the number of iterations)
//
//      = n * c + a sum_(i = 0)^(n - 1) (k + i * r)
//      = n * c + a k n + a r  sum_(i = 0)^(n - 1) i
//      = n * c + a k n + a r  n(n - 1)/2
//
//      for the x = x^2 example, we have: n = x, c = -1, k = n, a = 2, r = -1
//      = -n + 2 n^2 - n(n - 1)
//      = -n + 2 n^2 - n(n - 1)
//      = n (-1 + 2 n - n + 1)
//      = n n = x^2
//
//      we can also solve for n:
//
//      k + n * r = 0 => n = - k / r (therefore r has to be odd, otherwise, n might be infinite)
//
//      n * c + a k n + a r  n(n - 1)/2 becomes 
//      - k /r * c + a k (- k /r) + a r (- k / r) (- k/r - 1) / 2
//      = - c k / r - a k^2 / r + a k (k/r + 1) / 2
//      = - c k / r - a k  (k/r) 2 / 2 + a k (k/r + 1) / 2
//      = - c k / r + a k (-k/r + 1) / 2
//      = a k (-k/r + 1) / 2 - c k / r 
//
//
//      we can set s = -1/r at compile time and l = s k  at runtime
//      = a k (s k + 1) / 2 + c s k
////
//      (k is even -> k (s k + 1) is even -> divisible by 2)
//      (k is odd -> s k is odd -> s k + 1 is even -> k (s k + 1) is even -> divisible by 2)
//
//      How do we compute it? (as much in compile time as possible!)
//
//      if a is even: b = a/2 is computed at compiletime:
//
//          a k (l + 1) / 2 + c l
//          = b k (l + 1)  + c l
//
//          compiletime:
//          b = a / 2 
//          s = -1/r
//
//          runtime:
//          l = s * k
//          result += c * l
//          l += 1
//          l *= k 
//          result += b * l     => we only need one extra register (which we alr. do for
//                                 permutations) and 5 lin comb instructions
//
//          if s is 1 (which we can detect at compiletime):
//
//              b k (k + 1)  + c k = b k^2 + (b + c) k
//
//              l = k * k
//              result += b * l
//              result += (b + c) k      // b + c is computed at compiletime
//                                       // reduced to 3 lin comb inst
//
//      if a is odd, we can't incorporate /2 into a, but have to watch it at runtime:
//          if k % 2 == 0 { (k/2) * ... } else {(s k + 1)/2 * ...}
//
//          s = -1/r
//
//          runtime:
//          l = s * k
//          result += c * l
//          l += 1
//          if (k % 2 == 0) { // jump if even
//              l *= k;
//              l >>= 1;
//          } else {          // jump
//              l >>= 1;
//              l *= k;
//          }
//          result += a * l; // 6 lin comb inst, 2 jump inst
//
//
//  2. [OffsetMap o1, OffsetMap o2]
//
//    can be optimized to o2 * o1 (notice that matmul is the mathematical representation,
//    therefore o2 gets applied first)
//
// 3. [OffsetMap o, Input i]
//
//    Input overwrites the i-th entry anyways, so o can remove the i-th row
//
// 4. [OffsetMap o1, Output o2, OffsetMap o3]
//
//    if o3 is independant of o2 (i.e. doesn't contain anything other than identity in the o2-th
//    row), we can propagate it over to [o1, o3, o2]
//
// 5. [Branch(_, _, _), OffsetMap o1]
//
//    The value after a branch will always be 0, so we can set the column 0 in o1 to 0
//
// 6. [Branch(_, _, _), Branch(_, preshift = 0, _)]
//
//    if the preshift is 0, it can be left out, as the current val has to 0 to leave the first
//    loop, which would immediately terminate the second one.
//
// 7. [OffsetMap o1, Branch(_, preshift, _)]
//
//    if the preshift row of o1 is all 0 (not empty, as empty would be identity, but really 0)
//    Branch can be deleted
pub fn optimize<T: BFAffineT<isize> + std::fmt::Debug>(unoptimized: &mut Vec<Optree<T>>, layer: usize) -> (bool, isize) {
    let mut index = 0;
    let mut total_shift = 0;
    let mut changed = false;
    while index < unoptimized.len() {
        let mut prev = false;
        let l = unoptimized.len();
        match &mut unoptimized[index] {
            Optree::OffsetMap(_) if index < l - 1 => {
                let next_el = unoptimized.get(index + 1);
                // see 2.
                if let Some(Optree::OffsetMap(_)) = next_el {
                    let Optree::OffsetMap(mut el) = unoptimized.remove(index + 1) else { panic!("wrong check!") };
                    let Some(Optree::OffsetMap(ref mut bfaddmap)) = unoptimized.get_mut(index) else {panic!("wrong check!")};
                    el = el.matmul(bfaddmap);
                    std::mem::swap(&mut el, bfaddmap);
                    prev = true;

                    changed = true;
                }
                // see 3.
                else if let Some(Optree::Input(into)) = next_el {
                    let into = *into;
                    let Some(Optree::OffsetMap(ref mut bfaddmap)) = unoptimized.get_mut(index) else {panic!("wrong check!")};
                    bfaddmap.rm_row(into);

                    changed = true;
                }
                // see 7.
                else if let Some(Optree::Branch(_, preshift, _)) = next_el {
                    let preshift = *preshift;
                    let Some(Optree::OffsetMap(ref mut bfaddmap)) = unoptimized.get_mut(index) else {panic!("wrong check!")};
                    let should_remove = bfaddmap.is_sure0(preshift);
                    if should_remove {
                        unoptimized.remove(index + 1);
                        prev = true;

                        changed = true;
                    }
                }
            },
            // see 1.
            Optree::Branch(ref mut uo, preshift, 0) if uo.len() == 1 => {
                let preshift_ = *preshift;
                match &mut uo[0] {
                    Optree::OffsetMap(ref mut m) if m.is_affine() => {
                        let a = m.get_affine(0);
                        if a.0 % 2 == 1 {
                            let b = {
                                let mut b = T::new_ident();
                                std::mem::swap(m, &mut b);
                                let factor = multinv(256isize - a.0 as isize, 256) as u8;
                                b.mul_var(0, Wrapping(factor));
                                b.set_zero(0);
                                b
                            };

                            std::mem::swap(&mut unoptimized[index], &mut Optree::OffsetMap(b));
                            if preshift_ != 0 {
                                let mut j = index;
                                while j < l && unoptimized[j].shift(preshift_) {
                                    j += 1;
                                }
                                if j == l {
                                    total_shift += preshift_;
                                }
                            }
                            if index > 0 {
                                index -= 1;
                            }
                            prev = true;
                            changed = true;
                        }
                    }
                    Optree::OffsetMap(ref mut m) if m.is_affine_at0() => {
                        println!("hOi! {:?}", m.zero_dep());
                    }
                    _ => {
                        let Optree::Branch(uo, _preshift, ref mut iter_shift) = &mut unoptimized[index] else { panic!("wrong check!") };
                        let (inner_changed, inner_shift) = optimize(&mut *uo, layer + 1);
                        changed = changed || inner_changed;
                        *iter_shift += inner_shift;
                    }
                }
            },
            Optree::Branch(_uo, _, _) => {
                let (inner_changed, inner_shift) = {
                    let Optree::Branch(uo, _, _) = &mut unoptimized[index] else { panic!("wrong check!") };
                    optimize(&mut *uo, layer + 1)
                };
                changed = changed || inner_changed;
                let Optree::Branch(_, _, ref mut iter_shift) = &mut unoptimized[index] else { panic!("wrong check!") };
                *iter_shift += inner_shift;
                if inner_changed {
                    prev = true;
                }
            }
            _ => {}
        }
        if !prev {
            index += 1;
        }
    }
    if let Some(Optree::OffsetMap(ref mut m)) = unoptimized.get_mut(0) {
        if layer == 0 {
            m.unset_linear();
        }
    }
    (changed, total_shift)
}


// we can either store which instruction to jump to in Jnez/Jez/J (for byte code compilation)
// or store the label we want to jump to (for jit compilation)
//
// => for byte code compilation, the label gets ignored
//
// Structure:
//
// <OP> <w8> <i16|i16> <i32|i32|i32|i32>
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Opcode {
    // t and r have to be saved in a register
    AddConst(w8, i32),    // t[b] += b
                          // add BYTE [rdi], sil
    SetConst(w8, i32),    // t[b] = a
                          // mov BYTE [rdi], sil
    MulConst(w8, i32),    // t[b] *= a
                          // movzx   eax, BYTE PTR [rsi]
                          // imul    eax, edi
                          // mov     BYTE PTR [rsi], al
    AddMul(w8, i16, i32), // t[c] += t[c + b] * a
                          // movzx   eax, BYTE PTR [rdx+rcx]
                          // imul    eax, edi
                          // add     BYTE PTR [rsi+rcx], al
    AddCell(i16, i32),    // t[c] += t[c + b]
    SubCell(i16, i32),    // t[c] += t[c + b]
    Load(i32),            // t[a] = r
    LoadSwap(i32),        // r, t[a] = t[a], r
    Shift(i32),           // t = t + a
    ShiftBig(i32),        // t = t + a * i32::max
    Read(i32),            // t[a] = input_char()
    Write(i32),           // print(t[a])
    J(u32),               // goto a
    Jez(u32),             // if t[0] == 0: goto a
    Jnez(u32),            // if t[0] != 0: goto a
                          // jmp label;
    Label(u32),
    DebugBreakpoint,
    DebugData(u16, u32),  // range where in code we are
}

impl From<&Opcode> for u64 {
    fn from(value: &Opcode) -> Self {
        match value {
            Opcode::AddConst(v, c) =>  ((v.0 as u64) << 48) | ((*c as u32) as u64) ,
            Opcode::SetConst(v, c) =>  (1 << 56) | ((v.0 as u64) << 48) | ((*c as u32) as u64) ,
            Opcode::MulConst(v, c) =>  (2 << 56) | ((v.0 as u64) << 48) | ((*c as u32) as u64) ,
            Opcode::AddMul(v, o, c) =>  (3 << 56) | ((v.0 as u64) << 48) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::AddCell(o, c) =>  (4 << 56) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::SubCell(o, c) =>  (5 << 56) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::Load(c) =>  (6 << 56) | ((*c as u32) as u64) ,
            Opcode::LoadSwap(c) =>  (7 << 56) | ((*c as u32) as u64) ,
            Opcode::Shift(c) =>  (8 << 56) | ((*c as u32) as u64) ,
            Opcode::ShiftBig(c) =>  (9 << 56) | ((*c as u32) as u64) ,
            Opcode::Read(c) =>  (10 << 56) | ((*c as u32) as u64) ,
            Opcode::Write(c) =>  (11 << 56) | ((*c as u32) as u64) ,
            Opcode::J(c) =>  (12 << 56) | (*c as u64) ,
            Opcode::Jez(c) =>  (13 << 56) | (*c as u64) ,
            Opcode::Jnez(c) =>  (14 << 56) | (*c as u64) ,
            Opcode::Label(c) =>  (15 << 56) | (*c as u64) ,
            Opcode::DebugBreakpoint => 16 << 56,
            Opcode::DebugData(o, c) => (17 << 56) | ((*o as u64) << 32) | (*c as u64) ,
        }
    }
}

impl std::fmt::Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Opcode::Label(u) => write!(f, "label_{}:", u),
            Opcode::J(u) =>    write!(f, "\tj    label_{}", u),
            Opcode::Jez(u) =>  write!(f, "\tjez  label_{}", u),
            Opcode::Jnez(u) => write!(f, "\tjnez label_{}", u),
            Opcode::AddConst(val, wher) => write!(f, "\tt[{}] += {}", wher, val),
            Opcode::SetConst(val, wher) => write!(f, "\tt[{}]  = {}", wher, val),
            Opcode::MulConst(val, wher) => write!(f, "\tt[{}] *= {}", wher, val),
            Opcode::Shift(of) => write!(f, "\tt += {}", of),
            Opcode::ShiftBig(of) => write!(f, "\tt += {} * i16::max", of),
            Opcode::Read(to) => write!(f, "\tt[{}] = inputc()", to),
            Opcode::Write(from) => write!(f, "\tprintc(t[{}])", from),
            Opcode::Load(from) => write!(f, "\tr = t[{}]", from),
            Opcode::LoadSwap(from) => write!(f, "\tt[{}], r = r, t[{}]", from, from),
            Opcode::AddMul(val, offset, to) => write!(f, "\tt[{}] += t[{} + {} = {}] * {}", to, to, offset, *to + *offset as i32, val),
            Opcode::DebugBreakpoint => write!(f, "\tDEBUG!"),
            Opcode::AddCell(offset, to) => write!(f, "\tt[{}] += t[{} + {} = {}]", to, to, offset, *to + *offset as i32),
            Opcode::SubCell(offset, to) => write!(f, "\tt[{}] += t[{} + {} = {}]", to, to, offset, *to + *offset as i32),
            Opcode::DebugData(offset, from) => write!(f, "\t\twithin code[{} to {}]", from, from + *offset as u32)
        }
    }
}

fn optout_labels(opcode: &mut Vec<Opcode>) {
    let mut label_map = Vec::new(); // maps each label id to the index position
    let mut label_count = Vec::new(); // counts the amount of labels before that label
    let mut counter = 0;
    for (index, i) in opcode.iter().enumerate() {
        match i {
            Opcode::Label(k) => {
                counter += 1;
                while label_map.len() <= *k as usize {
                    label_map.push(0);
                    label_count.push(0);
                }
                label_map[*k as usize] = index;
                label_count[*k as usize] = counter;
            },
            _ => {}
        }
    }

    for i in opcode.iter_mut() {
        match i {
            Opcode::Jez(ref mut k) | Opcode::Jnez(ref mut k) | Opcode::J(ref mut k) => {
                *k = (label_map[*k as usize] - label_count[*k as usize]) as u32;
            },
            _ => {}
        }
    }

    opcode.retain(|x| match x { Opcode::Label(_) => false, _ => true });
}

fn simulate(mut opcode: Vec<Opcode>) {
    let mut tape: Vec<w8> = vec![w8::ZERO;u16::MAX as usize];
    let mut index = i16::MAX as usize;
    let mut pc = 0;
    let mut debug_counter = -1;
    let mut reg = w8::ZERO;

    optout_labels(&mut opcode);

    let Some(mut last_op) = opcode.first() else {return;};
    let mut print_buffer = String::new();
    let mut counter = 0;
    while let Some(op) = opcode.get(pc) {
        match op {
            Opcode::AddConst(val, to) => {tape[(index as isize + *to as isize) as usize] += *val;},
            Opcode::SetConst(val, to) => {tape[(index as isize + *to as isize) as usize] = *val;},
            Opcode::AddCell(offset, to) => {
                let readout = tape[(index as isize + *to as isize + *offset as isize) as usize];
                tape[(index as isize + *to as isize) as usize] += readout;
            },
            Opcode::SubCell(offset, to) => {
                let readout = tape[(index as isize + *to as isize + *offset as isize) as usize];
                tape[(index as isize + *to as isize) as usize] -= readout;
            },
            Opcode::AddMul(val, offset, to) => {
                let readout = tape[(index as isize + *to as isize + *offset as isize) as usize];
                tape[(index as isize + *to as isize) as usize] += readout * *val;
            },
            Opcode::MulConst(val, to) => {tape[(index as isize + *to as isize) as usize] *= *val;},
            Opcode::LoadSwap(from) => {
                let tmp = reg;
                reg = tape[(index as isize + *from as isize) as usize];
                tape[(index as isize + *from as isize) as usize] = tmp;  },
            Opcode::Load(from) => { reg = tape[(index as isize + *from as isize) as usize]; },
            Opcode::ShiftBig(_) => todo!(),
            Opcode::Shift(by) => {index = (index as isize + *by as isize) as usize;},
            Opcode::Read(to) => {
                tape[(index as isize + *to as isize) as usize] =  Wrapping(std::io::stdin().bytes().next().and_then(|x| x.ok()).unwrap_or_else(|| panic!("no input provided!")) );
            },
            Opcode::Write(from) => {
                let c = tape[(index as isize + *from as isize) as usize].0 as char;
                if c == '\n' {
                    println!("{}", print_buffer);
                    print_buffer.clear();
                } else {
                    print_buffer.push(c)
                }
            },
            Opcode::Jnez(a) => { if tape[index] != w8::ZERO { pc = *a as usize - 1; } },
            Opcode::Jez(a) => { if tape[index] == w8::ZERO { pc = *a as usize - 1; } },
            Opcode::J(a) => { pc = *a as usize - 1; },
            Opcode::Label(_) => {},
            Opcode::DebugBreakpoint => { debug_counter = 0; }
            Opcode::DebugData(_, _) => todo!(),

        }
        if debug_counter == 0 {
            let mut line = String::new();
            print!("{}", print_buffer);
            print_buffer.clear();
            loop {
                print!("\x1B[32mDEBUG[{}@ {}] > ", pc, format!("{}", last_op).trim());
                std::io::stdout().flush();
                line.clear();
                std::io::stdin().read_line(&mut line).expect("FAILED TO READ LINE!");

                let trimmed = line.trim();
                let a = trimmed.split_whitespace().collect::<Vec<_>>();
                if trimmed == "p" {
                    println!("\tindex: {}",  index as isize - i16::MAX as isize );
                    println!("\treg: {}", reg);
                    println!("\ttape [{} to {}]: {:?} < {} > {:?}", index as isize - i16::MAX as isize - 10, index as isize - i16::MAX as isize + 10, &tape[(index - 10)..index], tape[index], &tape[(index + 1)..(index + 10)]);
                } else if trimmed.starts_with("w") {
                    if a.len() != 3 {
                        println!("2 arguments needed");
                        continue;
                    }
                    let Ok(here) = a[1].parse::<isize>() else {
                        println!("invalid index as first argument");
                        continue;
                    };

                    let Ok(val) = a[2].parse::<u8>() else {
                        println!("invalid value as second argument");
                        continue;
                    };
                    tape[i16::MAX as usize + here as usize] = Wrapping(val);
                } else if trimmed.starts_with("r") {
                    if a.len() != 3 && a.len() != 2 {
                        println!("2 argument needed");
                        continue;
                    }
                    if a.len() == 3 {
                        let Ok(here) = a[1].parse::<isize>() else {
                            println!("invalid index as first argument");
                            continue;
                        };
                        let Ok(to) = a[2].parse::<isize>() else {
                            println!("invalid index as second argument");
                            continue;
                        };
                        if to < here {
                            println!("first argument has to be smaller or equal to the second");
                        }
                        println!("{:?}", &tape[i16::MAX as usize + here as usize..=i16::MAX as usize + to as usize]);
                    } else {
                        let Ok(here) = a[1].parse::<isize>() else {
                            println!("invalid index as first argument");
                            continue;
                        };
                        println!("{:?}", &tape[i16::MAX as usize + here as usize]);
                    }
                } else if trimmed.starts_with("s") {
                    if a.len() == 1 {
                        debug_counter = 1;
                        println!("STEP 1\x1B[0m");
                        break;
                    }
                    if a.len() == 2 {
                        let Ok(l) = a[1].parse::<isize>() else {
                            println!("invalid number of steps as first argument");
                            continue;
                        };

                        debug_counter = l;
                        println!("STEP {}\x1B[0m", l);
                        break;
                    }
                } else if trimmed == "q" {
                    debug_counter = -1;
                    println!("EXIT DEBUG\x1B[0m");
                    break;
                }
            }
        }
        last_op = op;
        pc += 1;
        debug_counter -= 1;
    }
}





// // BrainFuck Affine Linear Optimizing Data Structure
// pub trait BFAFLODS<T: Copy + Ord> : BFAffineT<T> {
//     // in order to compile the affine transforms, we need to make sure, we 
//     // do the least amount of unnecessary computations and caching
//     //
//     // for that we can make use of some graph theory: When we see the matrix as
//     // graph, we can first get all strongly connected components (which need
//     // some form of caching the result) and topologically sort the rest 
//     //
//     // we then go backwards in the topological sorting, because these elements 
//     // are the ones, that only dependent on other variables, while no other variables
//     // depend on them
//     fn new_graph_adapter(&self) -> impl Graphlike<T> + std::fmt::Debug;
//     fn linearize(&self) -> Vec<Opcode>;
// }

// #[derive(Clone, Debug)]
// pub struct BFAddMapGraphAdapter {
//     vertices: Vec<isize>,
//     edges: Vec<Vec<usize>>,
//     translator: BTreeMap<isize, usize>
// }

// impl BFAFLODS<isize> for BFAddMap {
//     fn new_graph_adapter(&self) -> impl Graphlike<isize> + std::fmt::Debug {
//         let mut keys: BTreeSet<isize> = BTreeSet::from_iter(self.matrix.keys().map(|x| *x));
//         keys.extend(self.affine.keys());

//         let mut vertices = Vec::new();
//         let mut translator = BTreeMap::new();
//         let mut edges = Vec::new();
//         for (index, i) in keys.iter().enumerate() {
//             vertices.push(*i);
//             translator.insert(*i, index);
//             edges.push(Vec::new());
//         }
//         for (i, j) in &self.matrix {
//             for (y, _) in j {
//                 edges[translator[&y]].push(translator[&i]);
//             }
//         }
//         BFAddMapGraphAdapter { vertices, edges, translator }
//     }
//     fn linearize(&self) -> Vec<Opcode> {
//         // 1. step: get strongly connected components from the hashmaps 
//         // 2. step: topologically sort them 
//         //
//         // we can combine the first two steps by using Tarjan's algorithm,
//         // that outputs the strongly connected components in reverse DAG-TS
//         // order anyway:
//         //


//         let out = self.new_graph_adapter();
//         dbg!(&out);
//         for strong_component in out.tarjan() {
//             dbg!(strong_component);
//         }
//         //
//         // 3. convert into opcode
//         //
//         todo!()
//     }
// }

pub fn write_opcode<P: AsRef<Path>>(opcodes: &Vec<Opcode>, file_path: P) -> Result<(), std::io::Error> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(file_path)?;
    let v: Vec<u64> = opcodes.iter().map(|x| u64::from(x)).collect();
    let v_bytes = unsafe {
        v.align_to::<u8>().1
    };
    let magic_numbers = 0x627261696e66636bu64; // brainfck
    file.write_all(&magic_numbers.to_ne_bytes())?;
    file.write_all(&v_bytes)?;
    Ok(())
}

pub fn main() {

    let args: Vec<String> = env::args().collect();

    let file_path = args.get(1).expect("No filepath given!");
    let contents = fs::read_to_string(file_path).expect("File doesn't exist");
    let mut tree = compile::<BFAddMap>(contents);

    // dbg!(&tree);
    // let opcodes = gen_opcode(&tree);
    // for (index, i) in opcodes.iter().enumerate() {
    //     println!("{}: {}", index, i);
    // }
    optimize(&mut tree, 0);
    // println!("after optimization:");
    dbg!(&tree);

    let mut optimized_opcodes = gen_opcode(&tree);
    optout_labels(&mut optimized_opcodes);
    println!("// {} opcodes", optimized_opcodes.len());
    write_opcode(&optimized_opcodes, "./result.op");
    // for (index, i) in optimized_opcodes.iter().enumerate() {
    //     println!("{}: {}", index, i);
    // }
    //
    // println!("end of rust part");
    // simulate(optimized_opcodes);
    // println!("{}", gen_ccode(&tree));
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn bfaffinet() {
        let mut addmap1: BFAddMap = BFAddMap::new_ident();
        addmap1.add_const(3, Wrapping(5));
        addmap1.add_const(5, Wrapping(5));
        let output = format!("{:?}", addmap1);
        assert_eq!(output, String::from("3: │     │ 5 │\n\
                                         5: │     │ 5 │\n"));
        
        addmap1.set_zero(2);
        addmap1.set_zero(5);
        let output = format!("{:?}", addmap1);
        assert_eq!(output, String::from("2: │ 0     │   │\n\
                                         3: │       │ 5 │\n\
                                         5: │     0 │   │\n"));
    }

    #[test]
    fn simple_compile(){
        let input = String::from("++++++++++>++++<<---<+.");
        let tree = compile::<BFAddMap>(input);
        let matrix_: Matrix = Matrix::new();
        let affine_ = BTreeMap::from_iter([(-2, 1), (-1, (256 - 3) as u8), (0, 10), (1, 4)].into_iter().map(|(x, y)| (x, Wrapping(y))));
        match &tree[0] {
            Optree::OffsetMap(
                BFAddMap { matrix, affine }
            ) => assert!(matrix == &matrix_ && affine == &affine_),
            _ => panic!("First optree doesn't match")
        }

        assert!(matches!(&tree[1], Optree::Output(-2)));
    }

    #[test]
    fn loop_compile(){
        let input = String::from(">>>[->>]>. [<<]");
        let tree = compile::<BFAddMap>(input);
        dbg!(tree);
    }
}
