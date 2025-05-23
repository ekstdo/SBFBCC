// Brainfuck bytecode compiler by ekstdo
#![allow(non_camel_case_types)]
#![allow(clippy::implicit_saturating_sub)]
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
    fn vec_mul(&self, v: &BTreeMap<T, w8>) -> BTreeMap<T, w8>;
    fn set_constants(&mut self, consts: &BTreeMap<T, w8>, set_0_diag: bool) -> bool;
    fn rm_row(&mut self, i: T) -> bool; // returns true on changed
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
        *self = self.matmul(&tmp);
    }

    fn is_ident(&self) -> bool;
    fn is_affine(&self) -> bool;
    fn is_affine_at0(&self) -> bool;
    fn affine_zero_lins(&self) -> Vec<T>;
    fn is_affine_zero_lin(&self, i: T) -> bool;
    fn zero_dep(&self) -> Vec<isize>; // indices, that only depend on 0 or a constant
    fn is_sure_ident(&self, i: T) -> bool;
    fn is_sure0(&self, i: T) -> bool;
    fn constants(&self, constants_so_far: &BTreeMap<T, w8>) -> (BTreeMap<isize, w8>, BTreeSet<isize>);
    fn squarable(&self) -> BTreeMap<isize, w8>;

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

    fn writes(&self) -> BTreeSet<T>;
    fn reads(&self) -> BTreeSet<T>;

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
        self.matrix.get(i, i) == w8::ZERO && self.matrix.inner.get(&i).is_some_and(|x| x.values().all(|x| *x == w8::ZERO))
    }

    fn vec_mul(&self, v: &BTreeMap<isize, w8>) -> BTreeMap<isize, w8> {
        let mut r = self.matrix.vecmul(v);
        addi_vec(&mut r, &self.affine);
        r
    }

    fn set_constants(&mut self, consts: &BTreeMap<isize, w8>, diag0: bool) -> bool {
        let consts_clone = consts.iter().map(|(x, v)| (*x, -*v)).collect::<BTreeMap<isize, w8>>();
        let diffs = self.matrix.vecmul(consts);
        for k in consts.keys() {
            self.matrix.rm_column(*k);
            if diag0 {
                self.matrix.set(*k, *k, w8::ZERO);
            }
        }
        addi_vec(&mut self.affine, &diffs);
        if !diag0 {
            addi_vec(&mut self.affine, &consts_clone);
        }
        !diffs.is_empty()
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

    fn rm_row(&mut self, i: isize) -> bool {
        let mut changed = false;
        changed =  self.affine.remove(&i).is_some() || changed;
        changed = self.matrix.inner.remove(&i).is_some() || changed;
        changed
    }

    fn is_sure_ident(&self, i: isize) -> bool {
        !self.affine.contains_key(&i) && !self.matrix.inner.contains_key(&i)
    }

    fn is_sure0(&self, i: isize) -> bool {
        !self.affine.contains_key(&i) && self.is_affine_zero_lin(i)
    }

    fn constants(&self, constants_so_far: &BTreeMap<isize, w8>) -> (BTreeMap<isize, w8>, BTreeSet<isize>) {
        let (zero_rows, mut non_zero_rows) = self.matrix.zero_rows(constants_so_far);
        let res = self.matrix.vecmul(constants_so_far);
        let mut constants = BTreeMap::new();
        for (k, v) in &self.affine {
            if zero_rows.contains(k) {
                constants.insert(*k, *v);
            } else {
                non_zero_rows.insert(*k);
            }
        }

        for i in &zero_rows {
            *constants.entry(*i).or_insert(w8::ZERO) += res.get(i).copied().unwrap_or(w8::ZERO) ;
        }
        (constants, non_zero_rows)
    }

    fn squarable(&self) -> BTreeMap<isize, w8> {
        self.matrix.squarable()
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
        aa_.cleanup_all();
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

    fn writes(&self) -> BTreeSet<isize> {
        let mut result = self.matrix.rows().cloned().collect::<BTreeSet<_>>();
        result.extend(self.affine.keys());
        result
    }

    fn reads(&self) -> BTreeSet<isize> {
        let (zc, mut nzc) = self.matrix.zero_columns();
        nzc.extend(self.affine.keys());
        nzc.difference(&zc).copied().collect()
    }

    fn unset_linear(&mut self) {
        self.matrix = Matrix::new();
    }

    fn cleanup(&mut self) {
        // adding 0 as a constant doesn't do anything
        self.affine.retain(|_, x| *x != w8::ZERO);
        self.matrix.cleanup_all();
    }

    fn to_opcode(mut self) -> Vec<Opcode> {
        self.cleanup();
        let (perm, mat_opcode) = self.matrix.convert_to_opcode();
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
                None => if v != w8::ZERO { out_opcode.push(Opcode::AddConst(v, k as i32)) }
            }
        }
        for cycle in perm.cycles() {
            if cycle.len() < 2 { // identity
                continue;
            } else if cycle.len() == 2 {
                out_opcode.push(Opcode::Swap((cycle[0] - cycle[1]) as i16, cycle[1] as i32));
            } else {
                let first = cycle.first().unwrap();
                out_opcode.push(Opcode::Load(*first as i32));
                for i in &cycle[1..] {
                    out_opcode.push(Opcode::LoadSwap(*i as i32));
                }
                out_opcode.push(Opcode::LoadSwap(*first as i32));
            }
        }

        // the only types to exist so far are: AddCell, SubCell, AddMul, SetConst, MulConst,
        // AddConst, Swap, Load, LoadSwap

        let mut index = 1;
        while index < out_opcode.len() {
            // println!("{:?}", &out_opcode[index]);
            match &out_opcode[index] {
                Opcode::SetConst(w8::ZERO, o) => {
                    let mut o = *o;
                    let mut back_index = index as isize - 1;
                    while back_index >= 0 {
                        // println!("{:?}", out_opcode);
                        let Some(Opcode::SetConst(_, _)) = out_opcode.get(index) else {panic!("Wrong check! {index} {back_index}")};
                        match &out_opcode[back_index as usize] {
                            Opcode::AddCell(offset, to) => {
                                let from = *offset as i32 + *to;
                                // it's being overwritten anyway
                                if o == *to {
                                    out_opcode.remove(back_index as usize);
                                    index -= 1;
                                } else if o == from {
                                    let offset = *offset;
                                    let to = *to;
                                    let Opcode::SetConst(_, _) = out_opcode.remove(index) else {panic!("Wrong check!")};
                                    index -= 1;
                                    let mut new_opcode = Opcode::AddnSet0(offset, to);
                                    std::mem::swap(&mut out_opcode[back_index as usize], &mut new_opcode);
                                    break;
                                }
                            }
                            Opcode::SubCell(offset, to) => {
                                let from = *offset as i32 + *to;
                                // it's being overwritten anyway
                                if o == *to {
                                    out_opcode.remove(back_index as usize);
                                    index -= 1;
                                } else if o == from {
                                    let offset = *offset;
                                    let to = *to;
                                    let Opcode::SetConst(_, _) = out_opcode.remove(index) else {panic!("Wrong check!")};
                                    index -= 1;
                                    let mut new_opcode = Opcode::SubnSet0(offset, to);
                                    std::mem::swap(&mut out_opcode[back_index as usize], &mut new_opcode);
                                    break;
                                }
                            }
                            Opcode::AddMul(_, offset, to) => {
                                let from = *offset as i32 + *to;
                                // it's being overwritten anyway
                                if o == *to {
                                    out_opcode.remove(back_index as usize);
                                    index -= 1;
                                } else if o == from {
                                    break;
                                }
                            },
                            Opcode::AddnSet0(offset, add_to) | Opcode::SubnSet0(offset, add_to) => {
                                let zero_to = *add_to + *offset as i32;
                                if o == zero_to {
                                    // this means the SetConst(w8::ZERO) is useless
                                    out_opcode.remove(index);
                                    index -= 1;
                                    break;
                                } else if o == *add_to {
                                    break;
                                }
                            },

                            // These cases shouldn't ever happen, as SetConst is generated instead
                            // of MulConst and AddConst, there shouldn't be duplicate SetConst's
                            Opcode::AddConst(_, to) | Opcode::SetConst(_, to) | Opcode::MulConst(_, to) => {
                                if o == *to {
                                    out_opcode.remove(back_index as usize); index -= 1;
                                }
                            },
                            // These cases also shouldn't happen, as they're generated AFTER Set0
                            Opcode::Swap(a, b) => {
                                o = if o == *a as i32 + *b { *b } else if o == *b { *a as i32 + *b } else { o };
                            },
                            Opcode::Load(from) | Opcode::LoadSwap(from) => {
                                // value is required, we can't set it to 0 any earlier
                                if o == *from { break; }
                            },
                            _ => { panic!("Illegal instruction within OffsetMap") }
                        }
                        back_index -= 1;
                    }
                },
                Opcode::Swap(offset_swap, to_swap) => {
                    let a_swap = *offset_swap as i32 + to_swap;
                    let b_swap = *to_swap;
                    // unlike OpCode::SetConst, I'll leave out every case, that doesn't happen this
                    // time
                    let Some(pos) = out_opcode.iter().position(|x| match x {
                        Opcode::AddnSet0(a, b) => {
                            let add_a = *a as i32 + *b;
                            let add_b = *b;
                            (a_swap == add_a && b_swap == add_b) || (a_swap == add_b && b_swap == add_a)
                        }, 
                        Opcode::SubnSet0(a, b) => {
                            let add_a = *a as i32 + *b;
                            let add_b = *b;
                            (a_swap == add_a && b_swap == add_b) || (a_swap == add_b && b_swap == add_a)
                        }, 
                        _ => false
                    }) else { index += 1; continue; };
                    let mut back_index = index - 1;
                    while back_index > pos {
                        match &mut out_opcode[back_index] {
                            Opcode::AddCell(offset, to) | Opcode::AddMul(_, offset, to) | Opcode::SubCell(offset, to) | Opcode::AddnSet0(offset, to) | Opcode::SubnSet0(offset, to) => {
                                let from_cell = *offset as i32 + *to;
                                let to_cell = *to;
                                if to_cell == a_swap {
                                    *to = b_swap;
                                } else if to_cell == b_swap {
                                    *to = a_swap;
                                }

                                if from_cell == a_swap {
                                    *offset = (b_swap - *to) as i16;
                                } else if from_cell == b_swap {
                                    *offset = (a_swap - *to) as i16;
                                }
                            }
                            // These cases shouldn't ever happen, as SetConst is generated instead
                            // of MulConst and AddConst, there shouldn't be duplicate SetConst's
                            Opcode::AddConst(_, to) | Opcode::SetConst(_, to) | Opcode::MulConst(_, to) => {
                                if a_swap == *to {
                                    *to = b_swap;
                                } else if b_swap == *to {
                                    *to = a_swap;
                                }
                            },
                            // Permutation splits it into mutually exclusive parts => no overlap
                            Opcode::Swap(_, _) | Opcode::Load(_) | Opcode::LoadSwap(_) => {}
                            _ => { panic!("Illegal instruction within OffsetMap") }
                        }
                        back_index -= 1;
                    }
                    match out_opcode.get_mut(pos) {
                        Some(Opcode::AddnSet0(a, b)) | Some(Opcode::SubnSet0(a, b)) => {
                            let from = *b + *a as i32;
                            *b = from;
                            *a = -*a;
                            out_opcode.remove(index);
                            index -= 1;
                        }
                        _ => {panic!("Wrong check")}
                    }

                },
                _ => {}
            }
            index += 1;
        }


        out_opcode
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugPosition {
    start_line: usize,
    start_col: usize,
    end_line: usize,
    end_col: usize
}

impl DebugPosition {
    fn combine(&self, other: &DebugPosition) -> DebugPosition {
        let (start_line, start_col) = if self.start_line == other.start_line {
            (self.start_line, self.start_col.min(other.start_col))
        } else {
            // using lexicographical sorting
            (self.start_line, self.start_col).min((other.start_line, other.start_col))
        };
        let (end_line, end_col) = if self.end_line == other.end_line {
            (self.end_line, self.end_col.max(other.end_col))
        } else {
            // using lexicographical sorting
            (self.end_line, self.end_col).max((other.end_line, other.end_col))
        };

        DebugPosition { start_line, start_col, end_line, end_col }
    }
}

#[derive(Debug, Clone)]
pub enum Optree<T: BFAffineT<isize>> {
    OffsetMap(T, DebugPosition),
    Branch { instructions: Vec<Optree<T>>, preshift: isize, itershift: isize, jumpback: bool, pos: DebugPosition },
    Input(isize, DebugPosition),
    Output(isize, DebugPosition),
    DebugBreakpoint(DebugPosition),
    Square { offset: isize, reciprocal: w8, map: BTreeMap<isize, (w8, w8)>, pos: DebugPosition }
}

fn gen_ccode<T: BFAffineT<isize> + Clone + Debug>(optrees: &Vec<Optree<T>>) -> String {
    let mut result = format!("#include<stdio.h>
unsigned char tape[{}];
int main() {{
    unsigned char* t = tape + {};
    unsigned char reg;
    unsigned char tmp;\n", u16::MAX, i16::MAX);

    for ot in optrees {
        result.push_str(&ot.gen_ccode_op(1));
    }

    result.push_str("    return 0;\n}");
    result
}

// There aren't that many optimizations, that we do at this point
//
// other than simplifying defaults
// and simplifying  [...[...]] constructs, as at the end, we can save a jnez
//                as this op ^ will never be executed
//
// so constructs like [...[...[...[...]]]] could be simplified to [...[...[...[...}
// where } jumps to the innermost [ only
//
// The jez [ is still required though
fn gen_opcode<T: BFAffineT<isize> + Clone + Debug>(optree: &[Optree<T>]) -> (Vec<Opcode>, Vec<DebugPosition>) {
    let mut result = Vec::new();

    let mut stack = optree.iter().rev().collect::<Vec<_>>();
    let mut jmp_stack: Vec<(usize, u32, i32, bool, &DebugPosition)> = Vec::new();
    let mut debug_result = Vec::new();
    let mut label_counter: u32 = 0;
    let mut i16_offset: usize = 0;
    let mut min_i16: isize = i16::MIN as isize;
    let mut max_i16: isize = i16::MAX as isize;
    let check_in_range = |x: isize| min_i16 <= x && x <= max_i16;
    loop {
        while let Some((when_len, cur_label_counter, shift, jumpback, ref pos)) = jmp_stack.last() {
            let pos = *pos;
            if *when_len == stack.len() {
                if *jumpback {
                    if *shift != 0 {
                        result.push(Opcode::JnezShift(*shift as i16, *cur_label_counter << 1));
                        debug_result.push(pos.clone());
                    } else {
                        result.push(Opcode::Jnez(*cur_label_counter << 1));
                        debug_result.push(pos.clone());
                    }
                } else if *shift != 0 {
                    result.push(Opcode::Shift(*shift));
                    debug_result.push(pos.clone());
                }
                result.push(Opcode::Label((*cur_label_counter << 1) + 1u32));
                debug_result.push(pos.clone());
                jmp_stack.pop();
            } else {
                break;
            }
        }

        let Some(cur_el) = stack.pop() else {break;};

        match cur_el {
            // todo: Check for i32 precision loss from isize
            Optree::Input(i, pos) => {
                result.push(Opcode::Read(*i as i32));
                debug_result.push(pos.clone());
            }
            Optree::DebugBreakpoint(pos) => {
                result.push(Opcode::DebugBreakpoint);
                debug_result.push(pos.clone());
            }
            Optree::Output(i, pos) => {
                result.push(Opcode::Write(*i as i32));
                debug_result.push(pos.clone());
            }
            Optree::OffsetMap(m, pos) => {
                // println!("{:?}", m);
                let opcodes = m.clone().to_opcode();
                // println!("{:?}", opcodes);
                for i in 0..opcodes.len() {
                    debug_result.push(pos.clone());
                }
                result.extend(opcodes);
            }
            Optree::Branch { instructions, preshift, itershift, jumpback, pos } => {
                if instructions.is_empty() {
                    if *preshift != 0 {
                        result.push(Opcode::Shift(*preshift as i32));
                        debug_result.push(pos.clone());
                    }
                    result.push(Opcode::SkipLoop(*itershift as i32));
                    debug_result.push(pos.clone());
                } else {
                    if *preshift != 0 {
                        result.push(Opcode::JezShift(*preshift as i16, (label_counter << 1) + 1));
                        debug_result.push(pos.clone());
                    } else {
                        result.push(Opcode::Jez((label_counter << 1) + 1));
                        debug_result.push(pos.clone());
                    }
                    result.push(Opcode::Label(label_counter << 1));
                    debug_result.push(pos.clone());
                    // when to insert the jump, where to jump to and how much to shift at each
                    // iteration
                    jmp_stack.push((stack.len(), label_counter, *itershift as i32, *jumpback, pos));
                    stack.extend(instructions.iter().rev());
                    label_counter += 1;
                }
            }
            Optree::Square { reciprocal, offset, pos, map } => {
                if map.is_empty() {
                    continue;
                }
                result.push(Opcode::LoadMul(*reciprocal, *offset as i32));
                debug_result.push(pos.clone());
                for (k_offset, (_, c)) in map {
                    result.push(Opcode::AddStoreMul(*c, *k_offset as i32));
                    debug_result.push(pos.clone());
                }
                result.push(Opcode::SquareAddReg(*offset as i32));
                debug_result.push(pos.clone());
                for (k_offset, (a, _)) in map {
                    result.push(Opcode::AddStoreMul(*a, *k_offset as i32));
                    debug_result.push(pos.clone());
                }
            },
        }
    }
    (result, debug_result)
}

impl<T: BFAffineT<isize>> Optree<T> {
    fn shift(&mut self, by: isize) -> bool { // returns whether the shift should propagate
        match self {
            Optree::OffsetMap(bfaddmap, _) => { bfaddmap.shift_keys(by); true },
            Optree::Input(x, _) | Optree::Output(x, _) => { *x += by; true },
            Optree::Branch { preshift, .. } => { *preshift += by; false },
            Optree::DebugBreakpoint(_) => true,
            Optree::Square { map, offset, .. } => { *offset += by; shift_bmap(map, by); true },
        }
    }

    fn size(&self) -> usize {
        match self {
            Optree::OffsetMap(_, _) | Optree::Input(_, _) | Optree::Output(_, _) | Optree::Square {..} => 1,
            Optree::Branch { instructions, .. } => 1 + instructions.iter().map(|x| x.size()).sum::<usize>(),
            Optree::DebugBreakpoint(_) => 0,
        }
    }

    fn reads(&self) -> Option<BTreeSet<isize>> {
        Some(match self {
            Optree::OffsetMap(map, _) => map.reads(),
            Optree::Input(_, _) => BTreeSet::new(),
            Optree::Output(i, _) => BTreeSet::from([*i]),
            Optree::Square {offset, map, ..} => map.keys().chain([offset].into_iter()).cloned().collect(),
            Optree::Branch { instructions, preshift, itershift: 0, .. } => {
                let mut result = BTreeSet::new();
                for set in instructions
                    .iter()
                    .map(|x| x.reads().map(|set| set.into_iter().map(|y: isize| y + preshift) )) {
                    let Some(set) = set else {return None;};
                    result.extend(set);
                }
                result.insert(*preshift);
                result
            },
            Optree::Branch { .. } => return None,
            Optree::DebugBreakpoint(_) => BTreeSet::new(),
        })
    }

}


impl<T: BFAffineT<isize> + std::fmt::Debug + Clone> Optree<T> {
    fn gen_ccode_op(&self, indent_level: usize) -> String {
        let indentation = "    ".repeat(indent_level);

        match self {
            Optree::Input(i, _) => format!("{}t[{}] = getchar();\n", indentation, i),
            Optree::Output(i, _) => format!("{}putchar(t[{}]);\n", indentation, i),
            Optree::OffsetMap(m, _) => {
                let offset_map_ops = m.clone().to_opcode();
                let mut result = String::new();
                for op in offset_map_ops {
                    result.push_str(&match op {
                        Opcode::AddCell(offset, to) => format!("{}t[{}] += t[{}];\n", indentation, to, to + offset as i32),
                        Opcode::AddnSet0(offset, to) => format!("{}t[{}] += t[{}]; t[{}] = 0;\n", indentation, to, to + offset as i32, to + offset as i32),
                        Opcode::SubnSet0(offset, to) => format!("{}t[{}] -= t[{}]; t[{}] = 0;\n", indentation, to, to + offset as i32, to + offset as i32),
                        Opcode::SubCell(offset, to) => format!("{}t[{}] -= t[{}];\n", indentation, to, to + offset as i32),
                        Opcode::AddMul(val, offset, to) => format!("{}t[{}] += {} * t[{}];\n", indentation, to, val, to + offset as i32),
                        Opcode::SetConst(val, to) => format!("{}t[{}] = {};\n", indentation, to, val),
                        Opcode::MulConst(val, to) => format!("{}t[{}] *= {};\n", indentation, to, val),
                        Opcode::AddConst(val, to) => format!("{}t[{}] += {};\n", indentation, to, val),
                        Opcode::Load(from) => format!("{}reg = t[{}];\n", indentation, from),
                        Opcode::LoadSwap(from) => format!("{}tmp = reg; reg = t[{}]; t[{}] = tmp;\n", indentation, from, from),
                        Opcode::Swap(offset, from) => format!("{0}tmp = t[{2}]; t[{2}] = t[{1}]; t[{1}] = tmp;\n", indentation, from + offset as i32, from),
                        _ => String::new()
                    });
                }
                result
            },
            Optree::Branch { instructions, preshift, itershift, jumpback, .. } => {
                let mut result = String::new();
                result.push_str(&format!("{}t += {};\n", indentation, preshift));
                if *jumpback {
                    result.push_str(&format!("{}while (*t) {{\n", indentation));
                } else {
                    result.push_str(&format!("{}if (*t) {{\n", indentation));
                }
                for o in instructions {
                    result.push_str(&o.gen_ccode_op(indent_level + 1));
                }
                result.push_str(&format!("{}    t += {};\n{}}}\n", indentation, itershift, indentation));
                result

            },
            Optree::DebugBreakpoint(_) => {String::new()},
            Optree::Square { reciprocal, offset, map, .. } => {
                let mut result = String::new();
                result.push_str(&format!("{indentation}reg = {reciprocal} * t[{offset}];\n"));
                for (k_offset, (_, c)) in map {
                    result.push_str(&format!("{indentation}t[{k_offset}] += reg * {c};\n"));
                }
                result.push_str(&format!("{indentation}reg += 1;\n"));
                result.push_str(&format!("{indentation}if (t[{offset}] % 2 == 0) {{\n"));
                result.push_str(&format!("{indentation}    reg = reg * (t[{offset}] >> 1);\n"));
                result.push_str(&format!("{indentation}}} else {{\n"));
                result.push_str(&format!("{indentation}    reg = (reg >> 1) * t[{offset}];\n"));
                result.push_str(&format!("{indentation}}}\n"));
                for (k_offset, (a, _)) in map {
                    result.push_str(&format!("{indentation}t[{k_offset}] += reg * {a};"));
                }
                result
            }
        }
    }
}


pub fn compile<T: BFAffineT<isize>>(s: &str) -> Vec<Optree<T>> {
    let mut inst = vec![vec![]];
    let mut current_inst = &mut inst[0];
    let mut current_offset_map = T::new_ident();
    let mut current_offset: isize = 0;
    let mut preshift = vec![0];
    let mut saved_pos: Vec<(usize, usize)> = vec![];
    let mut last_add_pos = None;
    let mut last_pos = (0, 0);
    let mut line = 0;
    let mut column = 0;
    for i in s.chars() {
        if "[].,".contains(i) && !current_offset_map.is_ident() {
            // println!("{:?}", last_add_pos);
            let unwrapped_last_add_pos: (usize, usize) = last_add_pos.unwrap();
            current_offset_map.cleanup();
            current_inst.push(Optree::OffsetMap(current_offset_map, DebugPosition { start_line: unwrapped_last_add_pos.0, start_col: unwrapped_last_add_pos.1, end_line: last_pos.0, end_col: last_pos.1 }));
            current_offset_map = T::new_ident();
        }
        if "[].,".contains(i) && last_add_pos.is_some() {
            last_add_pos = None;
        }
        if "+-<>".contains(i) && last_add_pos.is_none() {
            last_add_pos = Some((line, column));
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
                current_inst.push(Optree::Output(current_offset, DebugPosition { start_col: column, end_col: column, start_line: line, end_line: line }));
            },
            '#' => {
                current_inst.push(Optree::DebugBreakpoint(DebugPosition { start_col: column, end_col: column, start_line: line, end_line: line }));
            },
            ',' => {
                current_inst.push(Optree::Input(current_offset, DebugPosition { start_col: column, end_col: column, start_line: line, end_line: line }));
            },
            '[' => {
                preshift.push(current_offset);
                current_offset = 0;
                inst.push(vec![]);
                current_inst = inst.last_mut().unwrap();
                saved_pos.push((line, column));
            },
            ']' => {
                let itershift = current_offset;
                current_offset = 0;
                let instructions = inst.pop().unwrap();
                let start_pos = saved_pos.pop().unwrap();
                current_inst = inst.last_mut().unwrap();
                current_inst.push(Optree::Branch { instructions, preshift: preshift.pop().unwrap(), itershift, jumpback: true, pos: DebugPosition { start_line: start_pos.0, start_col: start_pos.1, end_line: line, end_col: column } });
            }
            _ => {}
        }


        last_pos = (line, column);
        if i == '\n' {
            column = 0;
            line += 1;
        } else {
            column += 1;
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
//      See README.md (explanation too long)
//
//      Branch[OffsetMap o, ...] -> Branch[OffsetMap o, Branch[ OffsetMap o.remove_affine_only(), ... ]]
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
//
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
//
//
// 8. [Branch(_, preshift, itershift=0)]


pub fn optimize<T: BFAffineT<isize> + std::fmt::Debug + Clone>(unoptimized: &mut Vec<Optree<T>>) {
    // applies 1 a/b, 2 and 3
    linearize(unoptimized);
    // applies 5/6/7
    constant_propagate(unoptimized);

    if let Some(Optree::OffsetMap(ref mut m, _)) = unoptimized.get_mut(0) {
        m.unset_linear();
    }

    rotate_affine_lines(unoptimized);
    linearize(unoptimized);
    square_detection(unoptimized);
}

fn shift_along<T: BFAffineT<isize> + std::fmt::Debug>(o: &mut [Optree<T>], by: isize, starting_at: usize) -> isize {
    if by != 0 {
        let mut j = starting_at;
        while j < o.len() && o[j].shift(by) {
            j += 1;
        }
        if j == o.len() {
            return by;
        }
    }
    0
}

pub struct LinearizationOutput {
    changed: bool,
    resulting_shift: isize,
}

// applies optimizations 1 a/b and 2 and 3
pub fn linearize<T: BFAffineT<isize> + std::fmt::Debug>(unoptimized: &mut Vec<Optree<T>>) -> LinearizationOutput {
    let mut index = 0;
    let mut result = LinearizationOutput {
        resulting_shift: 0,
        changed: false,
    };

    while index < unoptimized.len() {
        let mut prev = false;
        match &mut unoptimized[index] {
            Optree::OffsetMap(_, _) => {
                let next_el = unoptimized.get(index + 1);
                // see 2.
                if let Some(Optree::OffsetMap(_, _)) = next_el {
                    let Optree::OffsetMap(mut el, pos1) = unoptimized.remove(index + 1) else { panic!("wrong check!") };
                    let Some(Optree::OffsetMap(ref mut bfaddmap, ref mut pos2)) = unoptimized.get_mut(index) else {panic!("wrong check!")};
                    el = el.matmul(bfaddmap);
                    std::mem::swap(&mut el, bfaddmap);
                    *pos2 = pos1.combine(pos2);
                    prev = true;

                    result.changed = true;
                } else
                // see 3.
                if let Some(Optree::Input(into, _)) = next_el {
                    let into = *into;
                    let Some(Optree::OffsetMap(ref mut bfaddmap, _)) = unoptimized.get_mut(index) else {panic!("wrong check!")};
                    result.changed = bfaddmap.rm_row(into) || result.changed;
                }
            },
            // see 1.
            Optree::Branch {  ref mut instructions, preshift, itershift: 0, pos, jumpback: true, .. } if instructions.len() == 1 => {
                let preshift_ = *preshift;
                let pos = pos.clone();
                match &mut instructions[0] {
                    Optree::OffsetMap(ref mut m, _) if m.is_affine() => {
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

                            std::mem::swap(&mut unoptimized[index], &mut Optree::OffsetMap(b, pos));
                            result.resulting_shift += shift_along(unoptimized, preshift_, index);
                            if index > 0 {
                                index -= 1;
                            }
                            prev = true;
                            result.changed = true;
                        }
                    }
                    _ => {
                        let Optree::Branch { instructions, ref mut itershift, .. } = &mut unoptimized[index] else { panic!("wrong check!") };
                        let inner_result = linearize(&mut *instructions);
                        result.changed = result.changed || inner_result.changed;
                        *itershift += inner_result.resulting_shift;
                        if inner_result.changed {
                            prev = true;
                        }
                    }
                }
            },
            Optree::Branch { .. } => {
                let inner_result = {
                    let Optree::Branch { instructions, .. } = &mut unoptimized[index] else { panic!("wrong check!") };
                    linearize(&mut *instructions)
                };
                result.changed = result.changed || inner_result.changed;
                let Optree::Branch { ref mut itershift, .. } = &mut unoptimized[index] else { panic!("wrong check!") };
                *itershift += inner_result.resulting_shift;
                if inner_result.changed {
                    prev = true;
                }
            }

            Optree::Input(_, _) | Optree::DebugBreakpoint(_) | Optree::Output(_ , _) | Optree::Square { .. } => {}
        }
        if !prev {
            index += 1;
        }
    }

    result
}

pub struct ConstantPropagationOutput {
    changed: bool,
    resulting_shift: isize,
    constants: BTreeMap<isize, w8>
}

pub fn constant_propagate<T: BFAffineT<isize> + std::fmt::Debug>(unoptimized: &mut Vec<Optree<T>>) -> ConstantPropagationOutput {
    let mut result = ConstantPropagationOutput {
        changed: false,
        resulting_shift: 0,
        constants: BTreeMap::new()
    };


    // println!("{:?}", unoptimized);
    let mut index = 0;
    while index < unoptimized.len() {
        let el = &mut unoptimized[index];
        // println!("{:?} -> {:?}", result.constants, el);
        match el {
            Optree::OffsetMap(bfaddmap, _) => {
                // setting it to true may make things slower, unless you follow it up with other
                // optimizations

                let diff = bfaddmap.set_constants(&result.constants, false);
                bfaddmap.cleanup();
                result.changed = diff || result.changed;
                let (to_be_inserted, to_be_removed) = bfaddmap.constants(&result.constants);
                result.constants.retain(|x, _| !to_be_removed.contains(x));
                to_be_inserted.into_iter().for_each(|(k ,v)| {result.constants.insert(k ,v);});
            },
            Optree::Input(i, _) => { result.constants.remove(i); },
            Optree::Output(_, _) => {},
            Optree::Branch { preshift, .. } if result.constants.get(preshift) == Some(&w8::ZERO) => {
                let preshift = *preshift;
                unoptimized.remove(index);
                result.resulting_shift += shift_along(unoptimized, preshift, index);
                result.changed = true;
                continue;
            },
            Optree::Branch { instructions, itershift, jumpback, .. } => {
                let inner_result = constant_propagate(&mut *instructions);
                *itershift += inner_result.resulting_shift;
                result.changed = result.changed || inner_result.changed;

                if inner_result.constants.get(itershift) == Some(&w8::ZERO) {
                    *jumpback = false;
                }

                result.constants = BTreeMap::from([(0, w8::ZERO)]);
            },
            Optree::DebugBreakpoint(_) => {},
            Optree::Square {offset, reciprocal, map, ..} => {
                let k = result.constants.get(offset);
                if let Some(&k) = k {
                    let mut l = *reciprocal * k;
                    for (key, (_, c)) in map.iter() {
                        result.constants.entry(*key).and_modify(|x| *x += *c * l);
                    }
                    l += 1;
                    l = if k.0 % 2 == 0 { l * (k >> 1) } else { (l >> 1) * k };
                    for (key, (a, _)) in map.iter() {
                        result.constants.entry(*key).and_modify(|x| *x += *a * l);
                    }

                } else {
                    for key in map.keys() {
                        result.constants.remove(key);
                    }
                }
            },
        }
        index += 1;
    }
    result

}

// returns all constants, that are set once within a list of Optrees
//
// TODO: Better detection
//
// What we actually want: lock mechanism when values are being read
pub fn detect_constants_throughout<T: BFAffineT<isize> + std::fmt::Debug>(v: &Vec<Optree<T>>) -> BTreeMap<isize, w8> {
    let mut result = BTreeMap::new();
    let mut mask = BTreeSet::new();


    for i in v {
        match i {
            Optree::Input(index, _) => { mask.insert(*index); },
            Optree::OffsetMap(map, _) => {
                let (constants, non_constants) = map.constants(&result );
                for (const_index, val) in constants {
                    match result.get(&const_index) {
                        Some(existing_val) if *existing_val == val => { },
                        Some(_) => {
                            mask.insert(const_index);
                            continue;
                        },
                        None => {
                            result.insert(const_index, val);
                        }
                    }
                }
                mask.extend(non_constants);
            },
            // TODO: Better constant detection
            Optree::Branch { .. } => { return BTreeMap::new(); },
            Optree::Output(_, _) => {},
            Optree::DebugBreakpoint(_) => {},
            Optree::Square { .. } => { todo!() },
        }
    }

    for i in mask {
        result.remove(&i);
    }

    result
}

// This doesn't check, whether the constants are actually constant
// which should be done by `detect_constants_throughout`
pub fn get_constant_value<T: BFAffineT<isize> + std::fmt::Debug>(v: &[Optree<T>], constants: &BTreeSet<isize>) -> BTreeMap<isize, w8> {
    let mut result = BTreeMap::from_iter(constants.iter().map(|x| (*x, w8::ZERO)));
    for i in v {
        match i {
            // input only overwrites and we assume constants are constant
            //
            // and square only adds onto existing values, but never actually sets anything
            Optree::Input(_, _) | Optree::Output(_, _) | Optree::DebugBreakpoint(_) | Optree::Square {..} => { },
            Optree::OffsetMap(map, _) => { result = map.vec_mul(&result); },
            Optree::Branch { .. } => return result,

        }
    }
    result
}

// one of the optimizations, that actually increase the number of opcodes
pub fn rotate_affine_lines<T: BFAffineT<isize> + std::fmt::Debug + Clone>(unoptimized: &mut [Optree<T>]) {
    let mut index = 0;
    while index < unoptimized.len() {
        match &unoptimized[index] {
            Optree::Input(_, _) | Optree::Output(_, _) | Optree::DebugBreakpoint(_) | Optree::OffsetMap(_, _) => {},
            Optree::Branch { .. } => {
                {
                    let Optree::Branch { ref mut instructions, .. } = &mut unoptimized[index] else { panic!("Wrong check!") };
                    rotate_affine_lines(instructions);
                }
                let Optree::Branch { instructions, preshift, itershift, jumpback, pos } = &unoptimized[index] else { panic!("Wrong check!") };
                if !(itershift == &0 && *jumpback) {
                    index += 1;
                    continue;
                }
                let detected_constants = detect_constants_throughout(instructions);
                if detected_constants.is_empty(){
                    index += 1;
                    continue;
                }
                let mut inner_instructions = instructions.clone();
                for i in &mut inner_instructions {
                    match i {
                        Optree::OffsetMap(ref mut map, _) => {
                            map.set_constants(&detected_constants, false);
                            map.cleanup();
                        },
                        Optree::Branch { .. } => {},
                        Optree::Input(_, _) => {},
                        Optree::Output(_, _) => {},
                        Optree::DebugBreakpoint(_) => {},
                        Optree::Square { offset, reciprocal, map, pos } => todo!(),
                    }
                }
                let mut outer_instructions = instructions.clone();
                outer_instructions.push(Optree::Branch{
                    instructions: inner_instructions,
                    preshift: 0,
                    itershift: 0,
                    jumpback: *jumpback,
                    pos: pos.clone(),
                });
                let mut new_branch = Optree::Branch {
                    instructions: outer_instructions.clone(),
                    preshift: *preshift,
                    itershift: 0,
                    jumpback: false,
                    pos: pos.clone()
                };
                // println!("{:?}", detected_constants);

                // println!("{:?}", instructions);
                // println!("results in {:?}", new_branch);
                std::mem::swap(&mut new_branch, &mut unoptimized[index]);
            },
            Optree::Square{ .. } => todo!(),
        }
        index += 1;
    }
}

pub fn square_detection<T: BFAffineT<isize> + std::fmt::Debug + Clone>(unoptimized: &mut Vec<Optree<T>>) {
    let mut index = 0;
    while index < unoptimized.len() {
        match &unoptimized[index] {
            Optree::OffsetMap(_, _) | Optree::Input(_, _) | Optree::Output(_, _) | Optree::DebugBreakpoint(_) | Optree::Square{..} => {},
            Optree::Branch { .. } => {
                {
                    let Optree::Branch { ref mut instructions, .. } = &mut unoptimized[index] else {panic!("Wrong check")};
                    square_detection(instructions);
                }
                let Optree::Branch { instructions, preshift, itershift, jumpback, pos } = &unoptimized[index] else {panic!("Wrong check")};
                if !(itershift == &0 && *jumpback && instructions.len() == 1) {
                    index += 1;
                    continue;
                }
                if let Optree::OffsetMap(map,  _) = &instructions[0] {
                    if !map.is_affine_at0() {
                        index += 1;
                        continue;
                    }
                    let squarable_indices = map.squarable();
                    // println!("sq: {:?}", squarable_indices);
                    if squarable_indices.is_empty() {
                        index += 1;
                        continue;
                    }
                    let r = map.get_affine(0);

                    let squarable_with_c = squarable_indices.into_iter().map(|(x, v)| (x, (v, map.get_affine(x)))).collect::<BTreeMap<isize, (w8, w8)>>();
                    let num_squarables = squarable_with_c.len();
                    let s = multinv(256isize - r.0 as isize, 256) as u8;
                    let mut optree: Optree<T> = Optree::Square{ reciprocal: Wrapping(s),
                        offset: *preshift,
                        map: squarable_with_c,
                        pos: pos.clone()
                    };
                    let pos = pos.clone();
                    if num_squarables + 1 == map.get_involved_lin().len()  { // counter variable -> +1
                        // i.e. every element there is squarable
                        std::mem::swap(&mut optree, &mut unoptimized[index]);
                        let mut set0_offset_map = T::new_ident();
                        set0_offset_map.set_zero(0);
                        unoptimized.insert(index + 1, Optree::OffsetMap(set0_offset_map, pos));
                    } else {
                        todo!()
                    }

                }
            },

        }
        index += 1;
    }
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
    AddnSet0(i16, i32),   // t[b] += t[a + b]; t[a + b] = 0;
    SubnSet0(i16, i32),   // t[b] -= t[a + b]; t[a + b] = 0;
    Load(i32),            // r = t[a]
    LoadSwap(i32),        // r, t[a] = t[a], r
    // Store(i32),        // t[a] = r; // not very useful, as: 1. bytecode vm is faster avoiding
                                       // jumps by repeating LoadSwap instead of trying to save a
                                       // single read and store instruction 
                                       //
                                       // 2. jit compiler can just compile the last  LoadStore to a
                                       //    Store, rendering the bytecode useless
    LoadMul(w8, i32),     // r = a * t[b];
    AddStoreMul(w8, i32), // t[b] += a * r;
    SquareAddReg(i32),// r += 1;
                          // r = t[b] % 2 == 0 ? r * (t[b] >> 1) : (r >> 1) * t[b];
    // SquareReg 
    // SetReg(),
    // AddReg(),
    Swap(i16, i32),        // t[a + b], t[a] = t[a], t[a + b]
    Shift(i32),           // t = t + a
    ShiftBig(i32),        // t = t + a * i32::max
    Read(i32),            // t[a] = input_char()
    Write(i32),           // print(t[a])
    J(u32),               // goto a
    Jez(u32),             // if t[0] == 0: goto a
    Jnez(u32),             // if t[0] != 0: goto a
    JezShift(i16, u32),  // t = t + b; if t[0] != 0: goto a
                          // jmp label;
    JnezShift(i16, u32),  // t = t + b; if t[0] != 0: goto a
                          // jmp label;

    Label(u32),
    SkipLoop(i32),
    DebugBreakpoint,
    DebugData(u8, u8, u8, u16, u16),  // range where in code we are
                                      // 0: number of instructions, which correspond
                                      // 1,2: number of columns and lines
                                      // 3,4: starting column and line
}

impl From<&Opcode> for u64 {
    fn from(value: &Opcode) -> Self {
        match value {
            Opcode::AddConst(v, c) =>  ((v.0 as u64) << 48) | ((*c as u32) as u64) ,
            Opcode::SetConst(v, c) =>  (1 << 56) | ((v.0 as u64) << 48) | ((*c as u32) as u64) ,
            Opcode::MulConst(v, c) =>  (2 << 56) | ((v.0 as u64) << 48) | ((*c as u32) as u64) ,
            Opcode::AddMul(v, o, c) =>  (3 << 56) | ((v.0 as u64) << 48) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::AddCell(o, c) =>  (4 << 56) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::AddnSet0(o, c) =>  (5 << 56) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::SubnSet0(o, c) =>  (6 << 56) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::SubCell(o, c) =>  (7 << 56) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::Load(c) =>  (8 << 56) | ((*c as u32) as u64) ,
            Opcode::LoadSwap(c) =>  (9 << 56) | ((*c as u32) as u64) ,
            Opcode::Swap(o, c) =>  (10 << 56) | (((*o as u16) as u64) << 32) | ((*c as u32) as u64) ,
            Opcode::LoadMul(v, c) => (11 << 56) | ((v.0 as u64) << 48) | ((*c as u32) as u64) ,
            Opcode::AddStoreMul(v, c) => (12 << 56) | ((v.0 as u64) << 48) | ((*c as u32) as u64) ,
            Opcode::SquareAddReg(c) => (13 << 56)  | ((*c as u32) as u64) ,
            Opcode::Shift(c) =>  (14 << 56) | ((*c as u32) as u64) ,
            Opcode::ShiftBig(c) =>  (15 << 56) | ((*c as u32) as u64) ,
            Opcode::Read(c) =>  (16 << 56) | ((*c as u32) as u64) ,
            Opcode::Write(c) =>  (17 << 56) | ((*c as u32) as u64) ,
            Opcode::J(c) =>  (18 << 56) | (*c as u64) ,
            Opcode::Jez(c) =>  (19 << 56) | (*c as u64) ,
            Opcode::Jnez(c) =>  (20 << 56) | (*c as u64) ,
            Opcode::JezShift(o, c) =>  (21 << 56) | (((*o as u16) as u64) << 32) | (*c as u64) ,
            Opcode::JnezShift(o, c) =>  (22 << 56) | (((*o as u16) as u64) << 32) | (*c as u64) ,
            Opcode::SkipLoop(c) => (23 << 56) | ((*c as u32) as u64),
            Opcode::DebugBreakpoint => 25 << 56,
            Opcode::Label(c) =>  (128 << 56) | (*c as u64) ,
            Opcode::DebugData(num_inst, ol, oc, sl, sc) => (129 << 56) | ((*num_inst as u64) << 48) | ((*ol as u64) << 40) | ((*oc as u64) << 32) | ((*sl as u64) << 16) | (*sc as u64)  ,
        }
    }
}

impl std::fmt::Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Opcode::Label(u) => write!(f, "label_{}:", u),
            Opcode::J(u) =>    write!(f, "\tj    label_{}", u),
            Opcode::Jez(u) =>  write!(f, "\tjez  label_{}", u),
            Opcode::Jnez(u) =>  write!(f, "\tjnez  label_{}", u),
            Opcode::JezShift(o, u) => write!(f, "\tt += {}; jez label_{}", o, u),
            Opcode::JnezShift(o, u) => write!(f, "\tt += {}; jnez label_{}", o, u),
            Opcode::AddConst(val, wher) => write!(f, "\tt[{}] += {}", wher, val),
            Opcode::SetConst(val, wher) => write!(f, "\tt[{}]  = {}", wher, val),
            Opcode::MulConst(val, wher) => write!(f, "\tt[{}] *= {}", wher, val),
            Opcode::Shift(of) => write!(f, "\tt += {}", of),
            Opcode::ShiftBig(of) => write!(f, "\tt += {} * i16::max", of),
            Opcode::Read(to) => write!(f, "\tt[{}] = inputc()", to),
            Opcode::Write(from) => write!(f, "\tprintc(t[{}])", from),
            Opcode::Load(from) => write!(f, "\tr = t[{}]", from),
            Opcode::LoadSwap(from) => write!(f, "\tt[{}], r = r, t[{}]", from, from),
            Opcode::Swap(offset, from) => write!(f, "\tt[{0}], t[{1} + {0} = {2}] = t[{2}], t[{0}]", from, offset, *offset as i32 + *from),
            Opcode::LoadMul(val, offset) => write!(f, "\tr = {val} * t[{offset}];"),
            Opcode::AddStoreMul(val, offset) => write!(f, "\tt[{offset}] += {val} * r;"),
            Opcode::SquareAddReg(offset) => write!(f, "\tr += 1; r = t[{offset}] % 2 == 0 ? r * (t[{offset}] >> 1) : (r >> 1) * t[{offset}];"),
            Opcode::AddMul(val, offset, to) => write!(f, "\tt[{}] += t[{} + {} = {}] * {}", to, to, offset, *to + *offset as i32, val),
            Opcode::DebugBreakpoint => write!(f, "\tDEBUG!"),
            Opcode::AddCell(offset, to) => write!(f, "\tt[{}] += t[{} + {} = {}]", to, to, offset, *to + *offset as i32),
            Opcode::AddnSet0(offset, to) => write!(f, "\tt[{0}] += t[{0} + {1} = {2}]; t[{2}] = 0", to, offset, *to + *offset as i32),
            Opcode::SubnSet0(offset, to) => write!(f, "\tt[{0}] -= t[{0} + {1} = {2}]; t[{2}] = 0", to, offset, *to + *offset as i32),
            Opcode::SubCell(offset, to) => write!(f, "\tt[{}] += t[{} + {} = {}]", to, to, offset, *to + *offset as i32),
            Opcode::DebugData(num_inst, ol, oc, sl, sc) => write!(f, "\t\twithin code {}@[{}|{} to {}|{}]", num_inst, sl, sc, *sl + *ol as u16, *sc + *oc as u16),
            Opcode::SkipLoop(offset) => write!(f, "\twhile (t[0]) {{ t += {}; }}", offset),
        }
    }
}

fn optout_labels(opcode: &mut Vec<Opcode>, debug_symbols: Option<&mut Vec<DebugPosition>>, add1: bool) {
    let mut label_map = Vec::new(); // maps each label id to the index position
    let mut label_count = Vec::new(); // counts the amount of labels before that label
    let mut counter = 0;
    for (index, i) in opcode.iter().enumerate() {
        if let Opcode::Label(k) = i {
            counter += 1;
            while label_map.len() <= *k as usize {
                label_map.push(0);
                label_count.push(0);
            }
            label_map[*k as usize] = index;
            label_count[*k as usize] = counter;
        }
    }

    for i in opcode.iter_mut() {
        match i {
            Opcode::Jez(ref mut k) | Opcode::Jnez(ref mut k) | Opcode::JnezShift(_, ref mut k) | Opcode::JezShift(_, ref mut k) | Opcode::J(ref mut k) => {
                *k = (label_map[*k as usize] - label_count[*k as usize] + if add1 { 1 } else { 0 }) as u32;
            },
            _ => {}
        }
    }

    if let Some(debug_symbols ) = debug_symbols {
        let mut new_debug_symbols = Vec::new();
        let mut old_debug_symbols: Vec<DebugPosition> = Vec::new();
        std::mem::swap(debug_symbols, &mut old_debug_symbols);
        for (i, x) in old_debug_symbols.into_iter().enumerate() {
            if !matches!(opcode[i], Opcode::Label(_)) {
                new_debug_symbols.push(x);
            }
        }
        *debug_symbols = new_debug_symbols;
    };
    opcode.retain(|x| !matches!(x, Opcode::Label(_)));
}

fn simulate(mut opcode: Vec<Opcode>) -> Result<(), std::io::Error> {
    let mut tape: Vec<w8> = vec![w8::ZERO;u16::MAX as usize];
    let mut index = i16::MAX as usize;
    let mut pc = 0;
    let mut debug_counter = -1;
    let mut reg = w8::ZERO;

    optout_labels(&mut opcode, None, false);

    let Some(mut last_op) = opcode.first() else {return Ok(());};
    let mut print_buffer = String::new();
    while let Some(op) = opcode.get(pc) {
        match op {
            Opcode::AddConst(val, to) => {tape[(index as isize + *to as isize) as usize] += *val;},
            Opcode::SetConst(val, to) => {tape[(index as isize + *to as isize) as usize] = *val;},
            Opcode::AddCell(offset, to) => {
                let readout = tape[(index as isize + *to as isize + *offset as isize) as usize];
                tape[(index as isize + *to as isize) as usize] += readout;
            }
            Opcode::AddnSet0(offset, to) => {
                let readout = tape[(index as isize + *to as isize + *offset as isize) as usize];
                tape[(index as isize + *to as isize) as usize] += readout;
                tape[(index as isize + *to as isize + *offset as isize) as usize] = w8::ZERO;
            },
            Opcode::SubnSet0(offset, to) => {
                let readout = tape[(index as isize + *to as isize + *offset as isize) as usize];
                tape[(index as isize + *to as isize) as usize] -= readout;
                tape[(index as isize + *to as isize + *offset as isize) as usize] = w8::ZERO;
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
                std::mem::swap(&mut reg, &mut tape[(index as isize + *from as isize) as usize]);  },
            Opcode::Swap(offset, from) => {
                tape.swap((index as isize + *from as isize + *offset as isize) as usize, (index as isize + *from as isize + *offset as isize) as usize);
            }
            Opcode::Load(from) => { reg = tape[(index as isize + *from as isize) as usize]; },
            Opcode::LoadMul(val, offset) => { reg = val * tape[(index as isize + *offset as isize) as usize]; },
            Opcode::AddStoreMul(val, offset) => { tape[(index as isize + *offset as isize) as usize] += reg * val; },
            Opcode::SquareAddReg(offset) => {
                reg += 1;
                let readout = tape[(index as isize + *offset as isize) as usize];
                reg = if readout.0 % 2 == 0 { (readout >> 1) * reg } else { (reg >> 1) * readout };
            },
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
            Opcode::JnezShift(o, a) => {index = (index as isize + *o as isize) as usize;  if tape[index] != w8::ZERO { pc = *a as usize - 1; } },
            Opcode::JezShift(o, a) => {index = (index as isize + *o as isize) as usize;  if tape[index] == w8::ZERO { pc = *a as usize - 1; } },
            Opcode::Jez(a) => { if tape[index] == w8::ZERO { pc = *a as usize - 1; } },
            Opcode::Jnez(a) => { if tape[index] != w8::ZERO { pc = *a as usize - 1; } },
            Opcode::J(a) => { pc = *a as usize - 1; },
            Opcode::Label(_) => {},
            Opcode::DebugBreakpoint => { debug_counter = 0; }
            Opcode::DebugData(_, _, _, _, _) => {},
            Opcode::SkipLoop(c) => {
                while tape[index] != w8::ZERO {
                    index = (index as isize + *c as isize) as usize;
                }
            },
        }
        if debug_counter == 0 {
            let mut line = String::new();
            print!("{}", print_buffer);
            print_buffer.clear();
            loop {
                print!("\x1B[32mDEBUG[{}@ {}] > ", pc, format!("{}", last_op).trim());
                std::io::stdout().flush()?;
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
    Ok(())
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

// structure:
//
// 1. b"brainfck" in big endian for the VM to check for endianness (if it comes out as "kcfniarb" as
//   u64 on the other end, the VM knows that it should flip it, or handle it differently depending on
//   the system)
//
//   [8 BYTES]
//
// 2. OpCode
//
//   [8 BYTES * opcodes.len()]
//
// 3. Debug Positions
//
//   [8 BYTES * whatever]
//
// 4. Debug Source
//
//   an entire copy of the .bf source code
//
//
pub fn write_opcode<P: AsRef<Path>>(opcodes: &[Opcode], debug_symbols: Option<(&[u64], &String)>, file_path: P) -> Result<(), std::io::Error> {
    let mut file = std::fs::OpenOptions::new()
        .create(true).truncate(true)
        .write(true)
        .open(file_path)?;
    let v: Vec<u64> = opcodes.iter().map(u64::from).collect();
    let v_bytes = unsafe {
        v.align_to::<u8>().1
    };
    let magic_numbers = 0x627261696e66636bu64; // brainfck
    file.write_all(&magic_numbers.to_ne_bytes())?;
    file.write_all(v_bytes)?;
    let magic_numbers = 0x1818181818181818u64; // MARK_EOF
    file.write_all(&magic_numbers.to_ne_bytes())?;
    if let Some((debug_symbols, source)) = debug_symbols {
        let debug_bytes = unsafe { debug_symbols.align_to::<u8>().1 };
        file.write_all(debug_bytes)?;
        let magic_numbers = 0xFFFFFFFFFFFFFFFFu64; // -1
        file.write_all(&magic_numbers.to_ne_bytes())?;
        file.write_all(source.as_bytes())?;
    }
    Ok(())
}

pub fn summarize_opcode_symbols(v: &Vec<DebugPosition>) -> Vec<u64> {
    let index = 0;
    let mut last_el = &v[index];
    let mut result = Vec::new();
    let mut counter = 0;

    let conv = |x: &DebugPosition, n: usize| {
        ((n as u64) << 48) 
            | ((((x.end_line as isize - x.start_line as isize) as u8) as u64) << 40)
            | ((((x.end_col as isize - x.start_col as isize) as u8) as u64) << 32)
            | ((x.start_line as u64) << 16)
            | (x.start_col as u64)
    };
    for el in v {
        if el == last_el {
            counter += 1;
        } else {
            result.push(conv(last_el, counter));
            last_el = el;
            counter = 1;
        }
    }
    result.push(conv(last_el, counter));
    result
}

pub fn main() -> Result<(), std::io::Error> {

    let args: Vec<String> = env::args().collect();

    let file_path = args.get(1).expect("No filepath given!");
    let contents = fs::read_to_string(file_path).expect("File doesn't exist");
    let mut tree = compile::<BFAddMap>(&contents);

    // println!("{:?}", &tree);
    // let opcodes = gen_opcode(&tree);
    // for (index, i) in opcodes.iter().enumerate() {
    //     println!("{}: {}", index, i);
    // }
    optimize(&mut tree);
    println!("// after optimization:");
    // dbg!(&tree);

    let (mut optimized_opcodes, mut debug_positions) = gen_opcode(&tree);
    optout_labels(&mut optimized_opcodes, Some(&mut debug_positions), true);
    println!("// {} opcodes", optimized_opcodes.len());
    let out_debug = summarize_opcode_symbols(&debug_positions);
    write_opcode(&optimized_opcodes, Some((&out_debug, &contents)), Path::new(file_path).with_extension("lbf"))?;
    println!("/*\n");
    for (index, i) in optimized_opcodes.iter().enumerate() {
        println!("{}: {}", index, i);
    }
    println!("*/\n");
    println!("// end of rust part");
    // simulate(optimized_opcodes);
    println!("{}", gen_ccode(&tree));
    Ok(())
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
        let tree = compile::<BFAddMap>(&input);
        let matrix_: Matrix = Matrix::new();
        let affine_ = BTreeMap::from_iter([(-2, 1), (-1, (256 - 3) as u8), (0, 10), (1, 4)].into_iter().map(|(x, y)| (x, Wrapping(y))));
        match &tree[0] {
            Optree::OffsetMap(
                BFAddMap { matrix, affine }, _
            ) => assert!(matrix == &matrix_ && affine == &affine_),
            _ => panic!("First optree doesn't match")
        }

        assert!(matches!(&tree[1], Optree::Output(-2, _)));
    }

    #[test]
    fn loop_compile(){
        let input = String::from(">>>[->>]>. [<<]");
        let tree = compile::<BFAddMap>(&input);
        dbg!(tree);
    }
}
