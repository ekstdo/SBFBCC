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
mod ludecomp;
mod utils;
use ludecomp::{addi_vec, Matrix};

use crate::utils::{shift_bmap, multinv, w8, Zero, One};
mod simulate;

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
    fn is_sure_ident(&self, i: T) -> bool;

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
        self.is_affine() && self.affine.len() == 1 && self.affine.get(&0).map(|x| x.0 & 1 == 1).unwrap_or(false)
    }

    fn is_sure_ident(&self, i: isize) -> bool {
        !self.affine.contains_key(&i) && !self.matrix.inner.contains_key(&i)
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
            ludecomp::MatOpCode::Add(to, from, val) => Opcode::AddMul(val, (from - to) as i16, to as i32),
            ludecomp::MatOpCode::Mul(to, val) => 
                if val == w8::ZERO {
                    Opcode::SetConst(w8::ZERO, to as i32)
                } else {
                    Opcode::MulConst(val, to as i32)
                }
            
        }).collect::<Vec<_>>();
        let mut selected = BTreeSet::new();
        for (k, mut v) in &perm.inner {
            out_opcode.push(Opcode::Load(*k as i32));
            while !selected.contains(v) {
                selected.insert(v);
                out_opcode.push(Opcode::LoadSwap(*v as i32));
                v = perm.inner.get(v).unwrap();
            }
        }
        for (k, v) in self.affine {
            out_opcode.push(Opcode::AddConst(v, k as i32));
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
// 2. [OffsetMap o1, OffsetMap o2]
//
//    can be optimized to o2 * o1
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
pub fn optimize<T: BFAffineT<isize> + std::fmt::Debug>(unoptimized: &mut Vec<Optree<T>>, layer: usize) -> isize {
    let mut index = 0;
    let mut total_shift = 0;
    while index < unoptimized.len() {
        let mut prev = false;
        let l = unoptimized.len();
        match &mut unoptimized[index] {
            Optree::OffsetMap(ref mut _bfaddmap) if index < l - 1 => {
                if let Some(Optree::OffsetMap(_)) = unoptimized.get(index + 1) {
                    let Optree::OffsetMap(mut el) = unoptimized.remove(index + 1) else { panic!("wrong check!") };
                    let Some(Optree::OffsetMap(ref mut bfaddmap)) = unoptimized.get_mut(index) else {panic!("wrong check!")};
                    el = el.matmul(bfaddmap);
                    std::mem::swap(&mut el, bfaddmap);
                    prev = true;
                }
            },
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
                            prev = true;
                            if index > 0 {
                                index -= 1;
                            }
                        }
                    }
                    _ => {
                        let Optree::Branch(uo, _preshift, ref mut iter_shift) = &mut unoptimized[index] else { panic!("wrong check!") };
                        let inner_shift = optimize(&mut *uo, layer + 1);
                        *iter_shift += inner_shift;
                    }
                }
            },
            Optree::Branch(_uo, _, _) => {
                let prev_len = _uo.len();
                
                let (new_len, inner_shift) = {
                    let Optree::Branch(uo, _, _) = &mut unoptimized[index] else { panic!("wrong check!") };
                    (uo.len(), optimize(&mut *uo, layer + 1))
                };
                let Optree::Branch(_, _, ref mut iter_shift) = &mut unoptimized[index] else { panic!("wrong check!") };
                *iter_shift += inner_shift;
                if prev_len > new_len && new_len == 1 {
                    continue;
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
    total_shift
}



#[derive(Debug)]
pub enum Opcode {
    // t and r have to be saved in a register
    AddConst(w8, i32),    // t[b] += b
                          // add BYTE [rdi], sil
    SetConst(w8, i32),    // t[b] = a
                          // mov BYTE [rdi], sil
    AddMul(w8, i16, i32), // t[c] += t[c + b] * a
                          // movzx   eax, BYTE PTR [rdx+rcx]
                          // imul    eax, edi
                          // add     BYTE PTR [rsi+rcx], al
    MulConst(w8, i32),    // t[b] *= a
                          // movzx   eax, BYTE PTR [rsi]
                          // imul    eax, edi
                          // mov     BYTE PTR [rsi], al
    LoadSwap(i32),        // r, t[a] = t[a], r
    Load(i32),            // t[a] = r
    ShiftBig(i32),        // t = t + a * i32::max
    Shift(i32),           // t = t + a
    Read(i32),            // t[a] = input_char()
    Write(i32),           // print(t[a])
    Jnez(u32),            // if t[0] != 0: goto a
    Jez(u32),             // if t[0] == 0: goto a
    J(u32),               // goto a
                          // jmp label;
    Label(u32),
    DebugBreakpoint,
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
        }
    }
}



fn simulate(mut opcode: Vec<Opcode>) {
    let mut tape: Vec<w8> = vec![w8::ZERO;u16::MAX as usize];
    let mut index = i16::MAX as usize;
    let mut pc = 0;
    let mut label_map = Vec::new();
    let mut debug_counter = -1;
    let mut reg = w8::ZERO;

    for (index, i) in opcode.iter().enumerate() {
        match i {
            Opcode::Label(k) => {
                while label_map.len() <= *k as usize {
                    label_map.push(0);
                }
                label_map[*k as usize] = index;
            },
            _ => {}
        }
    }

    for i in opcode.iter_mut() {
        match i {
            Opcode::Jez(ref mut k) | Opcode::Jnez(ref mut k) | Opcode::J(ref mut k) => {
                *k = label_map[*k as usize] as u32;
            },
            _ => {}
        }
    }

    let Some(mut last_op) = opcode.first() else {return;};
    while let Some(op) = opcode.get(pc) {
        match op {
            Opcode::AddConst(val, to) => {tape[(index as isize + *to as isize) as usize] += *val;},
            Opcode::SetConst(val, to) => {tape[(index as isize + *to as isize) as usize] = *val;},
            Opcode::AddMul(w8::ONE, offset, to) => {
                let readout = tape[(index as isize + *to as isize + *offset as isize) as usize];
                tape[(index as isize + *to as isize) as usize] += readout;
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
            Opcode::Write(from) => print!("{}", tape[(index as isize + *from as isize) as usize].0 as char),
            Opcode::Jnez(a) => { if tape[index] != w8::ZERO { pc = *a as usize; } },
            Opcode::Jez(a) => { if tape[index] == w8::ZERO { pc = *a as usize; } },
            Opcode::J(a) => { pc = label_map[*a as usize]; },
            Opcode::Label(_) => {},
            Opcode::DebugBreakpoint => { debug_counter = 0; }
        }
        if debug_counter == 0 {
            let mut line = String::new();
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



pub fn main() {
    let args: Vec<String> = env::args().collect();

    let file_path = args.get(1).expect("No filepath given!");
    let contents = fs::read_to_string(file_path).expect("File doesn't exist");
    let mut tree = compile::<BFAddMap>(contents);

    dbg!(&tree);
    let opcodes = gen_opcode(&tree);
    for (index, i) in opcodes.iter().enumerate() {
        println!("{}: {}", index, i);
    }
    optimize(&mut tree, 0);
    println!("after optimization:");
    dbg!(&tree);

    let optimized_opcodes = gen_opcode(&tree);
    for (index, i) in optimized_opcodes.iter().enumerate() {
        println!("{}: {}", index, i);
    }
    //
    // +[-[-[>]+[<]]]


    simulate(optimized_opcodes);
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
