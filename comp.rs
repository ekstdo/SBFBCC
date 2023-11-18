// Brainfuck bytecode compiler by ekstdo

use std::env;
use std::fs;
use std::collections::{BTreeMap, BTreeSet};
use std::num::Wrapping;
use std::iter::FromIterator;
type w8 = Wrapping<u8>;

// Utility functions

fn shift_hashmap<T: Copy>(mut hashmap: &mut BTreeMap<isize, T>, by: isize) {
    let mut tbc = hashmap.keys().map(|x| *x).collect::<BTreeSet<isize>>();
    while!tbc.is_empty() {
        let mut old_key = *tbc.iter().next().unwrap();
        let mut new_key = old_key + by;
        let mut tmp = *hashmap.get(&old_key).unwrap();
        while let Some(v) = hashmap.get(&new_key) {
            tbc.remove(&old_key);
            hashmap.remove(&old_key);
            let tmp2 = *hashmap.get(&new_key).unwrap();
            hashmap.insert(new_key, tmp);
            tmp = tmp2;
            old_key = new_key;
            new_key = old_key + by;
        }
        tbc.remove(&old_key);
        hashmap.remove(&old_key);
        hashmap.insert(new_key, tmp);
    }
}

fn eeagcd(mut a: isize, mut b: isize) -> (isize, isize, isize, isize, isize) {
    let (mut x, mut v, mut y, mut u) = (1, 1, 0, 0);
    while b != 0 {
        let q = a / b;
        (a, b, x, y, u, v) = (b, a % b, u, v, x - q * u, y - q * v);
    }
    (a, x, y, u, v)
}

fn multinv(mut a: isize, mut n: isize) -> isize {
    eeagcd(a, n).1 % n
}



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
// general trait to represent the affine liner transformations
pub trait BFAffineT<T: Sized + Clone + Copy>: Sized {
    // generates the identity representation of 
    fn new_ident() -> Self; 
    fn add_const(&mut self, i: T, v: w8);
    fn set_zero(&mut self, i: T);
    fn set_const(&mut self, i: T, v: w8) {
        self.set_zero(i);
        self.add_const(i, v);
    }
    fn mul_const(&mut self, i: w8);
    // multiplies an affine matrix to a variable to get a affine linear
    // matrix
    fn mul_var(&mut self, i: isize, v: w8);
    // just adds the value to the matrix, disregarding the fact
    // that the src may have been transformed beforehand
    fn add_mul_raw(&mut self, dest: T, src: T, v: w8);
    // can be thought of as a merge operation
    fn matmul(&mut self, other: &Self);
    // actually adds the multiplied value to the matrix
    //
    // default implementation is a pretty slow approach
    fn add_mul(&mut self, dest: T, src: T, v: w8) {
        if v.0 == 0 { return; }
        let mut tmp = Self::new_ident();
        tmp.add_mul_raw(dest, src, v);
        self.matmul(&tmp);
    }
    fn set_mul(&mut self, dest: T, src: T, v: w8) {
        self.set_zero(dest);
        self.add_mul(dest, src, v);
    }

    fn is_ident(&self) -> bool;
    fn is_affine(&self) -> bool;
    fn is_affine_at0(&self) -> bool;
    fn is_sure0(&self, i: T) -> bool;

    fn get_affine(&self, i: T) -> w8;
    fn get_mat(&self, i:T, j:T) -> w8;

    fn shift_keys(&mut self, by: isize);
}

// simple and naive approach to represent the arbitrary dimensional matrix
#[derive(Debug)]
pub struct BFAddMap { 
    // default is the zero vector
    affine: BTreeMap<isize, w8>,

    // default is identity matrix
    matrix: BTreeMap<isize, BTreeMap<isize, w8>>
}

impl BFAffineT<isize> for BFAddMap {
    fn new_ident() -> Self {
        Self { affine: BTreeMap::new(), matrix: BTreeMap::new() }
    }

    fn add_const(&mut self, i: isize, v: w8) {
        self.affine.entry(i).and_modify(|mut e| *e += v).or_insert(v);
        if self.affine[&i] == Wrapping(0) {
            self.affine.remove(&i);
        }
    }

    fn set_zero(&mut self, i:isize) {
        self.affine.remove(&i);
        let x = self.matrix.entry(i).or_insert(BTreeMap::new());
        x.clear();
        x.insert(i, Wrapping(0));
    }

    fn add_mul_raw(&mut self, dest: isize, src: isize, v: w8) {
        let mut m = self.matrix.entry(dest).or_insert(BTreeMap::new());
        m.entry(src).and_modify(|mut e| *e += v).or_insert(v);
        if (m[&src] == Wrapping(0) && dest != src) || (m[&src] == Wrapping(1) && dest == src) {
            m.remove(&src);
        }
        if m.is_empty(){
            self.matrix.remove(&dest);
        }
    }

    fn get_affine(&self, i: isize) -> w8 {
        self.affine.get(&i).map(|x| *x).unwrap_or(Wrapping(0))
    }

    fn get_mat(&self, i: isize, j: isize) -> w8 {
        self.matrix.get(&i).and_then(|x| x.get(&j)).map(|x| *x).unwrap_or(Wrapping(if i == j { 1 } else { 0 }))
    }

    fn is_ident(&self) -> bool {
        self.affine.is_empty() && self.matrix.is_empty()
    }

    fn is_affine(&self) -> bool {
        self.matrix.is_empty()
    }

    fn is_affine_at0(&self) -> bool {
        self.is_affine() && self.affine.len() == 1 && self.affine.get(&0).map(|x| x.0 & 1 == 1).unwrap_or(false)
    }

    fn is_sure0(&self, i: isize) -> bool {
        self.affine.get(&i).is_none() && self.matrix.get(&i).is_none()
    }

    // might cache this instead
    fn shift_keys(&mut self, by: isize) {
        shift_hashmap(&mut self.affine, by);
        if self.matrix.is_empty() {
            return;
        }

        let mut new_multmap = BTreeMap::new();
        let keys = self.matrix.keys().map(|x| *x).collect::<Vec<_>>();
        for k in keys {
            let mut tmp_map = self.matrix.remove(&k).unwrap();
            shift_hashmap(&mut tmp_map, by);
            new_multmap.insert(k + by, tmp_map);
        }
        self.matrix = new_multmap;
    }

    // very inefficient, O(n^3), should be optimized in the future!
    fn matmul(&mut self, other: &Self) {
        // transposing the other matrix 
        let mut t_other: BTreeMap<isize, BTreeMap<isize, w8>> = BTreeMap::new();
        for (i, v) in &other.matrix {
            for (j, w) in v {
                let m = t_other.entry(*j).or_insert(BTreeMap::new());
                m.insert(*i, *w);
            }
        }

        let mut keys: BTreeSet<isize> = BTreeSet::from_iter(self.matrix.keys().map(|x| *x));
        keys.extend(t_other.keys());

        let mut resulting_mat = BTreeMap::new();
        let empty_mat = BTreeMap::new();
        for i in &keys {
            let v = self.matrix.get(&i).unwrap_or(&empty_mat);
            let mut row = BTreeMap::new();
            for j in &keys {
                match t_other.get(&j) {
                    Some(w) => {
                        let mut val = Wrapping(0);
                        for (x, a) in v {
                            val += a * w.get(x).map(|x| *x).unwrap_or(Wrapping(if i == j { 1 } else { 0 }));
                        }

                        if (i == j && val.0 != 1) || (i != j && val.0 != 0) {
                            row.insert(*j, val);
                        }
                    }
                    None => {
                        row = v.clone(); // if w doesn't exist, we default to v due to w being
                                         // ident
                    }
                }
            }
            if !row.is_empty() {
                resulting_mat.insert(*i, row);
            }

        }

        let mut keys: BTreeSet<isize> = BTreeSet::from_iter(self.matrix.keys().map(|x| *x));
        keys.extend(other.affine.keys());
        for i in &keys {
            let v = self.matrix.get(&i).unwrap_or(&empty_mat);
            let mut val = Wrapping(0);
            for (j, w) in &other.affine {
                val += w * v.get(&j).map(|x| *x).unwrap_or(Wrapping(if i == j {1} else {0}));
            }

            if val.0 != 0 {
                self.affine.entry(*i).and_modify(|mut e| *e += val).or_insert(val);
                if self.affine[i].0 == 0 {
                    self.affine.remove(i);
                }
            }
        }

        self.matrix = resulting_mat;
    }


    fn mul_const(&mut self, x: w8) {
        for (i, v) in &mut self.affine {
            *v = *v * x;
        }

        for (i, v) in &mut self.matrix {
            for (j, w) in v {
                *w = *w * x;
            }
        }
    }

    fn mul_var(&mut self, ind: isize, x: w8) {
        if !self.is_affine() {
            panic!("this is undefined!!!");
        }

        for (i, v) in &self.affine {
            let h = BTreeMap::from([(ind, v * x)]);
            self.matrix.insert(*i, h);
        }
        self.affine.clear();
    }
    
}

#[derive(Debug)]
pub enum Optree<T: BFAffineT<isize>> {
    OffsetMap(T),
    Branch(Vec<Optree<T>>, isize, isize),
    Input(isize),
    Output(isize),
}

impl<T: BFAffineT<isize> + std::fmt::Debug> Optree<T> {
    fn shift(&mut self, by: isize) -> bool { // returns whether the shift should propagate
        match self {
            Optree::OffsetMap(bfaddmap) => { bfaddmap.shift_keys(by); true },
            Optree::Input(x) | Optree::Output(x) => { *x += by; true },
            Optree::Branch(_, x, _) => { *x += by; false }
        }
    }

    fn size(&self) -> usize {
        match self {
            Optree::OffsetMap(_) | Optree::Input(_) | Optree::Output(_) => 1,
            Optree::Branch(a, _, _) => 1 + a.iter().map(|x| x.size()).sum::<usize>()
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
                current_offset_map.add_const(current_offset, Wrapping(1));
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

pub fn optimize<T: BFAffineT<isize> + std::fmt::Debug>(unoptimized: &mut Vec<Optree<T>>) {
    let mut index = 0;
    while index < unoptimized.len() {
        let mut prev = false;
        let l = unoptimized.len();
        match &unoptimized[index] {
            Optree::OffsetMap(bfaddmap) if index < l - 1 => {
                if let Some(Optree::OffsetMap(_)) = unoptimized.get(index + 1) {
                    let Optree::OffsetMap(mut el) = unoptimized.remove(index + 1) else { panic!("wrong check!") };
                    let Some(Optree::OffsetMap(bfaddmap)) = unoptimized.get_mut(index) else {panic!("wrong check!")};
                    el.matmul(&bfaddmap);
                    std::mem::swap(&mut el, bfaddmap);
                    prev = true;
                }
            },
            Optree::Branch(uo, preshift, 0) if uo.len() == 1 => {
                let preshift_ = *preshift;
                match &uo[0] {
                    // a classical [-] loop or variations of it
                    Optree::OffsetMap(m) if m.is_affine_at0() => {
                        std::mem::swap(&mut unoptimized[index], &mut Optree::OffsetMap({
                            let mut new_map = T::new_ident();
                            new_map.set_zero(preshift_);
                            new_map
                        }));
                        {
                            let mut j = index + 1;
                            while j < l && unoptimized[j].shift(preshift_) {
                                j += 1;
                            }
                        }
                        prev = true;
                        if index > 0 {
                            index -= 1;
                        }
                    },
                    Optree::OffsetMap(m) if m.is_affine() => {
                        let a = m.get_affine(0);
                        if a.0 & 1 == 1 {
                            let b = {
                                let Optree::Branch(uo, preshift, 0) = &mut unoptimized[index] else { panic!("wrong check!") };
                                let Optree::OffsetMap(m) = &mut uo[0] else { panic!("wrong check!") };
                                let mut b = T::new_ident();
                                std::mem::swap(m, &mut b);
                                let factor = multinv(256 - (a.0 as isize), 256) as u8;
                                b.mul_var(0, Wrapping(factor));
                                b.set_zero(0);
                                b
                            };

                            std::mem::swap(&mut unoptimized[index], &mut Optree::OffsetMap(b));
                            prev = true;
                            if index > 0 {
                                index -= 1;
                            }
                        }
                    }
                    _ => {
                        let Optree::Branch(uo, preshift, 0) = &mut unoptimized[index] else { panic!("wrong check!") };
                        optimize(&mut *uo);
                    }
                }
            }
            Optree::Branch(uo, _, _) => {
                let Optree::Branch(uo, _, _) = &mut unoptimized[index] else { panic!("wrong check!") };
                optimize(&mut *uo);
            }
            _ => {}
        }
        if !prev {
            index += 1;
        }
    }
}

#[derive(Debug)]
pub enum Opcode {
    Add(i16, w8),
    Set(i16, w8),
    ShiftBig(i16),
    Shift(i16),
    Read(i16),
    Write(i16),
    Jnez(i16, i16),
    Jez(i16, i16),
    Mult(i16, i16, w8),
}



pub fn linearize<T: BFAffineT<isize>>(tree: Vec<Optree<T>>) -> Vec<Opcode> {
    todo!()
}

pub fn main() {
    let args: Vec<String> = env::args().collect();

    let file_path = args.get(1).expect("No filepath given!");
    let contents = fs::read_to_string(file_path).expect("File doesn't exist");

    let mut tree = compile::<BFAddMap>(contents);
    let t1: usize = tree.iter().map(|x| x.size()).sum();
    dbg!(&tree);
    optimize(&mut tree);
    let t2: usize = tree.iter().map(|x| x.size()).sum();
    dbg!(&tree);
    dbg!(t1, t2);
}


