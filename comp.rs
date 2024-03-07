// Brainfuck bytecode compiler by ekstdo
#![feature(impl_trait_in_assoc_type)]
#![allow(non_camel_case_types)]
use std::env;
use std::fs;
use std::collections::{BTreeMap, BTreeSet, btree_map};
use std::num::Wrapping;
use std::iter::FromIterator;
use std::ops::{Add, Sub, Index};
use std::cmp::Ord;
use std::cell::RefCell;
use std::rc::Rc;
type w8 = Wrapping<u8>;

// Utility functions

fn shift_hashmap<T: Copy>(hashmap: &mut BTreeMap<isize, T>, by: isize) {
    let mut tbc = hashmap.keys().map(|x| *x).collect::<BTreeSet<isize>>();
    while!tbc.is_empty() {
        let mut old_key = *tbc.iter().next().unwrap();
        let mut new_key = old_key + by;
        let mut tmp = *hashmap.get(&old_key).unwrap();
        while let Some(_v) = hashmap.get(&new_key) {
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

fn multinv(a: isize, n: isize) -> isize {
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
pub trait BFAffineT<T: Copy + Ord>: Sized {
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
    fn get_involved(&self) -> BTreeSet<T> {
        self.get_involved_lin().union(&self.get_involved_aff()).map(|x| *x).collect()
    }
    fn get_involved_lin(&self) -> BTreeSet<T>;
    fn get_involved_aff(&self) -> BTreeSet<T>;

    fn shift_keys(&mut self, by: T);
    fn unset_linear(&mut self);
}

// Proxy Type to avoid shifting keys
#[derive(Debug, Clone)]
pub struct ShiftedMap<V> {
    inner: BTreeMap<isize, V>,
    by: Rc<RefCell<isize>>
}

impl<V> ShiftedMap<V> {
    fn new(shift: Rc<RefCell<isize>>) -> Self {
        Self { inner: BTreeMap::new(), by: shift }
    }

    fn get_by(&self) -> isize {
        *self.by.borrow()
    }

    fn keys(&self) -> impl Iterator<Item=isize> + '_{
        let a = self.get_by();
        return self.inner.keys().map(move |x| x + a);
    }

    fn clear(&mut self){
        self.inner.clear();
    }

    fn get(&self, key: &isize) -> Option<&V> {
        self.inner.get(&(key + self.get_by()))
    }

    fn remove(&mut self, key: &isize) -> Option<V> {
        self.inner.remove(&(key + self.get_by()))
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn entry(&mut self, key: isize) -> btree_map::Entry<isize, V> {
        self.inner.entry(key + &self.get_by())
    }

    fn insert(&mut self, key: isize, value: V) -> Option<V> {
        self.inner.insert(key + &self.get_by(), value)
    }
}

impl<'a, V> IntoIterator for &'a ShiftedMap<V> {
    type Item = (isize, &'a V);
    type IntoIter = impl Iterator<Item = Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        let a = self.get_by();
        self.inner.iter().map(move |(x, y)| (*x + a, y))
    }
}

impl<'a, V> IntoIterator for &'a mut ShiftedMap<V> {
    type Item = (isize, &'a mut V);
    type IntoIter = impl Iterator<Item = (isize, &'a mut V)>;
    fn into_iter(self) -> Self::IntoIter {
        let a = self.get_by();
        self.inner.iter_mut().map(move |(x, y)| (x + &a, y))
    }
}

impl<V> Index<&isize> for ShiftedMap<V> {
    type Output = V;
    fn index(&self, index: &isize) -> &Self::Output {
        self.inner.index(&(index + &self.get_by()))
    }
}

impl<V> Index<isize> for ShiftedMap<V> {
    type Output = V;
    fn index(&self, index: isize) -> &Self::Output {
        self.inner.index(&(index + &self.get_by()))
    }
}

// simple and naive approach to represent the arbitrary dimensional matrix
#[derive(Debug)]
pub struct BFAddMap { 
    // default is the zero vector
    affine: ShiftedMap<w8>,

    // default is identity matrix
    matrix: ShiftedMap<ShiftedMap<w8>>,
    shift: Rc<RefCell<isize>>
}

impl BFAddMap {
    fn new_map<T>(&self) -> ShiftedMap<T> {
        ShiftedMap::new(self.shift.clone())
    }

    fn transpose(&self) -> ShiftedMap<ShiftedMap<w8>> {
    }
}

impl BFAffineT<isize> for BFAddMap {
    fn new_ident() -> Self {
        let shift = Rc::new(RefCell::new(0));
        Self {
            affine: ShiftedMap::new(shift.clone()),
            matrix: ShiftedMap::new(shift.clone()),
            shift
        }
    }



    fn add_const(&mut self, i: isize, v: w8) {
        self.affine.entry(i).and_modify(|e| *e += v).or_insert(v);
        if self.affine[&i] == Wrapping(0) {
            self.affine.remove(&i);
        }
    }

    fn set_zero(&mut self, i:isize) {
        self.affine.remove(&i);
        let nm = self.new_map();
        let x = self.matrix.entry(i).or_insert(nm);
        x.clear();
        x.insert(i, Wrapping(0));
    }

    fn add_mul_raw(&mut self, dest: isize, src: isize, v: w8) {
        let nm = self.new_map();
        let m = self.matrix.entry(dest).or_insert(nm);
        m.entry(src).and_modify(|e| *e += v).or_insert(v);
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
        *self.shift.borrow_mut() += by;
        // shift_hashmap(&mut self.affine, by);
        // if self.matrix.is_empty() {
        //     return;
        // }

        // let mut new_multmap = BTreeMap::new();
        // let keys = self.matrix.keys().map(|x| *x).collect::<Vec<_>>();
        // for k in keys {
        //     let mut tmp_map = self.matrix.remove(&k).unwrap();
        //     shift_hashmap(&mut tmp_map, by);
        //     new_multmap.insert(k + by, tmp_map);
        // }
        // self.matrix = new_multmap;
    }

    // very inefficient, O(n^3), should be optimized in the future!
    fn matmul(&mut self, other: &Self) {
        // transposing the other matrix 
        let mut t_other: ShiftedMap<ShiftedMap<w8>> = ShiftedMap::new(self.shift.clone());
        for (i, v) in &other.matrix {
            for (j, w) in v {
                let m = t_other.entry(j).or_insert(ShiftedMap::new(self.shift.clone()));
                m.insert(i, *w);
            }
        }

        let mut keys: BTreeSet<isize> = BTreeSet::from_iter(self.matrix.keys());
        keys.extend(t_other.keys());

        let mut resulting_mat = self.new_map();
        let empty_mat = self.new_map();
        for i in &keys {
            let v = self.matrix.get(&i).unwrap_or(&empty_mat);
            let mut row = self.new_map();
            for j in &keys {
                match t_other.get(&j) {
                    Some(w) => {
                        let mut val = Wrapping(0);
                        for (x, a) in v {
                            val += a * w.get(&x).map(|x| *x).unwrap_or(Wrapping(if i == j { 1 } else { 0 }));
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

        let mut keys: BTreeSet<isize> = BTreeSet::from_iter(self.matrix.keys());
        keys.extend(other.affine.keys());
        for i in &keys {
            let v = self.matrix.get(&i).unwrap_or(&empty_mat);
            let mut val = Wrapping(0);
            for (j, w) in &other.affine {
                val += w * v.get(&j).map(|x| *x).unwrap_or(Wrapping(if *i == j {1} else {0}));
            }

            if val.0 != 0 {
                self.affine.entry(*i).and_modify(|e| *e += val).or_insert(val);
                if self.affine[i].0 == 0 {
                    self.affine.remove(&i);
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
            let mut h = self.new_map();
            h.insert(ind, v* x);
            dbg!(v, x);
            self.matrix.insert(i, h);
        }
        self.affine.clear();
    }
    
    fn get_involved_lin(&self) -> BTreeSet<isize> {
        self.affine.keys().map(|x| x).collect()
    }
    fn get_involved_aff(&self) -> BTreeSet<isize> {
        todo!()
    }

    fn unset_linear(&mut self) {
        self.matrix = ShiftedMap::new(self.shift.clone());
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
        match &mut unoptimized[index] {
            Optree::OffsetMap(bfaddmap) if index < l - 1 => {
                if let Some(Optree::OffsetMap(_)) = unoptimized.get(index + 1) {
                    let Optree::OffsetMap(mut el) = unoptimized.remove(index + 1) else { panic!("wrong check!") };
                    let Some(Optree::OffsetMap(bfaddmap)) = unoptimized.get_mut(index) else {panic!("wrong check!")};
                    el.matmul(&bfaddmap);
                    std::mem::swap(&mut el, bfaddmap);
                    prev = true;
                }
            },
            Optree::Branch(ref mut uo, preshift, 0) if uo.len() == 1 => {
                let preshift_ = *preshift;
                match &mut uo[0] {
                    // a classical [-] loop or variations of it
                    Optree::OffsetMap(m) if m.is_affine_at0() => {
                        std::mem::swap(&mut unoptimized[index], &mut Optree::OffsetMap({
                            let mut new_map = T::new_ident();
                            new_map.set_zero(preshift_);
                            new_map
                        }));
                        if preshift_ > 0 {
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
                    Optree::OffsetMap(ref mut m) if m.is_affine() => {
                        let a = m.get_affine(0);
                        if a.0 % 2 == 1 {
                            let mut b = {
                                let mut b = T::new_ident();
                                std::mem::swap(m, &mut b);
                                let factor = multinv(256 - (a.0 as isize), 256) as u8;
                                b.mul_var(0, Wrapping(factor));
                                b.set_zero(0);
                                b
                            };

                            b.shift_keys(preshift_);

                            std::mem::swap(&mut unoptimized[index], &mut Optree::OffsetMap(b));
                            if preshift_ > 0 {
                                let mut j = index + 1;
                                while j < l && unoptimized[j].shift(preshift_) {
                                    j += 1;
                                }
                            }
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
    if let Optree::OffsetMap(ref mut m) = &mut unoptimized[0] {
        m.unset_linear();
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

// BrainFuck Affine Linear Optimizing Data Structure
pub trait BFAFLODS<T: Copy + Ord> : BFAffineT<T> {
    // in order to compile the affine transforms, we need to make sure, we 
    // do the least amount of unnecessary computations and caching
    //
    // for that we can make use of some graph theory: When we see the matrix as
    // graph, we can first get all strongly connected components (which need
    // some form of caching the result) and topologically sort the rest 
    //
    // we then go backwards in the topological sorting, because these elements 
    // are the ones, that only dependent on other variables, while no other variables
    // depend on them
    fn new_graph_adapter(&self) -> impl Graphlike<T> + std::fmt::Debug;
    fn linearize(&self) -> Vec<Opcode>;
}

#[derive(Clone, Debug)]
pub struct BFAddMapGraphAdapter {
    vertices: Vec<isize>,
    edges: Vec<Vec<usize>>,
    translator: BTreeMap<isize, usize>
}

impl BFAFLODS<isize> for BFAddMap {
    fn new_graph_adapter(&self) -> impl Graphlike<isize> + std::fmt::Debug {
        let mut keys: BTreeSet<isize> = BTreeSet::from_iter(self.matrix.keys());
        keys.extend(self.affine.keys());

        let mut vertices = Vec::new();
        let mut translator = BTreeMap::new();
        let mut edges = Vec::new();
        for (index, i) in keys.iter().enumerate() {
            vertices.push(*i);
            translator.insert(*i, index);
            edges.push(Vec::new());
        }
        for (i, j) in &self.matrix {
            for (y, _) in j {
                edges[translator[&i]].push(translator[&y]);
            }
        }
        BFAddMapGraphAdapter { vertices, edges, translator }
    }
    fn linearize(&self) -> Vec<Opcode> {
        // 1. step: get strongly connected components from the hashmaps 
        // 2. step: topologically sort them 
        //
        // we can combine the first two steps by using Tarjan's algorithm,
        // that outputs the strongly connected components in reverse DAG-TS
        // order anyway:
        //


        let out = self.new_graph_adapter();
        dbg!(out);
        //
        // 3. convert into opcode
        //
        todo!()
    }
}



struct Graph<V> {
    vertices: Vec<V>,
    edges: Vec<Vec<usize>>
}

pub trait Graphlike<V>: Sized {
    fn get_num_verts(&self) -> usize;
    fn edges_iter(&self, i: usize) -> impl Iterator<Item = &usize>;

    fn tarjan(&self) -> Vec<Vec<usize>> {
        let mut index = 0;
        // the stack
        let mut s: Vec<usize> = Vec::new();
        // storing the tarjan states of index, lowlink and onStack
        let mut gstate: Vec<Option<TarjanState>> = Vec::new();
        // vector that stores the strongly connected components
        let mut connected: Vec<Vec<usize>> = Vec::new();
        gstate.resize(self.get_num_verts(), None);

        for vindex in 0..self.get_num_verts() {
            if let None = gstate[vindex] {
                strongconnect(self, vindex, &mut s, &mut gstate, &mut index, &mut connected);
            }
        }
        return connected;
    }
}

impl<V> Graphlike<V> for Graph<V> {
    fn get_num_verts(&self) -> usize {
        self.vertices.len()
    }

    fn edges_iter(&self, i: usize) -> impl Iterator<Item = &usize> {
        self.edges[i].iter()
    }
}

impl Graphlike<isize> for BFAddMapGraphAdapter {
    fn get_num_verts(&self) -> usize {
        self.vertices.len()
    }

    fn edges_iter(&self, i: usize) -> impl Iterator<Item = &usize> {
        self.edges[i].iter()
    }
}

#[derive(Clone)]
struct TarjanState {
    index: usize,
    lowlink: usize,
    onstack: bool
}




fn strongconnect<V, G: Graphlike<V>>(
    g: &G,
    vindex: usize,
    s: &mut Vec<usize>,
    gstate: &mut Vec<Option<TarjanState>>,
    index: &mut usize,
    connected: &mut Vec<Vec<usize>>
) -> Result<(), String> {
    gstate[vindex] = Some(TarjanState { index: *index, lowlink: *index, onstack: true });
    *index += 1;
    s.push(vindex);

    for &w in g.edges_iter(vindex) {
        match &gstate[w] {
            None => {
                let _ = strongconnect(g, w, s, gstate, index, connected);
                let l = gstate[w].as_ref().unwrap().lowlink;
                let v = gstate[vindex].as_mut().unwrap();
                v.lowlink = v.lowlink.min(l);
            },
            Some(x) if x.onstack => {
                let l = x.lowlink;
                let v = gstate[vindex].as_mut().unwrap();
                v.lowlink = v.lowlink.min(l);
            }
            Some(_) => {}
        }
    }

    let v = gstate[vindex]
        .as_ref()
        .ok_or(format!("ERROR: State of vertex {} got undefined (somehow)", vindex))?;
    if v.lowlink == v.index {
        let mut component = Vec::new();
        loop {
            let w = s
                .pop()
                .ok_or(format!("ERROR: Stack is empty, even though vertex {} is registered", vindex))?;
            match &mut gstate[w] {
                Some(x) => x.onstack = false,
                None => {}
            }
            component.push(w);
            if w == vindex { break; }
        }
        connected.push(component);
    }

    Ok(())
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
    if let Optree::OffsetMap(m) = &tree[0] {
        m.linearize();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    fn some_test() {
        let g = Graph {vertices: vec![1, 2, 3], edges: vec![vec![1], vec![0], vec![]]};
        println!("Hello, world! {:?}", g.tarjan());
    }
}
