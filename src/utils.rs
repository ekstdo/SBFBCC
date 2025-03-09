#![feature(trait_alias)]
use std::ops::{Add, Div, Index, Sub, Rem, Mul};
use std::collections::{BTreeMap, BTreeSet, btree_map};
use std::num::Wrapping;
pub type w8 = Wrapping<u8>;

pub struct BTreeChain<'a, K, V> {
    a: btree_map::Iter<'a, K, V>,
    b: btree_map::Iter<'a, K, V>,
    next_a: Option<(&'a K, &'a V)>,
    next_b: Option<(&'a K, &'a V)>
}

impl<'a, K, V> BTreeChain<'a, K, V> {
    pub fn new(a: &'a BTreeMap<K, V>, b: &'a BTreeMap<K, V>) -> Self {
        let mut a = a.iter();
        let next_a = a.next();
        let mut b = b.iter();
        let next_b = b.next();
        BTreeChain { a, b, next_a, next_b }
    }
}

impl<'a, K: Ord, V> Iterator for BTreeChain<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match (self.next_a, self.next_b) {
            (None, None) => None,
            (Some(i), None) => {
                self.next_a = self.a.next();
                Some(i)
            },
            (None, Some(i)) => {
                self.next_b = self.b.next();
                Some(i)
            },
            (Some(i), Some(j)) => {
                if i.0 <= j.0 {
                    self.next_a = self.a.next();
                    Some(i)
                } else {
                    self.next_b = self.b.next();
                    Some(j)
                }
            }
        }
    }
}












pub struct BTreeZipOr<'a, K, V> {
    a: btree_map::Iter<'a, K, V>,
    b: btree_map::Iter<'a, K, V>,
    next_a: Option<(&'a K, &'a V)>,
    next_b: Option<(&'a K, &'a V)>
}

impl<'a, K, V> BTreeZipOr<'a, K, V> {
    pub fn new(a: &'a BTreeMap<K, V>, b: &'a BTreeMap<K, V>) -> Self {
        let mut a = a.iter();
        let next_a = a.next();
        let mut b = b.iter();
        let next_b = b.next();
        Self { a, b, next_a, next_b }
    }
}

impl<'a, K: Ord, V> Iterator for BTreeZipOr<'a, K, V> {
    type Item = (&'a K, (Option<&'a V>, Option<&'a V>));
    fn next(&mut self) -> Option<Self::Item> {
        match (self.next_a, self.next_b) {
            (None, None) => None,
            (Some(i), None) => {
                self.next_a = self.a.next();
                Some((i.0, (Some(i.1), None)))
            },
            (None, Some(i)) => {
                self.next_b = self.b.next();
                Some((i.0, (None, Some(i.1))))
            },
            (Some(i), Some(j)) => {
                if i.0 == j.0 {
                    self.next_a = self.a.next();
                    self.next_b = self.b.next();
                    Some((i.0, (Some(i.1), Some(j.1))))
                } else if i.0 < j.0 {
                    self.next_a = self.a.next();
                    Some((i.0, (Some(i.1), None)))
                } else {
                    self.next_b = self.b.next();
                    Some((j.0, (None, Some(j.1))))
                }
            }
        }
    }
}


pub struct BTreeZipAnd<'a, K, V> {
    a: btree_map::Iter<'a, K, V>,
    b: btree_map::Iter<'a, K, V>,
    next_a: Option<(&'a K, &'a V)>,
    next_b: Option<(&'a K, &'a V)>
}

impl<'a, K, V> BTreeZipAnd<'a, K, V> {
    pub fn new(a: &'a BTreeMap<K, V>, b: &'a BTreeMap<K, V>) -> Self {
        let mut a = a.iter();
        let next_a = a.next();
        let mut b = b.iter();
        let next_b = b.next();
        Self { a, b, next_a, next_b }
    }
}

impl<'a, K: Ord, V> Iterator for BTreeZipAnd<'a, K, V> {
    type Item = (&'a K, (&'a V, &'a V));
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match (self.next_a, self.next_b) {
                (Some(i), Some(j)) => {
                    if i.0 == j.0 {
                        self.next_a = self.a.next();
                        self.next_b = self.b.next();
                        return Some((i.0, (i.1, j.1)))
                    } else if i.0 < j.0 {
                        self.next_a = self.a.next();
                    } else {
                        self.next_b = self.b.next();
                    }
                },
                _ => return None
            }
        }
    }
}







pub fn swap_entries<K: Ord + std::hash::Hash + Clone, V>(map: &mut BTreeMap<K, V>, key1: &K, key2: &K) {
    if key1 == key2 {
        return;
    }

    let ptr1 = map.get_mut(key1).map(|v| v as *mut V);
    let ptr2 = map.get_mut(key2).map(|v| v as *mut V);

    if let (Some(ptr1), Some(ptr2)) = (ptr1, ptr2) {
        unsafe {
            std::ptr::swap(ptr1, ptr2);
        }
    } else if let (Some(_), None) = (ptr1, ptr2) {
        let v = map.remove(key1).unwrap();
        map.insert(key2.clone(), v);
    } else if let (None, Some(_)) = (ptr1, ptr2) {
        let v = map.remove(key2).unwrap();
        map.insert(key1.clone(), v);
    }
}


pub fn shift_bmap<U: Ord + Add<Output = U> + Copy + Zero, T>(bmap: &mut BTreeMap<U, T>, by: U) {
    let keys = bmap.keys().copied().collect::<Vec<_>>();
    let fun = |key| {
        let val = bmap.remove(&key).unwrap();
        bmap.insert(key + by, val);
    };
    if by > U::ZERO { keys.into_iter().rev().for_each(fun) } else { keys.into_iter().for_each(fun) } 
}

pub trait Zero {
    const ZERO: Self;
}

pub trait One {
    const ONE: Self;
}

macro_rules! impl_zero_def {
    ($a:ty) => {
        impl Zero for $a {
            const ZERO: $a = 0;
        }
    };
}

macro_rules! impl_one_def {
    ($a:ty) => {
        impl One for $a {
            const ONE: $a = 1;
        }
    };
}

macro_rules! impl_field_def {
    ($a:ty) => {
        impl_zero_def!($a);
        impl_one_def!($a);
    };
}

impl_field_def!(i8);
impl_field_def!(i16);
impl_field_def!(i32);
impl_field_def!(i64);
impl_field_def!(i128);
impl_field_def!(u8);
impl_field_def!(u16);
impl_field_def!(u32);
impl_field_def!(u64);
impl_field_def!(u128);
impl_field_def!(isize);
impl_field_def!(usize);

impl<T:Zero> Zero for Wrapping<T> {
    const ZERO: Self = Wrapping(T::ZERO);
}

impl<T:One> One for Wrapping<T> {
    const ONE: Self = Wrapping(T::ONE);
}

trait Num<T> = Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Rem<Output=T> + Clone + PartialOrd + Zero + One;

pub fn eeagcd<T>(mut a: T, mut b: T) -> (T, T, T, T, T)
where T: Num<T> {
    let (mut x, mut v, mut y, mut u) = (T::ONE, T::ONE, T::ZERO, T::ZERO);
    while b != T::ZERO {
        let q = a.clone() / b.clone();
        (a, b, x, y, u, v) = (b.clone(), a % b, u.clone(), v.clone(), x - q.clone() * u, y - q * v);
    }
    (a, x, y, u, v)
}

pub fn multinv<T>(a: T, n: T) -> T
where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Rem<Output=T> + Copy + PartialOrd + Zero + One {
    eeagcd(a, n).1 % n
}

pub fn w8inv(a: w8) -> w8 {
    Wrapping(eeagcd(a.0 as isize, 256isize).1 as u8)
}

pub fn w8div(a: w8, b: w8) -> w8 {
    // a.trailing_zeros >= b.trailing_zeros  HAS to be true, but is unchecked
    if b == w8::ONE {
        return a;
    }
    let s = b.0.trailing_zeros();
    Wrapping(a.0 >> s) * w8inv(Wrapping(b.0 >> s))
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
        connected
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




#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_w8_div() {
        assert_eq!(w8div(Wrapping(1), Wrapping(3)) * Wrapping(3), Wrapping(1) ); // inverse
        assert_eq!(w8div(Wrapping(3), Wrapping(5)) * Wrapping(5), Wrapping(3) ); // all odd
        assert_eq!(w8div(Wrapping(2), Wrapping(5)) * Wrapping(5), Wrapping(2) ); // divident even
                                                                                 // (divisor even,
                                                                                 // dividend odd is
                                                                                 // undefined)
        assert_eq!(w8div(Wrapping(8), Wrapping(4)) * Wrapping(4), Wrapping(8) );
        assert_eq!(w8div(Wrapping(16), Wrapping(12)) * Wrapping(12), Wrapping(16) );
    }

    fn tarjan_test() {
        let g = Graph {vertices: vec![1, 2, 3], edges: vec![vec![1], vec![0], vec![]]};
        println!("Hello, world! {:?}", g.tarjan());
    }
}
