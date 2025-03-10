use std::{collections::{BTreeMap, BTreeSet}, hash::Hash, num::Wrapping};

use crate::utils::{multinv, shift_bmap, swap_entries, w8, w8div, w8inv, BTreeZipAnd, BTreeZipOr, One, Zero};

#[derive(Debug, PartialEq)]
pub struct Permutation<K> {
    pub inner: BTreeMap<K, K>
}

impl<K: Copy + Eq + Ord + Hash> Permutation<K> {
    pub fn swap(&mut self, row1: K, row2: K) {
        if row1 == row2 {
            return;
        }
        self.inner.entry(row1).or_insert(row1);
        self.inner.entry(row2).or_insert(row2);
        swap_entries(&mut self.inner, &row1, &row2);
        if self.inner.get(&row1) == Some(&row1) {
            self.inner.remove(&row1);
        }
        if self.inner.get(&row2) == Some(&row2) {
            self.inner.remove(&row2);
        }
    }

    pub fn reverse(&self) -> Permutation<K> {
        let mut new_inner = BTreeMap::new();
        for (k, v) in &self.inner {
            new_inner.insert(*v, *k);
        }
        Permutation { inner: new_inner }
    }

    pub fn get(&self, k: &K) -> K {
        *self.inner.get(k).unwrap_or(k)
    }

    pub fn apply<V>(&self, other: &mut BTreeMap<K, V>) {
        let mut swapped = BTreeMap::new();
        for &k in self.inner.keys() {
            swapped.insert(k, false);
        }
        for (&k, &v) in &self.inner {
            let mut tmp_v = v;
            swapped.insert(k, true);
            while !swapped[&tmp_v] {
                swapped.insert(tmp_v, true);
                swap_entries(other, &tmp_v, &k);
                tmp_v = self.inner[&tmp_v];
            }
        }
    }
}


impl Permutation<isize> {
    pub fn apply_matrix(&self, other: &mut Matrix) {
        let mut swapped = BTreeMap::new();
        for &k in self.inner.keys() {
            swapped.insert(k, false);
        }
        for (&k, &v) in &self.inner {
            let mut tmp_v = v;
            swapped.insert(k, true);
            while !swapped[&tmp_v] {
                swapped.insert(tmp_v, true);
                other.swap_rows(k, tmp_v);
                tmp_v = self.inner[&tmp_v];
            }
        }
    }
}

pub fn optimize_vec(vec: &mut BTreeMap<isize, w8>) -> bool {
    let mut changed = false;
    vec.retain(|&_, v| {
        let is_default = *v == w8::ZERO;
        changed = changed || is_default;
        !is_default // retain filters out false values, therefore we have to negate this
    });
    changed
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub inner: BTreeMap<isize, BTreeMap<isize, w8>>
}

impl Matrix {
    pub fn new() -> Matrix {
        Matrix {
            inner : BTreeMap::new()
        }
    }

    pub fn rows(&self) -> impl Iterator<Item=&isize> {
        self.inner.keys()
    }

    pub fn columns(&self) -> BTreeSet<isize> {
        let mut accum = BTreeSet::new();
        for row in self.inner.values() {
            for q in row.keys() {
                accum.insert(*q);
            }
        }
        accum
    }

    pub fn get_raw(&self, row: isize, column: isize) -> Option<w8> {
        self.inner.get(&row).and_then(|x| x.get(&column).copied())
    }

    pub fn get_row(row: &BTreeMap<isize, w8>, row_index: isize, column_index: isize) -> w8 {
        row.get(&column_index).copied().unwrap_or(if column_index == row_index {w8::ONE} else {w8::ZERO})
    }

    pub fn get(&self, row: isize, column: isize) -> w8 {
        self.get_raw(row, column).unwrap_or(if row == column { w8::ONE } else { w8::ZERO })
    }

    pub fn set(&mut self, row: isize, column: isize, value: w8) -> Option<w8> {
        let is_default = (row == column && value == w8::ONE) || (row != column && value == w8::ZERO);
        // if it already has that value by default, remove the entry
        if is_default {
            let row_mut = self.inner.get_mut(&row);
            return match row_mut {
                Some(r) => {
                    let v = r.remove(&column);
                    // if it's empty, we can remove the row :O
                    if r.is_empty() {
                        self.inner.remove(&row);
                    }
                    v
                },
                None => None
            };
        }
        // otherwise upsert it
        let entry = self.inner.entry(row).or_default();
        entry.insert(column, value)
    }

    pub fn optimize_all(&mut self) -> bool {
        let mut changed = false;
        for (&k, row) in self.inner.iter_mut() {
            row.retain(|&j, v| {
                let is_default = *v == if k == j { w8::ONE } else { w8::ZERO };
                changed = changed || is_default;
                !is_default // retain filters out false values, therefore we have to negate this
            });
        }
        self.inner.retain(|_, x| !x.is_empty());
        changed
    }

    pub fn optimize_row(&mut self, row_index: isize) -> bool {
        let mut changed = false;
        if let Some(row) = self.inner.get_mut(&row_index) {
            row.retain(|&j, v| {
                let is_default = (*v == w8::ZERO && row_index != j) || (*v == w8::ONE && row_index == j);
                changed = changed || is_default;
                !is_default // retain filters out false values, therefore we have to negate this
            });
            if row.is_empty() {
                self.inner.remove(&row_index);
                changed = true;
            }
        }
        changed
    }

    pub fn swap_rows(&mut self, row1: isize, row2: isize) {
        self.inner.entry(row1).or_insert(BTreeMap::from([(row1, w8::ONE)]));
        self.inner.entry(row2).or_insert(BTreeMap::from([(row2, w8::ONE)]));
        swap_entries(&mut self.inner, &row1, &row2);
        // correct out identity by default
        if let Some(row2_inner) = self.inner.get_mut(&row2) {
            // previously defaulted to 1, but it defaults to 0 now, so 
            // we need to set it to 1
            row2_inner.entry(row1).or_insert(w8::ONE);
            // same thing, just the other way around
            row2_inner.entry(row2).or_insert(w8::ZERO);
        }
        self.optimize_row(row2);
        if let Some(row1_inner) = self.inner.get_mut(&row1) {
            // previously defaulted to 1, but it defaults to 0 now, so 
            // we need to set it to 1
            row1_inner.entry(row2).or_insert(w8::ONE);
            // same thing, just the other way around
            row1_inner.entry(row1).or_insert(w8::ZERO);
        }
        self.optimize_row(row1);
    }

    pub fn zero_row(&mut self, row_index: isize) {
        let x = self.inner.entry(row_index).or_insert(BTreeMap::new());
        x.clear();
        x.insert(row_index, w8::ZERO);

    }

    pub fn mul_row(&mut self, row_index: isize, val: w8) {
        if let Some(row) = self.inner.get_mut(&row_index) {
            for v in row.values_mut() {
                *v *= val;
            }
            row.entry(row_index).or_insert(val);
        } else if val != w8::ONE { // if the val is 1, it's identity anyways
            let mut new_row = BTreeMap::new();
            new_row.insert(row_index, val);
            self.inner.insert(row_index, new_row);
        }
    }


    pub fn add_mul_row(&mut self, from: isize, dest: isize, factor: w8) {
        if self.inner.contains_key(&from) {
            let mut dest_row = self.inner.remove(&dest).unwrap_or_default();
            // default identity is multiplied by factor
            let from_row = self.inner.get(&from).unwrap();
            if !from_row.contains_key(&from) {
                dest_row.insert(from, factor + Self::get_row(&dest_row, dest, from));
            }
            // let mut new_row = BTreeMap::new();
            // for (&k, (v1, v2)) in BTreeZipOr::new(&from_row, &dest_row) {
            //     match (v1, v2) {
            //         (Some(v1), Some(v2)) => new_row.insert(k, v1 * factor + v2),
            //         (Some(v1), None) => new_row.insert(k, v1 * factor + if k == dest {w8::ONE} else {w8::ZERO} ),
            //         (None, Some(v2)) => new_row.insert(k, *v2),
            //         (None, None) => None
            //     };
            // }
            // self.inner.insert(dest, new_row);
            for (&i, j) in from_row {
                *dest_row.entry(i).or_insert(if dest == i {w8::ONE} else {w8::ZERO} ) += j * factor;
            }
            self.inner.insert(dest, dest_row);
        } else {
            self.set(dest, from, self.get(dest, from) + factor); // current value + factor * 1
        }
    }

    // This is a helper function to score which pivoting point to use. The more rows are linear
    // combination of a row, the more it's worth it to pivot that row.
    //
    // pivoting requires 1 additional permutation at most, so 2 read and 2 write instructions,
    // while avoiding a linear combination saves us 1 read, 1 write and 1 multiplication
    // instruction => if total lin comb saved > 2, it's worth it
    //
    //
    // this function counts how many of the first entries of a row in row2 are a factor apart from row1
    pub fn num_lin(&self, row1: isize, row2: isize) -> (usize, w8) {
        let mut lin_factor = None;
        let mut previous_k = isize::MIN;
        let mut counter = 0;
        let Some(row_entry1) = self.inner.get(&row1) else { return (0, w8::ZERO); };
        let Some(row_entry2) = self.inner.get(&row2) else { return (0, w8::ZERO); };
        for (k, (v1, v2)) in BTreeZipOr::new(row_entry1, row_entry2) {
            if (previous_k < row1 && row1 < *k) || (previous_k < row2 && row2 < *k) {
                // i.e. we skipped a default 1, so the other entry is 0
                // 0 * anything can't be 1 and only 1 * 0 = 0 mod 256 => lin factor has to be 0 => 
                // would return 0 anyways, so that would be invalid
                return (counter, lin_factor.unwrap_or(w8::ZERO));
            }
            let unwrapped_v1 = v1.copied().unwrap_or(if k == &row1 {w8::ONE} else {w8::ZERO});
            let unwrapped_v2 = v2.copied().unwrap_or(if k == &row2 {w8::ONE} else {w8::ZERO});
            match lin_factor {
                None => {
                    lin_factor = Some(w8div(unwrapped_v2, unwrapped_v1));
                    if lin_factor == Some(w8::ZERO) {
                        // if the other row starts with a 0, it's not helpful for comparison
                        return (0, w8::ZERO);
                    }
                },
                Some(lin) => {
                    if lin * unwrapped_v1 != unwrapped_v2 {
                        return (counter, lin);
                    }
                    counter += 1;
                }
            }
            previous_k = *k;

        }
        (counter, lin_factor.unwrap_or(w8::ZERO))
    }

    pub fn total_num_lin(&self, row: isize, starting_at: isize) -> usize {
        let mut counter = 0;
        for (k, _) in self.inner.range(starting_at..) {
            counter += self.num_lin(row , *k).0;
        }
        counter
    }


    pub fn first_row_with_oddest_entry(&self, column: isize, starting_at: isize) -> Option<(isize, w8)> {
        let default_identity = Some((column, w8::ONE));
        if self.inner.is_empty() { // default is the identity
            return default_identity;
        }
        let (&first_key, _) = self.inner.first_key_value().unwrap();
        let (&last_key, _) = self.inner.last_key_value().unwrap();
        // if column lies outside, default to identity
        if (column < first_key && column >= starting_at) || starting_at > last_key  {
            return default_identity;
        }
        // let mut candidate: Option<(isize, w8)> = None;
        let mut candidates: Vec<(isize, w8)> = Vec::new();
        let mut minimum_trailing_0 = u32::MAX;

        if !self.inner.contains_key(&column) && column >= starting_at {
            minimum_trailing_0 = 0;
            candidates.push((column, w8::ONE)); // Identity is a possibility
        }
        for (&row_index, row) in self.inner.range(starting_at..) {
            if let Some(&val) =  row.get(&column) {
                if val == w8::ZERO {
                    continue;
                }

                let val_t0 = val.0.trailing_zeros();

                if val_t0 < minimum_trailing_0 {
                    candidates = vec![(row_index, val)];
                    minimum_trailing_0 = val_t0;
                } else if val_t0 == minimum_trailing_0 {
                    candidates.push((row_index, val));
                }
            } else if row_index == column { // identity entry by default
                return default_identity;
            }
        }
        candidates.get(0).cloned()
    }

    pub fn first_row_with_oddest_entry_opt(&self, column: isize, starting_at: isize) -> Option<(isize, w8)> {
        let default_identity = Some((column, w8::ONE));
        if self.inner.is_empty() { // default is the identity
            return default_identity;
        }
        let (&first_key, _) = self.inner.first_key_value().unwrap();
        let (&last_key, _) = self.inner.last_key_value().unwrap();
        // if column lies outside, default to identity
        if (column < first_key && column >= starting_at) || starting_at > last_key  {
            return default_identity;
        }
        // let mut candidate: Option<(isize, w8)> = None;
        let mut candidates: Vec<(isize, w8)> = Vec::new();
        let mut minimum_trailing_0 = u32::MAX;

        if !self.inner.contains_key(&column) && column >= starting_at {
            minimum_trailing_0 = 0;
            candidates.push((column, w8::ONE)); // Identity is a possibility
        }
        for (&row_index, row) in self.inner.range(starting_at..) {
            if let Some(&val) =  row.get(&column) {
                if val == w8::ZERO {
                    continue;
                }

                let val_t0 = val.0.trailing_zeros();

                if val_t0 < minimum_trailing_0 {
                    candidates = vec![(row_index, val)];
                    minimum_trailing_0 = val_t0;
                } else if val_t0 == minimum_trailing_0 {
                    candidates.push((row_index, val));
                }
            } else if row_index == column { // identity entry by default
                return default_identity;
            }
        }
        let mut max_total_lin_comb = 0;
        let mut max_candidate = candidates.get(0).cloned();
        for i in &candidates {
            let total_lin_comb = self.total_num_lin(i.0, starting_at);
            if total_lin_comb > 3 && total_lin_comb > max_total_lin_comb {
                max_total_lin_comb = total_lin_comb;
                max_candidate = Some(i.clone());
            }
        }
        max_candidate
    }

    pub fn check_plu_equiv(mat: &Matrix, p: &Permutation<isize>, l: &Matrix, u: &Matrix) -> bool {
        let mut mat_clone = mat.clone();
        mat_clone.optimize_all();
        let mut lu = l.matmul(u);
        p.apply_matrix(&mut lu);
        lu.optimize_all();
        let result = mat_clone == lu;
        if !result {
            dbg!("LU: {:?}", lu);
        }
        result
    }

    // computes the PLU decomposition and returns a permutation, a lower triangular matrix and an
    // upper triangular matrix
    //
    // runs in O(n^3 log n) as there are O(n^2) add_mul_row calls, each taking O(n log n) time
    // could be made O(n^3)
    pub fn plu(mut self) -> (Permutation<isize>, Matrix, Matrix) {
        let mut perm = Permutation { inner: BTreeMap::new() };
        let mut left = Matrix { inner: BTreeMap::new() };

        // time for some gaussian elimination
        let row_indices = self.rows().copied().collect::<BTreeSet<_>>();
        let column_indices = self.columns();
        let all_indices = column_indices.union(&row_indices).collect::<Vec<_>>();
        for (index_index, &&index) in all_indices.iter().enumerate() {
            // notes on division
            //
            // PLU decomposition requires a field F to work in, which requires multiplicative
            // inverses. Because we work with modulo 256 (in a non polynomial field), this isn't
            // possible, as only odd (share no factors with 2 in mod 2^8) numbers have
            // multiplicative inverses.
            //
            // Therefore we get the following cases:
            //
            // All values in a column are odd -> amazing, any row will do
            //
            // There exists one odd value in the column -> use that
            //
            // There are no odd values in the column -> this is problematic, but not too bad
            //
            // if we have 12 (11_00) and 16 (100_00) for example, and want to divide 16 by 12, we can
            // still do 4/3 (odd quotient) * 12 = 4 * 171 * 12 = 16 mod 256
            // i.e. we just act like bitshifting all the values, until there is an odd number
            // (which is guaranteed for non zero entries)
            //
            // we're therefore looking for the row with the lowest trailing zeros in a bit
            // representation!

            // check in which row this column exists
            let Some((row, row_val)) = self.first_row_with_oddest_entry_opt(index, index) else {continue;};

            if row != index {
                self.swap_rows(row, index); // we swap the rows for U
                swap_entries(&mut left.inner, &row, &index); // swap the rows for L; unlike U we
                                                             // want to avoid switching the default
                                                             // identity, as you'd also have to
                                                             // switch the columns for L
                perm.swap(row, index);
            }
            for &&srow_index in &all_indices[index_index + 1..] {
                let mut factor = self.get(srow_index, index);
                if factor == w8::ZERO {
                    continue;
                }
                factor = w8div(factor, row_val);
                self.add_mul_row(index, srow_index, -factor);
                left.set(srow_index, index, factor);
            }
        }
        (perm, left, self)

        // we can further decompose the left matrix, by looking for 0 rows in
        // the right matrix. e.g. a row could be a multiple of another row
    }

    // converts an upper right triangle matrix to a list of 
    // in place op codes 
    //
    // when we start with 
    //
    // * * * * *    a
    //   * * * *    b
    //     * * *    c
    //       * *    d
    //         *    e 
    //
    // so we can replace a first, as nothing else depends on a
    pub fn u_op(&self) -> Vec<MatOpCode> {
        let mut result = Vec::new();
        for (&k, row) in &self.inner {
            for (&column, &val) in row {
                if (k == column) {
                    result.push(MatOpCode::Mul(k, val));
                } else {
                    result.push(MatOpCode::Add(k, column, val));
                }
            }
        }
        result
    }

    // converts an lower left triangle matrix to a list of 
    // in place op codes 
    //
    // when we start with 
    //
    // *            a
    // * *          b
    // * * *        c
    // * * * *      d
    // * * * * *    e 
    //
    // so we can replace e first, as nothing else depends on e
    pub fn l_op(&self) -> Vec<MatOpCode> {
        let mut result = Vec::new();
        for (&k, row) in self.inner.iter().rev() {
            for (&column, &val) in row.iter().rev() {
                if k == column {
                    result.push(MatOpCode::Mul(k, val));
                    continue;
                }
                result.push(MatOpCode::Add(k, column, val));
            }
        }
        result
    }

    pub fn to_opcode(self) -> (Permutation<isize>, Vec<MatOpCode>) {
        let (p, l, u) = self.plu();
        let mut u_op = u.u_op();
        let l_op = l.l_op();
        u_op.extend(l_op.iter().cloned());
        (p, u_op)
    }

    pub fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::new();
        for (&row_index, row) in &self.inner {
            for (&column_index, &val) in row {
                transposed.set(column_index, row_index, val);
            }
        }
        transposed
    }

    pub fn vecmul(&self, other: &BTreeMap<isize, w8>) -> BTreeMap<isize, w8> {
        let mut result = BTreeMap::new();
        for (k, en) in other.iter() {
            if self.inner.contains_key(k) {
                continue; // we handle this in the second for loop
            }
            result.insert(*k, *en); // identity rows by default
        }
        for (&row_index, row) in self.inner.iter() {
            let e = Self::row_vec_mul(row_index, row, other);
            
            if e != w8::ZERO {
                result.insert(row_index, e);
            }
        }
        result
    }

    pub fn row_vec_mul(row_index: isize, row: &BTreeMap<isize, w8>, column: &BTreeMap<isize, w8>) -> w8 {
        let mut result = w8::ZERO;
        // if row doesn't contain its index: 1 default entry
        if !row.contains_key(&row_index) && column.contains_key(&row_index) {
            result += column[&row_index];
        }
        for (_, (i, j)) in BTreeZipAnd::new(row, column) {
            result += i * j;
        }
        result
    }

    pub fn row_column_mul(row_index: isize, row: &BTreeMap<isize, w8>, column_index: isize, column: &BTreeMap<isize, w8>) -> w8 {
        let mut result = w8::ZERO;
        // if row doesn't contain its index: 1 default entry
        // if column doesn't contain its index: same thing
        // if both don't contain themselves, but are equal, add 1
        // else don't do anything
        if !column.contains_key(&column_index) {
            if let Some(r) = row.get(&column_index) {
                result += *r;
            } else if row_index == column_index {
                result += 1; // 1 * 1 is 1, both default to 1
            }
        }
        result + Self::row_vec_mul(row_index, row, column)
    }

    // note: that, as in mathematical notation, the right matrix get's applied first
    //
    // runs in O(n^3)
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        let mut new_inner = Matrix::new();

        let other_t = other.transpose();
        let self_keys = self.rows().collect::<BTreeSet<_>>();
        let other_keys = other_t.rows().collect::<BTreeSet<_>>();
        let exclusive_self_keys = self_keys.difference(&other_keys);
        let exclusive_other_keys = other_keys.difference(&self_keys);
        // O(n^3)
        for (&self_row_index, self_row) in &self.inner {
            for (&other_column_index, other_column) in &other_t.inner {
                new_inner.set(self_row_index, other_column_index, Self::row_column_mul(self_row_index, self_row, other_column_index, other_column));
            }
        }
        // O(n^2 log n)
        
        for &&exclusive_column_index in exclusive_other_keys { // therefore these are identity rows
            for (&other_column_index, other_column) in &other_t.inner {
                if let Some(&x) = other_column.get(&exclusive_column_index) {
                    new_inner.set(exclusive_column_index, other_column_index, x);
                }
            }
        }
        for &&exclusive_row_index in exclusive_self_keys { // therefore these are identity columns
            for (&self_row_index, self_row) in &self.inner {
                if let Some(&x) = self_row.get(&exclusive_row_index) {
                    new_inner.set(self_row_index, exclusive_row_index, x);
                }
            }
        }

        for (&other_column_index, other_column) in &other_t.inner {
            for (&other_row_index, other_val) in other_column {
                if !self_keys.contains(&other_row_index) {
                    new_inner.set(other_row_index, other_column_index, *other_val);
                }
            }
        }

        for (&self_row_index, self_row) in &self.inner {
            for (&self_column_index, self_val) in self_row {
                if !other_keys.contains(&self_column_index) {
                    new_inner.set(self_row_index, self_column_index, *self_val);
                }
            }
        }
        new_inner
    }

    pub fn and_then(&self, other: &Matrix) -> Matrix {
        other.matmul(self)
    }

    pub fn shift_key(&mut self, by: isize) {
        self.inner.iter_mut().for_each(|(_, r)| shift_bmap(r, by) );
        shift_bmap(&mut self.inner, by);
    }
}

impl From<Vec<Vec<u8>>> for Matrix {
    fn from(value: Vec<Vec<u8>>) -> Self {
        let mut inner = BTreeMap::new();
        for (j, row) in value.into_iter().enumerate() {
            let mut new_row = BTreeMap::new();
            for (k, x) in row.into_iter().enumerate() {
                let is_default = (j == k && x == 1) || (j != k && x == 0);
                if !is_default {
                    new_row.insert(k as isize, Wrapping(x));
                }
            }
            inner.insert(j as isize, new_row);
        }
        Matrix { inner }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatOpCode {
    Add(isize, isize, w8),
    Mul(isize, w8)
}

pub fn apply_mat_opcode(opcodes: &Vec<MatOpCode>, vector: &mut BTreeMap<isize, w8>) {
    for opcode in opcodes {
        match opcode {
            MatOpCode::Mul(i, v) => {vector.entry(*i).and_modify(|x| *x *= v);}
            MatOpCode::Add(to, from, v) => {
                let a = vector.get(from).copied().unwrap_or(w8::ZERO);
                *vector.entry(*to).or_insert(w8::ZERO) += a * v;
            }
        }
    }
}

pub fn addi_vec(a: &mut BTreeMap<isize, w8>, b: &BTreeMap<isize, w8>) {
    for (k, v) in b.iter() {
        a.entry(*k).and_modify(|e| *e += v).or_insert(*v);
    }
    a.retain(|&_, v| *v != w8::ZERO);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distr::{StandardUniform, Uniform}, prelude::*};

    #[test]
    fn test_mat_opcode(){
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..100 {
            println!("Iter: {i}");
            // let width = rng.next_u32() % 100; // I know modulo bias is a thing, but testing this is such a low
                                  // severity usecase, that I'll just modulo it
            // let height = rng.next_u32() % 100;
            let width = 50;
            let height = 50;
            let vec = gen_random_vec(height as usize, &mut rng);
            let mat = gen_random_matrix(width as usize, height as usize, &mut rng);

            let (p, mut l, mut u) = mat.clone().plu();
            l.optimize_all();
            u.optimize_all();

            let l_op = l.l_op();
            let u_op = u.u_op();
            let mut all_op = u_op.clone();
            all_op.append(&mut l_op.clone());

            let mut vec_lop = vec.clone();
            let mut vec_uop = vec.clone();
            let mut vec_allop = vec.clone();
            apply_mat_opcode(&l_op, &mut vec_lop);
            apply_mat_opcode(&u_op, &mut vec_uop);
            apply_mat_opcode(&all_op, &mut vec_allop);
            p.apply(&mut vec_allop);
            optimize_vec(&mut vec_lop);
            optimize_vec(&mut vec_uop);
            optimize_vec(&mut vec_allop);

            let mut vec2 = l.vecmul(&vec);
            let mut vec3 = u.vecmul(&vec);
            let mut vec4 = mat.vecmul(&vec);
            optimize_vec(&mut vec2);
            optimize_vec(&mut vec3);
            optimize_vec(&mut vec4);

            assert_eq!(vec2, vec_lop);
            assert_eq!(vec3, vec_uop);
            assert_eq!(vec4, vec_allop);
        }
    }


    #[test]
    fn test_vec_mul() {
        let test_matrix = Matrix::new();
        let test_vec1 = BTreeMap::from([(1, Wrapping(1)), (3, Wrapping(4))]);
        assert_eq!(&test_vec1, &test_matrix.vecmul(&test_vec1 ));
        let test_matrix = Matrix::from(vec![
            vec![0, 0, 0],
            vec![0, 0, 0],
            vec![0, 0, 0],
            vec![0, 2, 0, 4],
        ]);
        assert_eq!(BTreeMap::from([(3, Wrapping(2) * test_vec1[&1] + Wrapping(4) * test_vec1[&3])]), test_matrix.vecmul(&test_vec1 ));
    }

    fn gen_random_vec<T: RngCore>(height: usize, rng: &mut T) -> BTreeMap<isize, w8> {
        let range = StandardUniform;
        let mut result = BTreeMap::new();
        for (index, i) in rng.sample_iter(&range).take(height).enumerate() {
            result.insert(index as isize, Wrapping(i));
        }
        result
    }

    fn gen_random_matrix<T: RngCore>(width: usize, height: usize, rng: &mut T) -> Matrix {
        let range = StandardUniform;
        let mut mat = Vec::new();
        for i in 0..height {
            let row: Vec<u8> = rng.sample_iter(&range).take(width).collect();
            mat.push(row);
        }
        Matrix::from(mat)
    }

    #[test]
    fn test_transpose() {
        let mut rng = StdRng::seed_from_u64(42);
        use std::time::Instant;
        let mut times = Vec::new();
        for i in 0..100 {
            println!("Iter: {i}");
            // let width = rng.next_u32() % 100; // I know modulo bias is a thing, but testing this is such a low
                                  // severity usecase, that I'll just modulo it
            // let height = rng.next_u32() % 100;
            let width = 100;
            let height = 100;
            let mut mat = gen_random_matrix(width as usize, height as usize, &mut rng);

            println!("Generated: {i}");
            let now = Instant::now();
            let transposed = mat.transpose();
            let mut dtransposed = transposed.transpose();
            let elapsed = now.elapsed();
            times.push(elapsed);
            println!("Double transposed: {i}");

            dtransposed.optimize_all();
            mat.optimize_all();
            assert_eq!(dtransposed, mat);
        }
        let total: u128 = times.iter().map(|x| x.as_micros()).sum();
        println!("max time {:?}", times.iter().max());
        println!("avg time {:?}", total as f32 / 100. / 1000.);
    }

    #[test]
    fn test_lu_example() {
        let test_matrix = Matrix::from(vec![
            vec![1, 2, 3],
            vec![2, 4, 5],
            vec![1, 3, 4]
        ]);
        let (p, mut l, mut u) = test_matrix.clone().plu();
        let mut lu = l.matmul(&u);
        p.apply_matrix(&mut lu);
        assert_eq!(lu, test_matrix);


        let test_matrix = Matrix::from(vec![
            vec![1, 2, 3, 2],
            vec![2, 4, 5, 9],
            vec![2, 4, 2, 18],
            vec![1, 3, 4, 8, 2],
            vec![1, 2, 1, 9],
            vec![1, 3, 4, 8, 4],
            vec![2, 4, 2, 18],
            vec![1, 9, 9, 9, 3],

        ]);
        let (p, l, u) = test_matrix.clone().plu();
        let mut lu = l.matmul(&u);
        p.apply_matrix(&mut lu);
        assert_eq!(lu, test_matrix);

        let mut rng = StdRng::seed_from_u64(42);
        use std::time::Instant;
        let mut times = Vec::new();
        for i in 0..100 {
            println!("Iter: {i}");
            // let width = rng.next_u32() % 100; // I know modulo bias is a thing, but testing this is such a low
                                  // severity usecase, that I'll just modulo it
            // let height = rng.next_u32() % 100;
            let width = 100;
            let height = 100;
            let mut mat = gen_random_matrix(width as usize, height as usize, &mut rng);

            println!("Generated: {i}");
            let now = Instant::now();
            let (p, mut l, mut u) = mat.clone().plu();
            let elapsed = now.elapsed();
            times.push(elapsed);
            // p.apply_matrix(&mut l); // associativity ftw

            u.optimize_all();
            let mut lu = l.matmul(&u);
            p.apply_matrix(&mut lu);
            lu.optimize_all();
            mat.optimize_all();
            // p.reverse().apply_matrix(&mut mat);
            assert_eq!(lu, mat);
        }
        let total: u128 = times.iter().map(|x| x.as_micros()).sum();
        println!("max time {:?}", times.iter().max());
        println!("avg time {:?}", total as f32 / 100. / 1000.);
    }

    #[test]
    fn test_num_lin(){
        let mut m1 = Matrix::from(vec![
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 4], // identical case -> 3
            vec![2, 4, 6, 8], // with a multiple -> 3
            vec![2, 4, 6, 9], // partially a multiple of the first row -> 2
            vec![0, 4, 6, 9], // starts with 0 -> 0
        ]);
        m1.optimize_all();
        assert_eq!(m1.num_lin(0, 1), (3, w8::ONE));
        assert_eq!(m1.num_lin(0, 2), (3, Wrapping(2)));
        assert_eq!(m1.num_lin(0, 3), (2, Wrapping(2)));
        assert_eq!(m1.num_lin(0, 4), (0, Wrapping(0)));
        let mut m2 = Matrix::from(vec![
            vec![],
            vec![], // identical case -> 3
            vec![], // with a multiple -> 3
            vec![0, 0, 2], // partially a multiple of the first row -> 2
            vec![0, 4, 6, 9], // starts with 0 -> 0
        ]);
        m2.optimize_row(3);
        assert_eq!(m2.num_lin(2, 3), (0, Wrapping(2)));
        m2.optimize_all();
        assert_eq!(m2.num_lin(2, 3), (0, Wrapping(0)));
    }
}



