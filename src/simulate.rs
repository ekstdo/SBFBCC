#![allow(non_camel_case_types)]
// Utility functions
use std::env;
use std::fmt::Debug;
use std::fs;
use std::collections::{BTreeMap, BTreeSet, btree_map};
use std::num::Wrapping;
use std::ops::MulAssign;
use std::ops::{Add, Sub, Index};
use std::cmp::Ord;
use std::cell::RefCell;
use std::rc::Rc;
type w8 = Wrapping<u8>;

// inefficient tape datatype to debug the compiler
pub struct MapTape {
    tape: BTreeMap<isize, w8>
}

impl MapTape {
    pub fn new () -> Self  {
        MapTape {
            tape: BTreeMap::new()
        }
    }
    
    // does: tape[to] = tape[from] * factor + affine
    pub fn apply(&mut self, to: isize, from: Option<isize>, factor: w8, affine: w8) -> w8 {
        let new_val = *from.and_then(|from_ind| self.tape.get(&from_ind)).unwrap_or(&Wrapping(0)) * factor + affine;
        self.tape.insert(to, new_val);
        new_val
    }
}

impl Debug for MapTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ind_line = String::from("󰇘 ");
        let mut top_line = String::from(" -");
        let mut mid_line = String::from("󰇘 ");
        let mut bot_line = String::from(" -");

        if (self.tape.is_empty()) {
            ind_line.push_str("empty 󰇘 ");
            top_line.push_str("--------");
            mid_line.push_str("empty 󰇘 ");
            bot_line.push_str("--------");
            writeln!(f, "{}\n{}\n{}\n{}", ind_line, top_line, mid_line, bot_line)?;
            return Ok(());
        }

        let min_ind = self.tape.keys().min().unwrap(); // empty case handeled above
        let max_ind = self.tape.keys().min().unwrap();
        let mut last_ind = min_ind;
        for (ind, val) in &self.tape {
            if ind - last_ind > 10 {
                ind_line.push_str(" 󰇘 ");
                top_line.push_str("---");
                mid_line.push_str(" 󰇘 ");
                bot_line.push_str("---");
            } else if ind - last_ind > 1 {
                for i in (*last_ind + 1)..*ind {
                    let ind_str = i.to_string();
                    let val_str = " ";
                    let width = ind_str.len().max(val_str.len());
                    ind_line.push_str("  ");
                    ind_line.push_str(&ind_str);
                    ind_line.push_str(&" ".repeat(width - ind_str.len() + 1));
                    mid_line.push_str("| ");
                    mid_line.push_str(val_str);
                    mid_line.push_str(&" ".repeat(width - val_str.len() + 1));
                    top_line.push('+');
                    top_line.push_str(&"-".repeat(width + 2));
                    bot_line.push('+');
                    bot_line.push_str(&"-".repeat(width + 2));
                }
            }
            let ind_str = ind.to_string();
            let val_str = val.to_string();
            let width = ind_str.len().max(val_str.len());
            ind_line.push_str("  ");
            ind_line.push_str(&ind_str);
            ind_line.push_str(&" ".repeat(width - ind_str.len() + 1));
            mid_line.push_str("| ");
            mid_line.push_str(&val_str);
            mid_line.push_str(&" ".repeat(width - val_str.len() + 1));
            top_line.push('+');
            top_line.push_str(&"-".repeat(width + 2));
            bot_line.push('+');
            bot_line.push_str(&"-".repeat(width + 2));
            last_ind = ind;
        }
        writeln!(f, "{}\n{}+-\n{}| 󰇘\n{}+-", ind_line, top_line, mid_line, bot_line)?;
        Ok(())
    }

}


