use std::fmt::Display;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;
use arrayfire::{Array, assign_seq, constant, ConstGenerator, HasAfEnum, Seq, dim4};

// Dataset struct, contains samples and truth labels.
pub struct Dataset<T: HasAfEnum> {pub x: Array<T>, pub y: Array<u32>}
impl<T: HasAfEnum + ConstGenerator<OutType = T> + FromStr + Display> Dataset<T> {
    pub fn from_csv(path: &Path) -> Option<Dataset<T>> { // Stupid very slow function
        if !path.exists() {
            println!("Path does not exist!");
            return None;
        }
        let mut file: File = File::open(path).unwrap();
        let mut contents: String = String::new();
        file.read_to_string(&mut contents).unwrap();

        // Redefine contents as an empty vec
        let c: Vec<&str> = contents.split("\n").collect::<Vec<&str>>();
        let mut contents: Vec<Vec<&str>> = Vec::new();

        // preprocess contents to eliminate newlines or bad encodings
        for i in c {
            let line: Vec<&str> = i.trim_end_matches("\r").split(",").collect::<Vec<&str>>();
            if line.len() > 0 {
                for l in &line {
                    if l.parse::<u32>().is_err() || l.parse::<T>().is_err() {
                        continue;
                    }
                    contents.push(line.clone());
                }
            }
        }

        let mut labels: Array<u32> = Array::new_empty(dim4!(contents.len() as u64));
        let mut vals: Array<T> = Array::new_empty(dim4!(contents.len() as u64, (contents[0].len() - 1) as u64));

        for line_number in 0..contents.len() {
            let label: u32 = contents[line_number][0].parse::<u32>().unwrap();
            assign_seq(&mut labels, &[Seq::new(line_number as u32, line_number as u32, 1)], &constant(label, dim4!(1)));

            let mut values: Array<T> = Array::new_empty(dim4!(1, (contents[line_number].len() - 1) as u64));
            for val in 0..contents[line_number].len() - 1 {
                let v: T = contents[line_number][val + 1].parse::<T>().ok()?;
                assign_seq(&mut values, &[Seq::new(val as u32, val as u32, 1)], &constant(v, dim4!(1)));
            }
            assign_seq(&mut vals, &[Seq::new(line_number as u32, line_number as u32, 1), Seq::new(0, (contents[line_number].len() - 2) as u32, 1)], &values);
        }

        Some(Dataset::<T> {x: vals, y: labels})
    }
}