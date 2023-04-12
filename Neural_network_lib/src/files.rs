use super::types_and_errors::*;
use bincode::{self, deserialize, serialize};
use nalgebra::DVector;
use std::{
    fs::File,
    io::{self, BufReader, Read, Result, Write},
    path::Path,
};

pub fn save_network<P: AsRef<Path>>(path: P, network: &NeuralNetwork) -> Result<()> {
    let mut f = File::create(path)?;
    let buf = match serialize(network) {
        Ok(buf) => buf,
        Err(_) => {
            return Err(std::io::Error::new(
                io::ErrorKind::Other,
                "Serializing failed!",
            ))
        }
    };
    f.write_all(&buf[..])?;
    Ok(())
}
pub fn load_network<P: AsRef<Path>>(path: P) -> Result<NeuralNetwork> {
    let mut file = File::open(path)?;
    let mut buf = vec![];
    if file.read_to_end(&mut buf).is_ok() {
        match deserialize(&buf[..]) {
            Ok(neural_network) => Ok(neural_network),
            Err(_) => Err(std::io::Error::new(
                io::ErrorKind::Other,
                "Deserializing failed!",
            )),
        }
    } else {
        Err(std::io::Error::new(
            io::ErrorKind::Other,
            "Reading to buffer failed",
        ))
    }
}
pub fn load_inputs(filename: &str) -> io::Result<Vec<DVector<f32>>> {
    let mut file = BufReader::new(File::open(filename)?);
    ignore_magic_number(&mut file)?;
    let count = read_num_of_items(&mut file)?;
    let image_size = read_image_size(&mut file)?;
    let images = read_images_and_normalize(&mut file, count, image_size)?;
    Ok(images)
}

/// Read magic number. Ignore it; error handling is for noobs.
fn ignore_magic_number(file: &mut BufReader<File>) -> io::Result<()> {
    let mut byte_buf = [0u8; 4];
    file.read_exact(&mut byte_buf)?;
    Ok(())
}

fn read_num_of_items(file: &mut BufReader<File>) -> io::Result<i32> {
    let mut num_buf = [0u8; std::mem::size_of::<i32>()];
    file.read_exact(&mut num_buf)?;
    Ok(i32::from_be_bytes(num_buf))
}

fn read_image_size(file: &mut BufReader<File>) -> io::Result<i32> {
    let mut dimensions = [0, 0];
    let mut num_buf = [0u8; std::mem::size_of::<i32>()];
    for dim in dimensions.iter_mut() {
        file.read_exact(&mut num_buf)?;
        *dim = i32::from_be_bytes(num_buf);
    }
    Ok(dimensions.iter().product())
}

fn read_images_and_normalize(
    file: &mut BufReader<File>,
    count: i32,
    image_size: i32,
) -> io::Result<Vec<DVector<f32>>> {
    let mut images = Vec::new();
    let mut buffer = vec![0u8; image_size as usize];
    for _ in 0..count {
        file.read_exact(&mut buffer)?;
        let image = DVector::from_vec(
            buffer
                .iter()
                .map(|x| *x as f32 / 255_f32)
                .collect::<Vec<f32>>(),
        );
        images.push(image);
    }
    Ok(images)
}

pub fn load_expected_outputs(filename: &str) -> io::Result<Vec<u8>> {
    let mut file = BufReader::new(File::open(filename)?);
    ignore_magic_number(&mut file)?;
    let count = read_num_of_items(&mut file)?;
    let mut labels = vec![0u8; count as usize];
    file.read_exact(&mut labels)?;
    Ok(labels)
}
