use super::types_and_errors::*;
use bincode::{self, deserialize, serialize};
use nalgebra::{DMatrix, DVector};
use std::{
    fs::File,
    io::{self, BufReader, Read, Result, Write},
    path::Path,
};

pub fn save_network_bincode<P: AsRef<Path>>(path: P, network: &NeuralNetwork) -> Result<()> {
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
pub fn load_network_bincode<P: AsRef<Path>>(path: P) -> Result<NeuralNetwork> {
    let mut file = File::open(path)?;
    let mut buf = vec![];
    if file.read_to_end(&mut buf).is_ok() {
        match deserialize(&buf[..]) {
            Ok(neural_network) => return Ok(neural_network),
            Err(_) => {
                return Err(std::io::Error::new(
                    io::ErrorKind::Other,
                    "Deserializing failed!",
                ))
            }
        }
    } else {
        return Err(std::io::Error::new(
            io::ErrorKind::Other,
            "Reading to buffer failed",
        ));
    }
}
pub fn load_inputs(filename: &str) -> io::Result<Vec<DVector<f32>>> {
    let mut file = BufReader::new(File::open(filename)?);
    ignore_magic_number(&mut file)?;
    let count = read_num_of_items(&mut file)?;
    let image_size = read_image_size(&mut file)?;
    let images = read_images(&mut file, count, image_size)?;
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

fn read_images(
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
                .map(|x| *x as f32 / 255 as f32)
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

pub fn load_network(filename: &str) -> io::Result<Vec<Layer>> {
    let mut file = BufReader::new(File::open(filename)?);
    let count = read_num_of_items(&mut file)?;
    let layers = read_layers(file, count)?;
    Ok(layers)
}

fn read_layers(mut file: BufReader<File>, count: i32) -> io::Result<Vec<Layer>> {
    let mut layers = Vec::new();
    for _ in 0..count {
        let layer = read_layer(&mut file)?;
        layers.push(layer);
    }
    Ok(layers)
}

fn read_layer(file: &mut BufReader<File>) -> io::Result<Layer> {
    let mut dimensions = [0, 0];
    let mut num_buf = [0u8; std::mem::size_of::<i32>()];
    for dim in dimensions.iter_mut() {
        file.read_exact(&mut num_buf)?;
        *dim = i32::from_be_bytes(num_buf);
    }
    let a_vec = read_vec_f64(file, dimensions[0] as usize)?;
    let o_vec = read_vec_f64(file, dimensions[0] as usize)?;
    let w_vec = read_vec_f64(file, dimensions.iter().product::<i32>() as usize)?;
    let b_vec = read_vec_f64(file, dimensions[1] as usize)?;
    let layer = Layer {
        values: DVector::from_vec(a_vec),
        outputs: DVector::from_vec(o_vec),
        weights: DMatrix::from_vec(dimensions[1] as usize, dimensions[0] as usize, w_vec),
        biases: DVector::from_vec(b_vec),
    };
    Ok(layer)
}

fn read_vec_f64(file: &mut BufReader<File>, length: usize) -> io::Result<Vec<f32>> {
    let mut vector = Vec::new();
    let mut num_buf = [0u8; std::mem::size_of::<f32>()];
    for _ in 0..length {
        file.read_exact(&mut num_buf)?;
        vector.push(f32::from_be_bytes(num_buf));
    }
    Ok(vector)
}

pub fn save_network(filename: &str, data: Vec<Layer>) -> io::Result<()> {
    let mut file = File::create(filename)?;
    file.write_all(&(data.len() as i32).to_be_bytes())?;
    for layer in data {
        let dimensions = [layer.values.len(), layer.biases.len()];
        file.write_all(&(dimensions[0] as i32).to_be_bytes())?;
        file.write_all(&(dimensions[1] as i32).to_be_bytes())?;
        write_vec(&mut file, layer.values.data.as_vec())?;
        write_vec(&mut file, layer.outputs.data.as_vec())?;
        write_vec(&mut file, layer.weights.data.as_vec())?;
        write_vec(&mut file, layer.biases.data.as_vec())?;
    }
    Ok(())
}

fn write_vec(file: &mut File, vec: &Vec<f32>) -> io::Result<()> {
    for num in vec {
        file.write_all(&num.to_be_bytes())?;
    }
    Ok(())
}
