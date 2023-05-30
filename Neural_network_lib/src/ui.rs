use crate::neuralnetwork::Layer;
use crate::neuralnetwork::NeuralNetwork;
use nalgebra::DVector;
use std::fmt;

pub struct NumberImage {
    pub rows: i32,
    pub columns: i32,
    pub data: DVector<f32>,
}

impl fmt::Display for NumberImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let greyscale: &[u8; 10] = b" .:-=+*#%@";
        let mut row_vec = Vec::new();
        for i in 0..self.rows {
            let start = (self.columns * i) as usize;
            let end = start + self.columns as usize;
            let line = &self.data.data.as_vec()[start..end];
            let line: String = line
                .iter()
                .map(|x| greyscale[f32::round(*x * 9.0) as usize] as char)
                .collect();
            row_vec.push(line);
        }
        write!(f, "{}", row_vec.join("\n"))
    }
}
impl fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "values: {} outputs: {} weights: {} biases: {} activation function: {:#?}",
            self.values, self.outputs, self.weights, self.biases, self.activation_function
        )
    }
}

pub fn print_neural(neural_network: &NeuralNetwork) {
    for i in neural_network.neural_network.iter() {
        println!("{}", i);
    }
}
