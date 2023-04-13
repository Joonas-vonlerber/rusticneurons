use std::iter;

use crate::files;
use crate::loss_and_activation_functions::activation_function;
use crate::loss_and_activation_functions::loss_function;
use crate::types_and_errors::*;
use crate::ui;
use nalgebra::DMatrix as matrix;
use nalgebra::DVector as vector;
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};
use statrs::distribution::Normal;

impl Layer {
    fn zeros(from_dim: usize, to_dim: usize, activation_function: &ActivationFunction) -> Self {
        Layer {
            outputs: vector::zeros(from_dim),
            values: vector::zeros(from_dim),
            weights: matrix::zeros(to_dim, from_dim),
            biases: vector::zeros(to_dim),
            activation_function: activation_function.clone(),
        }
    }
    fn uniform(from_dim: usize, to_dim: usize, activation_function: &ActivationFunction) -> Self {
        let distribution = Uniform::new(-1.0, 1.0);
        let rng = &mut thread_rng();
        Layer {
            outputs: vector::zeros(from_dim),
            values: vector::zeros(from_dim),
            weights: matrix::from_distribution(to_dim, from_dim, &distribution, rng),
            biases: vector::from_distribution(to_dim, &distribution, rng),
            activation_function: activation_function.clone(),
        }
    }
    fn he_init(from_dim: usize, to_dim: usize, activation_function: &ActivationFunction) -> Self {
        let distribution = Normal::new(0.0f64, 2.0 / from_dim as f64).unwrap();
        let rng = &mut thread_rng();
        Layer {
            values: vector::zeros(from_dim),
            outputs: vector::zeros(from_dim),
            weights: matrix::from_distribution(to_dim, from_dim, &distribution, rng)
                .map(|i| i as f32),
            biases: vector::zeros(to_dim),
            activation_function: activation_function.clone(),
        }
    }
    fn xavier_init<T>(
        from_dim: usize,
        to_dim: usize,
        activation_function: &ActivationFunction,
        previous_dist: &T,
    ) -> Self
    where
        T: Distribution<f64>,
    {
        let rng = &mut thread_rng();
        Layer {
            values: vector::zeros(from_dim),
            outputs: vector::zeros(from_dim),
            weights: matrix::from_distribution(to_dim, from_dim, previous_dist, rng)
                .map(|i| i as f32),
            biases: vector::zeros(to_dim),
            activation_function: activation_function.clone(),
        }
    }
}

impl NeuralNetwork {
    pub fn input(&self, input: &vector<f32>) -> vector<f32> {
        let mut output: vector<f32> = input.clone();
        for layer in self.neural_network.iter() {
            output = activation_function(
                &layer.activation_function,
                &(&layer.weights * &output + &layer.biases),
                false,
            )
            .extract_vector();
        }
        output
    }
    pub fn loss_mnist(&self, test_data: &Vec<(&vector<f32>, &vector<f32>)>) -> (f64, f32) {
        let mut forward: vector<f32>;
        let mut loss: f64 = 0.0;
        let mut got_right: u16 = 0;
        for (input, expected) in test_data.iter() {
            forward = self.input(input);
            loss += loss_function(&self.loss_function, &forward, expected, false).sum() as f64;
            if forward.imax() == expected.imax() {
                got_right += 1;
            }
        }
        (
            loss / test_data.len() as f64,
            f32::from(got_right) / test_data.len() as f32,
        )
    }

    pub fn loss(&self, test_data: &Vec<(&vector<f32>, &vector<f32>)>) -> f32 {
        let mut forward: vector<f32>;
        let mut loss: f32 = 0.0;
        for (input, expected) in test_data.iter() {
            forward = self.input(input);
            loss += loss_function(&self.loss_function, &forward, expected, false).sum();
        }
        loss / test_data.len() as f32
    }

    fn activation_function_iterator<'a>(
        dims_len: usize,
        hidden_activation: &'a ActivationFunction,
        final_activation: &'a ActivationFunction,
    ) -> impl IntoIterator<Item = &'a ActivationFunction> {
        if let ActivationFunction::LayerByLayer(layerbylayer) = hidden_activation {
            layerbylayer
                .iter()
                .collect::<Vec<_>>()
                .into_iter()
                .chain(iter::once(final_activation))
        } else {
            iter::repeat(hidden_activation)
                .take(dims_len - 2usize)
                .collect::<Vec<_>>()
                .into_iter()
                .chain(iter::once(final_activation))
        }
    }
    pub fn zeros(
        dims: &[usize],
        loss_function: LossFunction,
        final_activation: ActivationFunction,
        hidden_activation: ActivationFunction,
        gradient_decent: GradientDecentType,
        optimizer: Optimizer,
        // dropout: Dropout,
    ) -> Self {
        let activation_iterator = NeuralNetwork::activation_function_iterator(
            dims.len(),
            &hidden_activation,
            &final_activation,
        );
        NeuralNetwork {
            neural_network: dims
                .windows(2)
                .zip(activation_iterator)
                .map(|(dims, activation_function)| {
                    Layer::zeros(dims[0], dims[1], activation_function)
                })
                .collect(),
            loss_function,
            gradient_decent,
            optimizer,
            // dropout,
        }
    }

    pub fn uniform(
        dims: &[usize],
        loss_function: LossFunction,
        final_activation: ActivationFunction,
        hidden_activation: ActivationFunction,
        gradient_decent: GradientDecentType,
        optimizer: Optimizer,
        // dropout: Dropout,
    ) -> Self {
        let activation_iterator = NeuralNetwork::activation_function_iterator(
            dims.len(),
            &hidden_activation,
            &final_activation,
        );
        NeuralNetwork {
            neural_network: dims
                .windows(2)
                .zip(activation_iterator)
                .map(|(dims, activation_function)| {
                    Layer::uniform(dims[0], dims[1], activation_function)
                })
                .collect(),
            loss_function,
            gradient_decent,
            optimizer,
            // dropout,
        }
    }

    pub fn he_init(
        dims: &[usize],
        loss_function: LossFunction,
        final_activation: ActivationFunction,
        hidden_activation: ActivationFunction,
        gradient_decent: GradientDecentType,
        optimizer: Optimizer,
        // dropout: Dropout,
    ) -> Self {
        let activation_iterator = NeuralNetwork::activation_function_iterator(
            dims.len(),
            &hidden_activation,
            &final_activation,
        );
        NeuralNetwork {
            neural_network: dims
                .windows(2)
                .zip(activation_iterator)
                .map(|(dims, activation_function)| {
                    Layer::he_init(dims[0], dims[1], activation_function)
                })
                .collect(),
            loss_function,
            gradient_decent,
            optimizer,
            // dropout,
        }
    }
    pub fn xavier_init(
        dims: &[usize],
        loss_function: LossFunction,
        final_activation: ActivationFunction,
        hidden_activation: ActivationFunction,
        gradient_decent: GradientDecentType,
        optimizer: Optimizer,
        // dropout: Dropout,
    ) -> Self {
        let activation_iterator = NeuralNetwork::activation_function_iterator(
            dims.len(),
            &hidden_activation,
            &final_activation,
        );
        let mut temp_neural_network = NeuralNetwork {
            neural_network: vec![],
            loss_function,
            gradient_decent,
            optimizer,
            // dropout,
        };
        let mut distribution = Normal::new(0.0, (1.0 / dims[0] as f64).sqrt()).unwrap();
        for (dim, activation_function) in dims.windows(2).zip(activation_iterator) {
            temp_neural_network.neural_network.push(Layer::xavier_init(
                dim[0],
                dim[1],
                activation_function,
                &distribution,
            ));
            distribution = Normal::new(0.0, (1.0 / dim[0] as f64).sqrt()).unwrap()
        }
        temp_neural_network
    }
}
pub fn initialize_expected_outputs_mnist(labels: &[u8]) -> Vec<vector<f32>> {
    let expected: Vec<vector<f32>> = labels
        .iter()
        .map(|f| {
            let mut expect = vector::zeros(10);
            expect[*f as usize] = 1.0;
            expect
        })
        .collect();
    expected
}

pub fn print_number_and_test(neural_network: &mut NeuralNetwork) {
    println!("Print nth image");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let data = files::load_inputs(r"resources/train-images-idx3-ubyte").unwrap();
    let n = line.trim().parse::<usize>().unwrap();
    let image = ui::NumberImage {
        rows: 28,
        columns: 28,
        data: data[n].clone(),
    };
    println!("{}", image);
    println!("{}", neural_network.forward_phase(&data[n].clone()).outputs);
    println!(
        "{}",
        neural_network
            .forward_phase(&data[n].clone())
            .outputs
            .imax()
    );
}
