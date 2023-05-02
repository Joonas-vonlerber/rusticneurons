use std::iter;

use crate::files;
use crate::loss_and_activation_functions::{activation_function, loss_function};
use crate::types_and_errors::*;
use crate::ui;
use nalgebra::DMatrix as matrix;
use nalgebra::DVector as vector;
use rand::thread_rng;
use statrs::distribution::Normal;

#[derive(Debug, Clone, Copy, PartialEq)]
/// Types of Normal distributions you can use when initializing the network
pub enum InitializaitonType {
    /// Choose your own Normal distribution, where you give the standard deviation. It will stay constant for the whole initialization
    Normal(f32),
    /// LeCun initialization is derived from keeping the same standard distribition to all of the weights in each layer.
    /// It results that the variance should be 1/n, where n in the number of input neurons. This initialization technique is useful for many types of neural networks.
    LeCun,
    /// Xavier Glorot initialization technique is mainly for tanh- or sigmoid-function. It is derived almost the same way LeCun is, but the assumption that
    /// a good approximation for tanh is just the identity function. It shows that the variance should be 1/(n_in + n_out), where n_in is the number of input
    /// neurons and n_out is the number of output neurons.
    XavierGlorot,
    /// He Kaiming initialization technique is usually used with ReLU or other types of linear units like Leaky ReLU, Parametric ReLU etc. In the research paper
    /// it is showed that the initialization variance 2/n, where n is the amount of input neurons, is better with funcitons like ReLU than 1/n.
    HeKaiming,
}

impl Layer {
    fn new(
        init_type: InitializaitonType,
        structure: &[usize],
        activation_function: &ActivationFunction,
    ) -> Self {
        let distribution: Normal = Layer::init_distribution(init_type, structure);
        let rng = &mut thread_rng();
        Layer {
            outputs: vector::zeros(structure[0]),
            values: vector::zeros(structure[0]),
            weights: matrix::from_distribution(structure[1], structure[0], &distribution, rng)
                .map(|i| i as f32),
            biases: vector::from_distribution(structure[1], &distribution, rng).map(|i| i as f32),
            activation_function: activation_function.clone(),
        }
    }
    fn init_distribution(init_type: InitializaitonType, dims: &[usize]) -> Normal {
        match init_type {
            InitializaitonType::Normal(std_dev) => Normal::new(0.0, std_dev as f64).unwrap(),
            InitializaitonType::LeCun => Normal::new(0.0, 1.0 / (dims[0] as f64).sqrt()).unwrap(),
            InitializaitonType::XavierGlorot => {
                Normal::new(0.0, 1.0 / ((dims[0] + dims[1]) as f64).sqrt()).unwrap()
            }
            InitializaitonType::HeKaiming => {
                Normal::new(0.0, (2.0 / (dims[0] as f64)).sqrt()).unwrap()
            }
        }
    }
}

impl NeuralNetwork {
    fn activation_function_iterator<'a>(
        structure_len: usize,
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
                .take(structure_len - 2usize)
                .collect::<Vec<_>>()
                .into_iter()
                .chain(iter::once(final_activation))
        }
    }
    pub fn new(
        structure: &[usize],
        initialization_type: InitializaitonType,
        loss_function: LossFunction,
        final_activation: ActivationFunction,
        hidden_activation: ActivationFunction,
        //dropout: Dropout,
    ) -> Self {
        let activation_iterator = NeuralNetwork::activation_function_iterator(
            structure.len(),
            &hidden_activation,
            &final_activation,
        );
        NeuralNetwork {
            neural_network: structure
                .windows(2)
                .zip(activation_iterator)
                .map(|(dims, activation_function)| {
                    Layer::new(initialization_type, dims, activation_function)
                })
                .collect(),
            loss_function,
            // dropout,
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
    println!("{}", neural_network.input(&data[n].clone()));
    println!("{}", neural_network.input(&data[n].clone()).imax());
}
