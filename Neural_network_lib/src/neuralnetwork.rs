use crate::files::load_inputs;
use crate::loss_and_activation_functions::{
    activation_function, loss_function, ActivationFunction, LossFunction,
};
use crate::ui::*;
use nalgebra::{DMatrix as matrix, DVector as vector};
use rand::thread_rng;
use statrs::distribution::Normal;
use std::iter;
use std::{
    fmt::Display,
    ops::{Add, Mul},
};
#[derive(Debug, Clone, PartialEq)]
/// One **layer** in the neural network, with the needed informaiton to calculate the next layer
pub struct Layer {
    pub values: vector<f32>,
    pub outputs: vector<f32>,
    pub weights: matrix<f32>,
    pub biases: vector<f32>,
    pub activation_function: ActivationFunction,
}
#[derive(Debug, Clone, PartialEq)]
/// **Feedforward Neural Newtork** with user defined Lossfunction and Activationfunctions.
pub struct NeuralNetwork {
    pub neural_network: Vec<Layer>,
    pub loss_function: LossFunction,
    // pub dropout: Dropout,
}

impl<'a, 'b> Add<&'b Layer> for &'a Layer {
    type Output = Layer;
    fn add(self, rhs: &'b Layer) -> Layer {
        Layer {
            values: &self.values + &rhs.values,
            outputs: &self.outputs + &rhs.outputs,
            weights: &self.weights + &rhs.weights,
            biases: &self.biases + &rhs.biases,
            activation_function: self.activation_function.clone(),
        }
    }
}

impl<'a, 'b> Mul<&'b f32> for &'a Layer {
    type Output = Layer;
    fn mul(self, rhs: &'b f32) -> Self::Output {
        Layer {
            values: &self.values * *rhs,
            outputs: &self.outputs * *rhs,
            weights: &self.weights * *rhs,
            biases: &self.biases * *rhs,
            activation_function: self.activation_function.clone(),
        }
    }
}

impl Mul<&Layer> for &Layer {
    type Output = Layer;
    fn mul(self, rhs: &Layer) -> Self::Output {
        Layer {
            values: self.values.component_mul(&rhs.values),
            outputs: self.outputs.component_mul(&rhs.outputs),
            weights: self.weights.component_mul(&rhs.weights),
            biases: self.biases.component_mul(&rhs.biases),
            activation_function: self.activation_function.clone(),
        }
    }
}

impl Layer {
    pub fn sum(&self) -> f32 {
        self.weights.sum() + self.biases.sum()
    }
}

impl NeuralNetwork {
    pub fn clone_clear(&self) -> Self {
        self.map(&|value| (value * 0.0).abs())
    }
    /// Creates a new Neural network by taking a mutable reference and multiplying every element by 0.0.
    pub fn clear(&mut self) -> Self {
        self.map(&|elem| (0.0 * elem).abs())
    }

    /// Creates a new Neural network by applying ```f``` to all of the elements.
    pub fn map<F>(&self, f: &F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        NeuralNetwork {
            neural_network: self
                .neural_network
                .iter()
                .map(|layer| Layer {
                    values: layer.values.map(f),
                    outputs: layer.outputs.map(f),
                    weights: layer.weights.map(f),
                    biases: layer.biases.map(f),
                    activation_function: layer.activation_function.clone(),
                })
                .collect(),
            loss_function: self.loss_function.clone(),
            // dropout: self.dropout.clone(),
        }
    }
    /// Gives the shape of the network.
    pub fn shape(&self) -> Vec<usize> {
        let mut shapes: Vec<usize> = self
            .neural_network
            .iter()
            .map(|layer| layer.weights.shape().1)
            .collect();
        shapes.push(self.neural_network.last().unwrap().weights.shape().0);
        shapes
    }
    pub fn sum(&self) -> f32 {
        self.neural_network.iter().map(|layer| layer.sum()).sum()
    }
    pub fn normalize(&self) -> f32 {
        1.0 / self.map(&|elem| elem * elem).sum().sqrt()
    }
}

impl<'a, 'b> Add<&'b NeuralNetwork> for &'a NeuralNetwork {
    type Output = NeuralNetwork;
    fn add(self, rhs: &'b NeuralNetwork) -> NeuralNetwork {
        NeuralNetwork {
            neural_network: self
                .neural_network
                .iter()
                .zip(rhs.neural_network.iter())
                .map(|(i, j)| i + j)
                .collect(),
            loss_function: self.loss_function.clone(),
            // dropout: self.dropout.clone(),
        }
    }
}

impl Mul<&NeuralNetwork> for &NeuralNetwork {
    type Output = NeuralNetwork;
    fn mul(self, rhs: &NeuralNetwork) -> Self::Output {
        NeuralNetwork {
            neural_network: self
                .neural_network
                .iter()
                .zip(rhs.neural_network.iter())
                .map(|(layer1, layer2)| layer1 * layer2)
                .collect(),
            loss_function: self.loss_function.clone(),
            // dropout: self.dropout.clone(),
        }
    }
}

impl<'a, 'b> Mul<&'b f32> for &'a NeuralNetwork {
    type Output = NeuralNetwork;
    fn mul(self, rhs: &'b f32) -> Self::Output {
        NeuralNetwork {
            neural_network: self.neural_network.iter().map(|i| i * rhs).collect(),
            loss_function: self.loss_function.clone(),
            // dropout: self.dropout.clone(),
        }
    }
}

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
    /**
    you can create Neural network by calling the method **new**, which takes in the structure of the neural network as **&[[usize]]**, **initialization type**,
    **loss function** and **activation functions**.<br>.

    You can make a Neural network for classifying the MNIST dataset with the parametres
    ```
    extern crate neural_network_lib as nnl;
    use nnl::types_and_errors::{
        InitializationType as Init,
        LossFunction as Loss,
        ActivationFunction as Actf,
    };

    let mut neural_network: NeuralNetwork = NeuralNetwork::new(
        &[784, 32, 32, 10],
        Init::HeKalming,
        Loss::CrossEntropy,
        Actf::SoftMax,
        Actf::ReLU,
    );
    ```
    or you may create a neural network with layer by layer defined activation functions
    ```
    let mut neural_network: NeuralNetwork = NeuralNetwork::new(
        &[784, 32, 32, 10],
        Init::HeKalming,
        Loss::CrossEntropy,
        Actf::SoftMax,
        Actf::LayerByLayer(vec![Actf::Tanh, Actf::LeReLU]))
    ```
    <br>
        */
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
    /// Input to a Neural network
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
    /// Calculate the loss and the accuracy of the Neural Network with the MNIST-dataset
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
    /// Calculate the loss of a Neural Network with a given dataset
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

/// Print a number from the MNIST dataset and print the output and the class of the neural network
pub fn mnist_print_number_and_test(neural_network: &mut NeuralNetwork) {
    println!("Print nth image");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let data = load_inputs(r"resources/train-images-idx3-ubyte").unwrap();
    let n = line.trim().parse::<usize>().unwrap();
    let image = NumberImage {
        rows: 28,
        columns: 28,
        data: data[n].clone(),
    };
    println!("{}", image);
    println!("{}", neural_network.input(&data[n].clone()));
    println!("{}", neural_network.input(&data[n].clone()).imax());
}

#[derive(Debug, Clone, PartialEq)]
/// Defines the way to run trough the data
pub enum GradientDecentType {
    /// [Stochastic gradient decent](https://www.ruder.io/optimizing-gradient-descent/#stochasticgradientdescent) runs trough the one data point, calculates the gradient and uses that gradient to update the weights.<br>
    Stochastic,
    /// [Mini-Batch gradient decent](https://www.ruder.io/optimizing-gradient-descent/#minibatchgradientdescent) is the beautiful lovechild of
    /// Stochastic- and Batch gradient decent. It divides the data into batches of specified size and
    /// calculates the gradient for each of them. It adds the gradients and uses the sum to update the weights.
    MiniBatch(usize),
    /// [Batch gradient decent](https://www.ruder.io/optimizing-gradient-descent/#batchgradientdescent) runs trough all the data points,
    /// calculating the gradient for each of them. It adds all the gradients together, after which it uses the sum to update the weights.
    Batch,
}
#[derive(Debug, Clone, PartialEq)]
pub enum Optimizer {
    /// Vanilla gradinet decent, where we use just the gradient of the loss function to update the weights.
    Vanilla,
    /// [Momentum](https://www.ruder.io/optimizing-gradient-descent/#momentum) is a gradient decent optimization algorithm
    /// which can reduce the time of convergence by adding a "velocity" term to it. It remembers the exponentially decaying sum of the gradients for the
    /// weight updates.
    Momentum(f32),
    /// [Nestrov Momentum](https://www.ruder.io/optimizing-gradient-descent/#nesterovacceleratedgradient) is an sometimes an improvement to the Momentum
    /// algorithm by adding a kind of prediction to the algorithm. Momentum takes the decay parameter of how long does it remember the gradients. In most cases set 0.9.
    NstMomentum(f32),
    /// [AdaGrad](https://www.ruder.io/optimizing-gradient-descent/#adagrad) is a adaptive optimization algorithm, which adapts the learning rates for each parameter.
    /// therefore the user defined learning rate doesn't really matter and should be set at 0.01 or if you're going by the paper that proposed this, it
    /// should be set at 1.0. Nestrov momentum has the same decay parameter, which is in most cases set 0.9.
    AdaGrad,
    /// [AdaDelta](https://www.ruder.io/optimizing-gradient-descent/#adadelta) is an adaptive optimization algorithm , which purpose is to fix the agressive
    /// reduction of the learning rate. It does this by keeping a exponentally decaying sum of the squared gradients. Very close to RMSProp and but
    /// with a slight difference in the formula. Input is a **Adadelta doesn't have a learning rate** so changing it won't affect anything. Adagrad takes a
    /// decay parameter which is in most cases set to 0.9.
    AdaDelta(f32),
    /// [RMSProp](https://www.ruder.io/optimizing-gradient-descent/#rmsprop) seeks to solve the same problems as AdaDelta but with a slighty different way.
    /// Both were invented around the same time but RMSProp wasn't ever published. RMSProp takes a decay parameter, which is in most cases set to 0.9.
    RMSprop(f32),
    /// [Adam](https://www.ruder.io/optimizing-gradient-descent/#adam) or Adaptive Moment Estimation is a adaptive gradient decent optimization algorithm,
    /// which is a sweet lovechild of the Momentum and RMSProp by keeping track of the sum of past squared gradients and gradients.
    /// It might be the most used algorithm for optimization due to its preformance and effectiveness. It takes in two parameters named beta1 and beta2.
    /// They can be thought as the decay parameters for Momentum and RMSProp. Beta1 is usually set at 0.9 and Beta2 is set at 0.99 or 0.999.
    Adam(f32, f32),
}
// #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
// pub enum Dropout {
//     NoDropout,
//     Dropout(f32),
// }
// impl Dropout {
//     pub fn get_probability(&self) -> f32 {
//         match self {
//             Self::NoDropout => 0.0,
//             Dropout::Dropout(probability) => *probability,
//         }
//     }
// }
#[derive(Debug)]
/// Errors for the Neural networks
pub enum NeuralNetworkError {
    CostError,
}

impl Display for NeuralNetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Reached a stable minima or saddle point")
    }
}
