use nalgebra::{DMatrix as matrix, DVector as vector};
use rand::{distributions::Uniform, thread_rng};
use serde::{Deserialize, Serialize};
use std::{
    default::Default,
    fmt::Display,
    ops::{Add, Mul},
};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub values: vector<f32>,
    pub outputs: vector<f32>,
    pub weights: matrix<f32>,
    pub biases: vector<f32>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub neural_network: Vec<Layer>,
    pub loss_function: LossFunction,
    pub final_activation: ActivationFunction,
    pub hidden_activation: ActivationFunction,
    pub gradient_decent: GradientDecentType,
    pub optimizer: Optimizer,
}

impl Layer {
    pub fn new(from_dim: usize, to_dim: usize) -> Self {
        let distribution = Uniform::new(-1.0, 1.0);
        let mut rng = thread_rng();
        Layer {
            outputs: vector::zeros(from_dim),
            values: vector::zeros(from_dim),
            weights: matrix::from_distribution(to_dim, from_dim, &distribution, &mut rng),
            biases: vector::from_distribution(to_dim, &distribution, &mut rng),
        }
    }
}

impl Default for Layer {
    fn default() -> Self {
        Layer {
            values: vector::zeros(1),
            outputs: vector::zeros(1),
            weights: matrix::zeros(1, 1),
            biases: vector::zeros(1),
        }
    }
}

impl<'a, 'b> Add<&'b Layer> for &'a Layer {
    type Output = Layer;
    fn add(self, rhs: &'b Layer) -> Layer {
        Layer {
            values: &self.values + &rhs.values,
            outputs: &self.outputs + &rhs.outputs,
            weights: &self.weights + &rhs.weights,
            biases: &self.biases + &rhs.biases,
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
        }
    }
}

impl NeuralNetwork {
    pub fn new(
        dims: &[usize],
        loss_function: LossFunction,
        final_activation: ActivationFunction,
        hidden_activation: ActivationFunction,
        gradient_decent: GradientDecentType,
        optimizer: Optimizer,
    ) -> Self {
        let mut temp_neural_network: NeuralNetwork = NeuralNetwork {
            neural_network: vec![],
            loss_function,
            final_activation,
            hidden_activation,
            gradient_decent,
            optimizer,
        };
        temp_neural_network.neural_network = dims
            .windows(2)
            .map(|dims| Layer::new(dims[0], dims[1]))
            .collect();
        temp_neural_network
    }
    pub fn clear(self) -> Self {
        self.map(&|elem| (0.0*elem).abs())
    }
    pub fn map<F>(&self, f: &F) -> Self 
    where F: Fn(f32) -> f32{
        NeuralNetwork { neural_network: self.neural_network.iter().map(|layer| Layer {
            values: layer.values.map(f),
            outputs: layer.outputs.map(f),
            weights: layer.weights.map(f),
            biases: layer.biases.map(f)}).collect(), 
            loss_function: self.loss_function, 
            final_activation: self.final_activation ,
            hidden_activation: self.hidden_activation ,
            gradient_decent: self.gradient_decent,
            optimizer: self.optimizer}

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
            loss_function: self.loss_function,
            final_activation: self.final_activation,
            hidden_activation: self.hidden_activation,
            gradient_decent: self.gradient_decent,
            optimizer: self.optimizer,
        }
    }
}

impl Mul<&NeuralNetwork> for &NeuralNetwork {
    type Output = NeuralNetwork;
    fn mul(self, rhs: &NeuralNetwork) -> Self::Output {
        NeuralNetwork {
            neural_network: self.neural_network.iter().zip(rhs.neural_network.iter()).map(|(layer1, layer2)| layer1*layer2).collect(),
            loss_function: self.loss_function,
            final_activation: self.final_activation,
            hidden_activation: self.hidden_activation,
            gradient_decent: self.gradient_decent,
            optimizer: self.optimizer,
        }
    }
}

impl<'a, 'b> Mul<&'b f32> for &'a NeuralNetwork {
    type Output = NeuralNetwork;
    fn mul(self, rhs: &'b f32) -> Self::Output {
        NeuralNetwork {
            neural_network: self.neural_network.iter().map(|i| i * rhs).collect(),
            loss_function: self.loss_function,
            final_activation: self.final_activation,
            hidden_activation: self.hidden_activation,
            gradient_decent: self.gradient_decent,
            optimizer: self.optimizer,
        }
    }
}

// impl Div<&NeuralNetwork> for &NeuralNetwork {
//     type Output = NeuralNetwork;
//     fn div(self, rhs: &NeuralNetwork) -> Self::Output {
//         NeuralNetwork {
//             neural_network: self.neural_network.iter().zip(rhs.neural_network.iter()).map(|(layer1, layer2)|
//         Layer {
//             values: layer1.values.component_div(&layer2.values),
//             outputs: layer1.outputs.component_div(&layer2.outputs),
//             weights: layer1.weights.component_div(&layer2.weights),
//             biases: layer1.biases.component_div(&layer2.biases),
//         }).collect(),
//         loss_function: self.loss_function,
//         final_activation: self.final_activation,
//         hidden_activation: self.hidden_activation,
//         gradient_decent: self.gradient_decent,

//         }
//     }
// }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Copy)]
pub enum LossFunction {
    MSE,
    CrossEntropy,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Copy)]
pub enum ActivationFunction {
    ReLU,
    PReLU(f32),
    LeReLU,
    Sigmoid,
    SoftMax,
    Linear,
    Tanh,
    GELU,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Copy)]
pub enum GradientDecentType {
    Stochastic,
    MiniBatch(usize),
    Batch,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Copy)]
pub enum Optimizer {
    SGD,
    Momentum(f32),
    NstMomentum(f32),
    AdaGrad,
    AdaDelta(f32),
    RMSprop(f32),
    Adam(f32,f32),
}

#[derive(Debug)]
pub enum NeuralNetworkError {
    CostError,
}

impl Display for NeuralNetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Reached a stable minima")
    }
}
