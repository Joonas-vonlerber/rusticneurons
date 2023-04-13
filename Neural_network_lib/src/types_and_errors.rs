use nalgebra::{DMatrix as matrix, DVector as vector};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    ops::{Add, Mul},
};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub values: vector<f32>,
    pub outputs: vector<f32>,
    pub weights: matrix<f32>,
    pub biases: vector<f32>,
    pub activation_function: ActivationFunction,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub neural_network: Vec<Layer>,
    pub loss_function: LossFunction,
    pub gradient_decent: GradientDecentType,
    pub optimizer: Optimizer,
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

impl NeuralNetwork {
    pub fn clear(self) -> Self {
        self.map(&|elem| (0.0 * elem).abs())
    }
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
            gradient_decent: self.gradient_decent.clone(),
            optimizer: self.optimizer.clone(),
            // dropout: self.dropout.clone(),
        }
    }
    pub fn shape(&self) -> Vec<usize> {
        let mut shapes: Vec<usize> = self
            .neural_network
            .iter()
            .map(|layer| layer.weights.shape().1)
            .collect();
        shapes.push(self.neural_network.last().unwrap().weights.shape().0);
        shapes
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
            gradient_decent: self.gradient_decent.clone(),
            optimizer: self.optimizer.clone(),
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
            gradient_decent: self.gradient_decent.clone(),
            optimizer: self.optimizer.clone(),
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
            gradient_decent: self.gradient_decent.clone(),
            optimizer: self.optimizer.clone(),
            // dropout: self.dropout.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LossFunction {
    MSE,
    MAE,
    CrossEntropy,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    PReLU(f32),
    LeReLU,
    Sigmoid,
    SoftMax,
    Linear,
    Tanh,
    GELU,
    ELU(f32),
    SoftPlus,
    LayerByLayer(Vec<ActivationFunction>),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GradientDecentType {
    Stochastic,
    MiniBatch(usize),
    Batch,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Optimizer {
    SGD,
    Momentum(f32),
    NstMomentum(f32),
    AdaGrad,
    AdaDelta(f32),
    RMSprop(f32),
    Adam(f32, f32),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Dropout {
    NoDropout,
    Dropout(f32),
}
impl Dropout {
    pub fn get_probability(&self) -> f32 {
        match self {
            Self::NoDropout => 0.0,
            Dropout::Dropout(probability) => *probability,
        }
    }
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
