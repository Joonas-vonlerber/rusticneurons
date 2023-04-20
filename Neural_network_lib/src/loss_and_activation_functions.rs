use crate::types_and_errors::{ActivationFunction, LossFunction};
use nalgebra::{DMatrix as matrix, DVector as vector};
use std::collections::VecDeque;

pub fn standard_deviation(data: &VecDeque<f32>) -> f32 {
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: Vec<f32> = data.iter().map(|elem| (elem - mean).powi(2)).collect();
    let standard_deviation = variance.iter().sum::<f32>() / variance.len() as f32;
    f32::sqrt(standard_deviation)
}

#[derive(Debug, Clone)]
pub enum ActivationOutput {
    Vector(vector<f32>),
    Matrix(matrix<f32>),
}

impl ActivationOutput {
    pub fn extract_matrix(self) -> matrix<f32> {
        match self {
            ActivationOutput::Vector(vector) => matrix::from_diagonal(&vector),
            ActivationOutput::Matrix(matrix) => matrix,
        }
    }
    pub fn extract_vector(self) -> vector<f32> {
        match self {
            ActivationOutput::Vector(vector) => vector,
            ActivationOutput::Matrix(matrix) => matrix.diagonal(),
        }
    }
    pub fn calculate_errorterm(&self, rhs: &vector<f32>) -> vector<f32> {
        match self {
            ActivationOutput::Vector(value) => value.component_mul(rhs),
            ActivationOutput::Matrix(value) => value * rhs,
        }
    }
}

//ACTIVATION FUNCTIONS

fn linear(input: &vector<f32>, derivative: bool) -> ActivationOutput {
    ActivationOutput::Vector(match derivative {
        true => vector::repeat(input.len(), 1.0),
        false => input.clone(),
    })
}

fn sigmoid(input: &vector<f32>, derivative: bool) -> ActivationOutput {
    ActivationOutput::Vector(match derivative {
        false => input.map(|value| 1.0 / (1.0 + (-value).exp())),
        true => sigmoid(input, false)
            .extract_vector()
            .component_mul(&sigmoid(&-input, false).extract_vector()),
    })
}
fn prelu(input: &vector<f32>, alpha: &f32, derivative: bool) -> ActivationOutput {
    ActivationOutput::Vector(match derivative {
        false => input.map(|element: f32| {
            if element > 0.0 {
                element
            } else {
                element * alpha
            }
        }),
        true => input.map(|element: f32| if element > 0.0 { 1.0 } else { *alpha }),
    })
}

fn tanh(input: &vector<f32>, derivative: bool) -> ActivationOutput {
    ActivationOutput::Vector(match derivative {
        false => input.map(|element| element.tanh()),
        true => input.map(|element| 1.0 - element.tanh().powi(2)),
    })
}

fn soft_max(input: &vector<f32>, derivative: bool) -> ActivationOutput {
    match derivative {
        false => {
            let exponential_vector: vector<f32> =
                input.map(|elements: f32| (elements - input.max()).exp());
            let exp_vec_sum: f32 = exponential_vector.iter().sum();
            ActivationOutput::Vector(exponential_vector / exp_vec_sum)
        }
        true => {
            let softmax = soft_max(input, false).extract_vector();
            let diagonal: vector<f32> = softmax.map(|f| f * (1.0 - f));
            let mut jacobian: matrix<f32> = -1.0 * (&softmax * &softmax.transpose());
            jacobian.set_diagonal(&diagonal);
            ActivationOutput::Matrix(jacobian)
        }
    }
}

fn softplus(input: &vector<f32>, derivative: bool) -> ActivationOutput {
    match derivative {
        false => ActivationOutput::Vector(input.map(|value| (value.exp() + 1.0).ln())),
        true => sigmoid(input, false),
    }
}

fn gelu(input: &vector<f32>, derivative: bool) -> ActivationOutput {
    ActivationOutput::Vector(match derivative {
        false => input.map(|value| {
            0.5 * value * (1.0 + (0.7978846 * (value + 0.044715 * value.powi(3))).tanh())
        }),
        true => input.map(|value| {
            (0.5 + 0.5 * (0.7978846 * (value + 0.044715 * value.powi(3))).tanh())
                + 0.3989422 * value * (-value * value / 2.0).exp()
        }),
    })
}

fn elu(input: &vector<f32>, alpha: &f32, derivative: bool) -> ActivationOutput {
    ActivationOutput::Vector(match derivative {
        false => input.map(|value| match value > 0.0 {
            true => value,
            false => *alpha * (value.exp() - 1.0),
        }),
        true => input.map(|value| match value > 0.0 {
            true => 1.0,
            false => *alpha * value.exp(),
        }),
    })
}

//LOSS FUNCTONS

fn mse(input: &vector<f32>, expected: &vector<f32>, derivative: bool) -> vector<f32> {
    let diff: vector<f32> = input - expected;
    match derivative {
        false => diff.map(|x| 0.5 * x * x),
        true => diff,
    }
}

fn mae(input: &vector<f32>, expected: &vector<f32>, derivative: bool) -> vector<f32> {
    let diff: vector<f32> = expected - input;
    match derivative {
        false => diff.abs(),
        true => diff.map(|value| {
            if value > 0.0 {
                1.0
            } else if value == 0.0 {
                0.0
            } else {
                -1.0
            }
        }),
    }
}

fn cross_entropy(input: &vector<f32>, expected: &vector<f32>, derivative: bool) -> vector<f32> {
    match derivative {
        false => {
            -1.0 * input
                .map(|element: f32| (element + f32::EPSILON).ln())
                .component_mul(expected)
        }
        true => {
            -1.0 * input
                .map(|element: f32| 1.0 / (element + f32::EPSILON))
                .component_mul(expected)
        }
    }
}

pub fn activation_function(
    activation_type: &ActivationFunction,
    input: &vector<f32>,
    derivative: bool,
) -> ActivationOutput {
    match activation_type {
        ActivationFunction::ReLU => prelu(input, &0.0, derivative),
        ActivationFunction::PReLU(alpha) => prelu(input, alpha, derivative),
        ActivationFunction::LeReLU => prelu(input, &0.01, derivative),
        ActivationFunction::GELU => gelu(input, derivative),
        ActivationFunction::Sigmoid => sigmoid(input, derivative),
        ActivationFunction::Linear => linear(input, derivative),
        ActivationFunction::SoftMax => soft_max(input, derivative),
        ActivationFunction::Tanh => tanh(input, derivative),
        ActivationFunction::ELU(alpha) => elu(input, alpha, derivative),
        ActivationFunction::SoftPlus => softplus(input, derivative),
        ActivationFunction::LayerByLayer(_) => panic!(
            "Layer by layer should not be inputted to activation function. Launch a git issue :DD"
        ),
    }
}

pub fn loss_function(
    loss_type: &LossFunction,
    input: &vector<f32>,
    expected: &vector<f32>,
    derivative: bool,
) -> vector<f32> {
    match loss_type {
        LossFunction::MSE => mse(input, expected, derivative),
        LossFunction::MAE => mae(input, expected, derivative),
        LossFunction::CrossEntropy => cross_entropy(input, expected, derivative),
    }
}
