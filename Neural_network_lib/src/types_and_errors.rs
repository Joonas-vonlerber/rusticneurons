use nalgebra::{DMatrix as matrix, DVector as vector};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    ops::{Add, Mul},
};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// One **layer** in the neural network, with the needed informaiton to calculate the next layer
pub struct Layer {
    pub values: vector<f32>,
    pub outputs: vector<f32>,
    pub weights: matrix<f32>,
    pub biases: vector<f32>,
    pub activation_function: ActivationFunction,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Enum to define all of the loss functions
pub enum LossFunction {
    /// MSE or Mean Squared Error is defined to be [this](https://en.wikipedia.org/wiki/Mean_squared_error)
    MSE,
    /// MAE or Mean Absolute Error is defined to be [this](https://en.wikipedia.org/wiki/Mean_absolute_error)
    MAE,
    /// Cross Entropy is defined to be [this](https://en.wikipedia.org/wiki/Cross_entropy)
    CrossEntropy,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
/// Enum to define all available activation functions
pub enum ActivationFunction {
    /// [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) or Rectified Linear Unit is a commonly used activation function for its benefits like faster learning.
    ReLU,
    /// [Parametric ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Parametric_ReLU) which is a generalized version of ReLU
    /// by adding a parameter of alpha which scales the input of the negative values. DOES NOT CHANGE WITH BACKPROP!! Maybe in the future ;) <br>
    PReLU(f32),
    /// [Leaky ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLU), which solves the dying ReLU problem, but
    /// **can** come with some learning slowdowns.
    LeReLU,
    /// [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) or Logistic regression was one of the most used activation functions
    /// but has fallen from grace due to ReLU and other better activation function. <br>
    /// Even though it has many disadvantages compared to ReLU, it still has its uses.
    Sigmoid,
    /// [Softmax](https://en.wikipedia.org/wiki/Softmax_function) takes in a distribution of numbers and outputs a propability distribution.
    /// Softmax is a function usually used in the output-layer and most commonly used in classification type problems.
    SoftMax,
    /// Identity function outputs the input.
    Identity,
    /// [Linear function](https://en.wikipedia.org/wiki/Linear_function) operates on the input by first scaling it by a scalar, after which it adds some constant term to it.
    /// The first value, depicts the scalar and the second value defines a constant.
    Linear(f32, f32),
    /// [The hyperbolic tangent](https://en.wikipedia.org/wiki/Hyperbolic_functions) activation function.
    ///  It is similar to the sigmoid activation function but ranging from -1 to 1 instead of 0 to 1.
    Tanh,
    /// [Gaussian Error Linear Unit](https://paperswithcode.com/method/gelu) is a new type of activation function, which has been noticed to improve learning time compared to ReLU.
    /// It is defined using the Normal cumulative distribution function and in my code I use an approximation of the function using tanh.
    GELU,
    /// [Exponential Linear Unit](https://paperswithcode.com/method/elu) is a another type of Linear unit,
    /// which allows for negative values thus pushing mean activations closer to zero. Input is the value of alpha, which is usually 1.0.
    ELU(f32),
    /// [SoftPlus](https://paperswithcode.com/method/softplus) is a smooth approximation of ReLU<br>
    SoftPlus,
    /// A way of defining the activation functions for hidden layers layer by layer. The lenght of the vec should be ```neural_network.shape().len()-2```.
    LayerByLayer(Vec<ActivationFunction>),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
