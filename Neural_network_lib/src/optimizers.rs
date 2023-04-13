use crate::loss_and_activation_functions::*;
use crate::types_and_errors::*;
use nalgebra::DMatrix;
use nalgebra::DVector as vector;
use rand::{seq::SliceRandom, thread_rng};
// use statrs::distribution::Bernoulli;
use std::{collections::VecDeque, f32};

impl NeuralNetwork {
    pub fn forward_phase(&mut self, inp: &vector<f32>) -> Layer {
        self.neural_network[0].outputs = inp.clone();
        let mut iterator = self.neural_network.iter_mut();
        let mut before = iterator.next().unwrap();
        // let dropout_distirbution =
        // Bernoulli::new(1.0f64 - self.dropout.get_probability() as f64).unwrap();
        for current in iterator {
            current.values = &before.weights * &before.outputs + &before.biases;
            current.outputs =
                activation_function(&current.activation_function, &current.values, false)
                    .extract_vector();
            // if let Dropout::Dropout(_) = self.dropout {
            //     current.outputs.component_mul_assign(
            //         &vector::from_distribution(
            //             current.outputs.len(),
            //             &dropout_distirbution,
            //             &mut thread_rng(),
            //         )
            //         .map(|elem| elem as f32),
            //     );
            // }
            before = current;
        }
        let last_layer = self.neural_network.last().unwrap();
        let output = &last_layer.weights * &last_layer.outputs + &last_layer.biases;
        Layer {
            outputs: activation_function(&last_layer.activation_function, &output, false)
                .extract_vector(),
            values: output,
            weights: DMatrix::zeros(1, 1),
            biases: vector::zeros(1),
            activation_function: ActivationFunction::Sigmoid,
        }
    }

    fn backward_phase(&self, forward: &Layer, expected: &vector<f32>) -> NeuralNetwork {
        let mut gradient = self.clone().clear();
        let mut iterator = self
            .neural_network
            .iter()
            .zip(gradient.neural_network.iter_mut())
            .rev();
        let (last_neural, last_gradient) = iterator.next().unwrap();
        let mut errorterm: vector<f32> =
            activation_function(&last_neural.activation_function, &forward.values, true)
                .calculate_errorterm(&loss_function(
                    &self.loss_function,
                    &forward.outputs,
                    expected,
                    true,
                ));

        last_gradient.biases = errorterm.clone();
        last_gradient.weights = &errorterm * last_neural.outputs.transpose();
        let mut dotvec: vector<f32> = last_neural.weights.transpose() * errorterm;
        errorterm = activation_function(
            &self.neural_network[self.neural_network.len() - 2usize].activation_function,
            &last_neural.values,
            true,
        )
        .calculate_errorterm(&dotvec);
        for (neural_layer, gradient_layer) in iterator {
            gradient_layer.biases = errorterm.clone();
            gradient_layer.weights = &errorterm * neural_layer.outputs.transpose();
            dotvec = neural_layer.weights.transpose() * errorterm;
            errorterm = activation_function(
                &neural_layer.activation_function,
                &neural_layer.values,
                true,
            )
            .calculate_errorterm(&dotvec);
        }
        gradient
    }

    pub fn minibatch(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let learn_scalar = -learning_rate / chunk_size as f32;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = self.backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                gradient = &gradient * &learn_scalar;
                self = &self + &gradient;
                gradient = gradient.clear();
            }
            loss_buffer.push_back(self.loss(data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!("epoch {} is done, loss: {}", epoch, self.loss(data));
        }
        self
    }
    pub fn momentum(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        momentum: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut weight_update: NeuralNetwork = self.clone().clear();
        let learn_scalar = -learning_rate / chunk_size as f32;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = self.backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                weight_update = &(&weight_update * &momentum) + &(&gradient * &learn_scalar);
                self = &self + &weight_update;
                gradient = gradient.clear();
            }
            loss_buffer.push_back(self.loss(data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!("epoch {} is done, loss: {}", epoch, self.loss(data));
        }
        self
    }
    pub fn nesterov_momentum(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        momentum: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut weight_update: NeuralNetwork = self.clone().clear();
        let learn_scalar = -learning_rate / chunk_size as f32;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward =
                        (&self + &(&weight_update * &-momentum)).backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                weight_update = &(&weight_update * &momentum) + &(&gradient * &learn_scalar);
                self = &self + &weight_update;
                gradient = gradient.clear();
            }
            loss_buffer.push_back(self.loss(data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!("epoch {} is done, loss: {}", epoch, self.loss(data));
        }
        self
    }
    pub fn adagrad(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut squared_gradient_sum: NeuralNetwork = self.clone().clear();
        let mut weight_update: NeuralNetwork;
        let learn_scalar = -learning_rate;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = self.backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                squared_gradient_sum = &squared_gradient_sum + &gradient.map(&|i: f32| i * i);
                weight_update = &squared_gradient_sum
                    .map(&|param| learn_scalar / (param.sqrt() + f32::EPSILON))
                    * &gradient;
                self = &self + &weight_update;
                gradient = gradient.clear();
            }
            loss_buffer.push_back(self.loss(data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!("epoch {} is done, loss: {}", epoch, self.loss(data));
        }
        self
    }

    pub fn adadelta(
        mut self,
        chunk_size: usize,
        decay: f32,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut weight_update: NeuralNetwork;
        let mut squared_gradient_sum: NeuralNetwork = self.clone().clear();
        let mut squared_weight_update_sum: NeuralNetwork = self.clone().clear();
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = self.backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                squared_gradient_sum = &(&squared_gradient_sum * &decay)
                    + &(&gradient.map(&|param| param.powi(2)) * &(1.0 - decay));
                weight_update = &(&(squared_weight_update_sum
                    .map(&|param| (param + f32::EPSILON).sqrt()))
                    * &(squared_gradient_sum.map(&|param| 1.0 / (param + f32::EPSILON).sqrt())))
                    * &gradient;
                squared_weight_update_sum = &(&squared_weight_update_sum * &decay)
                    + &(&weight_update.map(&|param| param.powi(2)) * &(1.0 - decay));
                self = &self + &(&weight_update * &-1.0);
                gradient = gradient.clear();
            }
            loss_buffer.push_back(self.loss(data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!("epoch {} is done, loss: {}", epoch, self.loss(data));
        }
        self
    }
    pub fn rmsprop(
        mut self,
        chunk_size: usize,
        decay: f32,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut squared_gradient_sum: NeuralNetwork = self.clone().clear();
        let learn_scalar = -learning_rate / chunk_size as f32;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = self.backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                squared_gradient_sum = &(&squared_gradient_sum * &decay)
                    + &(&gradient.map(&|param| param * param) * &(1.0 - decay));

                self = &self
                    + &(&(&gradient
                        * &squared_gradient_sum.map(&|param| 1.0 / (param.sqrt() + f32::EPSILON)))
                        * &learn_scalar);
                gradient = gradient.clear();
            }
            loss_buffer.push_back(self.loss(data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!("epoch {} is done, loss: {}", epoch, self.loss(data));
        }
        self
    }
    pub fn adam(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        beta1: f32,
        beta2: f32,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut first_moment: NeuralNetwork = self.clone().clear();
        let mut second_moment: NeuralNetwork = self.clone().clear();
        let mut norm_beta_1: f32 = beta1;
        let mut norm_beta_2: f32 = beta2;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = self.backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                first_moment = &(&first_moment * &beta1) + &(&gradient * &(1.0 - beta1));
                second_moment = &(&second_moment * &beta2)
                    + &(&gradient.map(&|param| param * param) * &(1.0 - beta2));
                norm_beta_1 *= beta1;
                norm_beta_2 *= beta2;
                self = &self
                    + &(&(&(&first_moment * &(1.0 / (1.0 - norm_beta_1))) * &-learning_rate)
                        * &((&second_moment * &(1.0 / (1.0 - norm_beta_2)))
                            .map(&|param| 1.0 / (param.sqrt() + f32::EPSILON))));
                gradient = gradient.clear();
            }
            loss_buffer.push_back(self.loss(data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!("epoch {} is done, loss: {}", epoch, self.loss(data));
        }
        self
    }
    pub fn train(
        self,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let chunk_size = match self.gradient_decent {
            GradientDecentType::Stochastic => 1,
            GradientDecentType::Batch => data.len(),
            GradientDecentType::MiniBatch(batch_size) => batch_size,
        };
        match self.optimizer {
            Optimizer::SGD => self.minibatch(chunk_size, data, learning_rate, epochs),
            Optimizer::Momentum(momentum) => {
                self.momentum(chunk_size, data, learning_rate, momentum, epochs)
            }
            Optimizer::NstMomentum(momentum) => {
                self.nesterov_momentum(chunk_size, data, learning_rate, momentum, epochs)
            }
            Optimizer::AdaGrad => self.adagrad(chunk_size, data, learning_rate, epochs),
            Optimizer::AdaDelta(decay) => self.adadelta(chunk_size, decay, data, epochs),
            Optimizer::RMSprop(decay) => {
                self.rmsprop(chunk_size, decay, data, learning_rate, epochs)
            }
            Optimizer::Adam(beta1, beta2) => {
                self.adam(chunk_size, data, beta1, beta2, learning_rate, epochs)
            }
        }
    }
}
