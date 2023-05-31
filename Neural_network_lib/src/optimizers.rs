use crate::loss_and_activation_functions::*;
use crate::neuralnetwork::*;
use nalgebra::DVector as vector;
use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::*;
use std::vec;
use std::{collections::VecDeque, f32};
#[derive(Debug, PartialEq, Clone)]
struct ValueOutputPair {
    value: vector<f32>,
    output: vector<f32>,
}

impl NeuralNetwork {
    fn forward_phase(&self, input: &vector<f32>) -> Vec<ValueOutputPair> {
        let mut value: vector<f32> = vector::zeros(input.len());
        let mut output: vector<f32> = input.clone();
        let input: Vec<ValueOutputPair> = vec![ValueOutputPair {
            value: value.clone(),
            output: output.clone(),
        }];
        let values_and_outputs: Vec<ValueOutputPair> = self
            .neural_network
            .iter()
            .map(|layer| {
                value = &layer.weights * &output + &layer.biases;
                output =
                    activation_function(&layer.activation_function, &value, false).extract_vector();
                ValueOutputPair {
                    value: value.clone(),
                    output: output.clone(),
                }
            })
            .collect::<Vec<ValueOutputPair>>();
        [input, values_and_outputs].concat()
    }

    fn backward_phase(&self, forward: &[ValueOutputPair], expected: &vector<f32>) -> NeuralNetwork {
        let mut iterator_values_and_outputs = forward.iter().rev();
        let neural_out = iterator_values_and_outputs.next().unwrap();
        let mut iterator = self
            .neural_network
            .iter()
            .rev()
            .zip(iterator_values_and_outputs);
        let (last_layer_neural, last_layer_values_outputs) = iterator.next().unwrap();
        let mut errorterm: vector<f32> = activation_function(
            &last_layer_neural.activation_function,
            &neural_out.value,
            true,
        )
        .calculate_errorterm(&loss_function(
            &self.loss_function,
            &neural_out.output,
            expected,
            true,
        ));
        let last_layer = Layer {
            weights: &errorterm * last_layer_values_outputs.output.transpose(),
            biases: errorterm.clone(),
            activation_function: last_layer_neural.activation_function.clone(),
        };
        errorterm = last_layer_neural.weights.transpose() * errorterm;
        let mut last_values: &vector<f32> = &last_layer_values_outputs.value;
        let mut gradient: Vec<Layer> = iterator
            .map(|(neural_layer, layer_values_outputs)| {
                errorterm =
                    activation_function(&neural_layer.activation_function, last_values, true)
                        .calculate_errorterm(&errorterm);
                let gradient_layer = Layer {
                    weights: &errorterm * layer_values_outputs.output.transpose(),
                    biases: errorterm.clone(),
                    activation_function: neural_layer.activation_function.clone(),
                };
                errorterm = neural_layer.weights.transpose() * errorterm.clone();
                last_values = &layer_values_outputs.value;
                gradient_layer
            })
            .collect();
        gradient.reverse();
        gradient.push(last_layer);
        NeuralNetwork {
            neural_network: gradient,
            loss_function: self.loss_function.clone(),
        }
    }

    fn set_gradient(
        &mut self,
        neural_network: &mut NeuralNetwork,
        batch: &[(&vector<f32>, &vector<f32>)],
    ) {
        *self = self.clear();
        let gradient: NeuralNetwork = batch
            .par_iter()
            .map(|(input, expected)| {
                neural_network.backward_phase(&neural_network.forward_phase(input), expected)
            })
            .reduce(|| neural_network.clone_clear(), |a, b| &a + &b);
        *self = &(&*self + &gradient) * &gradient.normalize();
    }

    fn set_gradient_nesterov(
        &mut self,
        neural_network: &mut Self,
        momentum_network: &Self,
        decay: f32,
        batch: &[(&vector<f32>, &vector<f32>)],
    ) {
        *self = self.clear();
        let gradient: NeuralNetwork = batch
            .par_iter()
            .map(|(input, expected)| {
                (&*neural_network + &(momentum_network * &-decay))
                    .backward_phase(&neural_network.forward_phase(input), expected)
            })
            .reduce(|| neural_network.clone_clear(), |a, b| &a + &b);
        *self = &(&*self + &gradient) * &gradient.normalize();
    }
    fn minibatch(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone_clear();
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for batch in data.chunks_exact(chunk_size) {
                gradient.set_gradient(&mut self, batch);
                gradient = &gradient * &(-learning_rate);
                self = &self + &gradient;
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
    fn momentum(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        decay: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone_clear();
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut weight_update: NeuralNetwork = self.clone_clear();
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for batch in data.chunks_exact(chunk_size) {
                gradient.set_gradient(&mut self, batch);
                weight_update = &(&weight_update * &decay) + &(&gradient * &(-learning_rate));
                self = &self + &weight_update;
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
    fn nesterov_momentum(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        decay: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone_clear();
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut weight_update: NeuralNetwork = self.clone_clear();
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for batch in data.chunks_exact(chunk_size) {
                gradient.set_gradient_nesterov(&mut self, &weight_update, decay, batch);
                weight_update = &(&weight_update * &decay) + &(&gradient * &(-learning_rate));
                self = &self + &weight_update;
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
    fn adagrad(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone_clear();
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut squared_gradient_sum: NeuralNetwork = self.clone_clear();
        let mut weight_update: NeuralNetwork;
        let learn_scalar = -learning_rate;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for batch in data.chunks_exact(chunk_size) {
                gradient.set_gradient(&mut self, batch);
                squared_gradient_sum = &squared_gradient_sum + &gradient.map(&|i: f32| i * i);
                weight_update = &squared_gradient_sum
                    .map(&|param| learn_scalar / (param.sqrt() + f32::EPSILON))
                    * &gradient;
                self = &self + &weight_update;
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

    fn adadelta(
        mut self,
        chunk_size: usize,
        decay: f32,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut gradient: NeuralNetwork = self.clone_clear();
        let mut weight_update: NeuralNetwork;
        let mut squared_gradient_sum: NeuralNetwork = self.clone_clear();
        let mut squared_weight_update_sum: NeuralNetwork = self.clone_clear();
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for batch in data.chunks_exact(chunk_size) {
                gradient.set_gradient(&mut self, batch);
                //yea ignore the following bullshit oneliners :DDD
                squared_gradient_sum = &(&squared_gradient_sum * &decay)
                    + &(&gradient.map(&|param| param.powi(2)) * &(1.0 - decay));
                weight_update = &(&(squared_weight_update_sum
                    .map(&|param| (param + f32::EPSILON).sqrt()))
                    * &(squared_gradient_sum.map(&|param| 1.0 / (param + f32::EPSILON).sqrt())))
                    * &gradient;
                squared_weight_update_sum = &(&squared_weight_update_sum * &decay)
                    + &(&weight_update.map(&|param| param.powi(2)) * &(1.0 - decay));
                self = &self + &(&weight_update * &-1.0);
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
    fn rmsprop(
        mut self,
        chunk_size: usize,
        decay: f32,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone_clear();
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut squared_gradient_sum: NeuralNetwork = self.clone_clear();
        let learn_scalar = -learning_rate / chunk_size as f32;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for batch in data.chunks_exact(chunk_size) {
                gradient.set_gradient(&mut self, batch);
                squared_gradient_sum = &(&squared_gradient_sum * &decay)
                    + &(&gradient.map(&|param| param * param) * &(1.0 - decay));

                self = &self
                    + &(&(&gradient
                        * &squared_gradient_sum.map(&|param| 1.0 / (param.sqrt() + f32::EPSILON)))
                        * &learn_scalar);
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
    fn adam(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        beta1: f32,
        beta2: f32,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone_clear();
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut first_moment: NeuralNetwork = self.clone_clear();
        let mut second_moment: NeuralNetwork = self.clone_clear();
        let mut norm_beta_1: f32 = beta1;
        let mut norm_beta_2: f32 = beta2;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for batch in data.chunks_exact(chunk_size) {
                gradient.set_gradient(&mut self, batch);
                first_moment = &(&first_moment * &beta1) + &(&gradient * &(1.0 - beta1));
                second_moment = &(&second_moment * &beta2)
                    + &(&gradient.map(&|param| param * param) * &(1.0 - beta2));
                norm_beta_1 *= beta1;
                norm_beta_2 *= beta2;
                //yea I knowww it is very ugly but I save memoryyyyyyy :DDDD
                //the whole shinanigance you can find here https://www.ruder.io/optimizing-gradient-descent/#adam
                self = &self
                    + &(&(&(&first_moment * &(1.0 / (1.0 - norm_beta_1))) * &-learning_rate)
                        * &((&second_moment * &(1.0 / (1.0 - norm_beta_2)))
                            .map(&|param| 1.0 / (param.sqrt() + f32::EPSILON))));
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
    /// Train networks with different types of gradient decent optimizations
    pub fn train(
        self,
        data: &mut Vec<(&vector<f32>, &vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
        batchtype: BatchType,
        optimizer: Optimizer,
    ) -> NeuralNetwork {
        let chunk_size = match batchtype {
            BatchType::Stochastic => 1,
            BatchType::Batch => data.len(),
            BatchType::MiniBatch(batch_size) => batch_size,
        };
        match optimizer {
            Optimizer::Vanilla => self.minibatch(chunk_size, data, learning_rate, epochs),
            Optimizer::Momentum(decay) => {
                self.momentum(chunk_size, data, learning_rate, decay, epochs)
            }
            Optimizer::NstMomentum(decay) => {
                self.nesterov_momentum(chunk_size, data, learning_rate, decay, epochs)
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
