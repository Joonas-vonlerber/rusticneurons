use nalgebra::DVector as vector;
use std::{collections::VecDeque, f32};
use rand::{thread_rng,seq::SliceRandom};
use super::generating_and_testing::*;
use super::loss_and_activation_functions::*;
use super::types_and_errors::*;


impl NeuralNetwork {
    pub fn forward_phase(&mut self, inp: &vector<f32>) -> Layer {
        self.neural_network[0].outputs = inp.clone();
        let mut iterator = self.neural_network.iter_mut();
        let mut before = iterator.next().unwrap();
        for current in iterator {
            current.values = &before.weights * &before.outputs + &before.biases;
            current.outputs =
                match activation_function(&self.hidden_activation, &current.values, false) {
                    ActivationOutput::Vector(output) => output,
                    ActivationOutput::Matrix(output) => output.diagonal(),
                };
            before = current;
        }
        let last_layer = self.neural_network.last().unwrap();
        let output = &last_layer.weights * &last_layer.outputs + &last_layer.biases;
        Layer {
            outputs: match activation_function(&self.final_activation, &output, false) {
                ActivationOutput::Vector(outputs) => outputs,
                ActivationOutput::Matrix(outputs) => outputs.diagonal(),
            },
            values: output,
            weights: Layer::default().weights,
            biases: Layer::default().biases,
        }
    }

    fn backward_phase(
        &self,
        forward: &Layer,
        expected: &vector<f32>,
    ) -> NeuralNetwork {
        let mut gradient = self.clone().clear();
        let mut iterator = self
            .neural_network
            .iter()
            .zip(gradient.neural_network.iter_mut())
            .rev();
        let last_element = iterator.next().unwrap();
        let last_node = last_element.0;
        let mut errorterm: vector<f32> = activation_function(&self.final_activation, &forward.values, true).calculate_errorterm(&loss_function(
            &self.loss_function,
            &forward.outputs,
            expected,
            true,
        ));
        
        last_element.1.biases = errorterm.clone();
        last_element.1.weights = &errorterm * last_node.outputs.transpose();
        let mut dotvec: vector<f32> = last_node.weights.transpose() * errorterm;
        errorterm = activation_function(&self.hidden_activation, &last_node.values, true).calculate_errorterm(&dotvec);
        for (neural_layer, gradient_layer) in iterator {
            gradient_layer.biases = errorterm.clone();
            gradient_layer.weights = &errorterm * neural_layer.outputs.transpose();
            dotvec = neural_layer.weights.transpose() * errorterm;
            errorterm = activation_function(
                &self.hidden_activation,
                &neural_layer.values,
                true,
            ).calculate_errorterm(&dotvec);
        }
        gradient
    }

    pub fn minibatch(
        mut self,
        chunk_size: usize,
        data: &mut Vec<(&vector<f32>,&vector<f32>)>,
        learning_rate: f32,
        epochs: u32,
    ) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let learn_scalar = -learning_rate/chunk_size as f32;
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
            loss_buffer.push_back(loss(&mut self, &data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!(
                "epoch {} is done, loss: {}",
                epoch,
                loss(&mut self, &data)
            );
        }
        self
    }
    pub fn momentum(mut self, chunk_size: usize, data: &mut Vec<(&vector<f32>,&vector<f32>)>, learning_rate: f32, momentum: f32, epochs: u32) -> NeuralNetwork{
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut weight_update: NeuralNetwork = self.clone().clear();
        let learn_scalar = -learning_rate/chunk_size as f32;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = self.backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                weight_update = &(&weight_update*&momentum) + &(&gradient * &learn_scalar);
                self = &self + &weight_update;
                gradient = gradient.clear();
            }
            loss_buffer.push_back(loss(&mut self, &data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!(
                "epoch {} is done, loss: {}",
                epoch,
                loss(&mut self, &data)
            );
        }
        self
    }
    pub fn nesterov_momentum(mut self, chunk_size: usize, data: &mut Vec<(&vector<f32>,&vector<f32>)>, learning_rate: f32, momentum: f32, epochs: u32) -> NeuralNetwork{
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut weight_update: NeuralNetwork = self.clone().clear();
        let learn_scalar = -learning_rate/chunk_size as f32;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = (&self + &(&weight_update * &-momentum)).backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                weight_update = &(&weight_update*&momentum) + &(&gradient * &learn_scalar);
                self = &self + &weight_update;
                gradient = gradient.clear();
            }
            loss_buffer.push_back(loss(&mut self, &data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!(
                "epoch {} is done, loss: {}",
                epoch,
                loss(&mut self, &data)
            );
        }
        self
    }
    pub fn adagrad(mut self, chunk_size: usize, data: &mut Vec<(&vector<f32>,&vector<f32>)>, learning_rate: f32, epochs: u32) -> NeuralNetwork{
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
                squared_gradient_sum = &squared_gradient_sum + &gradient.map(&|i:f32| i*i);
                weight_update = &squared_gradient_sum.map(&|param| learn_scalar / (param.sqrt()+f32::EPSILON)) * &gradient;
                self = &self + &weight_update;
                gradient = gradient.clear();
            }
            loss_buffer.push_back(loss(&mut self, &data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!(
                "epoch {} is done, loss: {}",
                epoch,
                loss(&mut self, &data)
            );
        }
        self
    }

    pub fn adadelta(
        mut self,
        chunk_size: usize,
        decay: f32,
        data: &mut Vec<(&vector<f32>,&vector<f32>)>,
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
                squared_gradient_sum = &(&squared_gradient_sum * &decay) + &(&gradient.map(&|param| param.powi(2)) * &(1.0-decay));
                weight_update = &(&(squared_weight_update_sum.map(&|param| (param+f32::EPSILON).sqrt())) * &(squared_gradient_sum.map(&|param| 1.0/(param+f32::EPSILON).sqrt()))) * &gradient;
                squared_weight_update_sum = &(&squared_weight_update_sum * &decay) + &(&weight_update.map(&|param| param.powi(2)) * &(1.0-decay));
                self = &self + &(&weight_update * &-1.0);
                gradient = gradient.clear();
            }
            loss_buffer.push_back(loss(&mut self, &data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!(
                "epoch {} is done, loss: {}",
                epoch,
                loss(&mut self, &data)
            );
        }
        self
    }
    pub fn rmsprop(mut self, chunk_size: usize ,alpha: f32,data: &mut Vec<(&vector<f32>,&vector<f32>)>, learning_rate: f32, epochs: u32) -> NeuralNetwork{
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut squared_gradient_sum: NeuralNetwork = self.clone().clear();
        let learn_scalar = -learning_rate/chunk_size as f32;
        for epoch in 0..epochs {
            data.shuffle(&mut thread_rng());
            for chunk in data.chunks_exact(chunk_size) {
                for (input, expect) in chunk.iter() {
                    forward = self.forward_phase(input);
                    backward = self.backward_phase(&forward, expect);
                    gradient = &gradient + &backward;
                }
                squared_gradient_sum = &(&squared_gradient_sum * &alpha) + &(&gradient.map(&|param| param*param) * &(1.0-alpha));
                
                self = &self + &(&(&gradient * &squared_gradient_sum.map(&|param| 1.0/(param.sqrt()+f32::EPSILON))) * &learn_scalar);
                gradient = gradient.clear();
            }
            loss_buffer.push_back(loss(&mut self, &data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!(
                "epoch {} is done, loss: {}",
                epoch,
                loss(&mut self, &data)
            );
        }
        self
    }
    pub fn adam(mut self, chunk_size: usize, data: &mut Vec<(&vector<f32>,&vector<f32>)>, beta1: f32, beta2: f32, learning_rate: f32, epochs: u32) -> NeuralNetwork {
        let mut gradient: NeuralNetwork = self.clone().clear();
        let mut forward: Layer;
        let mut backward: NeuralNetwork;
        let mut loss_buffer: VecDeque<f32> = VecDeque::with_capacity(9);
        let mut first_moment: NeuralNetwork = self.clone().clear();
        let mut second_moment: NeuralNetwork = self.clone().clear();
        let mut first_moment_norm: NeuralNetwork;
        let mut second_moment_norm: NeuralNetwork;
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
                first_moment = &(&first_moment * &beta1) + &(&gradient * &(1.0-beta1));
                second_moment = &(&second_moment * &beta2) + &(&gradient.map(&|param| param*param) * &(1.0-beta2));   
                norm_beta_1 *= beta1;
                norm_beta_2 *= beta2;
                first_moment_norm = &first_moment * &(1.0/(1.0-norm_beta_1));
                second_moment_norm = &second_moment * &(1.0/(1.0-norm_beta_2)); 
                self = &self + &(&(&first_moment_norm * &-learning_rate) * &(second_moment_norm.map(&|param| 1.0/(param.sqrt() + f32::EPSILON))));
                gradient = gradient.clear();
            }
            loss_buffer.push_back(loss(&mut self, &data));
            if epoch > 8 {
                if standard_deviation(&loss_buffer) < 0.0001 {
                    eprintln!("{}", NeuralNetworkError::CostError);
                    return self;
                }
                loss_buffer.pop_front();
            }
            println!(
                "epoch {} is done, loss: {}",
                epoch,
                loss(&mut self, &data)
            );
        }
        self
    }
    pub fn train(self, data: &mut Vec<(&vector<f32>,&vector<f32>)>, learning_rate: f32, epochs: u32) -> NeuralNetwork {
        match self.gradient_decent {
            GradientDecentType::MiniBatch(chunk_size) => self.minibatch(chunk_size, data, learning_rate, epochs),
            GradientDecentType::Batch => self.minibatch(data.len(), data, learning_rate, epochs),
            GradientDecentType::Stochastic => self.minibatch(1, data, learning_rate, epochs),
        }
    }
}
