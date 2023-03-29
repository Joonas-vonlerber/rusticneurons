extern crate neural_network_lib as nnl;
use nnl::files::*;
use nnl::types_and_errors::*;
use nnl::generating_and_testing::*;
use nalgebra::DVector as vector;

fn main() {
    let mut neural_network: NeuralNetwork = NeuralNetwork::new(
        &[784, 32, 32, 10],
        LossFunction::MSE,
        ActivationFunction::SoftMax,
        ActivationFunction::LeReLU,
        GradientDecentType::MiniBatch(64),
        Optimizer::Adam(0.9, 0.999)
    );
    let data = load_inputs("resources/train-images-idx3-ubyte").unwrap();
    let labels = initialize_expected_outputs_mnist(
        &load_expected_outputs("resources/train-labels-idx1-ubyte").unwrap(),
    );
    let test_data = load_inputs("resources/t10k-images-idx3-ubyte").unwrap();
    let test_labels = initialize_expected_outputs_mnist(
        &load_expected_outputs("resources/t10k-labels-idx1-ubyte").unwrap(),
    );
    let mut data: Vec<(&vector<f32>, &vector<f32>)> = data.iter().zip(labels.iter()).collect();
    let test_data: Vec<(&vector<f32>, &vector<f32>)> = test_data.iter().zip(test_labels.iter()).collect();
    println!(
        "{:?}",
        loss_mnist(&mut neural_network, &test_data)
    );
    neural_network = neural_network.train(&mut data, 0.001, 100);
    println!(
        "{:?}",
        loss_mnist(&mut neural_network, &test_data)
    );
    save_network_bincode("networks/MNIST_Adam", &neural_network).unwrap();
    // print_number_and_test(&mut neural_network);
}