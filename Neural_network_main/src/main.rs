use neural_network_lib::*;

fn main() {
    // let mut neural_network: NeuralNetwork = NeuralNetwork::new(
    //     &[784, 32, 32, 10],
    //     LossFunction::MSE,
    //     ActivationFunction::SoftMax,
    //     ActivationFunction::LeReLU,
    //     GradientDecentType::MiniBatch(64),
    // );
    // let mut neural_network = load_network_bincode("networks/adam_testing").unwrap();
    // let data = load_inputs("resources/train-images-idx3-ubyte").unwrap();
    // let labels = initialize_expected_outputs_mnist(
    //     &load_expected_outputs("resources/train-labels-idx1-ubyte").unwrap(),
    // );
    // let test_data = load_inputs("resources/t10k-images-idx3-ubyte").unwrap();
    // let test_labels = initialize_expected_outputs_mnist(
    //     &load_expected_outputs("resources/t10k-labels-idx1-ubyte").unwrap(),
    // );
    // let mut data: Vec<(&vector<f32>, &vector<f32>)> = data.iter().zip(labels.iter()).collect();
    // println!(
    //     "{:?}",
    //     loss_mnist(&mut neural_network, &test_data, &test_labels)
    // );
    // neural_network = neural_network.adam(64,&mut data,0.9, 0.999, 0.001, 100);
    // println!(
    //     "{:?}",
    //     loss_mnist(&mut neural_network, &test_data, &test_labels)
    // );
    // save_network_bincode("networks/adam_testing", &neural_network).unwrap();
    // print_number_and_test(&mut neural_network);
}