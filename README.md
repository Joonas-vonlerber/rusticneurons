# Rusticneurons
A machinelearning package for Rust with neural networks :D

It is not currently a public crate, but maybe in the future?

Really was just a way for me to try out Rust and learn about neural networks on the way.

**Run with release mode on**, because without it, it'll be very slow.
## Usage:

### Step 1:
Clone the repository
> `git clone https://github.com/Joonas-vonlerber/rusticneurons.git`

### Step 2:
In your projects Cargo.toml under dependencies add:
>`neural_network_lib = {path = 'path/to/Neural_network_lib'}`

## Example program

Provided you have the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in resources 

```
extern crate neural_network_lib as nnl;
use nalgebra::DVector as vector;
use nnl::files::*;
use nnl::generating_and_testing::*;
use nnl::loss_and_activation_functions::*;
use nnl::types_and_errors::*;
fn main() {
    let mut neural_network: NeuralNetwork = NeuralNetwork::new(
        &[784, 32, 32, 10],
        InitializaitonType::HeKaiming,
        LossFunction::CrossEntropy,
        ActivationFunction::SoftMax,
        ActivationFunction::ReLU,
    );
    let data: Vec<vector<f32>> = load_inputs("resources/train-images-idx3-ubyte").unwrap();
    let labels: Vec<vector<f32>> = initialize_expected_outputs_mnist(
        &load_expected_outputs("resources/train-labels-idx1-ubyte").unwrap(),
    );
    let test_data: Vec<vector<f32>> = load_inputs("resources/t10k-images-idx3-ubyte").unwrap();
    let test_labels: Vec<vector<f32>> = initialize_expected_outputs_mnist(
        &load_expected_outputs("resources/t10k-labels-idx1-ubyte").unwrap(),
    );
    let mut data: Vec<(&vector<f32>, &vector<f32>)> = data.iter().zip(labels.iter()).collect();
    let test_data: Vec<(&vector<f32>, &vector<f32>)> =
        test_data.iter().zip(test_labels.iter()).collect();
    println!("{:?}", neural_network.loss_mnist(&test_data));
    neural_network = neural_network.train(
        &mut data,
        0.001,
        10,
        GradientDecentType::MiniBatch(64),
        Optimizer::Adam(0.9, 0.999),
    );
    println!("{:?}", neural_network.loss_mnist(&test_data));
    save_network("networks/MNIST_Adam", &neural_network).unwrap();
}
```