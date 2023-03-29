use crate::files;
use crate::ui;
use nalgebra::DVector as vector;

use crate::loss_and_activation_functions::loss_function;
use crate::types_and_errors::NeuralNetwork;
pub fn initialize_expected_outputs_mnist(labels: &[u8]) -> Vec<vector<f32>> {
    let expected: Vec<vector<f32>> = labels
        .iter()
        .map(|f| {
            let mut expect = vector::zeros(10);
            expect[*f as usize] = 1.0;
            expect
        })
        .collect();
    expected
}

pub fn loss_mnist(
    neural_network: &mut NeuralNetwork,
    test_data: &Vec<(&vector<f32>, &vector<f32>)>,
) -> (f64, f32) {
    let mut forward: vector<f32>;
    let mut loss: f64 = 0.0;
    let mut got_right: u16 = 0;
    for (input, expected) in test_data.iter() {
        forward = neural_network.forward_phase(input).outputs;
        loss +=
            loss_function(&neural_network.loss_function, &forward, expected, false).sum() as f64;
        if forward.imax() == expected.imax() {
            got_right += 1;
        }
    }
    (
        loss / test_data.len() as f64,
        f32::from(got_right) / test_data.len() as f32,
    )
}

pub fn loss(
    neural_network: &mut NeuralNetwork,
    test_data: &Vec<(&vector<f32>, &vector<f32>)>,
) -> f32 {
    let mut forward: vector<f32>;
    let mut loss: f32 = 0.0;
    for (input, expected) in test_data.iter() {
        forward = neural_network.forward_phase(input).outputs;
        loss += loss_function(&neural_network.loss_function, &forward, expected, false).sum();
    }
    loss / test_data.len() as f32
}

pub fn print_number_and_test(neural_network: &mut NeuralNetwork) {
    println!("Print nth image");
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    let data = files::load_inputs(r"resources/train-images-idx3-ubyte").unwrap();
    let n = line.trim().parse::<usize>().unwrap();
    let image = ui::NumberImage {
        rows: 28,
        columns: 28,
        data: data[n].clone(),
    };
    println!("{}", image);
    println!("{}", neural_network.forward_phase(&data[n].clone()).outputs);
    println!(
        "{}",
        neural_network.forward_phase(&data[n].clone()).outputs.imax()
    );
}
