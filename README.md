# katana-rs
A rust-based neural network library powered by arrayfire

I created this project with the goal of broadening my understanding about neural networks.

module tree:

    - examples
        - feedforward.rs
        - spiral.csv
    - src
        - layer
            - activation
                - mod.rs
                - relu.rs
                - softmax.rs
            - mod.rs
            - dense.rs
        - lib.rs
        - dataset.rs
    