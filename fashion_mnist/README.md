# fmnist-sample in Rust

Fashion MNIST classification sample using TensorFlow Rust bindings.

## Prerequisite

```shell
pip install tensorflow numpy pillow
python generate.py
```

## Run

### Test accuracy of the model

```shell
cargo run -- test
```

### Classify a specified image

```shell
cargo run -- classify --file <file>
```
