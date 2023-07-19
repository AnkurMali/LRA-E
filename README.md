# LRA-E
AAAI 2019 LRA-E Model
Requires:- Tensorflow >=2.0, Python >=3.5, numpy

# We have provided scripts for training LRA-E on mnist and fmnist.

# How to run
* python mnist/mnist.py runs 4 layer MLP on mnist
* python fmnist/fmnist.py runs 4 layer MLP on fashion-mnist.

# Structure of code
We have structured code keeping the following things in mind
* easy to modify
* our codebase maps with pseudo-code provided in our paper
* easy to convert for researchers familiar with other deep learning libraries(Pytorch/Theano).
* Using basic ops to understand the update rule for LRA-E 

Class MLP creates all the necessary function
* Declare all the Weight and biases variables for the model.[How deeper you wish to go]
* forward :- Function does the forward pass for the model on cpu or gpu.
* compute_output:- gives you a set of post and preactivations for the model.
* backward :- will help you calculate backpropagation (backprop) gradients and perform one step of SGD [Note: Can be replaced with any TF optimizer].
* compute_lra_update :- Will compute LRA_updates (instead of backprop)


We will update this repo to add more examples and optimal training rules for LRA.

# Citation

If you use or adapt (portions of) this code/algorithm in any form in your project(s), or
find the LRA-E algorithm helpful in your own work; please cite this code's source paper:

```bibtex
@inproceedings{ororbia2019biologically,
  title={Biologically motivated algorithms for propagating local target representations},
  author={Ororbia, Alexander G and Mali, Ankur},
  booktitle={Proceedings of the aaai conference on artificial intelligence},
  volume={33},
  number={01},
  pages={4651--4658},
  year={2019}
}
```
