# goom_ssm_rnn

Reference implementation of a deep RNN that captures sequential dependencies in every layer with the following non-diagonal state-space system, _executed in parallel via a prefix scan without any form of stabilization_:

$$
\begin{aligned}
x_t & = A x_{t-1} + B u_t \\
y_t & = C x_t + D u_t, \\
\end{aligned}
$$

where $u_t$, $x_t$, and $y_t$ are input, hidden, and output state vectors, respectively, and $A$, $B$, $C$, and $D$ are matrix parameters. Parallel execution without stabilization is possible because we compute the recurrence over [generalized orders of magnitude](https://github.com/glassroom/generalized_orders_of_magnitude) (GOOMs), with greater dynamic range than previously possible.


## Installing

1. Clone this repository.

2. Install the Python dependencies in `requirements.txt`.

3. There is no third step.


## Instantiating the RNN

The following code instantiates a small RNN for generative language modeling tasks with GPT-2's vocabulary: 

```python
import torch
import tiktoken
import goom_ssm_rnn

DEVICE = 'cuda'  # change as needed

# Get GPT-2 encoder:
enc = tiktoken.get_encoding('gpt2')

# Instantiate an RNN for natural language generation:
model = goom_ssm_rnn.GenerativeRNN(
    vocab_sz=enc.n_vocab, d_emb=768, n_hid=24, d_hid=32, n_res=24)

# Move model to cuda device:
model.to(device=DEVICE)

# You must provide your own training code.
```

## Use of Complex-Typed GOOMs

Recurrent layers in the model capture sequential dependencies with a non-diagonal linear SSM, executed via a parallel prefix scan, over [GOOMs](https://github.com/glassroom/generalized_orders_of_magnitude), implemented as torch.complex64 tensors (_i.e._, with torch.float32 real and imaginary components). As we explain in our paper, the use of complex-typed GOOMs makes it possible for each layer to compute _non-diagonal recurrent states in parallel without requiring any form of stabilization_.

Otherwise, the rest of the model operates conventionally, over torch.float32 tensors, optionally autocasting to torch.float16, if you specify it. As we explain in our paper, each recurrent layer scales complex-typed GOOMs before exponentiating them to torch.float32 real tensors, because the GOOM magnitudes can be outside bounds representable by torch.float32.


## Convenience Methods

Besides the standard PyTorch `forward()` method, the model provides three additional methods, for convenience:

* `model.get_param_groups()`, which accepts a scalar weight_decay value as input, and returns two parameter groups for training, one with weight decay and one without without decay.

* `model.compute_loss_and_metrics()`, which accepts predicted scores over the model's vocabulary, and true token ids, and returns a cross-entropy loss and a dictionary with one metric: 'accuracy'.

* `model.generate()`, for generating new token ids, given a sequence of preceding token ids, after the model has been trained on a language-generation task. Please see our code for additional arguments.


## Training and Testing the Model

We have implemented the model as a standard PyTorch `nn.Module` that you can train and test on any task, using conventional techniques, including autocasting. However, at present the model can be only partially compiled, because PyTorch's compiler doesn't yet fully support complex tensors. For information on the current state of PyTorch's support for complex tensors, please see [this page on the PyTorch website](https://docs.pytorch.org/docs/stable/complex_numbers.html).

When we apply `torch.compile()` to the entire model and start training it, lazy compilation spits out a variety of warnings related to the lack of support of complex tensors, but compilation succeeds -- and significantly reduces execution time and memory use. Our implementation of GOOMs incorporates custom `torch.Autograd.function` transformations under-the-hood to ensure proper backpropagation of gradients, taking special care to handle the singularity at zero gracefully. (As a real number approaches zero, the real component of its complex logarithm approaches negative infinity.)

Note: We have tested autocasting of float tensors only to torch.float16.


## Replicating Published Results

We trained the RNN model in this repository on natural language generation and multiple other tasks.


### Natural Language Generation

We trained an instance of the RNN with 768 embedding dimensions (`d_emb=768`), 24 heads per token (`n_hid=24`), 32 features per head (`d_hid=32`), 24 recurrent residual layers (`n_res=24`), and GPT-2 vocabulary, resulting in 124M parameters, on approximately 10B tokens randomly sampled from [The Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted), with a sequence length of 1024 tokens. We trained the RNN with the hyper-parameters shown on the table below. Cross-entropy loss declined to approximately 2.7 after training on 10B tokens. For comparison, cross-entropy for state-of-the-art models of comparable size, trained on 30x or more tokens sampled from higher-quality datasets, is approximately 2.4, suggesting our RNN model can be scaled up to larger tasks.

| Hyper-parameter        | Value                                                            |
| :--------------------- | :--------------------------------------------------------------- |
| Batch size             | 960 sequences, split in micro-batches that accumulate gradients  |
| Micro-batch size       | Largest integer factor of 960 that fits in GPU memory            |
| Optimizer              | AdamW, using `torch.optim.AdamW`                                 |
| Weight decay           | 1e-1                                                             |
| Parameter groups       | 2, obtained with `model.get_param_groups(weight_decay=1e-1)`     |
| Learning rate schedule | One cycle, using `torch.optim.lr_scheduler.OneCycleLR`           |
| Maximum learning rate  | 3e-4                                                             |
| Ending learning rate   | 1e-5                                                             |
| Maximum momentum       | 0.99                                                             |
| Minimum momentum       | 0.85                                                             |
| Warm-up period         | 10 batches (9600 sample sequences)                               |
| Compilation            | Yes (applies only to operations on floats, not complex GOOMs)    |
| Autocasting            | Yes, to `torch.float16` (only floats, not complex GOOMs)         |
| Training iterations    | 10240 batches                                                    |
| Cumulative tokens      | 10B (1024 tokens/sequence x 960 sequences/batch x 10240 batches) |


### Other Tasks

Other tasks include Sequential [MNIST](https://huggingface.co/datasets/ylecun/mnist) generation (unrolling the images into sequences of 784 pixel-tokens, using a vocabulary size of 256 gray levels, and generating each next pixel), Sequential [MNIST](https://huggingface.co/datasets/ylecun/mnist) classification (replacing the generative-language-modeling head with a linear-classification head that predicts 10 classes from the last pixel-token's hidden state),  [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) (using the GPT-2 vocabulary for convenience), and Copy-Memory tasks.  For all such tasks, we instantiated the RNN with 512 embedding dimensions (`d_emb=512`), 16 heads per token (`n_hid=16`), 32 features per head (`d_hid=32`), eight residual recurrent layers (`n_res=8`), a task-specific vocabulary, and a task-specific model head, resulting in 12.8M to 38M parameters. We trained all models with the hyper-parameters shown on the table below. The models trained to competitive performance on all tasks we tested.

| Hyper-parameter        | Value                                                            |
| :--------------------- | :--------------------------------------------------------------- |
| Batch size             | 1000, split in micro-batches that accumulate gradients           |
| Micro-batch size       | Largest integer factor of 1000 that fits in GPU memory           |
| Optimizer              | AdamW, using `torch.optim.AdamW`                                 |
| Weight decay           | 1e-1                                                             |
| Parameter groups       | 2, obtained with `model.get_param_groups(weight_decay=1e-1)`     |
| Learning rate schedule | One cycle, using `torch.optim.lr_scheduler.OneCycleLR`           |
| Maximum learning rate  | 3e-4                                                             |
| Ending learning rate   | 1e-5                                                             |
| Maximum momentum       | 0.99                                                             |
| Minimum momentum       | 0.85                                                             |
| Warm-up period         | 10 batches (10,000 samples)                                      |
| Compilation            | Yes (applies only to operations on floats, not complex GOOMs)    |
| Autocasting            | Yes, to `torch.float16` (only floats, not complex GOOMs)         |
| Data augmentation      | Yes, conventional (_e.g._, affine transforms on training images) |
| Training iterations    | At least 1,800 (1.8M samples); harder tasks require more samples |


## Modifying the RNN for Other Tasks

You can modify or replace the model's language-modeling head, as needed, for tasks other than generative language modeling. All model components are defined in a single file:

[goom_ssm_rnn.py](goom_ssm_rnn.py)


## Citing

```
@article{
heinsen2025generalized,
title={Generalized Orders of Magnitude for Scalable, Parallel, High-Dynamic-Range Computation},
author={Franz A. Heinsen and Leo Kozachkov},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=SUuzb0SOGu},
note={}
}
```


## Notes

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the complex plane. Our casual conversations gradually evolved into the development of generalized orders of magnitude, along with an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan.

We hope others find our work and our code useful.
