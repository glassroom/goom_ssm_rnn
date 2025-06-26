# goom_ssm_rnn

Reference implementation of a deep RNN that captures dependencies with a non-diagonal linear state-space model (SSM), executable in parallel via a prefix scan, over [generalized orders of magnitude](https://github.com/glassroom/generalized_orders_of_magnitude) (GOOMs), allowing recurrent states to fluctuate freely over a greater dynamic range of real values than previously possible.


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

# Instantiate an RNN for generative language model with GPT-2 token ids:
model = goom_ssm_rnn.GenerativeRNN(
    vocab_sz=enc.n_vocab, d_emb=512, n_hid=16, d_hid=32, n_res=8)

# Move model to cuda device:
model.to(device=DEVICE)

# You must provide your own training code.
```

## Use of Complex-Typed GOOMs

Recurrent layers in the model capture sequential dependencies with a non-diagonal linear SSM, executed via a parallel prefix scan, over [GOOMs](https://github.com/glassroom/generalized_orders_of_magnitude), which are represented as torch.complex64 tensors (_i.e._, with torch.float32 real and imaginary components). As we explain in our paper, the use of complex-typed GOOMs makes it possible for us to allow recurrent states in each layer to fluctuate freely over a greater dynamic range of values than possible with torch.float32 or torch.float64, without numerical degradation, including in training.

Otherwise, the rest of the model operates conventionally, over torch.float32 tensors, optionally with autocasting to torch.float16, if you specify it. As we explain in our paper, each recurrent layer scales complex-typed GOOMs before exponentiating them to torch.float32 real tensors, because GOOM magnitudes can be be outside the bounds representable by torch.float32.


## Training and Testing the Model

We have implemented the model as a standard PyTorch `nn.Module` that you can train and test on any task, using conventional techniques, including autocasting. However, at present the model can be only partially compiled, because PyTorch's compiler doesn't yet fully support complex tensors. For information on the current state of PyTorch's support for complex tensors, please see [this page on the PyTorch website](https://docs.pytorch.org/docs/stable/complex_numbers.html).

When we apply `torch.compile()` to the entire model and start training it, lazy compilation spits out a variety of warnings related to the lack of support of complex tensors, but compilation succeeds -- and significantly reduces execution time and memory use. Our implementation of GOOMs incorporates custom `torch.Autograd.function` transformations under-the-hood to ensure proper backpropagation of gradients, taking special care to handle the singularity at zero gracefully. (As a real number approaches zero, the real component of its complex logarithm approaches negative infinity.)

Note: We have tested autocasting of float tensors only to torch.float16.


## Replicating Published Results

We successfully trained the RNN model in this repository on several toy tasks, including [Wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) (using the GPT-2 vocabulary), Sequential [MNIST](https://huggingface.co/datasets/ylecun/mnist) generation (unrolling the images into sequences of 784 pixel-tokens, and using a vocabulary size of 256 gray levels), Sequential [MNIST](https://huggingface.co/datasets/ylecun/mnist) classification (replacing the generative-language-modeling head with a linear-classification head that predicts 10 classes from the last pixel-token's hidden state), and simple Copy-Memory tasks.

For all toy tasks, we instantiated the model with 512 embedding dimensions (`d_emb=512`), 16 heads per token (`n_hid=16`), 32 features per head (`d_hid=32`), and eight residual recurrent layers (`n_res=8`), resulting in 13M to 38M parameters, and trained it on a recent mid-tier Nvidia GPU, with the following hyper-parameters:

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
| Compilation            | Yes (applies only to operations on float tensors, not GOOMs)     |
| Autocasting            | Yes, to `torch.float16` (only float tensors, not GOOMs)          |
| Data augmentation      | Yes, conventional (_e.g._, affine transforms on training images) |
| Training iterations    | At least 1,800 (1.8M samples); harder tasks require more samples |

The RNN trained to competitive performance on all toy tasks we tested.

Out of curiosity, we also partially trained a larger instance of the RNN (`d_emb=768`, `n_hid=24`, `d_hid=32`, `n_res=24`, GPT-2 vocabulary, 124M parameters) on approximately 10B tokens randomly sampled from [The Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted), with a sequence length of 1024 tokens, and saw cross-entropy loss decline to approximately 2.7. State-of-the-art cross-entropy for models of comparable size, with a similar vocabulary, trained on 30x or more tokens sampled from higher-quality datasets, is approximately 2.4. This partial experiment suggests our RNN model can be scaled up to non-toy tasks.


## Convenience Methods

Besides the standard PyTorch `forward()` method, the model provides three additional methods:

* `model.get_param_groups()`, which accepts a scalar weight_decay value as input, and returns two parameter groups for training, one with weight decay and one without without decay.

* `model.compute_loss_and_metrics()`, which accepts predicted scores over the model's vocabulary, and true token ids, and returns a cross-entropy loss and a dictionary with one metric: 'accuracy'.

* `model.generate()`, for generating new token ids, given a sequence of preceding token ids, after the model has been trained on a language-generation task. Please see our code for additional arguments.


## Modifying the RNN for Other Tasks

You can modify or replace the model's language-modeling head, as needed, for tasks other than generative language modeling. All model components are defined in a single file:

[goom_ssm_rnn.py](goom_ssm_rnn.py)


## Citing

TODO: Update citation.

```
@misc{heinsenkozachkov2025gooms,
    title={
        Generalized Orders of Magnitude for
        Scalable, Parallel, High-Dynamic-Range Computation},
    author={Franz A. Heinsen, Leo Kozachkov},
    year={2025},
}
```


## Notes

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the complex plane. Our casual conversations gradually evolved into the development of generalized orders of magnitude, along with an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan.

We hope others find our work and our code useful.
