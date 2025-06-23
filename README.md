# goom_ssm_rnn

Reference implementation of a deep RNN that captures dependencies with a non-diagonal state-space model (SSM) over [generalized orders of magnitude](https://github.com/glassroom/generalized_orders_of_magnitude) (GOOMs), executable in parallel via a prefix scan, allowing recurrent state magnitudes to fluctuate freely over a greater dynamic range of real values than previously possible.


## Installing

1. Clone this repository.

2. Install the Python dependencies in `requirements.txt`.

3. There is no third step.


## Instantiating the RNN

The following code instantiates a small RNN for generative language modeling with GPT-2's vocabulary: 

```python
import torch
import tiktoken
import goom_ssm_rnn

DEVICE = 'cuda'  # change as needed

# Get GPT-2 encoder:
enc = tiktoken.get_encoding('gpt2')

# Instantiate an RNN for generative language model with GPT-2 token ids:
model = goom_ssm_rnn.GenerativeRNN(vocab_sz=enc.n_vocab, d_emb=512, n_hid=16, d_hid=32, n_res=8)

# Move model to cuda device:
model.to(device=DEVICE)

# You must provide the training code.
```

The RNN model is implemented as a standard PyTorch module that you can train and test on any task, using conventional techniques, including autocasting, as with any other PyTorch module. However, the model can be only partially compiled, because at present PyTorch's compiler does not yet fully support complex-typed tensors. The recurrent layers capture sequential dependencies with an SSM over GOOMs, which are represented by torch.complex64 tensors. As we explain in our paper, the GOOMs computed by each recurrent layer can fluctuate freely over a dynamic range of values that is not representable by torch.float32 or torch.float64, requiring that we scale the GOOMs before exponentiating them to real values represented by float tensors.

Besides the standard PyTorch `forward()` method, the model provides three nice-to-have methods, for convenience:

* `model.get_param_groups()`, which accepts a scalar weight_decay value as input, and returns two parameter groups for training, one with weight decay and one without. 

* `model.compute_loss_and_metrics()`, which accepts predicted scores over the model's vocabulary and true token ids, and returns a cross-entropy loss and a dictionary with useful metrics.

* `model.generate()`, for generating token ids after the model has been trained on a language generation task.


## Modifying the RNN for Other Tasks

You can modify or replace the model's embedding layer and/or head, as needed, for tasks other than generative language model. All model components are defined in a single file, [goom_ssm_rrn.py](goom_ssm_rrn.py), for your convenience.


## Training the RNN

We successfully trained this RNN model, and variants of it, on several toy tasks, including Wikitext-103 (using the GPT-2 vocabulary), Sequential MNIST generation (using a vocabulary size of 256 gray levels per pixel-token), and Sequential MNIST classification (replacing the language modeling head with a linear classification head that predicts 10 classes from the last pixel-token), and a simple Copy Memory task.

For all tasks, we instantiated the model with 512 embedding dimensions (`d_emb=512`), 16 heads per token (`n_hid=16`), 32 features per head (`d_hid=32`), and eight residual recurrent layers (`n_res=8`). For all tasks, we trained the model on a recent mid-tier Nvidia GPU, with a single-cycle learning schedule, and the following training hyper-parameters:

* optimizer: AdamW.
* maximum learning rate: 3e-4.
* minimum learning rate: 1e-5.
* maximum momentum rate: 0.99.
* minimum momentum rate: 0.85.
* batch size: 1000, split into the largest micro-batch size that fits in memory.
* warm-up period: 10 batches (10,000 samples).

We found that the model trains to competitive performance on all tasks we tested.


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
