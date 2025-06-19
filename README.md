# goom_ssm_rnn

TODO: Write a brief introduction to the RNN.

## Installation

TODO


## Instantiating the Model

```python
import torch
import goom_ssm_rnn

model = goom_ssm_rnn.ModelClass(vocab_sz=256, d_emb=512, n_hid=16, d_hid=32, n_res=8)
```

## Training the Model

TODO: Describe training hyper-parameters for one task (e.g., MNIST generation pixel by pixel, or Wikitext-103), and provide guidance for other model variants and tasks.


## Background

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the domain of complex logarithms. Our casual conversations gradually evolved into the development of generalized orders of magnitude, an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan. We hope others find our work and our code useful.
