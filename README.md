This repo contains the code for generating and using the AF4 code as described in (TODO).

### Generating codes
The script `generate_af4.py` will produce the AF4 code for a variety of block sizes. It's really slow to run since a ton of numerical integration is needed, so you may want to swap in the approximate CDF as described in the appendix.

### JAX transform
I used a modified version of the quantization code from [my GPTQ implementation](https://github.com/davisyoshida/jax-gptq) for the experiments I reported.
I'll post a usable version here once I clean it up and disentangle it from the GPTQ stuff.
