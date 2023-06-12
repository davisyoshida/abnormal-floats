This repo contains the code for generating and using the AF4 code as described in (TODO).

### Generating codes
The script `generate_af4.py` will produce the AF4 code for a variety of block sizes. It's really slow to run since a ton of numerical integration is needed, so you may want to swap in the approximate CDF as described in the appendix.

### JAX transform
I've put helper code for doing absmax quantization with 4-bit codes in `transform.py`. It doesn't use a custom matmul kernel, so it will be a few times slower than solutions which do.
On the othe hand it's extremely easy to use and modify if you're interested in messing around with different types of quantization.
