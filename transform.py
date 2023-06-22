"""
JAX code for absmax based blockwise quantization.This doesn't use a custom matmul kernel,
matrices are simply unpacked prior to use. This should be useful for ease of iteration
in quantization research, but shouldn't be used if latency is critical.
"""

from dataclasses import dataclass
from functools import partial
from itertools import chain

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

from qax import use_implicit_args, ImplicitArray, aux_field

@dataclass
class CodeQuantMatrix(ImplicitArray):
    int_weight : jax.Array
    absmaxes : jax.Array
    code : jax.Array

    block_size : int = aux_field()
    contraction_axis : int = aux_field()

    def materialize(self):
        return dequantize(
            int_weight=self.int_weight,
            absmaxes=self.absmaxes,
            code=self.code,
            block_size=self.block_size,
            contraction_axis=self.contraction_axis,
            orig_shape=self.shape,
            dtype=self.dtype,
        )


def put_axis_last(weight, axis):
    new_order = chain(
        range(axis),
        range(axis+1, weight.ndim),
        [axis],
    )

    return weight.transpose(*new_order)

def pack(arr):
    assert arr.ndim == 1
    return arr[::2] << 4 | arr[1::2]

def unpack(arr):
    high_bits = arr >> 4
    low_bits = arr & 0xF

    return jnp.stack([high_bits, low_bits], axis=-1).reshape(-1)

@partial(jax.jit, static_argnums=(2, 3))
def quantize(weight, code, block_size, contraction_axis):
    orig_shape = weight.shape
    dtype = weight.dtype

    transposed = put_axis_last(weight, contraction_axis)

    grouped = transposed.reshape(-1, block_size)
    absmaxes = jnp.max(jnp.abs(grouped), axis=1, keepdims=True)

    scaled = grouped / absmaxes

    assert scaled.ndim == 2
    code_vals = jnp.argmin(jnp.abs(scaled[..., None] - code), axis=-1).astype(jnp.uint8)

    int_weight = pack(code_vals.reshape(-1))

    return CodeQuantMatrix(
        int_weight=int_weight,
        absmaxes=absmaxes,
        code=code,
        block_size=block_size,
        contraction_axis=contraction_axis,
        shape=orig_shape,
        dtype=dtype,
    )

def dequantize(int_weight, absmaxes, code, block_size, contraction_axis, orig_shape, dtype):
    unpacked = unpack(int_weight)
    decoded = code[unpacked].reshape(-1, block_size).astype(dtype)

    unscaled = decoded * absmaxes

    transposed_shape = orig_shape[:contraction_axis] + orig_shape[contraction_axis + 1:] + (orig_shape[contraction_axis],)

    transposed = unscaled.reshape(transposed_shape)

    untranspose = chain(
        range(contraction_axis),
        [transposed.ndim - 1],
        range(contraction_axis, transposed.ndim - 1),
    )

    return transposed.transpose(*untranspose)

def quantize_params(param_tree, block_size, code, contraction_axis=0):
    skip_patterns = ['wte', 'wpe', 'lm_head', 'emb']
    bar = tqdm(total=jax.tree_util.tree_structure(param_tree).num_leaves, desc='Quantizing')
    def quantizer(path, param):
        bar.update()
        param = jax.device_put(param, jax.devices('gpu')[0])
        if param.ndim < 2:
            return param

        if any(
            isinstance(node, jax.tree_util.DictKey)
            and any(pattern in node.key for pattern in skip_patterns)
            for node in path
        ):
            return param
        return quantize(param, code, block_size, contraction_axis)

    qparams = jax.tree_util.tree_map_with_path(quantizer, param_tree)
    bar.close()
    return qparams

def main():
    model_name = 'EleutherAI/gpt-neo-2.7B'
    block_size = 32

    # Most Flax models do left multiplication (i.e. activations @ W), so the contractoin
    # axis is 0. For GPT-2 in particular it should be set to 1.
    contraction_axis = 0
    code = jnp.asarray(np.load(f'af4_{block_size}.npy'))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model, params = FlaxAutoModelForCausalLM.from_pretrained(model_name, _do_init=False)

    @jax.jit
    @use_implicit_args
    def call_model_with_quantized_params(params, *args):
        return model(*args, params=params)

    prefix = (
        'We introduce Adam, an algorithm for first-order gradient-based optimization'
        ' of stochastic objective functions, based on adaptive estimates of lower-order'
    )

    input_ids = jnp.asarray(tokenizer.encode(prefix))[None]

    quantized_params = quantize_params(params, block_size, code, contraction_axis)

    logits_out = call_model_with_quantized_params(quantized_params, input_ids).logits

    vals, indices = jax.lax.top_k(logits_out[0, -1], 10)
    print(f'Most likely next tokens:')
    for i, index in enumerate(indices, 1):
        token = tokenizer.decode(index)
        print(f'\t{i}. {token}')

if __name__ == '__main__':
    main()
