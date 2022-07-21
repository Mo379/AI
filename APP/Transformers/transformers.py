from typing import NamedTuple, Optional, Any, Mapping
from dataclasses import dataclass
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import functools
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclass
class LanguageDataset:
    records: tf.data.dataset
    vocab_size: int


def load(batch_size: int, sequence_length: int) -> LanguageDataset:
    """Load LM1B dataset, returning it and the vocab_size."""
    dataset, dataset_info = tfds.load(
            'lm1b/subwords32k',
            split=tfds.Split.TRAIN,
            shuffle_files=True,
            with_info=True
            )
    crop_size = sequence_length + 1
    dataset = dataset.repeat()
    # Convert the dataset to constant-size int32 tensors.
    dataset = dataset.map(lambda d: tf.cast(d['text'], tf.int32))
    dataset = dataset.map(lambda t: _crop_or_pad(t, crop_size, pad_token=0))
    dataset = dataset.shuffle(batch_size * 10)
    # Create the language modeling observation/target pairs and batch them up.
    dataset = dataset.map(lambda t: dict(obs=t[:-1], target=t[1:]))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = iter(tfds.as_numpy(dataset))
    return LanguageDataset(
            dataset, dataset_info.features['text'].encoder.vocab_size
            )


def _crop_or_pad(value, size, pad_token):
    """Either crop or pad value to be of size size."""
    val_size = tf.size(value)
    pad = lambda: tf.pad(  # pylint: disable=g-long-lambda
        value, [[0, size - val_size]],
        'CONSTANT',
        constant_values=pad_token)
    return tf.cond(val_size < size, pad, lambda: value[:size])

# ##############   MODEL ##########################


# self attention layer class
class SelfAttention(hk.MultiHeadAttention):
    """Self attention with a casual mask applied"""

    def __call__(
            self,
            query: jnp.ndarray,
            key: Optional[jnp.ndarray] = None,
            value: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        key = key if not None else query
        value = value if not None else query
        sequence_length = query.shape[0]
        casual_mask = np.trill(np.ones((sequence_length, sequence_length)))
        mask = mask * casual_mask if mask is not None else casual_mask
        return super().__call__(query, key, value, mask)


# dense layer class
class DenseBlock(hk.Module):
    """A 2-layer MLP"""

    def __init__(
            self,
            init_scale: float,
            widening_factor: int = 4,
            name: Optional[str] = None
    ):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initialiser = hk.intilizers.VarianceScaleing(self.init_scale)
        x = hk.Linear(
                self._widening_factor * hiddens,
                w_init=initialiser
            )(x)
        x = jax.nn.relu(x)
        x = hk.Linear(
                hiddens,
                w_init=initialiser
            )(x)
        x = jax.nn.relu(x)
        return x


# Normalisation layer
def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique layer normilisation to x with default settings"""
    return hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name=name
            )(x)


# Transformer class
class Transformer(hk.Module):
    """A transformer stack."""

    def __init__(
            self,
            num_heads: int,
            num_layers: int,
            dropout_rate: float,
            name: Optional[str] = None,
            ):
        super.__init__(name=name)
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate
        pass

    def __call__(
            self,
            h: int,
            mask: Optional[jnp.ndarray],
            is_training: bool
            ) -> jnp.ndarray:
        """ Connect and call the transformer
        Args:
            h: Inputs, [B,T,H].
            mask: Padding mask, [B,T].
            is_training: wheather we're training or not.
        Returns:
            Array of shape [B,T,H].
        """

        init_scale = 2.0/self._num_layers
        dropout_rate = self._dropout_rate if is_training else 0.0
        if mask is not None:
            mask = mask[:, None, None, :]
        # Layers loop
        for i in range(self._num_layers):
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            h_attn = SelfAttention(
                    num_heads=self._num_heads,
                    key_size=64,
                    w_init_scale=init_scale,
                    name=f'h{i}_attn'
                )(h_norm, mask=mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_attn
        h = layer_norm(h, name='ln_fn')
        return h

# ######################################## TRAIN ##########################


batch_size = 16  # Train batch size per core
sequence_length = 128  # Sequence length to learn on

d_model = 256  # model width
num_heads = 4  # Number of attention heads
num_layers = 6  # Number of transformer layers
dropout_rate = 0.1  # Dropout rate

learning_rate = 2e-4  # Max learning-rate
grad_clip_value = 0.25  # Gradient norm clip value

checkpoint_dir = '/jax-transformer'  # Directory to store checkpoints
LOG_EVERY = 50
MAX_STEPS = 10 ** 6


# Embedding layer
def embeddings(data: Mapping[str, jnp.ndarray], vocab_size: int):
    tokens = data['obs']
    input_mask = jnp.grater(tokens, 0)
    seq_length = tokens.shape[1]

    # embed the input tokens and positions
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
    token_embs = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter(
                'pos_embs',
                [seq_length, d_model],
                init=embed_init
            )
    input_embeddings = token_embs + positional_embeddings
    return input_embeddings, input_mask


# Forward function
def build_forward_fn(
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float
        ):
    """Constructs the model's forward pass."""
    def forward_fn(
            data: Mapping[str, jnp.ndarray],
            is_training: bool = True
            ) -> jnp.ndarray:
        """Forward pass."""
        input_embeddings, input_mask = embeddings(data, vocab_size)

        # Run the transformer over the inputs
        transformer = Transformer(
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_rate=dropout_rate
                )
        output_embeddings = transformer(
                input_embeddings,
                input_mask,
                is_training
                )
        # Reverse the embeddings (untied).
        return hk.Linear(vocab_size)(output_embeddings)
    return forward_fn


# Loss function
def lm_loss_fn(
        forward_fn,
        vocab_size: int,
        params,
        rng,
        data: Mapping[str, jnp.ndarray],
        is_training: bool = True
        ) -> jnp.ndarray:
    """Computes the loss on data give the params."""
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['targets'], vocab_size)
    assert logits.shape == targets.shape
    mask = jnp.greater(data['obs'], 0)
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask) / jnp.sum(mask)
    return loss


# defining the gradient updater
class GradientUpdater:
    """A stateless obstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """
    def __init__(
            self,
            net_init,
            loss_fn,
            optimizer: optax.GradientTransformation
            ):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        """Initilizes the state of the updater"""
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
                step=np.array(0),
                rng=out_rng,
                opt_state=opt_state,
                params=params
                )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(
            self,
            state: Mapping[str, Any],
            data: Mapping[str, jnp.ndarray]
            ):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)
        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_update(params, updates)
        new_state = {
                'step': state['step'] + 1,
                'rng': rng,
                'opt_state': opt_state,
                'params': params
                }
        metrics = {
                'step': state['step'],
                'loss': loss
                }
        return new_state, metrics


# Main training loop
def main():
    # Create the dataset.
    training_dataset, vocab_size = load(batch_size, sequence_length)
    # Setup the model, loss and updater.
    forward_fn = build_forward_fn(
            vocab_size,
            d_model,
            num_heads,
            num_layers,
            dropout_rate
            )
    forward_fn = hk.transform(forward_fn)
    loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size)
    optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip_value),
            optax.adam(learning_rate, b1=0.9, b2=0.99)
            )
    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)
    # Initialise parameters
    rng = jax.random.PRNGKey(0)
    data = next(training_dataset)
    state = updater.init(rng, data)
    # Run loop
    # prev_time = time.time()
    for step in range(MAX_STEPS):
        print(f'step-{step}')
        data = next(training_dataset)
        state, metrics = updater.update(state, data)
        print(metrics)



