"""Makes scalar conditioner for conditional normalizing flow model."""

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk

from typing import Sequence, Optional

ACTIVATION_FUNCTIONS = {
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jax.nn.tanh,
    "softplus": jax.nn.softplus,
    "selu": jax.nn.selu,
    "gelu": jax.nn.gelu,
}

class GLU(hk.Module):
    def __init__(self, input_dim: int, condition_dim: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.fc = hk.Linear(input_dim)
        self.gate = hk.Linear(input_dim)

    def __call__(self, x: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
        combined_input = jnp.concatenate([x, condition], axis=-1)
        fc_output = self.fc(combined_input)
        gate_output = self.gate(combined_input)
        return fc_output * jax.nn.sigmoid(gate_output)


def scalar_conditioner_mlp(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    resnet: bool = True,
) -> hk.Module:
    class ScalarConditionerModule(hk.Module):
        def __call__(self, x):
            """z represents the conditioned values."""
            x = hk.Flatten()(x)
            if resnet:
                for i, hidden in enumerate(hidden_sizes):
                    x_temp = hk.nets.MLP([hidden], activate_final=True)(x)
                    if i > 0:
                        x += x_temp
                        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
                    else:
                        x = x_temp
                        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            else:
                x = hk.nets.MLP(hidden_sizes, activate_final=True)(x)

            x = hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            )(x)
            x = hk.Reshape(
                tuple(event_shape) + (num_bijector_params,), preserve_dims=-1
            )(x)
            return x

    return ScalarConditionerModule()


def conditional_scalar_conditioner_mlp(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    standardize_theta: bool = False,
    resnet: bool = True,
    activation: str = "relu",
    dropout_rate: float = 0.0,
) -> hk.Module:
    class ScalarConditionerModule(hk.Module):
        def __init__(self, name=None):
            super().__init__(name=name)
            if activation in ACTIVATION_FUNCTIONS:
                self.activation_func = ACTIVATION_FUNCTIONS[activation]
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            self.event_shape = event_shape
            self.hidden_sizes = hidden_sizes
            self.num_bijector_params = num_bijector_params
            self.standardize_theta = standardize_theta
            self.resnet = resnet
            self.dropout_rate = dropout_rate

        def __call__(self, x, theta, xi):
            """z represents the conditioned values."""
            if standardize_theta:
                raise NotImplementedError("lol standardizing this way is a bad idea.")
            theta = hk.Flatten()(theta)
            x = hk.Flatten()(x)
            xi = hk.Flatten()(xi)
            z = jnp.concatenate((x, theta, xi), axis=1)

            if resnet:
                for i, hidden in enumerate(hidden_sizes):
                    z_temp = hk.Linear(hidden)(z)
                    z_temp = self.activation_func(z_temp)
                    if self.dropout_rate > 0.0:
                        z_temp = hk.dropout(hk.next_rng_key(), self.dropout_rate, z_temp)
                    if i > 0:
                        z += z_temp
                        z = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(z)
                    else:
                        z = z_temp
                        z = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(z)
            else:
                for i, hidden in enumerate(hidden_sizes):
                    z = hk.Linear(hidden)(z)
                    z = self.activation_func(z)
                    if self.dropout_rate > 0.0:
                        z = hk.dropout(hk.next_rng_key(), self.dropout_rate, z)
                    z = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(z)

            z = hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal"),
                b_init=hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal"),
            )(z)
            z = hk.Reshape(
                tuple(event_shape) + (num_bijector_params,), preserve_dims=-1
            )(z)
            return z

    return ScalarConditionerModule()

def conditional_scalar_conditioner_transformer(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    standardize_theta: bool = True,
) -> hk.Module:
    class TransformerConditionerModule(hk.Module):
        def __init__(self, name=None):
            super().__init__(name=name)
            self.model_dim = model_dim
            self.num_heads = num_heads
            self.num_layers = num_layers

        def __call__(self, x, theta, xi):
            # Optionally standardize theta
            if standardize_theta:
                theta_mean = jnp.mean(theta, axis=0, keepdims=True)
                theta_std = jnp.std(theta, axis=0, keepdims=True) + 1e-10
                theta = (theta - theta_mean) / theta_std

            # Flatten inputs
            theta = hk.Flatten()(theta)
            x = hk.Flatten()(x)
            xi = hk.Flatten()(xi)
            z = jnp.concatenate((theta, xi), axis=-1)

            # Embed inputs to model_dim
            embed = hk.Linear(self.model_dim)
            z = jax.nn.relu(embed(z))

            # Positional encoding (optional, depending on your data)
            # If you decide to use positional encodings, you'll need to implement or use a predefined function here.
            # z += positional_encoding(z.shape, self.model_dim)

            # Apply Transformer layers
            for _ in range(self.num_layers):
                z = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.model_dim // self.num_heads, model_size=self.model_dim)(z, z)
                # Consider adding Feed-forward layers, LayerNorm, and dropout here as needed.

            # Reshape output to match the bijector parameters
            output_size = np.prod(event_shape) * num_bijector_params
            z = hk.Linear(output_size)(z)
            z = hk.Reshape(tuple(event_shape) + (num_bijector_params,))(z)

            return z

    return TransformerConditionerModule()

def conditional_glu_mlp(
    event_shape: Sequence[int],
    hidden_sizes: Sequence[int],
    num_bijector_params: int,
    x_shape: Sequence[int],
    theta_shape: Sequence[int],
    xi_shape: Sequence[int],
    activation: str = "relu",
    dropout_rate: float = 0.0,
) -> hk.Module:
    class GLUConditionerModule(hk.Module):
        def __init__(self, name=None):
            super().__init__(name=name)
            if activation in ACTIVATION_FUNCTIONS:
                self.activation_func = ACTIVATION_FUNCTIONS[activation]
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            self.event_shape = event_shape
            self.hidden_sizes = hidden_sizes
            self.num_bijector_params = num_bijector_params
            self.dropout_rate = dropout_rate

            # Flattened dimensions
            self.x_dim = np.prod(x_shape)
            self.theta_dim = np.prod(theta_shape)
            self.xi_dim = np.prod(xi_shape)
            self.condition_dim = self.x_dim + self.xi_dim
            self.input_dim = self.theta_dim

            # Create layers
            self.layers = []
            self.glu_layers = []
            input_dim = self.input_dim

            for layer_size in self.hidden_sizes:
                # Linear layer
                layer = hk.Linear(layer_size)
                self.layers.append(layer)
                # GLU layer
                glu = GLU(layer_size, self.condition_dim)
                self.glu_layers.append(glu)
                input_dim = layer_size

            self.out_layer = hk.Linear(
                np.prod(self.event_shape) * self.num_bijector_params,
                w_init=hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal"),
                b_init=hk.initializers.Constant(0.0),
            )

        def __call__(self, x, theta, xi):
            x = hk.Flatten()(x)
            xi = hk.Flatten()(xi)
            theta = hk.Flatten()(theta)
            h = theta  # Input is theta
            glu_condition = jnp.concatenate([x, xi], axis=-1)
            assert glu_condition.shape[-1] == self.condition_dim, f"Expected condition dim {self.condition_dim}, got {glu_condition.shape[-1]}"
            assert h.shape[-1] == self.input_dim, f"Expected input dim {self.input_dim}, got {h.shape[-1]}"

            # Forward pass
            for layer, glu in zip(self.layers, self.glu_layers):
                h = layer(h)
                h = glu(h, glu_condition)
                h = self.activation_func(h)
                if self.dropout_rate > 0.0:
                    h = hk.dropout(hk.next_rng_key(), self.dropout_rate, h)

            z = self.out_layer(h)
            z = hk.Reshape(tuple(self.event_shape) + (self.num_bijector_params,), preserve_dims=-1)(z)
            return z

    return GLUConditionerModule()
