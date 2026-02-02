import dataclasses 
import math
from typing import Optional, Tuple,

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import PartitionSpec as P

#--- 1. CONFIGURATION ---#

@dataclasses.dataclass(slots=True, frozen=True)
class ShardingCfg:
	"""Controls how the model weights are distributed across the devices."""
	emb_sharding: P = P(None,None)
	qkv_kernel: P = P(None,None)
	o_kernel: P = P(None,None)
	mlp_kernel: P = P(None,None)
	activation: P = P(None,None)

	@staticmethod
	def no_sharding():
		return ShardingCfg()

	@staticmethod
	def default():
		return ShardingCfg(
			emb_sharding=P("tp","fsdp"),
			qkv_kernel=P("tp","fsdp"),
			q_kernel=P("tp","fsdp"),
			mlp_kernel=P("tp","fsdp"),
			activation=P("fsdp","tp"))

@dataclasses.dataclass(frozen=True)
class BartConfig:

	#--- 1. Model Architecture ---
	vocab_size: int = 50265
	d_model: int = 1024
	encoder_layers: int = 12
	decoder_layers: int = 12
	encoder_attention_heads: int = 16
	decoder_attention_heads: int = 16
	encoder_ffn_dim: int = 4096
	decoder_ffn_dim: int = 4096
	activation_function: str = "gelu"
	
	#--- 2. Regularization ---
	dropout: float = 0.1
	attention_dropout: float = 0.1
	activation_dropout: float = 0.1
	classifier_dropout: float = 0.0

	#--- 3. Initialization --- 
	init_std: float = 0.2

	#--- 4. Position & Embeddings ---
	max_position_embeddings: int = 1024
	scale_embedding: bool = False
	tie_word_embeddings: bool = True

	#--- 5. Token IDs ---
	pad_token_id: int = 1
	bos_token_id: int = 0
	eos_token_id: int = 2
	decoder_start_token_id: int = 2

	#--- 6. Metadata & Task Config ---
	is_encoder_decoder: bool = True
	num_labels: int = 3
	use_cache: bool = True

	#--- 7. JAX Sharding ---
	shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

	@classmethod
	def bart_base(cls):
		return cls(
			d_model = 768,
			encoder_layers = 6,
			decoder_layers = 6,
			encoder_attention_heads = 12,
			decoder_attention_heads = 12,
			encoder_ffn_dim = 3072,
			decoder_ffn_dim = 3072,
			scale_embedding = False
			)

#--- 2. HELPER LAYERS ---#
