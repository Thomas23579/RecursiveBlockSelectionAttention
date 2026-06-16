"""
Hierarchical Sparse Attention – Monkey-patch for Gemma 4 E2B
=============================================================
Replaces the forward method of Gemma4TextAttention on global layers only.
Everything except the attention computation is identical to the original.
"""

import types
import torch
import gc

# Will be implemented separately
from .RecursiveBlockSelectionAttention import recursive_block_selection_attention
from .HierarchyCache import HierarchyCache
# from .findFullAttentionLayersGemma import get_global_layers


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
B = 8  # branching factor
C = 256 # C/B = 32
H = 4
N_FULL_ATTN = C * (B**(H-1))  # = 131072   for   B=8, C=256, H=4
# !!! _BLOCK_SIZE_DEFAULT is only active when `self.config.block_size` is not set !!!
_BLOCK_SIZE_DEFAULT = 512 # max number of elements per instance

_cache_init = lambda x: HierarchyCache(num_levels = H, active_context_size = C, branching_factor = B, device = x.device, dtype = x.dtype)

_ORANGE, _RESET = "\033[38;5;208m", "\033[0m"
_GIBI_BYTE = 1024**3

def _print_debug_dashboard(attn, hidden_states, position_embeddings, block_size):
    dev = hidden_states.device

    # process-wide cache instances (leak monitor)
    caches = [o for o in gc.get_objects() if isinstance(o, HierarchyCache)]
    cache_bytes = sum(c.vram_bytes() for c in caches)

    # device VRAM
    alloc     = torch.cuda.memory_allocated(dev)
    reserved  = torch.cuda.memory_reserved(dev)
    free, tot = torch.cuda.mem_get_info(dev)

    # attention geometry — everything derived from the projection weights
    n_batch, seq_len = hidden_states.shape[:2]
    hd      = attn.head_dim
    hidden  = attn.q_proj.weight.shape[1]      # in_features
    kv_out  = attn.k_proj.weight.shape[0]      # d_head * n_kv
    q_out   = attn.q_proj.weight.shape[0]      # d_head * n_q
    n_kv    = kv_out // hd
    n_q     = q_out // hd
    groups  = n_q // max(n_kv, 1)
    cos, sin = position_embeddings

    lines = [
        "+- LogAttention debug --------------------------------",
        f"| VRAM: {alloc/_GIBI_BYTE:.1f}GiB ({reserved/_GIBI_BYTE:.0f}GiB)   free: {free/_GIBI_BYTE:.0f}GiB / {tot/_GIBI_BYTE:.0f}GiB    Cache {cache_bytes/_GIBI_BYTE:.2f}GiB ({len(caches)})",
        f"| n_batch={n_batch}   seq_len={seq_len}   block_size={block_size}   hidden_dim={hidden}   kv_proj->{kv_out}   q_proj->{q_out}  head_dim={hd}   q_heads={n_q}   kv_heads={n_kv}   GQA_groups={groups}   rope cos/sin = {tuple(cos.shape)} / {tuple(sin.shape)}",
        "+------------------------------------------------------",
    ]
    print(_ORANGE + "\n".join(lines) + _RESET)

def gemma4_attn_forward(self, hidden_states, position_embeddings,
                        attention_mask, shared_kv_states,
                        past_key_values=None, apply_rotary_pos_emb=None, **kwargs):
    """Drop-in replacement for Gemma4TextAttention.forward on global layers."""
    # hidden_states: [batch, seq_len, hidden_dim=1536]
    # position_embeddings: (cos, sin) each [batch, seq_len, head_dim=512]
    # attention_mask: [batch, 1, q_len, kv_len] or None
    # shared_kv_states: dict[int, tuple[Tensor, Tensor]]

    block_size = self.config.block_size if hasattr(self.config, 'block_size') else _BLOCK_SIZE_DEFAULT

    # `past_key_values._hierarchy_state` is a workaround to remember the hierarchy cache.
    # When the cache `past_key_values` is cleared, the attribute `_hierarchy_state` is reset as well.
    # Consequently, the `_hierarchy_state`'s livecycle is tied to the `past_key_values` cache.
    # As long as the `past_key_values` cache persists, also the hierarchical cache persists.
    if past_key_values is None:
        print("\033[38;5;208m" + "Unexpected: `past_key_values` is None!" + "\033[0m")
    elif not hasattr(past_key_values, "_hierarchy_state"):
        torch.cuda.empty_cache() # prevents accumulating unused allocations
        _print_debug_dashboard(self, hidden_states, position_embeddings, block_size)

    # retrieve caches field
    cache_idx = self.kv_shared_layer_index if self.is_kv_shared_layer else self.layer_idx
    if past_key_values is None:
        cache = _cache_init(hidden_states) # but, can't store it anywhere
    elif not hasattr(past_key_values, "_hierarchy_state"):
        cache = _cache_init(hidden_states)
        caches = {cache_idx: cache}
        past_key_values._hierarchy_state = caches
    else:
        cache = past_key_values._hierarchy_state.get(cache_idx)
        if cache is None:
            cache = _cache_init(hidden_states)
            past_key_values._hierarchy_state[cache_idx] = cache

    cos, sin = position_embeddings

    # the sequence is tiled into chunks to prevent allocation of large tensors (which could OOM the GPU)
    n_batch, seq_len = hidden_states.shape[:-1]
    block_size_seq = block_size//n_batch
    attn_output = torch.zeros((n_batch, seq_len, self.config.hidden_size), device=hidden_states.device, dtype=hidden_states.dtype) # (n_batch, seq_len, config.hidden_size)
    for offset in range(0, seq_len, block_size_seq):
        hidden_states_block = hidden_states[:,offset:min(offset+block_size_seq, seq_len), :] # (n_batch, block_size, hidden_dim)

        input_shape = hidden_states_block.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos_block = cos[:,offset:min(offset+block_size_seq, seq_len), :]
        sin_block = sin[:,offset:min(offset+block_size_seq, seq_len), :]

        # --- Q projection (identical to original) ---
        query_states = self.q_proj(hidden_states_block).view(hidden_shape) # [batch, seq_len, 8, 512]
        query_states = self.q_norm(query_states) # [batch, seq_len, 8, 512]
        if apply_rotary_pos_emb is not None:
            query_states = apply_rotary_pos_emb(query_states, cos_block, sin_block, unsqueeze_dim=2) # [batch, seq_len, 8, 512]
        query_states = query_states.transpose(1, 2) # [batch, 8, seq_len, 512]
        # query_states is (n_batch, n_head, seq_len, d_head)

        if not self.is_kv_shared_layer:
            # --- KV projection
            key_states = self.k_proj(hidden_states_block).view(hidden_shape)  # [batch, seq_len, 1, 512]
            value_states = self.v_proj(hidden_states_block).view(hidden_shape) if self.v_proj is not None else key_states  # [batch, seq_len, 1, 512]

            key_states = self.k_norm(key_states)  # [batch, seq_len, 1, 512]
            if apply_rotary_pos_emb is not None:
                key_states = apply_rotary_pos_emb(key_states, cos_block, sin_block, unsqueeze_dim=2)  # [batch, seq_len, 1, 512]
            key_states = key_states.transpose(1, 2)  # [batch, 1, seq_len, 512]

            value_states = self.v_norm(value_states)  # [batch, seq_len, 1, 512]
            value_states = value_states.transpose(1, 2)  # [batch, 1, seq_len, 512]

            #   key_states is (n_batch, n_head_kv, seq_len, d_head) # shared KV over heads
            # value_states is (n_batch, n_head_kv, seq_len, d_head) # shared KV over heads
            cache.update(key_states, value_states)

        # --- Recursive Block-Selection Attention (replaces attention_interface call) ---
        # query must be  (n_batch, n_head, seq_len, d_head)
        attn_output_block = recursive_block_selection_attention(query_states, cache, self.scaling, B, C) # (n_batch, n_head, seq_len, d_head)
        #attn_output = normal_attention(query_states, cache, self.scaling, B, C) # (n_batch, n_head, seq_len, d_head)
        attn_output_block = attn_output_block.transpose(1,2)  # (n_batch, seq_len, n_head, d_head)

        # --- Output projection (identical to original) ---
        attn_output_block = attn_output_block.reshape(*input_shape, -1).contiguous()
        attn_output_block = self.o_proj(attn_output_block) # (n_batch, seq_len, config.hidden_size)
        # print(f"Attention output: {attn_output_block.shape}")
        attn_output[:,offset:min(offset+block_size_seq, seq_len), :] = attn_output_block

    attn_weights = None
    return attn_output, attn_weights


def clear_hierarchy_cache(cache):
    """Call before each new forward pass / generation to reset cached hierarchies."""
    cache.clear()

def _get_global_layers(model, debug=False):
    full_attention_layers = []
    lines = []
    layers = model.get_submodule("model.language_model.layers")
    for i, layer in enumerate(layers):
        if type(layer).__name__ != "Gemma4TextDecoderLayer":
            continue

        attn = layer.self_attn
        if attn.layer_type == "full_attention":
            full_attention_layers.append(attn.layer_idx)

        if debug:
            num_key_value_heads = (
                attn.config.num_global_key_value_heads
                if attn.use_alternative_attention
                else attn.config.num_key_value_heads
            )
            lines.append(f"Layer: #{i} (id: {attn.layer_idx})")
            lines.append(f"   layer_type: {attn.layer_type}")
            lines.append(
                f"   is_sliding: {attn.is_sliding}"
                + (f" --- sliding_window: {attn.sliding_window}" if attn.is_sliding else "")
            )
            lines.append(f"   head_dim: {attn.head_dim}")
            lines.append(f"   num_key_value_groups: {attn.num_key_value_groups} (i.e. how often is one KV projection reused)")
            lines.append(f"   config.num_attention_heads: {attn.config.num_attention_heads} (number of query heads)")
            lines.append(f"   ~num_key_value_heads: {num_key_value_heads} (number of KV projections)")
            lines.append(f"   use_alternative_attention: {attn.use_alternative_attention}")
            lines.append(f"   scaling: {attn.scaling}")
            lines.append(
                f"   is_kv_shared_layer: {attn.is_kv_shared_layer}"
                + (f" --- kv_shared_layer_index: {attn.kv_shared_layer_index}" if attn.is_kv_shared_layer else "")
            )
            lines.append("\n")

    if debug:
        lines.append(f"Full Attention Layers: {full_attention_layers}")

    return full_attention_layers, "\n".join(lines)

def patch_global_attention(model):
    """Monkey-patch all global attention layers to use hierarchical sparse attention."""
    global_layers, _ = _get_global_layers(model)
    for idx in global_layers:
        attn = model.get_submodule(f"model.language_model.layers.{idx}.self_attn")
        attn.forward = types.MethodType(gemma4_attn_forward, attn)
    print(f"Patched global attention layers: {global_layers}")

def unpatch_global_attention(model):
    """Restore original forward methods on all global attention layers."""
    global_layers, _ = _get_global_layers(model)
    # Deleting the instance method reveals the class method underneath
    for idx in global_layers:
        attn = model.get_submodule(f"model.language_model.layers.{idx}.self_attn")
        if "forward" in attn.__dict__:
            del attn.__dict__["forward"]
    clear_hierarchy_cache()
    print(f"Restored original attention on layers: {global_layers}")