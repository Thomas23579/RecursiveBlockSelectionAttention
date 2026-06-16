import torch

def dense_attention(query, cache, scale, B, C):
    # query is (n_batch, n_head, seq_len, d_head)
    n_batch, n_head, seq_len, d_head = query.shape
    n_head_kv = cache.levels[-1].shape[1] # If this is 1, then the kv is shared among the heads!
    n_rep = n_head // n_head_kv

    N = cache.n_valid
    keys = cache.levels[0][:,:,:N,:] # (n_batch, n_head_kv, N, d_head)
    values = cache.values[:,:,:N,:] # (n_batch, n_head_kv, N, d_head)

    # match KV heads to query heads
    keys = keys[:,:,None,:,:].expand(-1, -1, n_rep, -1, -1).reshape(n_batch, -1, N, d_head) # (n_batch, n_head, N, d_head)
    values = values[:,:,None,:,:].expand(-1, -1, n_rep, -1, -1).reshape(n_batch, -1, N, d_head) # (n_batch, n_head, N, d_head)

    # query is (n_batch, n_head, seq_len, d_head)
    #   key is (n_batch, n_head, N, d_head)
    # value is (n_batch, n_head, N, d_head)
    # returns (n_batch, n_head, seq_len, d_head)
    return torch.nn.functional.scaled_dot_product_attention(query, keys, values, is_causal=True, scale=scale)

def recursive_block_selection_attention(query, cache, scale, B, C):
    n_batch, n_head, seq_len, d_head = query.shape
    n_head_kv = cache.levels[-1].shape[1]
    n_rep = n_head // n_head_kv

    survivor_indices = None

    n_valid = cache.n_valid
    index_start = cache.wrap(-seq_len)
    index_end = cache.current_as_end()
    N = cache.capacity
    if n_valid == 0:
        return torch.zeros_like(query)

    H = len(cache.levels)

    for s in range(H - 1, -1, -1):
        # Retrieve the composite keys for the current level
        if s == H - 1: # for the top layer use a view of the entire top-level composite-key cache
            composite_keys, indices, values = cache.retrieve_top_level(n_rep, seq_len) # (n_batch, n_head, seq_len, C, d_head)
        else: # On lower levels gather the surviving groups
            # level is (n_batch, n_head_kv, L^{s}, d_head)
            # survivor_indices is (n_batch, n_head, seq_len, C//B)
            composite_keys, indices, values = cache.retrieve(survivor_indices, s, n_rep, gather_values = s==0) # (n_batch, n_head, seq_len, C, d_head)

        # composite_keys is (n_batch, n_head, seq_len, C, d_head)
        # indices is (n_batch, n_head, seq_len, C)

        # Build mask:
        # invalid is (n_batch, n_head, seq_len, C)
        invalid = torch.zeros_like(indices, dtype=torch.bool)
        # (1) Buffer-not-full mask
        if n_valid < N:
            invalid |= indices > (n_valid - 1) // (B ** s)
        # (2) Future-tokens mask (causal at group granularity)
        upper = (index_end - 1) // (B ** s) + 1
        if index_end > index_start:
            lower = torch.arange(index_start, index_end, device=query.device)[None, None, :, None] // (B ** s)
            invalid |= (indices > lower) & (indices < upper)
        else:  # wrap around
            lower = torch.arange(index_start, index_start + seq_len, device=query.device)[None, None, :, None] // (B ** s)
            lower_wrapped = torch.arange(index_end - seq_len, index_end, device=query.device)[None, None, :, None] // (B ** s)
            invalid |= (indices > lower) | ((indices > lower_wrapped) & (indices < upper))
        mask = torch.zeros_like(indices, dtype=query.dtype)
        mask.masked_fill_(invalid, float('-inf'))


        # Attention scores and output
        #            query is (n_batch, n_head, seq_len, d_head)
        #   composite_keys is (n_batch, n_head, seq_len, C, d_head)
        #           values is (n_batch, n_head, seq_len, C, d_head)
        #             mask is (n_batch, n_head, seq_len, C)
        #             returns (n_batch, n_head, seq_len, d_head)

        # If on the final level then compute the softmax attention directly
        if s==0: # in this branch is the actual return path!
            return (torch.softmax((query.unsqueeze(-2) @ composite_keys.transpose(-2, -1)) * scale + mask.unsqueeze(-2), dim=-1) @ values).squeeze(-2)
            # out: (n_batch, n_head, seq_len, d_head)

        attn_score = (composite_keys @ query.unsqueeze(-1)).squeeze(-1) + mask  # (n_batch, n_head, seq_len, C)

        # top-k attention scores
        _, idx = torch.topk(attn_score, C//B, dim=-1, sorted=False) # indexed wrt survivor_indices and not keys
        # map to the indices of keys
        survivor_indices = torch.gather(indices, dim=-1, index=idx) # (n_batch, n_head, seq_len, C//B)
    assert False, "unreachable: loop must return at s=0" # I'll push my luck here


"""
About writing all of this into a triton kernel:
What's in the kernel: The entire loop

One tile: batch_idx, head_idx, seq_idx
    this means one tile is
        d_head*C*2 for keys
        d_head*C*2 for values
        d_head*2 for the query
        C*4 for indices
        C*1 for invalid (for mask)
        C*4 for lower
        C*4 for lower_wrapped
        C*2 for mask
        C*2 for attn_score
        C//B * 4 for survivor_indices
        C//B * 4 for idx
        
    Concurrently in shared memory:
    * indices, invalid, lower, mask           => C*(4 + 1 + 4 + 2) = C*11 Bytes
    * indices, keys, query, attn_score        => C*(4 + 2*d_head + 2) + 2*d_head = C*(2*d_head + 6) + 2*d_head
    * indices, attn_score, mask               => C*(4 + 2 + 2) = 8*C
    * indices, attn_score, values, out        => C*(4 + 2 + 2*d_head + 2) = C*(2*d_head + 8)
    * indices, attn_score, idx                => C*(4 + 2 + 4/B) = C*(6 + 4/B)
    * indices, idx, survivor_indices          => C*(4 + 2*4/B) = C*(4 + 8/B)
    Max: 2*(C*(d_head + 3) + d_head) = 2*d_head*(C+1) + 6*C   with d_head=512 and C=256 => 265216 B = 258.5 KiB (more than H200 can offer)
    
    => MUST TILE ALONG C DIRECTION
       The smallest unit becomes: 1 batch_idx, 1 head_idx, 1 seq_idx, 1 survivor_idx
       Optimization then chooses n_batch_idx, n_head_idx, n_seq_idx, n_survivor_idx
       But prefer to have all survivors in one tile!
"""
