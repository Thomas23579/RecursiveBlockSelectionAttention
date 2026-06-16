import torch

class HierarchyCache():
    def __init__(self, num_levels = 4, active_context_size = 256, branching_factor = 8, device = "cuda", dtype = torch.bfloat16):
        self.num_levels = num_levels
        self.active_context_size = active_context_size
        self.branching_factor = branching_factor

        # index related variables
        self.capacity = active_context_size * (branching_factor ** (num_levels-1))
        self.current = 0  # the next overwritten target
        self.n_valid = 0

        # device and dtype
        self.device = device
        self.dtype = dtype

        # levels is dict[int, list[torch.Tensor]] with tensors (n_batch, n_head_kv, page_size, head_dim)
        self.levels: list[torch.Tensor | None] = []
        # values is list[torch.Tensor] with tensors (n_batch, n_head_kv, page_size, head_dim)
        self.values: torch.Tensor | None = None
        self.clear_cache()# instantiates the caches

    def _as_exclusive_index(self, i):
        return 1 + ((i - 1) % self.capacity)

    def clear_cache(self):
        self.values = None
        self.levels = [None] * self.num_levels
        self.current = 0
        self.n_valid = 0

    def wrap(self, n):
        return (self.current + n) % self.capacity

    def current_as_end(self):
        return self._as_exclusive_index(self.current)

    def _incr_index(self, n=1):
        self.current = (self.current + n) % self.capacity
        self.n_valid = min(self.n_valid + n, self.capacity)
        return self.current

    def combineKeys(self, keys):
        # keys is (n_batch, n_head_kv, L//B, B, d_head)
        # combine along dim=3
        return torch.sum(keys, dim=3, keepdim=False)

    def update(self, keys, values):
        # keys (n_batch, n_head_kv, n_keys, d_head)
        # values (n_batch, n_head_kv, n_keys, d_head)
        assert keys.shape == values.shape, f"Keys and values must have the same shape: {keys.shape} vs. {values.shape}"
        n_keys = keys.shape[-2]
        assert n_keys > 0, "Given empty Key--Value pair. There must be at least one element."

        if n_keys == 1:
            self._single_update(keys[:,:,0,:], values[:,:,0,:])
            return

        if n_keys >= self.capacity:
            keys = keys[:, :, -self.capacity:, :]
            values = values[:, :, -self.capacity:, :]
            n_keys = self.capacity
            self.current = 0
            self.n_valid = self.capacity

        start_index = self.current
        end_index = self._as_exclusive_index(self.current + n_keys)

        if start_index >= end_index: # wrap around!
            iterm_len = self.capacity - start_index # group size of the first block
            self._update_contiguous(keys[..., :iterm_len, :], values[..., :iterm_len, :])
            self._update_contiguous(keys[..., iterm_len:, :], values[..., iterm_len:, :])
        else:
            self._update_contiguous(keys, values)

    def _update_contiguous(self, keys, values):
        # It is guaranteed that the group fits into the cache without wrapping around
        n_batch, n_head_kv, n_keys, d_head = keys.shape

        assert n_keys > 0, "Number of keys should be greater than 0"

        B = self.branching_factor
        H = self.num_levels
        C = self.active_context_size

        inx_cache_0 = self.current
        # end_index = self._as_exclusive_index(self.current + n_keys)
        self._incr_index(n_keys)

        composite_keys = keys
        for s in range(H):
            len_keys_s = composite_keys.shape[2]  # sequence length of composite keys in the current level
            inx_cache_s_start = inx_cache_0 // (B**s) # index in the current level of the cache
            inx_cache_s_end_incl = inx_cache_s_start+len_keys_s-1 # last index (inclusive) in the current level of the cache

            if self.levels[s] is None:
                L = C * (B ** (H-1-s))
                self.levels[s] = torch.zeros((n_batch, n_head_kv, L, d_head), device = self.device, dtype = self.dtype)
            self.levels[s][:,:,inx_cache_s_start:inx_cache_s_end_incl+1,:] = composite_keys

            if s==0: # also set values
                if self.values is None:
                    self.values = torch.zeros((n_batch, n_head_kv, self.capacity, d_head), device = self.device, dtype = self.dtype)
                self.values[:,:,inx_cache_s_start:inx_cache_s_start+len_keys_s,:] = values

            # prepare next iteration
            if s < H-1: # not necessary on last layer
                inx_next_first = inx_cache_s_start // B
                inx_next_last_excl = (inx_cache_s_end_incl // B)+1
                composite_keys = self.combineKeys(self.levels[s].view(n_batch, n_head_kv, -1, B, d_head)[:,:,inx_next_first:inx_next_last_excl,:,:])

    def _single_update(self, key, value):
        # key (n_batch, n_head_kv, d_head)
        # value (n_batch, n_head_kv, d_head)

        inx_cache_0 = self.current
        self._incr_index()

        B = self.branching_factor
        H = self.num_levels
        C = self.active_context_size

        n_batch, n_head_kv, d_head = key.shape

        composite_key = key
        for s in range(H):
            inx_cache_s_start = inx_cache_0 // (B ** s)
            if self.levels[s] is None:
                L = C * (B ** (H - 1 - s))
                self.levels[s] = torch.zeros((n_batch, n_head_kv, L, d_head), device=self.device, dtype=self.dtype)
            self.levels[s][:, :, inx_cache_s_start, :] = composite_key

            if s == 0:  # also set values
                if self.values is None:
                    self.values = torch.zeros((n_batch, n_head_kv, self.capacity, d_head), device=self.device,
                                              dtype=self.dtype)
                self.values[:, :, inx_cache_s_start, :] = value

            # prepare next iteration
            if s < H - 1:  # not necessary on last layer
                start = (inx_cache_s_start // B) * B
                end = start + B
                composite_key = self.combineKeys(
                    self.levels[s][:, :, None, start:end, :]  # (n_batch, n_head_kv, 1, B, head_dim)
                ).squeeze(2)  # (n_batch, n_head_kv, head_dim)

    def retrieve(self, group_indices, s, n_rep, gather_values=False):
        # group_indices: (n_batch, n_head, seq_len, num_ret)  -- groups to fetch
        # level is (n_batch, n_head_kv, L, head_dim)
        # n_rep: query heads per kv head (n_head_kv * n_rep = n_head)
        # each group covers B contiguous children -> returns (n_batch, n_head, seq_len, num_ret*B, head_dim)
        level = self.levels[s]
        B = self.branching_factor

        n_batch, n_head_kv, L, head_dim = level.shape
        _, n_head, seq_len, num_ret = group_indices.shape

        # n_batch*n_head*seq_len*num_ret*B = 256*8*1 = 2048 * seq_len items
        indices = group_indices[..., None] * B + torch.arange(B, device=self.device)
        indices = indices.reshape(n_batch, n_head, seq_len, num_ret * B)  # (n_batch, n_head, seq_len, num_ret * B)

        # broadcast index tensors; map query head -> kv head for GQA (no repeat_kv)
        b_idx = torch.arange(n_batch, device=self.device).view(n_batch, 1, 1, 1) # n_batch items
        hkv_idx = (torch.arange(n_head, device=self.device) // n_rep).view(1, n_head, 1, 1) # n_head items

        # (n_batch, n_head, seq_len, num_ret*B, head_dim), (n_batch, n_head, seq_len, num_ret * B)
        if gather_values:
            assert s==0, "Can only gather values at level 0."
            return level[b_idx, hkv_idx, indices, :], indices, self.values[b_idx, hkv_idx, indices, :]
        else:
            return level[b_idx, hkv_idx, indices, :], indices, None

    def retrieve_top_level(self, n_rep, seq_len):
        # level is (n_batch, n_head_kv, L, head_dim)
        # n_rep: query heads per kv head (n_head_kv * n_rep = n_head)
        # each group covers B contiguous children -> returns (n_batch, n_head, seq_len, L, head_dim)

        level = self.levels[-1]
        n_batch, n_head_kv, L, head_dim = level.shape
        n_head = n_head_kv * n_rep

        indices = torch.arange(L, device=self.device)[None,None,None,:].expand(n_batch, n_head, seq_len, L)
        if self.num_levels == 1:
            return (level[:,:,None,None,:,:].expand(-1,-1,n_rep,seq_len,-1,-1).reshape(n_batch, n_head, seq_len, L, head_dim),
                    indices,
                    self.values[:,:,None,None,:,:].expand(-1,-1,n_rep,seq_len,-1,-1).reshape(n_batch, n_head, seq_len, L, head_dim))
        else:
            return level[:,:,None,None,:,:].expand(-1,-1,n_rep,seq_len,-1,-1).reshape(n_batch, n_head, seq_len, L, head_dim), indices, None

    def vram_bytes(self):
        """Total device bytes held by this cache's level + value tensors."""
        total = 0
        for lvl in self.levels:
            if lvl is not None:
                total += lvl.numel() * lvl.element_size()
        if self.values is not None:
            total += self.values.numel() * self.values.element_size()
        return total









