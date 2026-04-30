import torch


class GraphCache:
    """
    Manages graph feature caching on target device to speed up training.
    Mimics PaGraph's caching strategy by storing features of high-degree nodes on accelerator.
    """

    def __init__(self, data, device, percent=None):
        """
        Args:
            data (Data): PyG Data object containing x (features) and edge_index.
            device (torch.device): Device to cache features on.
            percent (float, optional): Percentage of nodes to cache (0.0 to 1.0).
                                     If None, defaults to 0.0 (disabled) unless explicitly set.
        """
        self.device = device
        self.num_nodes = data.num_nodes

        # Ensure cpu_x is pinned for faster transfer
        self.cpu_x = data.x
        if hasattr(self.cpu_x, "pin_memory") and not self.cpu_x.is_pinned():
            try:
                self.cpu_x = self.cpu_x.pin_memory()
            except Exception:
                pass

        self.feature_dim = self.cpu_x.size(1)
        self.cached_x = None
        self.node_to_cache_idx = None
        self.is_cached = None

        self.auto_cache(data, percent)

    def auto_cache(self, data, percent=None):
        # Determine Cache Size
        if percent is None:
            # We cannot reliably auto-detect memory without torch.cuda if we want to abide by non-cuda strictness.
            # Defaulting to 0.0 as per plan.
            percent = 0.0

        percent = max(0.0, min(1.0, percent))
        num_cache = int(self.num_nodes * percent)

        if num_cache == 0:
            print("[GraphCache] Caching disabled (0 nodes).")
            return

        print(
            f"[GraphCache] Caching {num_cache} / {self.num_nodes} nodes ({percent*100:.1f}%) on {self.device}..."
        )

        # Sort by out-degree (frequency of being a source/neighbor)
        # edge_index[0] are source nodes.
        # Note: If edge_index is on GPU, this is fast. If CPU, also fine.
        # Use CPU for sorting to save GPU mem during init
        edge_index = data.edge_index.cpu()
        deg = torch.bincount(edge_index[0], minlength=self.num_nodes)

        # Sort descending
        sorted_idx = torch.argsort(deg, descending=True)
        cache_idx = sorted_idx[:num_cache]

        # Store Mapping: Global ID -> Cache Index
        # Initialize with -1
        self.node_to_cache_idx = torch.full(
            (self.num_nodes,), -1, dtype=torch.int32, device=self.device
        )

        # Assign indices 0 to num_cache-1
        cache_range = torch.arange(num_cache, dtype=torch.int32, device=self.device)
        cache_idx_gpu = cache_idx.to(self.device)

        self.node_to_cache_idx[cache_idx_gpu] = cache_range
        self.is_cached = self.node_to_cache_idx >= 0

        # Move features to target device
        print(f"[GraphCache] Moving features to {self.device}...")
        self.cached_x = self.cpu_x[cache_idx].to(self.device)

        print(
            f"[GraphCache] Layout complete. Cached features size: {self.cached_x.size()}"
        )

    def fetch(self, n_id):
        """
        Fetch features for given global node IDs.
        Args:
            n_id (Tensor): Global node IDs (can be on CPU or accelerator).
        Returns:
            Tensor: Features on target device.
        """
        n_id = n_id.to(self.device)

        # If caching is disabled
        if self.cached_x is None:
            return self.cpu_x[n_id.cpu()].to(self.device, non_blocking=True)

        # Check entries
        masks = self.is_cached[n_id]

        # Prepare output
        out = torch.empty(
            (n_id.size(0), self.feature_dim),
            device=self.device,
            dtype=self.cached_x.dtype,
        )

        # 1. Cache Hit
        if masks.any():
            # Get cache indices
            hit_nodes = n_id[masks]
            cache_indices = self.node_to_cache_idx[hit_nodes]
            out[masks] = self.cached_x[cache_indices]

        # 2. Cache Miss
        if (~masks).any():
            miss_nodes = n_id[~masks]
            # Fetch from CPU
            # Note: Indexing into CPU tensor with GPU tensor is not supported directly in older Pytorch
            # So move indices to CPU
            cpu_indices = miss_nodes.cpu()
            fetched = self.cpu_x[cpu_indices].to(self.device, non_blocking=True)
            out[~masks] = fetched

        return out
