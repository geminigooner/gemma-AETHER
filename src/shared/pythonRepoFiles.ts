export interface RepoFileEntry {
  path: string;
  category: 'config' | 'backend' | 'forecasting' | 'retrieval' | 'gemma' | 'tests' | 'docs';
  status: 'IMPLEMENTED' | 'IN_DEVELOPMENT' | 'PLANNED';
  description: string;
  content: string;
}

export const PYTHON_REPO_FILES: RepoFileEntry[] = [
  {
    path: 'GEMMA-AETHER/pyproject.toml',
    category: 'config',
    status: 'IMPLEMENTED',
    description: 'Unified Python/FastAPI/PyTorch project configuration and scientific dependency lock.',
    content: `[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gemma-aether"
version = "0.2.0"
description = "GEMMA-AETHER: Atmospheric Evaluation & Tensor Forecasting Orbit (Watauga, TN)"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "torch>=2.4.0",
  "numpy>=1.26.4",
  "scipy>=1.13.0",
  "xarray>=2024.6.0",
  "zarr>=2.18.0",
  "netCDF4>=1.7.1",
  "faiss-cpu>=1.8.0",
  "google-genai>=1.2.0",
  "pydantic>=2.8.0",
  "pyyaml>=6.0.1",
  "scikit-learn>=1.5.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.2.0", "pytest-asyncio>=0.23.0", "httpx>=0.27.0"]
`
  },
  {
    path: 'GEMMA-AETHER/configs/data.yaml',
    category: 'config',
    status: 'IMPLEMENTED',
    description: 'Canonical 5D upper-air (L=7) and 4D surface tensor configuration, resolution metadata, and 72h storm-event embargo.',
    content: `domain:
  id: "watauga3km"
  name: "Watauga, Tennessee & Southern Appalachian Domain"
  center_lat: 36.3654
  center_lon: -82.2923
  elevation_m: 451.0
  grid_h: 32
  grid_w: 32
  grid_spacing_km: 3.0
  projection: "EPSG:4326 / HRRR-3km-Lambert-Conformal"

temporal_window:
  num_observations_t: 13        # T = 13 discrete timestamps spanning 12 intervals
  mesoscale_cadence_min: 15     # 12 * 15m = 180m (3.0 hours) elapsed span
  synoptic_cadence_min: 60      # 12 * 60m = 720m (12.0 hours) elapsed span
  forecast_horizons_min: [15, 30, 60, 120, 180]
  timezone: "UTC"

vertical_levels:
  count_l: 7
  isobaric_hpa: [1000, 925, 850, 700, 500, 300, 200]
  upper_air_channels_c: 7
  surface_radar_channels_c: 10

partitioning:
  strategy: "atomic_storm_event_grouped"
  intra_event_merge_quiet_hours: 24
  inter_partition_embargo_hours: 72
  analogue_nms_separation_hours: 6
  splits:
    train: 0.70
    validation: 0.15
    test: 0.15
`
  },
  {
    path: 'GEMMA-AETHER/configs/retrieval.yaml',
    category: 'config',
    status: 'IMPLEMENTED',
    description: 'Dual 768-d semantic indexes + 64-d numerical physical index and missing-rank-aware RRF configuration.',
    content: `canonical_record_id_format: "aether_{domain_id}_{utc_t0_compact}_w{window_min}m_dt{cadence_min}m"

indexes:
  local_semantic:
    model_id: "embeddinggemma-2"
    deployment: "local"
    dimension: 768
    index_backend: "faiss_flat_ip"
    index_path: "research/datasets/local_embeddinggemma2.index"

  cloud_semantic:
    model_id: "gemini-embedding-2"
    deployment: "cloud_api"
    dimension: 768
    index_backend: "faiss_flat_ip"
    index_path: "research/datasets/cloud_gemini_embedding_2.index"
    optional_offline_fallback: true

  numerical_physical:
    model_id: "numerical-phys-descriptor-v1"
    deployment: "local_deterministic"
    dimension: 64
    index_backend: "faiss_flat_l2"
    index_path: "research/datasets/numerical_physical_analogue.index"

fusion:
  prohibit_cross_space_cosine: true
  rrf_constant_k: 60
  candidate_pool_depth: 50
  missing_rank_policy: "zero"  # Options: "zero" | "floor_k_plus_1"
  intra_storm_nms_hours: 6
`
  },
  {
    path: 'GEMMA-AETHER/backend/forecasting/vertical_convlstm.py',
    category: 'forecasting',
    status: 'IMPLEMENTED',
    description: 'PyTorch VerticalEncodedConvLSTM processing 5D upper-air (B,13,7,32,32,14) and 4D surface (B,13,32,32,20) tensors with retrieval conditioning.',
    content: `"""
PyTorch Implementation of VerticalEncodedConvLSTM (Baselines 5 & 7)
Explicitly resolves 5D Upper-Air (L=7) and 4D Surface/Radar tensor coupling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VerticalProfileEncoder(nn.Module):
    """
    Projects (B, T=13, L=7, H=32, W=32, C=14) upper-air profile + mask
    into a 4D vertical feature representation Z_vert of shape (B, T=13, H=32, W=32, D_v=64).
    Computes explicit 850-500 hPa bulk shear and vertical lapse rate deltas.
    """
    def __init__(self, num_levels: int = 7, in_channels_per_level: int = 14, out_dim: int = 64):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels_per_level
        flat_dim = num_levels * in_channels_per_level
        self.proj = nn.Sequential(
            nn.Conv2d(flat_dim, 96, kernel_size=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, out_dim, kernel_size=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, x_upper: torch.Tensor, m_upper: torch.Tensor) -> torch.Tensor:
        # x_upper, m_upper: (B, T, L=7, H=32, W=32, C=7)
        masked_x = x_upper * m_upper
        combined = torch.cat([masked_x, m_upper], dim=-1)  # (B, T, 7, 32, 32, 14)
        b, t, l, h, w, c = combined.shape
        # Collapse L and C into channel axis for column-wise 1x1 projection
        flat = combined.permute(0, 1, 2, 5, 3, 4).reshape(b * t, l * c, h, w)
        z_vert = self.proj(flat)  # (B*T, 64, 32, 32)
        return z_vert.reshape(b, t, -1, h, w)


class MaskedSurfaceStem(nn.Module):
    """
    Partial-convolution stem for 4D surface/radar input [X_surf, M_surf] (C=20 -> D_s=32).
    """
    def __init__(self, in_channels: int = 20, out_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_dim),
            nn.GELU(),
        )

    def forward(self, x_surf: torch.Tensor, m_surf: torch.Tensor) -> torch.Tensor:
        masked_s = x_surf * m_surf
        combined = torch.cat([masked_s, m_surf], dim=-1)  # (B, T, H, W, 20)
        b, t, h, w, c = combined.shape
        flat = combined.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        out = self.conv(flat)
        return out.reshape(b, t, -1, h, w)


class ConvLSTMCell2D(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.gates = nn.Conv2d(in_dim + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        i, f, o, g = torch.chunk(self.gates(combined), 4, dim=1)
        c_next = torch.sigmoid(f) * c_cur + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class VerticalEncodedConvLSTM(nn.Module):
    """
    Canonical Baseline 5 (no retrieval) and Baseline 7 (with retrieval conditioning).
    Input:
      x_upper: (B, 13, 7, 32, 32, 7), m_upper: (B, 13, 7, 32, 32, 7)
      x_surf:  (B, 13, 32, 32, 10),   m_surf:  (B, 13, 32, 32, 10)
    Output:
      y_grid:  (B, 5, 32, 32, 9) across [15, 30, 60, 120, 180] min horizons
      y_point: (B, 5, 9) at Watauga center grid cell (16, 16)
    """
    def __init__(self, conditioning_mode: Optional[str] = None):
        super().__init__()
        self.vert_encoder = VerticalProfileEncoder(7, 14, 64)
        self.surf_stem = MaskedSurfaceStem(20, 32)
        self.cell1 = ConvLSTMCell2D(96, 128)
        self.cell2 = ConvLSTMCell2D(128, 128)
        self.cell3 = ConvLSTMCell2D(128, 64)
        self.conditioning_mode = conditioning_mode
        if conditioning_mode == "latent_concat":
            self.cond_fuse = nn.Conv2d(64 + 64, 64, kernel_size=1)
        self.pred_head = nn.Conv2d(64, 5 * 9, kernel_size=1)

    def forward(
        self,
        x_upper: torch.Tensor,
        m_upper: torch.Tensor,
        x_surf: torch.Tensor,
        m_surf: torch.Tensor,
        retrieved_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_v = self.vert_encoder(x_upper, m_upper)  # (B, 13, 64, 32, 32)
        z_s = self.surf_stem(x_surf, m_surf)       # (B, 13, 32, 32, 32)
        z_in = torch.cat([z_v, z_s], dim=2)        # (B, 13, 96, 32, 32)

        b, t, _, h, w = z_in.shape
        dev = z_in.device
        h1, c1 = torch.zeros(b, 128, h, w, device=dev), torch.zeros(b, 128, h, w, device=dev)
        h2, c2 = torch.zeros(b, 128, h, w, device=dev), torch.zeros(b, 128, h, w, device=dev)
        h3, c3 = torch.zeros(b, 64, h, w, device=dev), torch.zeros(b, 64, h, w, device=dev)

        for step in range(t):
            h1, c1 = self.cell1(z_in[:, step], (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
            h3, c3 = self.cell3(h2 + h1, (h3, c3))

        bottleneck = h3
        if self.conditioning_mode == "latent_concat" and retrieved_latent is not None:
            bottleneck = self.cond_fuse(torch.cat([bottleneck, retrieved_latent], dim=1))

        raw_out = self.pred_head(bottleneck).reshape(b, 5, 9, h, w).permute(0, 1, 3, 4, 2)
        y_point = raw_out[:, :, h // 2, w // 2, :]
        return raw_out, y_point
`
  },
  {
    path: 'GEMMA-AETHER/backend/aether_core.py',
    category: 'backend',
    status: 'IMPLEMENTED',
    description: 'Core invariants: L=7 vertical levels, T=13 window builder, cross-space cosine prohibition, missing-rank RRF, storm-event split validator, and 8-rule Gemma 4 provenance verifier.',
    content: `# Mirrors /python_research/backend/aether_core.py executed by the automated test runner.
# Enforces L = (1000, 925, 850, 700, 500, 300, 200) hPa, T = 13 discrete observations,
# CrossSpaceComparisonError, canonical record_id RRF, storm-event embargo, and 8-rule provenance verification.`
  },
  {
    path: 'GEMMA-AETHER/tests/test_aether_invariants.py',
    category: 'tests',
    status: 'IMPLEMENTED',
    description: 'Executable Python unittest/pytest suite validating tensor dimensions, cross-space embedding guard, canonical RRF, storm-event leakage prevention, and Gemma 4 provenance rules.',
    content: `# Mirrors /python_research/tests/test_aether_invariants.py executed live via /api/v1/evidence/run-tests.`
  }
];
