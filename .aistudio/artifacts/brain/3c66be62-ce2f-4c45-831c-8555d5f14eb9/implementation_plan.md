# GEMMA-AETHER: Atmospheric Evaluation & Tensor Forecasting Orbit — Authoritative Engineering Specification (Revision 2)

GEMMA-AETHER is a localized atmospheric intelligence, dual-embedding + numerical analogue retrieval, and spatiotemporal precipitation forecasting research observatory centered on **Watauga, Tennessee (`36.3654° N, 82.2923° W`, elevation `451 m`)**. This document serves as the authoritative, pre-implementation engineering baseline incorporating all 12 corrections from the Second-Pass Technical Review.

---

### 1. Evidence-Based Implementation Status & Verification Gate

> [!IMPORTANT]
> **Pre-Implementation Workspace Audit (Zero Fabricated Statuses)**: A declaration in an implementation plan does not establish that a component exists. An audit of the current workspace confirms that **only the bare project scaffold and specification assets exist on disk**. Therefore, every GEMMA-AETHER scientific, backend, model, and Observatory UI subsystem is currently classified as **`PLANNED`** (scheduled for implementation upon approval of this specification).

#### A. Current Verified Workspace State (`IMPLEMENTED` Scaffold Only)

| Component | Status | Actual Source Files on Disk | Verified Symbols / Contents | Executed Tests & Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Frontend Build Scaffold** | `IMPLEMENTED` | `/package.json`, `/vite.config.ts`, `/tsconfig.json`, `/index.html`, `/src/main.tsx`, `/src/App.tsx`, `/src/index.css` | `App()` returning placeholder `<div></div>`; Vite 8 + React 19 + Tailwind 4 configuration | Workspace file inspection verified; zero GEMMA-AETHER domain modules exist yet. |
| **Reference Specification Assets** | `IMPLEMENTED` | `/public/assets/aistudio/GEMMA_AETHER_IMPL_1c4bd7ed.pdf`, `/public/assets/aistudio/D63422BE-6D34-4A4C-A54A-18F4D1E5F95B_97b1b4af.png` | 21-page architectural specification PDF & Kawaii Observatory UI reference mockup | Inspected via OCR and visual verification. |

#### B. GEMMA-AETHER Target Subsystems (All Currently `PLANNED` Pending Build Approval)

No subsystem below may be relabeled `IN DEVELOPMENT` or `IMPLEMENTED` in the repository or UI until it satisfies the **Six-Point Evidence Gate**:
1. **Actual Source File Path(s)** created on disk.
2. **Implemented Classes / Functions** with complete type signatures and zero mock stubs.
3. **API Endpoint(s)** wired and responding with validated payloads.
4. **Automated Tests Executed** via test runner (`pytest` / `vitest`).
5. **Recorded Test Results** (pass/fail count, assertion logs, execution timestamp).
6. **Real-Data Provenance Verification** confirming live API or authentic meteorological dataset ingestion rather than synthetic fixtures masquerading as live measurements.

| Target Subsystem | Current Status | Target Phase | Required Evidence to Promote to `IMPLEMENTED` |
| :--- | :--- | :--- | :--- |
| **Layer 1: Multi-Source Data Ingestion (`ERA5`, `HRRR`, `MRMS`, `Open-Meteo`)** | `PLANNED` | Phase 1 | Ingestion adapters on disk, schema validation tests passing, and live/archival provenance tags (`DIRECT_OBSERVATION`, `RADAR_DERIVED_ESTIMATE`, `MODEL_DERIVED_CURRENT`, `REANALYSIS_FIELD`, `MODEL_FORECAST`) verified. |
| **Layer 2: 5D Upper-Air & 4D Surface Tensor Pipeline + Storm Event Splitter** | `PLANNED` | Phase 2 | Tensor builder outputting $X_{\text{upper}} \in \mathbb{R}^{T \times 7 \times 32 \times 32 \times 7}$ and $X_{\text{surf}} \in \mathbb{R}^{T \times 32 \times 32 \times 10}$, unit tests proving zero storm-event leakage across splits. |
| **Layer 3: Independent `EmbeddingGemma 2` (Local) & `gemini-embedding-2` (Cloud) Indexes** | `PLANNED` | Phase 3 | Separate 768-d vector indexes keyed by canonical `record_id`, passing cross-space cosine prohibition tests and live `gemini-embedding-2` API integration tests. |
| **Layer 4: Retrieval Evaluation (`Exps A–E`) & Numerical Analogue Pathway** | `PLANNED` | Phase 4 | Executed retrieval benchmarks (`Recall@K`, `Precision@K`, `MRR`, `NDCG@K`) comparing Local, Cloud, Numerical Physical Analogue, and RRF over held-out storm events. |
| **Layer 5: Seven Forecasting Baselines & `VerticalEncodedConvLSTM`** | `PLANNED` | Phase 5 & 6 | Executable `Persistence`, `Climatology`, `Optical-Flow`, `HRRR`, `VerticalEncodedConvLSTM`, `ST-Transformer`, and `Retrieval-Conditioned` modules passing shape and mask-invariance tests. |
| **Layer 6: `Gemma 4` Runtime Adapter & Structured Claim-Tuple Provenance Verifier** | `PLANNED` | Phase 7 | `StructuredProvenanceVerifier` unit tests proving rejection of mismatched fields, units, horizons, locations, and numeric tolerances. |
| **Layer 7: Seven-Workspace Kawaii Scientific Observatory & Research Exporter** | `PLANNED` | Phase 8 | All 7 workspaces wired to real backend API endpoints, rendering dynamic status badges (`IMPLEMENTED` / `IN DEVELOPMENT` / `PLANNED`) backed by runtime readiness checks. |

---

### 2. Data Source Semantics, Resolution Metadata & Regridding Contract

#### A. Five-Class Data Source Provenance Semantics
Every API response, tensor slice, database record, UI badge, and generated briefing must classify data into one of five mutually exclusive semantic categories. **Open-Meteo `current` API values must never be labeled as direct thermometer/station observations** when they originate from NWP assimilation/0-hour analysis grids:

| Provenance Class | Definition | Applicable Sources | UI & Schema Badge |
| :--- | :--- | :--- | :--- |
| `DIRECT_OBSERVATION` | In-situ physical instrument measurement at a weather station | Local ASOS/AWOS / NWS station (`KTRI` Tri-Cities / Watauga mesonets) | `Direct Station Obs` |
| `RADAR_DERIVED_ESTIMATE` | Remote-sensing precipitation or reflectivity derived from weather radar | NOAA MRMS (`PrecipRate`, `QPE_01H`, `RALA`) | `Radar Estimate (MRMS)` |
| `MODEL_DERIVED_CURRENT` | Assimilated or 0-hour NWP model analysis representing current conditions at a point | Open-Meteo `/v1/forecast?current=...` (derived from HRRR/GFS/IFS analysis) | `Model-Derived Current` |
| `REANALYSIS_FIELD` | Retrospective 4D-Var data assimilation combining historical observations and NWP | ECMWF ERA5 Reanalysis | `Reanalysis (ERA5)` |
| `MODEL_FORECAST` | Forward-in-time prediction generated by an NWP or neural forecasting model | NOAA HRRR forecast cycles, Open-Meteo hourly forecast, GEMMA-AETHER models | `Model Forecast (t+Δτ)` |

#### B. Native vs. Regridded vs. Effective Resolution Contract
Co-registering `ERA5`, `NOAA HRRR`, and `NOAA MRMS` onto a common `3 km × 3 km` Watauga grid ($32 \times 32$ cells covering $96\text{ km} \times 96\text{ km}$) is purely a **geometric co-registration operation** for tensor alignment. **Interpolating 0.25° ERA5 fields onto a 3 km grid does not create 3 km mesoscale atmospheric information.** Every variable channel in the tensor metadata carries three mandatory resolution fields:

| Dataset | Native Horizontal Resolution (`native_resolution_km`) | Co-Registered Grid Spacing (`grid_spacing_km`) | Effective Information Scale (`effective_resolution_km`) | Regridding Operator & Physical Implications |
| :--- | :--- | :--- | :--- | :--- |
| **ERA5** | `0.25° × 0.25°` ($\approx 27.8\text{ km N-S} \times 22.4\text{ km E-W}$ at $36.36^\circ\text{N}$) | `3.0 km` (oversampled $7.5\times$–$9.2\times$) | **$\ge 55\text{--}100\text{ km}$** (Synoptic background scale) | Bicubic interpolation onto the 3 km grid provides smooth large-scale synoptic pressure, geopotential, and upper-air moisture boundary conditions only; sub-25 km convective structures are absent. |
| **NOAA HRRR** | `3.0 km × 3.0 km` Lambert Conformal | `3.0 km` (1:1 native match) | **$\approx 9\text{--}15\text{ km}$** ($3\Delta x\text{--}5\Delta x$ NWP dynamical filter scale) | Bilinear coordinate alignment to canonical Watauga grid; resolves mesoscale convective systems and Appalachian terrain-forced flows. |
| **NOAA MRMS** | `0.01° × 0.01°` ($\approx 1.11\text{ km N-S} \times 0.89\text{ km E-W}$) | `3.0 km` (aggregated $3\times 3$ cells) | **$\approx 3.0\text{ km}$** (Convective precipitation scale) | Conservative area-weighted flux averaging ($\text{mm/hr}$) and maximum pooling for `RALA` ($\text{dBZ}$); preserves total basin precipitation volume while matching the 3 km forecasting grid. Pixels with `RQI < 0.25` or `-999` set validity mask $M = 0$. |
| **Open-Meteo** | Point coordinate (`36.3654° N, 82.2923° W`) | `0D Point` (Not regridded to 3D/4D grid) | Underlying NWP grid (`3 km` HRRR / `9 km` IFS) | Stored strictly in a separate `PointTelemetryRecord` schema; never broadcast into spatial tensors. |

---

### 3. Corrected Tensor Schema, Vertical Levels & Temporal Indexing

#### A. Unified 7-Level Vertical Pressure Array & Dual-Tensor Coupling
To eliminate any discrepancy between the upper-air tensor and the `0–12 km` vertical cross-section visualization, the canonical vertical dimension is fixed at **$L = 7$ standard isobaric pressure levels**:
$$\mathcal{L} = [1000, \; 925, \; 850, \; 700, \; 500, \; 300, \; 200]\text{ hPa}$$
corresponding to approximate standard atmosphere altitudes $z \approx [0.11, \; 0.76, \; 1.46, \; 3.01, \; 5.57, \; 9.16, \; 11.78]\text{ km MSL}$.

Rather than mixing 1D surface/radar variables into the 5D isobaric array (which would require duplicating surface fields across 7 pressure levels), GEMMA-AETHER maintains an explicit **Paired Dual-Tensor Representation** aligned on time $T$ and horizontal grid $H \times W$:
1. **5D Isobaric Upper-Air Tensor**:
   $$X_{\text{upper}} \in \mathbb{R}^{T \times L \times H \times W \times C_{\text{upper}}}, \quad M_{\text{upper}} \in \{0, 1\}^{T \times L \times H \times W \times C_{\text{upper}}}$$
   - Dimensions: $T = 13, \; L = 7, \; H = 32, \; W = 32, \; C_{\text{upper}} = 7$.
   - Channels ($C_{\text{upper}} = 7$):
     0. `air_temperature_k` ($\text{K}$)
     1. `geopotential_height_m` ($\text{m}$)
     2. `u_wind_ms` ($\text{m}\cdot\text{s}^{-1}$)
     3. `v_wind_ms` ($\text{m}\cdot\text{s}^{-1}$)
     4. `relative_humidity_pct` ($\%$)
     5. `specific_humidity_g_kg` ($\text{g}\cdot\text{kg}^{-1}$)
     6. `vertical_velocity_omega_pa_s` ($\text{Pa}\cdot\text{s}^{-1}$)
2. **4D Surface, Terrain & Radar Tensor**:
   $$X_{\text{surf}} \in \mathbb{R}^{T \times H \times W \times C_{\text{surf}}}, \quad M_{\text{surf}} \in \{0, 1\}^{T \times H \times W \times C_{\text{surf}}}$$
   - Dimensions: $T = 13, \; H = 32, \; W = 32, \; C_{\text{surf}} = 10$.
   - Channels ($C_{\text{surf}} = 10$):
     0. `mrms_precip_rate_mm_hr` ($\text{mm}\cdot\text{hr}^{-1}$)
     1. `mrms_qpe_1h_mm` ($\text{mm}$)
     2. `mrms_reflectivity_rala_dbz` ($\text{dBZ}$)
     3. `mrms_radar_quality_index` ($[0, 1]$)
     4. `temperature_2m_k` ($\text{K}$)
     5. `dewpoint_2m_k` ($\text{K}$)
     6. `surface_pressure_hpa` ($\text{hPa}$)
     7. `u_wind_10m_ms` ($\text{m}\cdot\text{s}^{-1}$)
     8. `v_wind_10m_ms` ($\text{m}\cdot\text{s}^{-1}$)
     9. `static_terrain_elevation_m` ($\text{m}$, USGS/SRTM 3 km elevation over Watauga domain)

#### B. Exact Temporal Window Indexing ($T = 13$ Observations vs. $12$ Intervals)
Throughout the pipeline, $T$ denotes the **number of discrete observation timestamps** in an input window anchored at initialization time $t_0$ (UTC), while $N_{\text{int}} = T - 1$ denotes the **number of elapsed temporal intervals** between the first and last observation:
$$\mathcal{T}(t_0, \Delta t, T=13) = \bigl[\, t_0 - 12\Delta t, \;\; t_0 - 11\Delta t, \;\; \dots, \;\; t_0 - \Delta t, \;\; t_0 \,\bigr]$$
- **Primary Mesoscale Nowcasting Mode ($\Delta t = 15\text{ min}, \; T = 13$)**:
  - Contains **$13$ discrete timestamps** spanning **$12$ intervals of $15\text{ minutes}$**, yielding an exact **$180\text{-minute}$ ($3.0\text{-hour}$) elapsed historical window** $[t_0 - 180\text{m}, \; t_0]$.
  - Example for $t_0 = \texttt{2026-11-02T18:00:00Z}$: index $0$ is `15:00Z`, index $1$ is `15:15Z`, ..., index $11$ is `17:45Z`, and index $12$ is `18:00Z`.
- **Synoptic Reanalysis Context Mode ($\Delta t = 60\text{ min}, \; T = 13$)**:
  - Contains **$13$ discrete hourly timestamps** spanning **$12$ elapsed 1-hour intervals**, yielding an exact **$12.0\text{-hour}$ elapsed synoptic window** $[t_0 - 12\text{h}, \; t_0]$.
- **Target Forecast Horizon Indexing ($H_{\text{lead}} = 5$)**:
  - Relative to $t_0$, predictions are evaluated at exact future timestamps:
    $$\mathcal{T}_{\text{target}}(t_0) = \bigl[\, t_0 + 15\text{m}, \;\; t_0 + 30\text{m}, \;\; t_0 + 60\text{m}, \;\; t_0 + 120\text{m}, \;\; t_0 + 180\text{m} \,\bigr]$$

---

### 4. Storm-Event Segmentation & Event-Isolated Partitioning

A fixed 72-hour buffer alone cannot prevent data leakage if a stalled frontal system, tropical remnant, or multi-day cutoff low persists longer than 72 hours across a split boundary. GEMMA-AETHER enforces a **Four-Stage Event-Grouped Partitioning Protocol**:

1. **Automated Storm-Event Identification**:
   - Let $I_{\text{active}}(t) \in \{0, 1\}$ be a domain-level meteorological activity indicator over the Watauga $96\text{ km} \times 96\text{ km}$ grid that equals $1$ at timestamp $t$ if **any** of three physical criteria hold:
     - *Precipitation Criterion*: $\text{P95}_{h,w}(\text{MRMS\_PrecipRate}(t)) \ge 0.25\text{ mm/hr}$ or domain-max reflectivity $\ge 20\text{ dBZ}$.
     - *Moisture Transport Criterion*: Domain-mean $850\text{ hPa}$ integrated moisture flux $\Vert \mathbf{V}_{850} \cdot q_{850} \Vert_2 \ge \mu_{\text{clim}} + 1.5\sigma_{\text{clim}}$.
     - *Synoptic Forcing Criterion*: Domain-mean surface pressure tendency $\Delta P_{3\text{h}} \le -2.0\text{ hPa}$ or $500\text{ hPa}$ height anomaly $\le -1.5\sigma$.
2. **Contiguous Event Grouping (`event_id`)**:
   - Consecutive active timestamps are merged across any quiescent gap shorter than $\tau_{\text{merge}} = 24\text{ hours}$ into a single discrete **Storm Event** $E_k = [\text{start}(E_k), \text{end}(E_k)]$ assigned a unique `storm_event_id` (even if $E_k$ spans 4 to 10 days). Intervening fair-weather periods longer than $24\text{ hours}$ are segmented as `quiescent_event_id` blocks.
3. **Atomic Event-Grouped Split Assignment & Temporal Embargo**:
   - The chronological sequence of complete events $\{E_1, E_2, \dots, E_M\}$ is partitioned **at the whole-event level** (`70% TRAIN`, `15% VALIDATION`, `15% TEST`). **An individual storm event $E_k$ is atomic and is never divided across two splits.**
   - At the chronological boundary between the last event of `TRAIN` and the first event of `VALIDATION` (and between `VALIDATION` and `TEST`), a mandatory **$\Delta t_{\text{embargo}} = 72\text{-hour}$ embargo buffer** discards any boundary-adjacent quiescent windows.
   - Normalization statistics ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed strictly over `TRAIN` events.
4. **Analogue Eligibility & Intra-Storm Window Deduplication**:
   - When forecasting or evaluating a query window $q$ at time $t_0$ belonging to event $E(t_0)$:
     - **Candidate Pool Restriction**: Only records $d$ belonging to the `TRAIN` partition that satisfy $E(t_d) \neq E(t_0)$ and $t_0 - t_d \ge 72\text{ hours}$ are eligible for retrieval.
     - **Intra-Storm Non-Maximum Suppression (NMS)**: Because sliding windows indexed every 15 minutes overlap heavily, retrieval over the candidate index applies temporal NMS within each historical storm event $E_m$: once the highest-scoring window $t_d^* \in E_m$ is selected, all overlapping or adjacent windows in $E_m$ within $|t_d - t_d^*| < 6\text{ hours}$ are suppressed so the top-$K$ analogues represent $K$ independent historical atmospheric evolutions.

---

### 5. Precise Forecasting Network Architecture (`VerticalEncodedConvLSTM`)

To remove any ambiguity about how the 5D upper-air tensor $X_{\text{upper}}$ and 4D surface tensor $X_{\text{surf}}$ are processed Together, Baseline 5 (`ConvLSTM`) is explicitly defined as a **Two-Stage Vertical-Column Encoder + 2D Spatiotemporal ConvLSTM (`VerticalEncodedConvLSTM`)**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│              VERTICAL-ENCODED SPATIOTEMPORAL CONVLSTM ARCHITECTURE                     │
│                                                                                        │
│  Upper-Air Input [X_upper, M_upper]             Surface Input [X_surf, M_surf]         │
│  Shape: (B, T=13, L=7, H=32, W=32, C=14)        Shape: (B, T=13, H=32, W=32, C=20)     │
│                   │                                              │                     │
│                   ▼                                              ▼                     │
│  ┌────────────────────────────────────┐        ┌────────────────────────────────────┐  │
│  │ Stage 1: VerticalProfileEncoder    │        │ Stage 2A: Masked Surface Stem      │  │
│  │ • Explicit Shear/Lapse Deltas (+6) │        │ • PartialConv2D(20 → 32, k=3×3)    │  │
│  │ • Flatten Levels: (7 × 20 = 140)   │        │ • GroupNorm + GELU                 │  │
│  │ • 1×1 Conv MLP: (140 → 64 channels)│        │ • Output Z_surf:                   │  │
│  │ • Output Z_vert:                   │        │   (B, T=13, H=32, W=32, D_s=32)    │  │
│  │   (B, T=13, H=32, W=32, D_v=64)    │        └─────────────────┬──────────────────┘  │
│  └────────────────┬───────────────────┘                          │                     │
│                   └──────────────────────┬───────────────────────┘                     │
│                                          ▼                                             │
│             ┌─────────────────────────────────────────────────────────┐                │
│             │ Stage 2B: Channel Concatenation & Mask Conditioning     │                │
│             │ Z_in = Concat(Z_vert, Z_surf) → (B, 13, 32, 32, D=96)   │                │
│             └────────────────────────────┬────────────────────────────┘                │
│                                          ▼                                             │
│             ┌─────────────────────────────────────────────────────────┐                │
│             │ Stage 3: 3-Layer 2D Spatiotemporal ConvLSTM Core        │                │
│             │ • Layer 1: ConvLSTM2D(96 → 128, k=3×3)                  │                │
│             │ • Layer 2: ConvLSTM2D(128 → 128, k=3×3) + Residual      │                │
│             │ • Layer 3: ConvLSTM2D(128 → 64, k=3×3)                  │                │
│             │ • Final Bottleneck State H_T: (B, 32, 32, 64)           │                │
│             └────────────────────────────┬────────────────────────────┘                │
│                                          ▼                                             │
│             ┌─────────────────────────────────────────────────────────┐                │
│             │ Optional Stage 3.5 (Baseline 7 Only):                   │                │
│             │ Retrieval-Conditioned Cross-Attention / Concatenation   │                │
│             │ H_cond = CrossAttn(Q=H_T, K=H_hist[1:K], V=Y_hist[1:K]) │                │
│             └────────────────────────────┬────────────────────────────┘                │
│                                          ▼                                             │
│             ┌─────────────────────────────────────────────────────────┐                │
│             │ Stage 4: Multi-Horizon Prediction Heads (5 Horizons)    │                │
│             │ • Grid Output Ŷ_grid: (B, 5, 32, 32, 9)                 │                │
│             │ • Watauga Basin/Point Output Ŷ_watauga: (B, 5, 9)       │                │
│             └─────────────────────────────────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Masking Behavior**: At every input cell where $M = 0$ (missing radar or station coverage), the input value is zeroed prior to convolution and the convolution output is scaled by the partial-convolution ratio $\frac{\Vert \mathbf{1} \Vert_1}{\max(1, \Vert M_{\text{patch}} \Vert_1)}$ so missing values never bias learned spatial gradients.
- **Output Tensor Channels ($C_{\text{out}} = 9$ per lead time $\Delta \tau \in \{15, 30, 60, 120, 180\}\text{ min}$)**:
  - Channels `0..2`: Logits/probabilities for precipitation exceeding thresholds $\tau \in \{0.25, \; 2.5, \; 7.6\}\text{ mm/hr}$.
  - Channel `3`: `expected_precipitation_mm` ($\text{Softplus}$ activation, $\ge 0$).
  - Channel `4`: `precipitation_intensity_mm_hr` ($\text{Softplus}$ activation, $\ge 0$).
  - Channel `5`: `temperature_2m_celsius` ($\text{linear}$).
  - Channel `6`: `relative_humidity_2m_pct` ($\text{Sigmoid} \times 100$).
  - Channels `7..8`: `u_wind_10m_ms` and `v_wind_10m_ms` ($\text{linear}$).

---

### 6. Retrieval Index Contract, Canonical Record Identity & Fusion Semantics

#### A. Canonical Atmospheric Record Identifier (`record_id`)
Independent vector indexes cannot be fused via RRF unless they share a deterministic primary key for every historical time window. Every indexed atmospheric window is identified by a canonical `record_id`:
$$\texttt{record\_id} = \texttt{aether\_watauga3km\_\{utc\_t0\_compact\}\_w180m\_dt15m}$$
For example, the 3-hour window ending at `2024-09-27T15:00:00Z` has `record_id = "aether_watauga3km_20240927T150000Z_w180m_dt15m"`.

#### B. Index Partitioning & Metadata Schema
The system maintains **three separate vector indexes** over the exact same `TRAIN`-split catalogue of canonical `record_id` entries:
1. `Index A — local_embeddinggemma2`: `768`-d float32 cosine/inner-product index populated by local `EmbeddingGemma 2` over structured meteorological text + rendered synoptic/radar plots.
2. `Index B — cloud_gemini_embedding_2`: `768`-d float32 cosine/inner-product index populated by cloud `gemini-embedding-2` (`output_dimensionality=768`).
3. `Index C — numerical_physical_analogue`: `64`-d float32 standardized Euclidean ($L_2$) index populated directly from physical atmospheric state descriptors ($500\text{ hPa}$ geopotential height anomaly PCA coefficients, $850\text{ hPa}$ moisture flux vector, $850\text{--}500\text{ hPa}$ bulk wind shear vector, cross-Appalachian pressure gradient, and domain precipitation fraction).

Every index entry stores an identical 12-field metadata document:
`{ record_id, storm_event_id, split_partition, utc_t0, temporal_window_minutes: 180, cadence_minutes: 15, model_id, model_version, embedding_dim, input_modality, preprocessing_version, normalization_hash }`.

#### C. Candidate-Set Construction & Missing-Result RRF Contract
For a query window $q$ at time $t_0$:
1. **Per-Index Retrieval & Storm NMS**: Each active index $m \in \mathcal{M}$ independently retrieves its raw top-500 records, filters out any record violating the storm-eligibility rule ($E(t_d) = E(t_0)$ or $|t_0 - t_d| < 72\text{h}$), and applies intra-storm NMS ($\Delta t_{\text{NMS}} = 6\text{h}$) to produce a deduplicated ranked list $R_m = [d_{m,1}, d_{m,2}, \dots, d_{m,K_{\text{pool}}}]$ of length up to $K_{\text{pool}} = 50$.
2. **Union Candidate Set**: Construct the merged candidate set $\mathcal{U} = \bigcup_{m \in \mathcal{M}} R_m$ keyed by canonical `record_id`.
3. **RRF Scoring with Explicit Missing-Record Rule**:
   For each candidate $d \in \mathcal{U}$, its Reciprocal Rank Fusion score across active pipelines $\mathcal{M}$ is:
   $$\text{RRF}_{\mathcal{M}}(d) = \sum_{m \in \mathcal{M}} w_m \cdot \phi_m(d), \qquad \phi_m(d) = \begin{cases} \dfrac{1}{k + \text{rank}_m(d)} & \text{if } d \in R_m \\[8pt] 0 & \text{if } d \notin R_m \text{ (absent from top-}K_{\text{pool}}\text{)} \end{cases}$$
   where $k = 60$ and default uniform weights $w_m = 1.0$. (An optional pessimistic floor $\phi_m(d) = \frac{1}{k + K_{\text{pool}} + 1}$ for absent records is configurable via `missing_rank_policy: "zero" | "floor_k_plus_1"` in `configs/retrieval.yaml`).
4. **Incorporation of Numerical Analogue Retrieval into Fusion Experiments**:
   - **Experiment A**: `Index A (EmbeddingGemma 2 Local)` alone.
   - **Experiment B**: `Index B (gemini-embedding-2 Cloud)` alone.
   - **Experiment N (Numerical Baseline)**: `Index C (Numerical Physical Analogue)` alone.
   - **Experiment C1 (Dual Semantic RRF)**: $\text{RRF}_{\{A, B\}}(d)$ combining Local and Cloud semantic embeddings.
   - **Experiment C2 (Tri-Modal Hybrid RRF)**: $\text{RRF}_{\{A, B, C\}}(d)$ combining both semantic embedding indexes with the physical numerical analogue index to test whether combining semantic and physical similarity improves downstream retrieval and forecasting over either pathway alone.

---

### 7. Structured Gemma 4 Provenance Verification

Free-form prose regex matching cannot guarantee that a number is attached to the right meteorological variable, unit, forecast horizon, or geographic location. GEMMA-AETHER replaces free-form number extraction with **Schema-Enforced Claim-Tuple Verification & Deterministic Value Interpolation**:

1. **Structured Output Schema (`BriefingSynthesisResponse`)**:
   `Gemma 4` is constrained via JSON schema to emit structured analytical claims where numerical slots are referenced symbolically via `{claim_ref:c_id}` rather than embedded as unverified free-text strings:
   ```json
   {
     "summary_headline": "Moderate frontal precipitation expected over Watauga within 60 minutes.",
     "claims": [
       {
         "claim_id": "c_01",
         "statement_prefix": "At the 60-minute horizon, precipitation probability (>= 0.25 mm/hr) reaches",
         "statement_suffix": "with expected accumulation of",
         "bindings": [
           {
             "source_field": "forecast.horizons.60m.precip_prob_025mm",
             "variable_name": "precipitation_probability_025mm",
             "reported_value": 0.87,
             "unit": "probability_0_to_1",
             "horizon_minutes": 60,
             "location_id": "watauga_tn_3km_center",
             "provenance_class": "MODEL_FORECAST"
           },
           {
             "source_field": "forecast.horizons.60m.expected_precip_mm",
             "variable_name": "expected_precipitation_mm",
             "reported_value": 4.2,
             "unit": "mm",
             "horizon_minutes": 60,
             "location_id": "watauga_tn_3km_center",
             "provenance_class": "MODEL_FORECAST"
           }
         ]
       }
     ],
     "qualitative_uncertainty_note": "Ensemble spread increases after +120 minutes due to Blue Ridge terrain blocking."
   }
   ```
2. **Eight-Rule Deterministic Verifier (`StructuredProvenanceVerifier`)**:
   Before any briefing is returned by `/api/v1/gemma/interpret`, the deterministic verifier validates every `binding` and every qualitative string:
   - **Rule 1 (Path Resolution)**: `source_field` must exist in the input `ForecastVerificationContext` dictionary.
   - **Rule 2 (Variable Identity Match)**: `variable_name` must match the schema type registered for `source_field`.
   - **Rule 3 (Unit Exactness)**: `unit` must match the canonical unit of `source_field` (or an approved deterministic unit transform such as $\text{K} \to {}^{\circ}\text{C} / {}^{\circ}\text{F}$).
   - **Rule 4 (Horizon Alignment)**: `horizon_minutes` must equal the temporal lead time of `source_field`.
   - **Rule 5 (Geographic Scope Match)**: `location_id` must match the spatial target (`watauga_tn_3km_center` or `watauga_96km_basin`).
   - **Rule 6 (Provenance Class Match)**: `provenance_class` must match the source object's provenance (`DIRECT_OBSERVATION`, `RADAR_DERIVED_ESTIMATE`, `MODEL_DERIVED_CURRENT`, `REANALYSIS_FIELD`, `MODEL_FORECAST`).
   - **Rule 7 (Numeric Tolerance & Deterministic Rendering)**: $|\text{reported\_value} - \text{true\_value}| \le \epsilon$, AND the final UI badge renders the value **deterministically formatted from `true_value`** alongside an interactive provenance chip showing `source_field`.
   - **Rule 8 (Qualitative Digit Guard)**: Free-text strings (`summary_headline`, `statement_prefix`, `qualitative_uncertainty_note`) are scanned to ensure they contain no un-bound floating-point measurements that bypass `bindings`.

---

### 8. Diagnostic Mascot Triggers & Seven-Workspace Navigation

#### A. Scientifically Grounded Mascot State Machine
The **Purple Gradient Cloud Mascot** never infers weather phenomena from an isolated single variable (such as 10 m wind speed for wind shear, or surface RH alone for rain). Its expression state is computed deterministically by `evaluateMascotDiagnosticState()` from multi-level atmospheric diagnostics and live pipeline statuses:

| Mascot State & Expression | Required Multi-Variable Physical or System Trigger |
| :--- | :--- |
| **`CONVECTIVE_PRECIP_ACTIVE`** *(Determined / Storm Cloud)* | Triggered iff observed MRMS `mrms_precip_rate_mm_hr >= 1.0` OR `mrms_reflectivity_rala_dbz >= 30.0 dBZ` OR forecast `precip_prob_2_5mm >= 0.60` at $\Delta\tau \le 60\text{ min}$. |
| **`VERTICAL_WIND_SHEAR_ALERT`** *(Wide-Eyed Surprised Cloud `!!`)* | Triggered iff deep-layer bulk vector wind shear $\Vert \mathbf{V}_{500\text{ hPa}} - \mathbf{V}_{850\text{ hPa}} \Vert_2 \ge 15.0\text{ m/s}$ OR low-level vector shear $\Vert \mathbf{V}_{850\text{ hPa}} - \mathbf{V}_{10\text{m}} \Vert_2 \ge 10.0\text{ m/s}$. |
| **`COLUMN_MOISTURE_SATURATION`** *(Teary / Droplet Cloud)* | Triggered on the vertical cross-section panel iff mean `850–700 hPa` layer $\text{RH} \ge 85\%$ **AND** upward vertical velocity $\omega_{700\text{ hPa}} \le -0.20\text{ Pa/s}$ (active synoptic/orographic ascent). |
| **`RETRIEVAL_FUSION_RUNNING`** *(Focused / Star-Goggles Cloud)* | Triggered during active dual-index query execution or RRF comparison across `Index A`, `Index B`, and `Index C`. |
| **`PROVENANCE_OR_PIPELINE_WARNING`** *(Cautious / Magnifying-Glass Cloud)* | Triggered if `StructuredProvenanceVerifier` flags a claim binding mismatch, or if cloud `gemini-embedding-2` is unreachable and local `EmbeddingGemma 2` fallback is active. |
| **`QUIESCENT_NOMINAL`** *(Happy Heart-Hugging Cloud)* | Default state when precipitation rate $< 0.25\text{ mm/hr}$, vertical shear is below threshold, and all active pipeline health checks pass. |

#### B. Consolidated Seven-Workspace Navigation Architecture
Both the left navigation rail and the system architecture strictly adhere to the **seven canonical workspaces** (with sub-tabs housed inside each workspace rather than expanding the primary navigation rail):

```
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                   GEMMA-AETHER: KAWAII SCIENTIFIC OBSERVATORY (7 WORKSPACES)               │
│  ┌────────────────────────────────────┐  ┌──────────────────────────────────────────────┐  │
│  │ Left Navigation Rail (7 Items)     │  │ Active Workspace Canvas & Sub-Tabs           │  │
│  │ 1. Observatory Overview            │  │ • Top Banner: Watauga Point Card (Labeled    │  │
│  │ 2. Atmospheric Conditions          │  │   by Provenance Class) + Diagnostic Mascot   │  │
│  │ 3. Vector Field Explorer           │  │ • Main Scientific Stage: Interactive 5D      │  │
│  │ 4. Semantic Embedding Explorer     │  │   Tensors, Wind/Shear Fields, RRF Tables,    │  │
│  │ 5. Forecast Laboratory             │  │   7-Baseline Forecast Curves, & Verifier     │  │
│  │ 6. Historical Data & Experiments   │  │ • Explicit Readiness Badges:                 │  │
│  │ 7. Model & Pipeline Monitoring     │  │   [IMPLEMENTED | IN DEVELOPMENT | PLANNED]   │  │
│  └────────────────────────────────────┘  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Workspace 1 — Observatory Overview**: System summary, Watauga point conditions (explicitly badged with `Provenance Class`), quick summary of the 5D domain, active model/index readiness table, and diagnostic Mascot briefing.
2. **Workspace 2 — Atmospheric Conditions**: Source inspector (`ERA5`, `NOAA HRRR`, `NOAA MRMS`, `Open-Meteo`) displaying native vs. regridded vs. effective resolution metadata, the $T=13 \times L=7 \times 32 \times 32 \times 7$ upper-air tensor slicer, the $0\text{--}12\text{ km}$ (`1000–200 hPa`) vertical cross-section chart, and binary missing-data mask $M$ viewer.
3. **Workspace 3 — Vector Field Explorer**: Interactive 2D/3D spatial field canvas over the `96 km × 96 km` Watauga grid with switchable level (`10m Surface`, `925`, `850`, `700`, `500`, `300`, `200 hPa`), variable (`Wind (u,v)`, `Vertical Shear`, `Temp`, `Geopotential/Pressure`, `MRMS Precip`), streamlines, and $T=13$ (`-180m` to `0m`) / forecast (`+15m` to `+180m`) timeline scrubber.
4. **Workspace 4 — Semantic Embedding Explorer**: Side-by-side inspection of `Index A (EmbeddingGemma 2 Local 768-d)`, `Index B (gemini-embedding-2 Cloud 768-d)`, and `Index C (Numerical Physical Analogue 64-d)` keyed by canonical `record_id`, exploratory PCA/UMAP projections (with non-equivalence disclaimers), and live RRF (`Exps A, B, N, C1, C2`) rank fusion inspector.
5. **Workspace 5 — Forecast Laboratory**: Comparative multi-horizon (`15, 30, 60, 120, 180 min`) predictions across all **7 Baselines** (`Persistence`, `Climatology`, `Optical-Flow`, `HRRR`, `VerticalEncodedConvLSTM`, `ST-Transformer`, and `Retrieval-Conditioned`), conditioning mode selector, and the **Gemma 4 Structured Briefing & Provenance Verifier** inspector.
6. **Workspace 6 — Historical Data & Experiments**: Event-grouped storm partition browser (`storm_event_id`, 72h embargo verification), controlled evaluation tables (`MAE`, `RMSE`, `Bias`, `POD`, `FAR`, `CSI`, `Brier`, `CRPS`, `FSS` across thresholds and neighborhood scales with 95% storm-event bootstrap confidence intervals), Reliability Diagrams, and Precision-Recall curves.
7. **Workspace 7 — Model & Pipeline Monitoring / Research Export**: Immutable Experiment Registry viewer, runtime latency/health diagnostics, automated test suite runner & evidence log, and one-click export of the complete `GEMMA-AETHER/` Python/FastAPI/PyTorch research repository.

---

### 9. Four-Tier Scientific Claims Policy & Acceptance Criteria

To ensure scientific integrity across code comments, UI copy, and generated experiment reports, GEMMA-AETHER enforces a strict **Four-Tier Claims Policy**:

- **Tier A — Functional Implementation**: A module or baseline executes end-to-end on real data structures without runtime errors. *(Permitted claim: "Reciprocal Rank Fusion and VerticalEncodedConvLSTM are implemented and executable.")*
- **Tier B — Technical Validation**: Automated tests verify mathematical and architectural invariants (e.g., $T=13, L=7$ tensor shapes, zero cross-space cosine similarity between `EmbeddingGemma 2` and `gemini-embedding-2`, zero storm-event overlap across splits, 100% deterministic binding verification in `StructuredProvenanceVerifier`). *(Permitted claim: "Pipeline satisfies leakage-prevention and provenance-verification invariants.")*
- **Tier C — Controlled Scientific Evaluation**: Quantitative verification metrics (`Recall@K`, `NDCG@K`, `POD`, `FAR`, `CSI`, `Brier Score`, `CRPS`, `FSS`) have been computed over held-out test storm events alongside all required baselines. *(Permitted claim: "Benchmark evaluation completed across $N$ independent held-out storm events.")*
- **Tier D — Demonstrated Research Finding**: Controlled empirical results on held-out storm events show a statistically significant improvement (non-overlapping 95% bootstrap confidence intervals or paired test $p < 0.05$) over the relevant baseline (`RRF > Individual Index`, `Retrieval-Conditioned > Non-Retrieval Backbone`, or `ConvLSTM > Persistence/HRRR`). **Until Tier D empirical evidence is logged in the Experiment Registry, no UI screen, documentation file, or Gemma 4 briefing may claim that dual-embedding retrieval or neural forecasting outperforms baselines.**

#### Phase-by-Phase Execution Roadmap (Upon Plan Approval)
1. **Phase 1**: Repository scaffolding, 5-class provenance schemas, and ERA5 / HRRR / MRMS / Open-Meteo ingestion with native/effective resolution metadata.
2. **Phase 2**: $L=7$ (`1000–200 hPa`) + surface dual-tensor construction ($T=13$), storm-event segmentation, and atomic event-grouped split verification.
3. **Phase 3**: Independent `EmbeddingGemma 2` (local) and `gemini-embedding-2` (cloud) 768-d pipelines + `Numerical Physical Analogue` index keyed by canonical `record_id`.
4. **Phase 4**: Independent retrieval evaluation (`Exps A, B, N`) and missing-rank-aware Reciprocal Rank Fusion (`Exps C1, C2`).
5. **Phase 5**: Baselines 1–4 (`Persistence`, `Climatology`, `Optical-Flow`, `HRRR`) and `VerticalEncodedConvLSTM` (Baseline 5).
6. **Phase 6**: Retrieval-conditioned forecasting (latent concatenation, cross-attention, gated FiLM) and `Spatiotemporal Transformer` (Baselines 6 & 7).
7. **Phase 7**: `Gemma 4` runtime adapter and 8-rule `StructuredProvenanceVerifier`.
8. **Phase 8**: Seven-workspace Kawaii Scientific Observatory UI, diagnostic Purple Cloud Mascot state machine, Experiment Registry, and full Python/FastAPI/PyTorch repository export.
