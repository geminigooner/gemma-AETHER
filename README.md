<div align="center">

☁️ GEMMA-AETHER

Atmospheric Evaluation & Tensor Forecasting Orbit

A Multimodal Atmospheric Intelligence & Scientific Forecasting Observatory

<img src="https://img.shields.io/badge/STATUS-FOUNDATION-blueviolet?style=for-the-badge" />
<img src="https://img.shields.io/badge/RESEARCH-MULTIMODAL-ff85c0?style=for-the-badge" />
<img src="https://img.shields.io/badge/FORECASTING-SPATIOTEMPORAL-c8a8e9?style=for-the-badge" />
<img src="https://img.shields.io/badge/LOCATION-WATAUGA%2C%20TN-ffd166?style=for-the-badge" />
<br />

Where atmospheric physics meets multimodal representation learning.

An experimental scientific observatory investigating whether semantic retrieval, physical atmospheric analogues, and neural forecasting can work together to improve localized precipitation prediction.

☁️　✦　🌧️　✦　🪻　✦　⚡

</div>

⸻

🪻 Overview

GEMMA-AETHER is a localized atmospheric intelligence and machine-learning research platform centered on Watauga, Tennessee.

The project investigates the intersection of:

* Multimodal embedding models.
* Independent semantic retrieval systems.
* Numerical atmospheric analogue retrieval.
* Spatiotemporal neural forecasting.
* Meteorological data analysis.
* Structured generative interpretation.
* Reproducible scientific evaluation.

Rather than treating weather forecasting as a single-model prediction problem, GEMMA-AETHER explores a layered architecture in which atmospheric observations, historical analogues, learned representations, and numerical forecasting models contribute to a unified research pipeline.

Research objective: Determine whether combining independent multimodal semantic representations with physically grounded atmospheric analogue retrieval can provide measurable improvements to localized precipitation forecasting.

⸻

🧠 Research Architecture

GEMMA-AETHER uses three independent retrieval pathways coordinated with a numerical forecasting architecture.

01 — Dual Multimodal Embedding Systems

Component	Model	Representation
Index A	EmbeddingGemma 2	Local, 768-dimensional
Index B	Gemini Embedding 2	Cloud, 768-dimensional
Index C	Numerical Physical Analogue	Local, 64-dimensional

Important architectural distinction:

EmbeddingGemma 2 and Gemini Embedding 2 produce vectors in independent representation spaces.

Equal dimensionality does not imply vector-space equivalence.

GEMMA-AETHER prohibits direct cross-model cosine similarity. Queries are embedded independently, searched against their respective indexes, and combined through Reciprocal Rank Fusion (RRF).

Calibrated score fusion and learned alignment remain subsequent research experiments.

02 — Numerical Atmospheric Analogue Retrieval

A third retrieval pathway represents atmospheric conditions through physically meaningful descriptors.

The numerical representation incorporates:

* 500 hPa geopotential height anomaly coefficients.
* 850 hPa moisture flux vectors.
* 850–500 hPa bulk wind shear.
* Cross-Appalachian pressure gradients.
* Domain precipitation fractions.

This creates a distinct research question:

Can semantic similarity and physical atmospheric similarity provide complementary information when retrieving historical weather analogues?

03 — Spatiotemporal Forecasting

Seven independent forecasting baselines:

#	Forecasting System
01	Persistence
02	Climatology
03	Optical Flow / Advection
04	NOAA HRRR
05	VerticalEncodedConvLSTM
06	Spatiotemporal Transformer
07	Retrieval-Conditioned VerticalEncodedConvLSTM

The retrieval-conditioned model investigates whether historical atmospheric representations can contribute useful information to neural forecasting through latent concatenation, cross-attention, and gated feature modulation.

Retrieval improvements are experimental hypotheses, not assumed outcomes.

⸻

🌎 Atmospheric Data Infrastructure

Data Source	Scientific Responsibility
ERA5	Historical atmospheric reanalysis
NOAA HRRR	High-resolution numerical weather prediction
NOAA MRMS	Radar-derived precipitation observations and verification
Open-Meteo	Point-based current-condition analysis and forecast comparison
NWS KTRI ASOS	Direct station observations

Resolution Awareness

All atmospheric data is associated with explicit resolution metadata:

* Native resolution.
* Co-registered grid spacing.
* Effective information resolution.

ERA5, HRRR, and MRMS are aligned onto a common Watauga research grid.

Interpolation does not create new atmospheric information. Regridded ERA5 fields retain their underlying coarse-resolution characteristics.

Open-Meteo point telemetry is maintained separately rather than artificially broadcast into spatial tensors.

Provenance Classification

Classification	Meaning
DIRECT_OBSERVATION	Physical station measurement
RADAR_DERIVED_ESTIMATE	Radar-derived precipitation estimate
MODEL_DERIVED_CURRENT	Current NWP analysis
REANALYSIS_FIELD	Historical assimilated atmospheric field
MODEL_FORECAST	Forward numerical or neural prediction

This classification is preserved throughout ingestion, processing, visualization, and generated meteorological briefings.

⸻

🧮 Atmospheric Tensor Architecture

GEMMA-AETHER uses paired tensor representations to distinguish upper-air atmospheric profiles from surface, terrain, and radar information.

Upper-Air Tensor

Dimension ordering:

[T, L, H, W, C]

Where:

Symbol	Dimension	Description
T	Time	Number of temporal observations
L	Levels	Atmospheric pressure levels
H	Height	Spatial grid height
W	Width	Spatial grid width
C	Channels	Atmospheric variables

Canonical configuration:

Dimension	Value
Time steps	13
Pressure levels	7
Spatial height	32
Spatial width	32
Atmospheric channels	7

Pressure levels:

1000, 925, 850, 700, 500, 300, 200 hPa

Variables include:

* Air temperature.
* Geopotential height.
* U/V wind components.
* Relative humidity.
* Specific humidity.
* Vertical velocity.

Surface, Terrain & Radar Tensor

Dimension ordering:

[T, H, W, C]

Ten channels represent:

* MRMS precipitation rate.
* One-hour quantitative precipitation estimates.
* Radar reflectivity.
* Radar quality index.
* 2 m temperature.
* 2 m dewpoint.
* Surface pressure.
* 10 m U/V wind.
* Terrain elevation.

Temporal Configuration

Primary mesoscale nowcasting mode:

Parameter	Configuration
Observations	13
Sampling interval	15 minutes
Elapsed intervals	12
Total observation window	180 minutes
Forecast horizons	15, 30, 60, 120, 180 minutes

Binary missing-data masks are maintained alongside atmospheric tensors.

⸻

🧠 VerticalEncodedConvLSTM

The primary neural forecasting architecture combines explicit vertical-profile encoding with spatial-temporal convolutional recurrence.

Architecture

Stage 1 — VerticalProfileEncoder

Processes upper-air atmospheric profiles across seven pressure levels.

Extracts vertical relationships, including:

* Wind shear.
* Temperature lapse relationships.
* Moisture gradients.
* Vertical atmospheric structure.

Produces a 64-channel spatial feature representation.

Stage 2 — Masked Surface Encoder

Processes radar, surface meteorology, and terrain through a masked convolutional stem.

Produces a 32-channel spatial representation.

Stage 3 — Feature Fusion

Combines vertical and surface representations into a 96-channel feature tensor.

Stage 4 — ConvLSTM Core

Three convolutional recurrent layers:

Layer	Input Channels	Output Channels
ConvLSTM 1	96	128
ConvLSTM 2	128	128
ConvLSTM 3	128	64

Stage 5 — Retrieval Conditioning

The retrieval-conditioned baseline introduces historical atmospheric representations through cross-attention or feature concatenation.

Stage 6 — Multi-Horizon Prediction

Produces spatial-grid and Watauga-specific predictions across five forecast horizons.

Outputs include precipitation probabilities, expected accumulation, precipitation intensity, temperature, humidity, and wind components.

⸻

🔎 Retrieval Fusion & Experiments

GEMMA-AETHER uses a canonical atmospheric record identifier to maintain correspondence between independent indexes.

aether_watauga3km_{utc_t0}_w180m_dt15m

All three indexes reference the same canonical training catalogue.

Retrieval Experiments

Experiment	Configuration
A	EmbeddingGemma 2 retrieval
B	Gemini Embedding 2 retrieval
N	Numerical Physical Analogue
C1	Dual Semantic RRF
C2	Tri-Modal Hybrid RRF

Reciprocal Rank Fusion

The RRF score for a document is calculated as:

RRF(d) = Σ [ w_m / (k + rank_m(d)) ]

Where:

* d = candidate document.
* m = retrieval system.
* w_m = retrieval-system weight.
* k = rank smoothing constant.
* rank_m(d) = document rank in retrieval system m.

Default configuration:

* k = 60
* Uniform pipeline weights.
* Canonical record-ID matching.
* Configurable missing-rank handling.
* Independent candidate retrieval.
* Storm-event filtering.
* Intra-event temporal deduplication.

The objective is to measure whether combining retrieval systems improves analogue relevance and downstream forecasting performance.

⸻

🌧️ Forecast Verification & Scientific Evaluation

Scientific performance is evaluated independently of interface presentation and generative interpretation.

Continuous Forecast Metrics

* Mean Absolute Error (MAE).
* Root Mean Square Error (RMSE).
* Bias.

Precipitation Detection

* Probability of Detection (POD).
* False Alarm Ratio (FAR).
* Critical Success Index (CSI).

Probabilistic Forecasting

* Brier Score.
* Continuous Ranked Probability Score (CRPS).
* Reliability diagrams.

Spatial Verification

Fractions Skill Score (FSS) across:

3 km · 9 km · 15 km · 27 km

Retrieval Evaluation

* Recall@K.
* Precision@K.
* Mean Reciprocal Rank (MRR).
* Normalized Discounted Cumulative Gain (NDCG).

Dataset Leakage Prevention

Historical observations are grouped by meteorological storm event before partitioning.

Target partitioning:

Partition	Target Share
Training	70%
Validation	15%
Testing	15%

Whole storm events remain atomic across partitions, with a 72-hour temporal embargo at split boundaries.

Normalization statistics are computed exclusively from training data.

Evaluation emphasizes independent held-out storm events rather than randomly distributed adjacent weather frames.

⸻

✨ Gemma 4 — Structured Meteorological Interpretation

Gemma 4 provides the generative interpretation layer.

Its responsibility is to translate numerical forecasting outputs into structured meteorological briefings.

It does not replace the numerical forecasting models.

The interpretation layer supports:

* Atmospheric condition summaries.
* Precipitation explanations.
* Forecast comparisons.
* Multi-horizon briefings.
* Qualitative uncertainty descriptions.
* Contextual interpretation of atmospheric diagnostics.

Structured Provenance Verification

Generated claims must reference explicit forecast fields.

The deterministic verifier checks:

1. Source-field resolution.
2. Variable identity.
3. Unit consistency.
4. Forecast-horizon alignment.
5. Geographic scope.
6. Provenance classification.
7. Numerical tolerance and deterministic rendering.
8. Unbound numerical claims in generated prose.

Numerical values displayed in the interface are rendered from verified source values rather than trusted directly from generated prose.

⸻

☁️ The Observatory Interface

GEMMA-AETHER presents its research infrastructure through seven consolidated workspaces.

Workspace	Purpose
01 — Observatory Overview	Atmospheric overview and system readiness
02 — Atmospheric Conditions	Tensor inspection and atmospheric profiles
03 — Vector Field Explorer	Wind fields, streamlines, and shear diagnostics
04 — Semantic Embedding Explorer	Independent embedding indexes and retrieval fusion
05 — Forecast Laboratory	Seven-baseline forecast comparisons
06 — Historical Data & Experiments	Event partitions and scientific verification
07 — Model & Pipeline Monitoring	Runtime diagnostics, evidence, and research export

Visual Identity

The interface follows the Kawaii Scientific Observatory aesthetic.

Its visual language combines:

* Twilight celestial banners.
* Warm cream scientific panels.
* Deep-plum beveled borders.
* Lavender and violet accents.
* Soft pink and celestial-gold details.
* Rounded observatory cards.
* A purple gradient cloud mascot.

The mascot is integrated into the diagnostic system rather than functioning as purely decorative artwork.

Its expressions correspond to actual atmospheric diagnostics, retrieval activity, and pipeline health.

⸻

🏗️ Technology Architecture

Layer	Technology
Frontend	React
Language	TypeScript
Build System	Vite
Styling	Tailwind CSS
Backend	Python
API	FastAPI
Machine Learning	PyTorch
Numerical Processing	NumPy / SciPy
Atmospheric Data	xarray
Embeddings	EmbeddingGemma 2 / Gemini Embedding 2
Forecast Interpretation	Gemma 4
Visualization	Interactive 2D / 3D WebGL

⸻

📁 Repository Architecture

GEMMA-AETHER/
│
├── frontend/
│   ├── observatory/
│   ├── components/
│   ├── visualizations/
│   └── mascot/
│
├── backend/
│   ├── api/
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   ├── forecasting/
│   ├── gemma/
│   └── evaluation/
│
├── research/
│   ├── datasets/
│   ├── experiments/
│   ├── notebooks/
│   └── results/
│
├── configs/
├── tests/
└── pyproject.toml

⸻

📊 Scientific Claims Policy

GEMMA-AETHER distinguishes functional implementation from experimentally demonstrated research findings.

Tier	Classification	Permitted Claim
A	Functional Implementation	Component executes successfully
B	Technical Validation	Architectural invariants pass testing
C	Controlled Scientific Evaluation	Held-out benchmark evaluation completed
D	Demonstrated Research Finding	Statistically supported improvement established

A working retrieval pipeline does not establish retrieval superiority.

A working neural forecasting model does not establish forecasting superiority.

Tier D claims require controlled evaluation against relevant baselines on independent held-out storm events.

Scientific claims must follow experimental evidence.

⸻

🚧 Development Roadmap

Phase	Milestone	Status
01	Repository, schemas & atmospheric ingestion	Planned
02	Tensor construction & event-isolated partitioning	Planned
03	Independent embedding pipelines	Planned
04	Retrieval evaluation & RRF fusion	Planned
05	Numerical baselines & VerticalEncodedConvLSTM	Planned
06	Retrieval-conditioned forecasting & Transformer	Planned
07	Gemma 4 interpretation & provenance verification	Planned
08	Observatory interface & research export	Planned

Current verified state: Project scaffold and reference specification assets. Scientific subsystems are awaiting implementation.

⸻

🔬 Research Questions

01 — Representation Complementarity

Do independent local and cloud multimodal embedding models retrieve complementary atmospheric analogues?

02 — Hybrid Retrieval

Does combining semantic retrieval with numerical physical analogue retrieval improve historical analogue relevance?

03 — Retrieval-Conditioned Forecasting

Can retrieved historical atmospheric states improve precipitation forecasting compared with an otherwise identical non-retrieval neural baseline?

04 — Multimodal Atmospheric Context

Can structured atmospheric descriptions and meteorological visualizations contribute useful information beyond numerical physical descriptors alone?

05 — Forecast Interpretability

Can generative interpretation provide useful meteorological context while maintaining deterministic numerical provenance?

These questions are evaluated through controlled experiments rather than assumed architectural benefits.

⸻

🌌 Project Philosophy

GEMMA-AETHER is built around a simple principle:

Scientific instrumentation should make its reasoning, measurements, and limitations inspectable.

The observatory is not designed to hide uncertainty behind an attractive interface or substitute generative explanations for physical measurements.

Every atmospheric value should have a source.

Every forecast should have a model.

Every retrieval result should have an identifiable record.

Every experimental claim should have evidence.

And every claimed improvement should survive comparison against a meaningful baseline.

⸻

<div align="center">

☁️ GEMMA-AETHER

Atmospheric Intelligence · Multimodal Retrieval · Scientific Forecasting

An experimental observatory for exploring the geometry of atmospheric information.

🪻　✦　🌧️　✦　⚡　✦　☁️

</div>