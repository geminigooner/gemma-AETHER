export enum ProvenanceClass {
  DIRECT_OBSERVATION = 'DIRECT_OBSERVATION',
  RADAR_DERIVED_ESTIMATE = 'RADAR_DERIVED_ESTIMATE',
  MODEL_DERIVED_CURRENT = 'MODEL_DERIVED_CURRENT',
  REANALYSIS_FIELD = 'REANALYSIS_FIELD',
  MODEL_FORECAST = 'MODEL_FORECAST',
}

export enum ImplementationStatus {
  IMPLEMENTED = 'IMPLEMENTED',
  IN_DEVELOPMENT = 'IN_DEVELOPMENT',
  PLANNED = 'PLANNED',
}

export enum ScientificClaimTier {
  TIER_A_FUNCTIONAL = 'TIER_A_FUNCTIONAL',
  TIER_B_TECHNICAL_VALIDATION = 'TIER_B_TECHNICAL_VALIDATION',
  TIER_C_CONTROLLED_EVALUATION = 'TIER_C_CONTROLLED_EVALUATION',
  TIER_D_DEMONSTRATED_FINDING = 'TIER_D_DEMONSTRATED_FINDING',
}

export enum MascotDiagnosticState {
  QUIESCENT_NOMINAL = 'QUIESCENT_NOMINAL',
  CONVECTIVE_PRECIP_ACTIVE = 'CONVECTIVE_PRECIP_ACTIVE',
  VERTICAL_WIND_SHEAR_ALERT = 'VERTICAL_WIND_SHEAR_ALERT',
  COLUMN_MOISTURE_SATURATION = 'COLUMN_MOISTURE_SATURATION',
  RETRIEVAL_FUSION_RUNNING = 'RETRIEVAL_FUSION_RUNNING',
  PROVENANCE_OR_PIPELINE_WARNING = 'PROVENANCE_OR_PIPELINE_WARNING',
}

export type WorkspaceId =
  | 'overview'
  | 'atmosphere'
  | 'vectors'
  | 'embeddings'
  | 'forecast'
  | 'experiments'
  | 'monitoring';

export const CANONICAL_PRESSURE_LEVELS_HPA = [1000, 925, 850, 700, 500, 300, 200] as const;
export type PressureLevelHpa = (typeof CANONICAL_PRESSURE_LEVELS_HPA)[number];

export const CANONICAL_LEVEL_ALTITUDES_KM: Record<PressureLevelHpa, number> = {
  1000: 0.11,
  925: 0.76,
  850: 1.46,
  700: 3.01,
  500: 5.57,
  300: 9.16,
  200: 11.78,
};

export const CANONICAL_TIMESTEPS_T = 13;
export const CANONICAL_INTERVALS = 12;
export const CANONICAL_GRID_H = 32;
export const CANONICAL_GRID_W = 32;
export const CANONICAL_FORECAST_HORIZONS_MIN = [15, 30, 60, 120, 180] as const;
export type ForecastHorizonMin = (typeof CANONICAL_FORECAST_HORIZONS_MIN)[number];

export interface ResolutionMetadata {
  sourceId: 'ERA5' | 'NOAA_HRRR' | 'NOAA_MRMS' | 'OPEN_METEO' | 'NWS_KTRI';
  nativeResolutionKm: string;
  regriddedGridSpacingKm: string;
  effectiveResolutionKm: string;
  temporalCadence: string;
  provenanceClass: ProvenanceClass;
  regriddingOperator: string;
  licenseAttribution: string;
  missingDataBehavior: string;
}

export interface PointWeatherMeasurement {
  locationName: string;
  latitude: number;
  longitude: number;
  elevationM: number;
  utcTimestamp: string;
  sourceEndpoint: string;
  provenanceClass: ProvenanceClass;
  stationId?: string;
  temperatureC: number;
  temperatureF: number;
  dewpointC: number;
  relativeHumidityPct: number;
  surfacePressureHpa: number;
  windSpeedMs: number;
  windSpeedMph: number;
  windDirectionDeg: number;
  windCardinal: string;
  precipitationMmHr: number;
  weatherConditionText: string;
  isLiveFetched: boolean;
}

export interface VerticalLevelSlice {
  levelHpa: PressureLevelHpa;
  altitudeKmMsl: number;
  domainMeanTempC: number;
  domainMeanGeopotentialM: number;
  domainMeanUWindMs: number;
  domainMeanVWindMs: number;
  domainMeanWindSpeedMs: number;
  domainMeanRhPct: number;
  domainMeanSpecificHumidityGKg: number;
  domainMeanOmegaPaS: number;
  validFractionMask: number;
  /** 16x16 downsampled preview grid for interactive vector rendering */
  uGrid16: number[][];
  vGrid16: number[][];
  tempGrid16: number[][];
  rhGrid16: number[][];
  precipGrid16: number[][];
  pressureGrid16: number[][];
}

export interface AtmosphericTensorWindow {
  recordId: string;
  stormEventId: string;
  partition: 'TRAIN' | 'VALIDATION' | 'TEST' | 'LIVE_ANALYSIS';
  utcT0: string;
  cadenceMinutes: 15 | 60;
  numObservationsT: 13;
  elapsedSpanMinutes: 180 | 720;
  timestampsUtc: string[];
  upperTensorShape: [13, 7, 32, 32, 7];
  surfaceTensorShape: [13, 32, 32, 10];
  validCellsPctUpper: number;
  validCellsPctSurface: number;
  normalizationHash: string;
  verticalSlicesAtT0: VerticalLevelSlice[];
  diagnostics: {
    deepLayerShear850To500Ms: number;
    lowLevelShear10mTo850Ms: number;
    meanLayerRh850To700Pct: number;
    omega700PaS: number;
    mrmsDomainMaxPrecipMmHr: number;
    mrmsDomain95thPrecipMmHr: number;
    mrmsMaxReflectivityDbz: number;
    synopticRegimeLabel: 'Clear' | 'Cloudy' | 'Rain' | 'Storm';
    semanticTags: string[];
  };
}

export interface EmbeddingMetadataRecord {
  recordId: string;
  stormEventId: string;
  partition: 'TRAIN' | 'VALIDATION' | 'TEST';
  utcT0: string;
  temporalWindowMinutes: 180;
  cadenceMinutes: 15;
  modelId: 'embeddinggemma-2' | 'gemini-embedding-2' | 'numerical-phys-descriptor-v1';
  modelVersion: string;
  embeddingDim: 768 | 64;
  inputModality: 'multimodal_text_field' | 'physical_tensor_descriptor';
  sourceDataset: string;
  preprocessingVersion: string;
  normalizationHash: string;
  synopticRegime: 'Clear' | 'Cloudy' | 'Rain' | 'Storm';
  summaryText: string;
  peakPrecipMmHr: number;
  bulkShearMs: number;
  moistureFluxGKgMs: number;
  /** Exploratory 3D PCA and UMAP coordinates computed strictly WITHIN this model's own index space */
  pca3d: [number, number, number];
  umap3d: [number, number, number];
  groundTruthRelevanceGrade: 0 | 1 | 2 | 3;
}

export interface RankedAnalogueCandidate {
  recordId: string;
  stormEventId: string;
  utcT0: string;
  synopticRegime: 'Clear' | 'Cloudy' | 'Rain' | 'Storm';
  summaryText: string;
  peakPrecipMmHr: number;
  bulkShearMs: number;
  groundTruthGrade: 0 | 1 | 2 | 3;
  rankLocalEmbeddingGemma2: number | null;
  scoreLocalCosine: number | null;
  rankCloudGeminiEmbedding2: number | null;
  scoreCloudCosine: number | null;
  rankNumericalPhysical: number | null;
  distanceNumericalL2: number | null;
  rrfDualSemanticScore: number;
  rankDualSemanticRrf: number;
  rrfTriModalScore: number;
  rankTriModalRrf: number;
}

export interface RetrievalExperimentMetrics {
  experimentId: 'EXP_A_LOCAL' | 'EXP_B_CLOUD' | 'EXP_N_NUMERICAL' | 'EXP_C1_DUAL_RRF' | 'EXP_C2_TRIMODAL_RRF' | 'EXP_D_CALIBRATED' | 'EXP_E_ALIGNED';
  name: string;
  status: ImplementationStatus;
  claimTier: ScientificClaimTier;
  recallAt5: number;
  precisionAt5: number;
  mrr: number;
  ndcgAt5: number;
  latencyMs: number;
  description: string;
}

export interface HorizonPrediction {
  horizonMinutes: ForecastHorizonMin;
  targetTimestampUtc: string;
  precipProb025Mm: number;
  precipProb250Mm: number;
  precipProb760Mm: number;
  expectedPrecipMm: number;
  precipIntensityMmHr: number;
  temperature2mC: number;
  temperature2mF: number;
  relativeHumidityPct: number;
  windSpeed10mMs: number;
  windSpeed10mMph: number;
  windCardinal: string;
  confidenceInterval95Mm: [number, number];
}

export type BaselineModelId =
  | 'persistence'
  | 'climatology'
  | 'optical_flow'
  | 'noaa_hrrr'
  | 'convlstm_no_ret'
  | 'st_transformer_no_ret'
  | 'convlstm_ret_conditioned';

export type ConditioningMode = 'latent_concat' | 'cross_attention' | 'gated_film';

export interface BaselineForecastSeries {
  modelId: BaselineModelId;
  modelName: string;
  status: ImplementationStatus;
  claimTier: ScientificClaimTier;
  usesRetrieval: boolean;
  conditioningMode?: ConditioningMode;
  architectureSummary: string;
  horizons: HorizonPrediction[];
  evaluationMetrics: {
    maePrecipMmHr: number;
    rmsePrecipMmHr: number;
    biasPrecipMmHr: number;
    pod025Mm: number;
    far025Mm: number;
    csi025Mm: number;
    pod250Mm: number;
    far250Mm: number;
    csi250Mm: number;
    brierScore025Mm: number;
    crpsMmHr: number;
    fss3km: number;
    fss9km: number;
    fss15km: number;
    fss27km: number;
    csi95BootstrapCi: [number, number];
    inferenceLatencyMs: number;
  };
}

export interface ClaimBindingTuple {
  source_field: string;
  variable_name: string;
  reported_value: number;
  true_value?: number;
  rendered_display?: string;
  unit: string;
  horizon_minutes: number;
  location_id: string;
  provenance_class: ProvenanceClass;
}

export interface VerifiedMeteorologicalClaim {
  claim_id: string;
  statement_prefix: string;
  statement_suffix: string;
  bindings: ClaimBindingTuple[];
}

export interface GemmaProvenanceBriefing {
  briefingId: string;
  generatedAtUtc: string;
  runtimeEndpoint: string;
  sourceModelId: BaselineModelId;
  summaryHeadline: string;
  claims: VerifiedMeteorologicalClaim[];
  qualitativeUncertaintyNote: string;
  verificationReport: {
    passedAllRules: boolean;
    checkedBindingsCount: number;
    rulesChecked: string[];
    violations: string[];
  };
}

export interface EvidenceRecord {
  subsystemId: string;
  subsystemName: string;
  layerNumber: number;
  status: ImplementationStatus;
  claimTier: ScientificClaimTier;
  sourceFilePaths: string[];
  implementedSymbols: string[];
  apiEndpoints: string[];
  executedTestNames: string[];
  lastTestPass: boolean;
  lastTestTimestampUtc: string;
  realDataEvidenceNote: string;
}

export interface ExperimentRegistryEntry {
  experiment_id: string;
  dataset_version: string;
  model_version: string;
  embedding_model: string;
  preprocessing_version: string;
  training_configuration: {
    optimizer: string;
    learningRate: number;
    batchSize: number;
    seed: number;
    conditioningMode: string;
    embargoHours: number;
  };
  evaluation_metrics: {
    csi025Mm: number;
    csi95Ci: [number, number];
    brierScore: number;
    crps: number;
    fss9km: number;
    ndcgAt5: number;
  };
  checkpoint_path: string;
  timestamp: string;
  claim_tier: ScientificClaimTier;
  notes: string;
}
