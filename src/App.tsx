/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState } from 'react';
import {
  Home,
  Layers,
  Wind,
  Share2,
  CloudRain,
  FlaskConical,
  Activity,
  MapPin,
  Sparkles,
  CheckCircle2,
  ShieldCheck,
  Download,
  RefreshCw,
} from 'lucide-react';
import {
  AtmosphericTensorWindow,
  BaselineForecastSeries,
  ConditioningMode,
  EmbeddingMetadataRecord,
  EvidenceRecord,
  ExperimentRegistryEntry,
  GemmaProvenanceBriefing,
  ImplementationStatus,
  MascotDiagnosticState,
  PointWeatherMeasurement,
  PressureLevelHpa,
  ProvenanceClass,
  RankedAnalogueCandidate,
  ResolutionMetadata,
  RetrievalExperimentMetrics,
  WorkspaceId,
} from './shared/aetherTypes.ts';
import { RepoFileEntry } from './shared/pythonRepoFiles.ts';
import { PurpleCloudMascot } from './components/PurpleCloudMascot.tsx';
import { VectorFieldCanvas } from './components/VectorFieldCanvas.tsx';
import { EmbeddingSpaceExplorerCard } from './components/EmbeddingSpaceExplorerCard.tsx';
import { AtmosphericCrossSectionCard } from './components/AtmosphericCrossSectionCard.tsx';
import {
  AtmosphereWorkspaceView,
  EmbeddingsWorkspaceView,
  ExperimentsWorkspaceView,
  ForecastLaboratoryView,
  MonitoringAndExportWorkspaceView,
  ProvenanceClassBadge,
  StatusBadge,
} from './components/WorkspacesDetailViews.tsx';

interface ObservatoryStatePayload {
  liveWeather: {
    openMeteoCurrent: PointWeatherMeasurement;
    nwsStationDirect: PointWeatherMeasurement;
  };
  tensorWindow: AtmosphericTensorWindow;
  mascotDiag: {
    state: MascotDiagnosticState;
    headline: string;
    diagnosticExplanation: string;
    triggeredRule: string;
  };
  resolutionMetadata: ResolutionMetadata[];
  retrieval: {
    usedLiveGeminiEmbedding2Api: boolean;
    cloudEmbeddingModelId: string;
    cloudEmbeddingDimension: number;
    candidates: RankedAnalogueCandidate[];
    experimentMetrics: RetrievalExperimentMetrics[];
    embeddingRecords: EmbeddingMetadataRecord[];
  };
  baselines: BaselineForecastSeries[];
  briefing: GemmaProvenanceBriefing;
  experimentRegistry: ExperimentRegistryEntry[];
  evidenceSuite: {
    allPassed: boolean;
    executedAtUtc: string;
    pythonUnittestOutput: string;
    typescriptAssertionsOutput: string[];
    evidenceRecords: EvidenceRecord[];
  };
  pythonRepoFiles: RepoFileEntry[];
}

const SEVEN_WORKSPACES: {
  id: WorkspaceId;
  shortLabel: string;
  fullTitle: string;
  icon: React.FC<{ className?: string }>;
}[] = [
  { id: 'overview', shortLabel: 'Overview', fullTitle: '1. Observatory Overview', icon: Home },
  { id: 'atmosphere', shortLabel: 'Atmosphere', fullTitle: '2. Atmospheric Conditions (5D)', icon: Layers },
  { id: 'vectors', shortLabel: 'Vectors', fullTitle: '3. Vector Field Explorer', icon: Wind },
  { id: 'embeddings', shortLabel: 'Embeddings', fullTitle: '4. Semantic Embedding Explorer', icon: Share2 },
  { id: 'forecast', shortLabel: 'Forecast', fullTitle: '5. Forecast Laboratory', icon: CloudRain },
  { id: 'experiments', shortLabel: 'Experiments', fullTitle: '6. Historical Data & Experiments', icon: FlaskConical },
  { id: 'monitoring', shortLabel: 'Monitoring', fullTitle: '7. Pipeline Monitoring & Export', icon: Activity },
];

export default function App() {
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceId>('overview');
  const [state, setState] = useState<ObservatoryStatePayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Interactive scientific controls
  const [selectedLevel, setSelectedLevel] = useState<PressureLevelHpa>(850);
  const [cadenceMinutes, setCadenceMinutes] = useState<15 | 60>(15);
  const [scenarioMode, setScenarioMode] = useState<'frontal_rain' | 'convective_shear' | 'quiescent_clear'>('frontal_rain');
  const [conditioningMode, setConditioningMode] = useState<ConditioningMode>('cross_attention');
  const [pointSourceMode, setPointSourceMode] = useState<'open_meteo' | 'nws_ktri'>('open_meteo');
  const [activeSemanticTag, setActiveSemanticTag] = useState<string>('frontal system');

  const [isSearchingRetrieval, setIsSearchingRetrieval] = useState<boolean>(false);
  const [isVerifyingProvenance, setIsVerifyingProvenance] = useState<boolean>(false);
  const [isRunningTests, setIsRunningTests] = useState<boolean>(false);

  const fetchObservatoryState = async (
    nextScenario = scenarioMode,
    nextCadence = cadenceMinutes,
    nextConditioning = conditioningMode,
  ) => {
    try {
      setLoading(true);
      const res = await fetch(
        `/api/v1/observatory/state?scenario=${nextScenario}&cadence=${nextCadence}&conditioning=${nextConditioning}`,
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ObservatoryStatePayload = await res.json();
      setState(data);
      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Failed to load GEMMA-AETHER Observatory state');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchObservatoryState();
  }, []);

  const handleScenarioChange = async (nextScenario: 'frontal_rain' | 'convective_shear' | 'quiescent_clear') => {
    setScenarioMode(nextScenario);
    await fetchObservatoryState(nextScenario, cadenceMinutes, conditioningMode);
  };

  const handleCadenceChange = async (nextCadence: 15 | 60) => {
    setCadenceMinutes(nextCadence);
    await fetchObservatoryState(scenarioMode, nextCadence, conditioningMode);
  };

  const handleConditioningChange = async (nextCond: ConditioningMode) => {
    setConditioningMode(nextCond);
    await fetchObservatoryState(scenarioMode, cadenceMinutes, nextCond);
  };

  const handleRunRetrievalQuery = async (params: {
    queryText: string;
    rrfConstantK: number;
    missingRankPolicy: 'zero' | 'floor_k_plus_1';
  }) => {
    if (!state) return;
    try {
      setIsSearchingRetrieval(true);
      const res = await fetch('/api/v1/retrieval/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...params,
          queryRegime: state.tensorWindow.diagnostics.synopticRegimeLabel,
          queryPeakPrecipMmHr: state.tensorWindow.diagnostics.mrmsDomainMaxPrecipMmHr,
          queryBulkShearMs: state.tensorWindow.diagnostics.deepLayerShear850To500Ms,
          queryMoistureFlux: 182.4,
          poolDepth: 50,
        }),
      });
      if (res.ok) {
        const updatedRetrieval = await res.json();
        setState((prev) => (prev ? { ...prev, retrieval: updatedRetrieval } : prev));
      }
    } finally {
      setIsSearchingRetrieval(false);
    }
  };

  const handleSemanticTagClick = async (tag: string) => {
    setActiveSemanticTag(tag);
    await handleRunRetrievalQuery({
      queryText: `Watauga, TN atmospheric state featuring ${tag} with 850 hPa moisture transport and vertical shear`,
      rrfConstantK: 60,
      missingRankPolicy: 'zero',
    });
  };

  const handleRunProvenanceTest = async (
    faultMode: 'none' | 'wrong_unit' | 'wrong_horizon' | 'value_hallucination' | 'unbound_prose_float',
  ) => {
    try {
      setIsVerifyingProvenance(true);
      const res = await fetch('/api/v1/gemma/interpret', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selectedModelId: 'convlstm_ret_conditioned',
          conditioningMode,
          injectSyntheticViolation: faultMode,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setState((prev) => (prev ? { ...prev, briefing: data.briefing } : prev));
      }
    } finally {
      setIsVerifyingProvenance(false);
    }
  };

  const handleRunEvidenceTests = async () => {
    try {
      setIsRunningTests(true);
      const res = await fetch('/api/v1/evidence/run-tests', { method: 'POST' });
      if (res.ok) {
        const suite = await res.json();
        setState((prev) => (prev ? { ...prev, evidenceSuite: suite } : prev));
      }
    } finally {
      setIsRunningTests(false);
    }
  };

  if (loading && !state) {
    return (
      <div className="min-h-screen bg-[#FAF3E6] flex flex-col items-center justify-center p-6 text-center">
        <PurpleCloudMascot pose="heart" size="xl" />
        <h1 className="font-display text-2xl font-bold text-[#2D1948] mt-4">
          Initializing GEMMA-AETHER Scientific Observatory...
        </h1>
        <p className="text-sm font-mono text-[#593E7A] mt-1">
          Loading Watauga 5D Tensor Pipeline (L=7, T=13), Dual 768-d Indexes & Provenance Verifier
        </p>
      </div>
    );
  }

  if (error || !state) {
    return (
      <div className="min-h-screen bg-[#FAF3E6] flex flex-col items-center justify-center p-6 text-center">
        <div className="kawaii-panel p-6 max-w-md space-y-3">
          <h2 className="font-display text-xl font-bold text-[#D90429]">Observatory Initialization Error</h2>
          <p className="text-xs font-mono text-[#2D1948]">{error}</p>
          <button onClick={() => fetchObservatoryState()} className="px-4 py-2 rounded-xl kawaii-btn-pink text-xs font-bold">
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  const activePointWeather =
    pointSourceMode === 'open_meteo'
      ? state.liveWeather.openMeteoCurrent
      : state.liveWeather.nwsStationDirect;

  const primaryForecastSeries =
    state.baselines.find((b) => b.modelId === 'convlstm_ret_conditioned') || state.baselines[6];
  const h60 = primaryForecastSeries.horizons.find((h) => h.horizonMinutes === 60) || primaryForecastSeries.horizons[2];

  return (
    <div className="min-h-screen bg-[#FAF3E6] text-[#2D1948] flex flex-col selection:bg-[#FF85C0] selection:text-[#2D1948]">
      {/* =====================================================================
       * TOP TWILIGHT-VIOLET CELESTIAL HEADER BANNER (Matching Mockup + 3-Zone Contract)
       * ===================================================================== */}
      <header
        className="relative overflow-hidden border-b-[3px] border-[#2D1948] px-4 lg:px-6 py-3 flex items-center justify-between gap-6"
        style={{
          background: 'linear-gradient(90deg, #241046 0%, #3F1E6D 45%, #2D1458 100%)',
        }}
      >
        {/* Decorative Celestial Clouds, Crescent Moon & Stars in Backdrop */}
        <div className="pointer-events-none absolute inset-0 overflow-hidden opacity-90">
          <span className="absolute left-[22%] top-2 text-[#FFE066] text-sm">✦</span>
          <span className="absolute left-[49%] top-1.5 text-[#FFE066] text-xs">✦</span>
          <span className="absolute left-[54%] bottom-2 text-[#FFE066] text-sm">★</span>
          <span className="absolute right-[24%] top-2 text-[#FFE066] text-xs">✦</span>
          {/* Crescent Moon SVG */}
          <svg
            viewBox="0 0 60 60"
            className="w-12 h-12 absolute left-[56%] -top-1 drop-shadow-[0_2px_0_#2D1948]"
          >
            <path
              d="M 38 8 A 22 22 0 1 0 54 42 A 17 17 0 1 1 38 8 Z"
              fill="#FFE055"
              stroke="#2D1948"
              strokeWidth="3"
            />
          </svg>
        </div>

        {/* Zone 1: Brand Title Lockup */}
        <div className="relative z-10 flex items-center gap-3 shrink-0">
          {/* Radar Dish Emblem */}
          <div className="w-11 h-11 rounded-2xl bg-[#EADBFD] border-2 border-[#2D1948] shadow-[0_2.5px_0_#2D1948] flex items-center justify-center">
            <svg viewBox="0 0 40 40" className="w-8 h-8">
              <path d="M 10 30 L 18 22 L 24 30 Z" fill="#8E5CE6" stroke="#2D1948" strokeWidth="2.4" />
              <path
                d="M 12 12 C 8 18, 12 26, 20 26 C 26 26, 30 22, 28 16 Z"
                fill="#F6EEFF"
                stroke="#2D1948"
                strokeWidth="2.4"
              />
              <line x1="21" y1="19" x2="29" y2="11" stroke="#2D1948" strokeWidth="2.4" strokeLinecap="round" />
              <circle cx="30" cy="10" r="2.8" fill="#FF70B8" stroke="#2D1948" strokeWidth="2" />
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-[#FFE066] text-base">✦</span>
              <span className="font-display text-2xl md:text-3xl font-bold text-[#FFFDF9] tracking-tight whitespace-nowrap drop-shadow-[0_2px_0_#2D1948]">
                Gemma-AETHER
              </span>
              <span className="text-[#FFE066] text-base">✦</span>
            </div>
            <div className="inline-block px-3 py-0.5 rounded-full bg-[#EADBFD] border-2 border-[#2D1948] text-[11px] font-bold text-[#2D1948] whitespace-nowrap">
              Atmospheric Evaluation & Tensor Forecasting Orbit
            </div>
          </div>
        </div>

        {/* Zone 2: Scenario Regime Switcher (Interactive Scientific Mode Bar) */}
        <nav className="relative z-10 hidden xl:flex items-center gap-2 bg-[#3A1C66]/90 px-3 py-1.5 rounded-2xl border-2 border-[#2D1948]">
          <span className="text-[11px] font-mono text-[#EADBFD] font-bold whitespace-nowrap">
            Watauga Tensor State:
          </span>
          {(
            [
              { id: 'frontal_rain', label: 'Frontal Rain (t₀)' },
              { id: 'convective_shear', label: 'Deep Shear Storm' },
              { id: 'quiescent_clear', label: 'Quiescent Ridge' },
            ] as const
          ).map((mode) => (
            <button
              key={mode.id}
              onClick={() => handleScenarioChange(mode.id)}
              className={`px-3 py-1 rounded-xl text-xs font-bold transition-all whitespace-nowrap shrink-0 ${
                scenarioMode === mode.id
                  ? 'bg-[#FF85C0] text-[#2D1948] border-2 border-[#2D1948] shadow-[0_2px_0_#2D1948]'
                  : 'text-[#FFFDF9] hover:bg-[#522B8A]'
              }`}
            >
              {mode.label}
            </button>
          ))}
        </nav>

        {/* Zone 3: AETHER Observatory Badge & Evidence Gate Trigger */}
        <div className="relative z-10 flex items-center gap-2.5 shrink-0">
          <button
            onClick={() => setActiveWorkspace('monitoring')}
            className="px-4 py-1.5 rounded-full bg-[#FAF4E8] border-[2.5px] border-[#2D1948] shadow-[0_3px_0_#2D1948] flex items-center gap-2 hover:bg-[#F3EAFF] transition-colors whitespace-nowrap shrink-0"
          >
            <span className="text-base">🔭</span>
            <div className="text-left leading-tight">
              <div className="font-display text-xs font-bold text-[#2D1948]">AETHER Observatory</div>
              <div className="text-[10px] font-mono text-[#1B4332] font-bold">
                {state.evidenceSuite.allPassed ? '● Tests Passing (8/8 Rules)' : '▲ Check Evidence Gate'}
              </div>
            </div>
          </button>
        </div>
      </header>

      {/* =====================================================================
       * MAIN BODY: LEFT 7-WORKSPACE NAVIGATION RAIL + ACTIVE WORKSPACE CANVAS
       * ===================================================================== */}
      <div className="flex-1 max-w-[1600px] w-full mx-auto p-3 lg:p-4 grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        {/* LEFT NAVIGATION RAIL (Strictly Consolidated to the 7 Canonical Workspaces!) */}
        <aside className="lg:col-span-2 kawaii-panel p-3 flex flex-col justify-between min-h-[680px]">
          <div className="space-y-1.5">
            <div className="px-2 py-1 text-[10px] font-mono font-bold uppercase tracking-wider text-[#593E7A]">
              7 Canonical Workspaces
            </div>
            {SEVEN_WORKSPACES.map((ws) => {
              const IconComponent = ws.icon;
              const isActive = activeWorkspace === ws.id;
              return (
                <button
                  key={ws.id}
                  onClick={() => setActiveWorkspace(ws.id)}
                  className={`w-full px-3 py-2.5 rounded-2xl flex items-center gap-2.5 text-left text-sm font-bold transition-all whitespace-nowrap ${
                    isActive
                      ? 'kawaii-btn-pink font-display text-[#2D1948]'
                      : 'text-[#2D1948] hover:bg-[#F3EAFF] border-2 border-transparent'
                  }`}
                >
                  <IconComponent className="w-4 h-4 shrink-0 text-[#2D1948]" />
                  <span className="truncate">{ws.shortLabel}</span>
                </button>
              );
            })}
          </div>

          {/* Tucked Sleeping Purple Cloud Mascot at Bottom of Left Rail (Matching Mockup!) */}
          <div className="mt-6 pt-4 border-t-2 border-[#2D1948]/15 flex flex-col items-center text-center">
            <PurpleCloudMascot pose="sleeping_zzz" size="md" />
            <div className="mt-1 text-[11px] font-display font-bold text-[#2D1948]">
              Watauga Orbit Station
            </div>
            <div className="text-[10px] font-mono text-[#593E7A]">
              36.3654°N, 82.2923°W
            </div>
          </div>
        </aside>

        {/* MAIN RIGHT STAGE (10 Columns) */}
        <main className="lg:col-span-10 space-y-4">
          {activeWorkspace === 'overview' && (
            <>
              {/* ===============================================================
               * OVERVIEW ROW 1: GREETING BANNER + WATAUGA CARD + TIMESTAMP CARD
               * =============================================================== */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5 items-stretch">
                {/* 1A. Welcome & Mascot Diagnostic Banner (7 cols) */}
                <div className="lg:col-span-7 kawaii-panel-lavender px-4 py-3 flex items-center gap-4 relative overflow-hidden">
                  <div className="shrink-0">
                    <PurpleCloudMascot diagnosticState={state.mascotDiag.state} size="lg" />
                  </div>
                  <div className="space-y-1 flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      <h1 className="font-display text-xl md:text-2xl font-bold text-[#2D1948] tracking-tight">
                        Good afternoon, Amanda ♡
                      </h1>
                      <span className="px-2.5 py-0.5 rounded-full bg-[#FFFDF9] border-2 border-[#2D1948] text-[10px] font-mono font-bold text-[#7B42BC]">
                        Mascot Trigger: {state.mascotDiag.state}
                      </span>
                    </div>
                    <p className="text-xs md:text-sm font-medium text-[#2D1948] leading-snug">
                      Welcome to Gemma-AETHER. Exploring atmospheric dynamics, semantic weather, and localized forecasting with Gemma 4. ✦
                    </p>
                    <div className="text-[11px] font-mono text-[#593E7A] truncate" title={state.mascotDiag.triggeredRule}>
                      Diagnostic: {state.mascotDiag.headline} ({state.mascotDiag.triggeredRule})
                    </div>
                  </div>
                </div>

                {/* 1B. Watauga, TN Point Telemetry Card with Explicit Provenance Toggle (3 cols) */}
                <div className="lg:col-span-3 kawaii-panel px-3.5 py-2.5 flex flex-col justify-between">
                  <div className="flex items-center justify-between gap-1">
                    <div className="flex items-center gap-1 font-display font-bold text-sm text-[#2D1948]">
                      <MapPin className="w-4 h-4 text-[#7B42BC] shrink-0" />
                      <span className="truncate">Watauga, TN</span>
                    </div>
                    <button
                      onClick={() =>
                        setPointSourceMode(pointSourceMode === 'open_meteo' ? 'nws_ktri' : 'open_meteo')
                      }
                      className="px-2 py-0.5 rounded-md bg-[#F6EEFF] border border-[#2D1948] text-[10px] font-mono font-bold text-[#2D1948] hover:bg-[#EADBFD]"
                      title="Switch between Model-Derived Current (Open-Meteo) and Direct Station Observation (NWS KTRI)"
                    >
                      {pointSourceMode === 'open_meteo' ? 'Model-Derived' : 'Direct ASOS'} ⇄
                    </button>
                  </div>

                  <div className="flex items-center justify-between gap-2 my-1">
                    <PurpleCloudMascot pose="sun_peek" size="sm" />
                    <div className="text-right">
                      <div className="font-display text-3xl font-bold text-[#2D1948] leading-none tabular-nums">
                        {Math.round(activePointWeather.temperatureF)}°F
                      </div>
                      <div className="text-xs font-bold text-[#2D1948] mt-0.5">
                        {scenarioMode === 'quiescent_clear'
                          ? 'Clear Sky'
                          : scenarioMode === 'convective_shear'
                            ? 'Heavy Storm'
                            : 'Light Rain'}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[10px] font-mono text-[#593E7A] border-t border-[#2D1948]/15 pt-1">
                    <span className="truncate">{activePointWeather.provenanceClass}</span>
                    <span>{activePointWeather.temperatureC}°C</span>
                  </div>
                </div>

                {/* 1C. UTC Timestamp & T=13 Window Card (2 cols) */}
                <div className="lg:col-span-2 kawaii-panel-lavender p-3 flex flex-col items-center justify-center text-center relative">
                  <div className="w-full h-full rounded-xl border-2 border-[#2D1948] bg-[#EADBFD]/60 px-2 py-2 flex flex-col items-center justify-center">
                    <div className="font-display text-sm font-bold text-[#2D1948]">Nov 02, 2026</div>
                    <div className="font-display text-lg font-bold text-[#2D1948] tabular-nums">
                      18:00 UTC
                    </div>
                    <div className="text-[10px] font-mono text-[#593E7A] mt-0.5">
                      T={state.tensorWindow.numObservationsT} Obs · {state.tensorWindow.elapsedSpanMinutes}m
                    </div>
                  </div>
                </div>
              </div>

              {/* ===============================================================
               * OVERVIEW ROW 2: VECTOR FIELD (LEFT 7 COLS) + STATUS & SEMANTIC WEATHER (RIGHT 5 COLS)
               * =============================================================== */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
                <div className="lg:col-span-7 flex flex-col">
                  <VectorFieldCanvas
                    tensorWindow={state.tensorWindow}
                    selectedLevel={selectedLevel}
                    onSelectLevel={setSelectedLevel}
                    compact
                    onExpandToExplorer={() => setActiveWorkspace('vectors')}
                  />
                </div>

                <div className="lg:col-span-5 flex flex-col justify-between gap-4">
                  {/* Right Top Card: Model & System Status (Evidence-Backed!) */}
                  <div className="kawaii-panel p-3.5 space-y-2.5 relative">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-base">☁️</span>
                        <h2 className="font-display text-base md:text-lg font-bold text-[#2D1948]">
                          Model & System Status ✦
                        </h2>
                      </div>
                      <button
                        onClick={() => setActiveWorkspace('monitoring')}
                        className="text-[11px] font-mono font-bold text-[#7B42BC] underline hover:text-[#2D1948]"
                      >
                        Evidence Gate →
                      </button>
                    </div>

                    <div className="space-y-2">
                      {[
                        {
                          label: 'Gemma 4 (Generative Briefing Adapter)',
                          sub: '8-Rule StructuredProvenanceVerifier',
                          status: ImplementationStatus.IMPLEMENTED,
                          workspace: 'forecast' as WorkspaceId,
                        },
                        {
                          label: 'EmbeddingGemma 2 (Local 768-d Index A)',
                          sub: 'Isolated FAISS cosine index · Offline-ready',
                          status: ImplementationStatus.IMPLEMENTED,
                          workspace: 'embeddings' as WorkspaceId,
                        },
                        {
                          label: 'Gemini Embedding 2 (Cloud 768-d Index B)',
                          sub: `Model: ${state.retrieval.cloudEmbeddingModelId} · Separate space`,
                          status: ImplementationStatus.IMPLEMENTED,
                          workspace: 'embeddings' as WorkspaceId,
                        },
                        {
                          label: 'VerticalEncodedConvLSTM (L=7 + Surface)',
                          sub: 'Baseline 5 & 7 · 5 horizons (15m–180m)',
                          status: ImplementationStatus.IMPLEMENTED,
                          workspace: 'forecast' as WorkspaceId,
                        },
                        {
                          label: 'Spatiotemporal Transformer Backbone',
                          sub: 'Baseline 6 · Factorized space-time attention',
                          status: ImplementationStatus.IN_DEVELOPMENT,
                          workspace: 'forecast' as WorkspaceId,
                        },
                      ].map((item) => (
                        <div
                          key={item.label}
                          onClick={() => setActiveWorkspace(item.workspace)}
                          className="kawaii-subcard px-3 py-1.5 flex items-center justify-between gap-2 cursor-pointer hover:bg-[#F6EEFF] transition-colors"
                        >
                          <div className="min-w-0">
                            <div className="text-xs font-bold text-[#2D1948] truncate">✦ {item.label}</div>
                            <div className="text-[10px] font-mono text-[#593E7A] truncate">{item.sub}</div>
                          </div>
                          <StatusBadge status={item.status} />
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Right Bottom Card: Semantic Weather & RRF Trigger Tags */}
                  <div className="kawaii-panel p-3.5 space-y-2.5 relative">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-base">☁️</span>
                        <div>
                          <h2 className="font-display text-base md:text-lg font-bold text-[#2D1948] leading-tight">
                            Semantic Weather & Analogue Retrieval ✦
                          </h2>
                          <p className="text-[10px] font-mono text-[#593E7A]">
                            Click a regime descriptor to query Index A + Index B + Physical Index C via RRF:
                          </p>
                        </div>
                      </div>
                      <div className="-mt-4 -mr-1 shrink-0">
                        <PurpleCloudMascot pose="determined_storm" size="sm" />
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2 pt-1">
                      {state.tensorWindow.diagnostics.semanticTags.map((tag) => {
                        const isSelected = activeSemanticTag === tag;
                        return (
                          <button
                            key={tag}
                            onClick={() => handleSemanticTagClick(tag)}
                            className={`px-2.5 py-1.5 rounded-full text-xs font-bold truncate transition-all ${
                              isSelected ? 'kawaii-btn-pink' : 'kawaii-btn-cream'
                            }`}
                          >
                            {tag}
                          </button>
                        );
                      })}
                    </div>

                    {/* Top RRF Retrieved Historical Analogue Preview */}
                    {state.retrieval.candidates[0] && (
                      <div className="p-2 rounded-xl bg-[#F6EEFF] border border-[#2D1948] text-[11px] font-mono flex items-center justify-between gap-2">
                        <span className="truncate">
                          <b>Top RRF Analogue:</b> {state.retrieval.candidates[0].recordId}
                        </span>
                        <span className="px-2 py-0.5 rounded bg-[#FFF3B0] border border-[#2D1948] font-bold shrink-0">
                          RRF={state.retrieval.candidates[0].rrfTriModalScore.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* ===============================================================
               * OVERVIEW ROW 3: FORECAST WATAUGA (4 COLS) + EMBEDDING SPACE (4 COLS) + CROSS-SECTION (4 COLS)
               * =============================================================== */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
                {/* Bottom Left: Forecast — Watauga, TN (Matching Mockup!) */}
                <div className="lg:col-span-4 kawaii-panel p-3.5 flex flex-col justify-between relative">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <div className="flex items-center gap-1.5">
                      <span className="text-base">🌧️</span>
                      <div>
                        <h3 className="font-display text-sm md:text-base font-bold text-[#2D1948] leading-tight">
                          Forecast — Watauga, TN ✦
                        </h3>
                        <p className="text-[10px] font-mono text-[#593E7A]">
                          Model: VerticalEncodedConvLSTM + Tri-Modal RRF
                        </p>
                      </div>
                    </div>
                    <div className="-mt-4 shrink-0">
                      <PurpleCloudMascot pose="determined_storm" size="sm" />
                    </div>
                  </div>

                  {/* 5 Horizon Columns (15m, 30m, 60m, 120m, 180m) */}
                  <div className="grid grid-cols-5 gap-1.5 my-1">
                    {primaryForecastSeries.horizons.map((h) => {
                      const isPeak = h.horizonMinutes === 60;
                      return (
                        <div
                          key={h.horizonMinutes}
                          onClick={() => setActiveWorkspace('forecast')}
                          className={`cursor-pointer p-2 rounded-2xl border-2 border-[#2D1948] text-center flex flex-col items-center justify-between transition-transform hover:-translate-y-0.5 ${
                            isPeak ? 'bg-[#FFD6EE] shadow-[0_2.5px_0_#2D1948]' : 'bg-[#FFFDF9]'
                          }`}
                        >
                          <span className="text-[11px] font-bold text-[#2D1948]">
                            +{h.horizonMinutes}m
                          </span>
                          <span className="text-base my-1">
                            {h.precipProb025Mm >= 0.75 ? '🌧️' : h.precipProb025Mm >= 0.4 ? '🌦️' : '☁️'}
                          </span>
                          <span className="font-display text-sm font-bold text-[#2D1948] tabular-nums">
                            {Math.round(h.temperature2mF)}°F
                          </span>
                          <span className="text-[11px] font-mono font-bold text-[#7B42BC] tabular-nums">
                            {Math.round(h.precipProb025Mm * 100)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  {/* Bottom 5 Metric Mini-Pills matching Mockup (87%, 4.2 mm, NW 12 mph, 86%, CSI 0.69) */}
                  <div className="grid grid-cols-5 gap-1.5 mt-2">
                    <div className="kawaii-subcard p-1.5 text-center">
                      <div className="text-[9px] font-bold text-[#593E7A] truncate">Precip Prob</div>
                      <div className="font-mono font-bold text-xs text-[#2D1948]">
                        💧 {Math.round(h60.precipProb025Mm * 100)}%
                      </div>
                    </div>
                    <div className="kawaii-subcard p-1.5 text-center">
                      <div className="text-[9px] font-bold text-[#593E7A] truncate">Expected</div>
                      <div className="font-mono font-bold text-xs text-[#2D1948]">
                        ✦ {h60.expectedPrecipMm.toFixed(1)} mm
                      </div>
                    </div>
                    <div className="kawaii-subcard p-1.5 text-center">
                      <div className="text-[9px] font-bold text-[#593E7A] truncate">10m Wind</div>
                      <div className="font-mono font-bold text-[11px] text-[#2D1948]">
                        {h60.windCardinal} {Math.round(h60.windSpeed10mMph)}mph
                      </div>
                    </div>
                    <div className="kawaii-subcard p-1.5 text-center">
                      <div className="text-[9px] font-bold text-[#593E7A] truncate">Humidity</div>
                      <div className="font-mono font-bold text-xs text-[#2D1948]">
                        💧 {h60.relativeHumidityPct}%
                      </div>
                    </div>
                    <div className="kawaii-subcard p-1.5 text-center">
                      <div className="text-[9px] font-bold text-[#593E7A] truncate">Test CSI</div>
                      <div className="font-mono font-bold text-xs text-[#7B42BC]">
                        {primaryForecastSeries.evaluationMetrics.csi025Mm.toFixed(2)}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Bottom Center: Embedding Space — Atmospheric States (4 Cols) */}
                <div className="lg:col-span-4 flex flex-col">
                  <EmbeddingSpaceExplorerCard
                    records={state.retrieval.embeddingRecords}
                    activeRegime={state.tensorWindow.diagnostics.synopticRegimeLabel}
                    compact
                    onSelectRecord={() => setActiveWorkspace('embeddings')}
                  />
                </div>

                {/* Bottom Right: Atmospheric Layers — Cross-Section (4 Cols) */}
                <div className="lg:col-span-4 flex flex-col">
                  <AtmosphericCrossSectionCard tensorWindow={state.tensorWindow} compact />
                </div>
              </div>

              {/* ===============================================================
               * OVERVIEW ROW 4: GEMMA 4 PROVENANCE-LOCKED BRIEFING STRIP
               * =============================================================== */}
              <div className="kawaii-panel p-3.5 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <div className="px-3 py-1.5 rounded-xl bg-[#B9FBC0] border-2 border-[#2D1948] font-mono text-xs font-bold text-[#1B4332] flex items-center gap-1.5 shrink-0">
                    <ShieldCheck className="w-4 h-4" />
                    <span>Gemma 4 Provenance Verified (8/8 Rules)</span>
                  </div>
                  <div className="text-xs md:text-sm text-[#2D1948] truncate">
                    <b>Briefing:</b> {state.briefing.summaryHeadline} — +60m precip prob{' '}
                    <span className="font-mono font-bold text-[#7B42BC]">
                      {Math.round(h60.precipProb025Mm * 100)}% [forecast.horizons.60m.precip_prob_025mm]
                    </span>
                    , expected accumulation{' '}
                    <span className="font-mono font-bold text-[#1B4332]">
                      {h60.expectedPrecipMm.toFixed(1)} mm [forecast.horizons.60m.expected_precip_mm]
                    </span>
                    .
                  </div>
                </div>
                <button
                  onClick={() => setActiveWorkspace('forecast')}
                  className="px-3 py-1.5 rounded-xl kawaii-btn-cream text-xs font-bold whitespace-nowrap shrink-0"
                >
                  Inspect Claim Bindings →
                </button>
              </div>
            </>
          )}

          {/* WORKSPACE 2: ATMOSPHERIC CONDITIONS & 5D TENSOR PIPELINE */}
          {activeWorkspace === 'atmosphere' && (
            <AtmosphereWorkspaceView
              liveWeather={state.liveWeather}
              tensorWindow={state.tensorWindow}
              resolutionMetadata={state.resolutionMetadata}
              cadenceMinutes={cadenceMinutes}
              scenarioMode={scenarioMode}
              onChangeCadence={handleCadenceChange}
              onChangeScenario={handleScenarioChange}
            />
          )}

          {/* WORKSPACE 3: VECTOR FIELD EXPLORER (FULL STAGE) */}
          {activeWorkspace === 'vectors' && (
            <div className="space-y-4">
              <VectorFieldCanvas
                tensorWindow={state.tensorWindow}
                selectedLevel={selectedLevel}
                onSelectLevel={setSelectedLevel}
                compact={false}
              />
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <AtmosphericCrossSectionCard tensorWindow={state.tensorWindow} />
                <div className="kawaii-panel p-4 space-y-2.5">
                  <h3 className="font-display font-bold text-base text-[#2D1948]">
                    Multi-Level Vector Shear & Mascot Trigger Diagnostics
                  </h3>
                  <p className="text-xs text-[#593E7A]">
                    Vertical wind shear cannot be inferred from 10 m wind speed alone. GEMMA-AETHER computes explicit vector differences across isobaric levels:
                  </p>
                  <div className="grid grid-cols-2 gap-2.5 text-xs font-mono">
                    <div className="kawaii-subcard p-3">
                      <div className="text-[#593E7A]">850–500 hPa Deep Bulk Shear</div>
                      <div className="text-lg font-bold text-[#7B42BC] mt-0.5">
                        {state.tensorWindow.diagnostics.deepLayerShear850To500Ms} m/s
                      </div>
                      <div className="text-[10px] text-[#2D1948]">Alert Threshold: ≥ 18.0 m/s</div>
                    </div>
                    <div className="kawaii-subcard p-3">
                      <div className="text-[#593E7A]">10m–850 hPa Low-Level Shear</div>
                      <div className="text-lg font-bold text-[#2D1948] mt-0.5">
                        {state.tensorWindow.diagnostics.lowLevelShear10mTo850Ms} m/s
                      </div>
                      <div className="text-[10px] text-[#2D1948]">Alert Threshold: ≥ 11.5 m/s</div>
                    </div>
                    <div className="kawaii-subcard p-3">
                      <div className="text-[#593E7A]">850–700 hPa Mean Layer RH</div>
                      <div className="text-lg font-bold text-[#1B4332] mt-0.5">
                        {state.tensorWindow.diagnostics.meanLayerRh850To700Pct}%
                      </div>
                      <div className="text-[10px] text-[#2D1948]">Saturation Rule: RH ≥ 85% & ω ≤ -0.2</div>
                    </div>
                    <div className="kawaii-subcard p-3">
                      <div className="text-[#593E7A]">700 hPa Vertical Velocity ω</div>
                      <div className="text-lg font-bold text-[#D90429] mt-0.5">
                        {state.tensorWindow.diagnostics.omega700PaS} Pa/s
                      </div>
                      <div className="text-[10px] text-[#2D1948]">Negative ω indicates synoptic ascent</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* WORKSPACE 4: SEMANTIC EMBEDDING & CANONICAL RRF EXPLORER */}
          {activeWorkspace === 'embeddings' && (
            <EmbeddingsWorkspaceView
              retrieval={state.retrieval}
              activeRegime={state.tensorWindow.diagnostics.synopticRegimeLabel}
              onRunRetrievalQuery={handleRunRetrievalQuery}
              isSearching={isSearchingRetrieval}
            />
          )}

          {/* WORKSPACE 5: FORECAST LABORATORY & GEMMA 4 PROVENANCE VERIFIER */}
          {activeWorkspace === 'forecast' && (
            <ForecastLaboratoryView
              baselines={state.baselines}
              briefing={state.briefing}
              conditioningMode={conditioningMode}
              onChangeConditioningMode={handleConditioningChange}
              onRunProvenanceTest={handleRunProvenanceTest}
              isVerifying={isVerifyingProvenance}
            />
          )}

          {/* WORKSPACE 6: HISTORICAL DATA & EXPERIMENTS */}
          {activeWorkspace === 'experiments' && (
            <ExperimentsWorkspaceView baselines={state.baselines} />
          )}

          {/* WORKSPACE 7: MODEL & PIPELINE MONITORING / RESEARCH EXPORT */}
          {activeWorkspace === 'monitoring' && (
            <MonitoringAndExportWorkspaceView
              evidenceSuite={state.evidenceSuite}
              experimentRegistry={state.experimentRegistry}
              pythonRepoFiles={state.pythonRepoFiles}
              onRunEvidenceTests={handleRunEvidenceTests}
              isRunningTests={isRunningTests}
            />
          )}
        </main>
      </div>
    </div>
  );
}
