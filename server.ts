import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import {
  buildWatauga5DTensorWindow,
  computeSevenForecastBaselines,
  evaluateMascotDiagnosticState,
  EXPERIMENT_REGISTRY_LOG,
  fetchLiveWataugaObservations,
  generateVerifiedGemma4Briefing,
  HISTORICAL_STORM_CATALOGUE,
  RESOLUTION_METADATA_CATALOG,
  runAutomatedEvidenceTestSuite,
  runDualEmbeddingAndPhysicalRetrieval,
  verifyCrossSpaceGuard,
} from './src/server/aetherEngine.ts';
import { PYTHON_REPO_FILES } from './src/shared/pythonRepoFiles.ts';
import { BaselineModelId, ConditioningMode } from './src/shared/aetherTypes.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '5mb' }));

  // 1. Unified Observatory Bootstrap State
  app.get('/api/v1/observatory/state', async (req, res) => {
    try {
      const scenarioMode = (req.query.scenario as 'frontal_rain' | 'convective_shear' | 'quiescent_clear') || 'frontal_rain';
      const cadenceMinutes = Number(req.query.cadence) === 60 ? 60 : 15;
      const conditioningMode = (req.query.conditioning as ConditioningMode) || 'cross_attention';

      const [liveWeather, retrievalRes, briefing, evidenceSuite] = await Promise.all([
        fetchLiveWataugaObservations(),
        runDualEmbeddingAndPhysicalRetrieval({
          queryText:
            scenarioMode === 'convective_shear'
              ? 'Deep wind shear mesoscale vortex and heavy orographic rain over Watauga and Roan Mountain'
              : scenarioMode === 'quiescent_clear'
                ? 'Broad 500 hPa anticyclonic ridge with dry subsiding air and clear sky over Watauga'
                : 'Frontal rainband with 850 hPa moisture influx, low pressure trough, and 86% relative humidity over Watauga, TN',
          queryRegime: scenarioMode === 'convective_shear' ? 'Storm' : scenarioMode === 'quiescent_clear' ? 'Clear' : 'Rain',
          queryPeakPrecipMmHr: scenarioMode === 'convective_shear' ? 28.4 : scenarioMode === 'quiescent_clear' ? 0.0 : 9.8,
          queryBulkShearMs: scenarioMode === 'convective_shear' ? 23.8 : scenarioMode === 'quiescent_clear' ? 6.2 : 16.4,
          queryMoistureFlux: scenarioMode === 'convective_shear' ? 280.0 : scenarioMode === 'quiescent_clear' ? 42.0 : 182.4,
          rrfConstantK: 60,
          missingRankPolicy: 'zero',
          poolDepth: 50,
        }),
        generateVerifiedGemma4Briefing({
          selectedModelId: 'convlstm_ret_conditioned',
          conditioningMode,
          injectSyntheticViolation: 'none',
        }),
        runAutomatedEvidenceTestSuite(),
      ]);

      const tensorWindow = buildWatauga5DTensorWindow(cadenceMinutes, scenarioMode);
      const mascotDiag = evaluateMascotDiagnosticState(
        tensorWindow,
        false,
        !briefing.verificationReport.passedAllRules,
      );
      const baselines = computeSevenForecastBaselines(conditioningMode, tensorWindow.utcT0);

      res.json({
        liveWeather,
        tensorWindow,
        mascotDiag,
        resolutionMetadata: RESOLUTION_METADATA_CATALOG,
        retrieval: retrievalRes,
        baselines,
        briefing,
        experimentRegistry: EXPERIMENT_REGISTRY_LOG,
        evidenceSuite,
        pythonRepoFiles: PYTHON_REPO_FILES,
      });
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Failed to compute observatory state' });
    }
  });

  // 2. Reconstruct 5D Upper-Air + 4D Surface Tensor Window
  app.get('/api/v1/atmosphere/tensor', (req, res) => {
    try {
      const scenarioMode = (req.query.scenario as 'frontal_rain' | 'convective_shear' | 'quiescent_clear') || 'frontal_rain';
      const cadenceMinutes = Number(req.query.cadence) === 60 ? 60 : 15;
      const tensorWindow = buildWatauga5DTensorWindow(cadenceMinutes, scenarioMode);
      const mascotDiag = evaluateMascotDiagnosticState(tensorWindow);
      res.json({ tensorWindow, mascotDiag });
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Failed to build 5D tensor window' });
    }
  });

  // 3. Execute Dual-Embedding + Numerical Physical Analogue Search & RRF
  app.post('/api/v1/retrieval/search', async (req, res) => {
    try {
      const {
        queryText = 'Frontal rainband with 850 hPa moisture influx over Watauga, TN',
        queryRegime = 'Rain',
        queryPeakPrecipMmHr = 9.8,
        queryBulkShearMs = 16.4,
        queryMoistureFlux = 182.4,
        rrfConstantK = 60,
        missingRankPolicy = 'zero',
        poolDepth = 50,
      } = req.body || {};

      const result = await runDualEmbeddingAndPhysicalRetrieval({
        queryText,
        queryRegime,
        queryPeakPrecipMmHr: Number(queryPeakPrecipMmHr),
        queryBulkShearMs: Number(queryBulkShearMs),
        queryMoistureFlux: Number(queryMoistureFlux),
        rrfConstantK: Number(rrfConstantK),
        missingRankPolicy: missingRankPolicy === 'floor_k_plus_1' ? 'floor_k_plus_1' : 'zero',
        poolDepth: Number(poolDepth),
      });

      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Retrieval search failed' });
    }
  });

  // 4. Test Cross-Space Embedding Prohibition Guard
  app.post('/api/v1/retrieval/test-cross-space-guard', (_req, res) => {
    try {
      const sample = HISTORICAL_STORM_CATALOGUE[0];
      verifyCrossSpaceGuard(
        'embeddinggemma-2',
        'gemini-embedding-2',
        sample.vecLocal768,
        sample.vecCloud768,
      );
      res.json({ blocked: false, message: 'Unexpectedly permitted cross-space comparison' });
    } catch (err: any) {
      res.json({
        blocked: true,
        errorName: 'CrossSpaceComparisonError',
        message: err?.message || 'Cross-space cosine similarity prohibited.',
        timestampUtc: new Date().toISOString(),
      });
    }
  });

  // 5. Generate & Verify Gemma 4 Structured Briefing (with optional Fault Injection test)
  app.post('/api/v1/gemma/interpret', async (req, res) => {
    try {
      const {
        selectedModelId = 'convlstm_ret_conditioned',
        conditioningMode = 'cross_attention',
        injectSyntheticViolation = 'none',
      } = req.body || {};

      const briefing = await generateVerifiedGemma4Briefing({
        selectedModelId: selectedModelId as BaselineModelId,
        conditioningMode: conditioningMode as ConditioningMode,
        injectSyntheticViolation,
      });

      res.json({ briefing });
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Gemma 4 briefing synthesis failed' });
    }
  });

  // 6. Execute Live Python + TypeScript Evidence Verification Suite
  app.post('/api/v1/evidence/run-tests', async (_req, res) => {
    try {
      const result = await runAutomatedEvidenceTestSuite();
      res.json(result);
    } catch (err: any) {
      res.status(500).json({ error: err?.message || 'Failed to execute evidence test suite' });
    }
  });

  // 7. Export Reproducible Python/FastAPI/PyTorch Repository Bundle
  app.get('/api/v1/repository/export', (_req, res) => {
    res.json({
      projectName: 'GEMMA-AETHER',
      version: '0.2.0-second-pass-baseline',
      exportedAtUtc: new Date().toISOString(),
      files: PYTHON_REPO_FILES,
    });
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`GEMMA-AETHER Scientific Observatory Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
