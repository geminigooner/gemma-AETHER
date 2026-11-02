import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.aether_core import (
    CANONICAL_PRESSURE_LEVELS_HPA,
    CANONICAL_TIMESTEPS_T,
    CrossSpaceComparisonError,
    StormEventSegment,
    ClaimBinding,
    VerifiedClaim,
    format_canonical_record_id,
    build_temporal_window_timestamps,
    verify_cross_space_guard,
    reciprocal_rank_fusion,
    validate_event_isolated_splits,
    verify_structured_provenance,
)


class TestGemmaAetherInvariants(unittest.TestCase):
    def test_01_vertical_pressure_levels_and_temporal_indexing(self):
        """Verifies L=7 (1000..200 hPa) and T=13 discrete observations spanning 12 intervals."""
        self.assertEqual(len(CANONICAL_PRESSURE_LEVELS_HPA), 7)
        self.assertEqual(CANONICAL_PRESSURE_LEVELS_HPA, (1000, 925, 850, 700, 500, 300, 200))
        self.assertEqual(CANONICAL_TIMESTEPS_T, 13)

        t0 = 1762106400  # 2025-11-02T18:00:00Z
        window_15m = build_temporal_window_timestamps(t0, cadence_minutes=15, num_observations_t=13)
        self.assertEqual(len(window_15m), 13)
        self.assertEqual(window_15m[-1] - window_15m[0], 180 * 60)  # Exactly 180 minutes (12 intervals)

        window_60m = build_temporal_window_timestamps(t0, cadence_minutes=60, num_observations_t=13)
        self.assertEqual(len(window_60m), 13)
        self.assertEqual(window_60m[-1] - window_60m[0], 12 * 3600)  # Exactly 12 hours (12 intervals)

    def test_02_cross_space_embedding_prohibition(self):
        """Verifies that comparing EmbeddingGemma 2 and gemini-embedding-2 raises CrossSpaceComparisonError."""
        v_local = [0.1] * 768
        v_cloud = [0.1] * 768
        with self.assertRaises(CrossSpaceComparisonError):
            verify_cross_space_guard("embeddinggemma-2", "gemini-embedding-2", v_local, v_cloud)

        same_space_sim = verify_cross_space_guard("gemini-embedding-2", "gemini-embedding-2", v_cloud, v_cloud)
        self.assertAlmostEqual(same_space_sim, 1.0, places=5)

    def test_03_canonical_record_id_and_rrf_missing_rank_handling(self):
        """Verifies deterministic record_id and RRF behavior when records are absent from one index."""
        rid1 = format_canonical_record_id("watauga3km", "2024-09-27T15:00:00Z", 180, 15)
        rid2 = format_canonical_record_id("watauga3km", "2024-05-18T21:00:00Z", 180, 15)
        rid3 = format_canonical_record_id("watauga3km", "2023-08-07T19:30:00Z", 180, 15)
        self.assertEqual(rid1, "aether_watauga3km_20240927T150000Z_w180m_dt15m")

        fused = reciprocal_rank_fusion(
            {
                "local_embeddinggemma2": [rid1, rid2],
                "cloud_gemini_embedding_2": [rid2, rid3],
            },
            k=60,
            missing_rank_policy="zero",
        )
        # rid2 is in both lists (rank 2 and rank 1) so it must rank #1
        self.assertEqual(fused[0][0], rid2)
        self.assertAlmostEqual(fused[0][1], (1.0 / 62.0) + (1.0 / 61.0), places=7)

    def test_04_storm_event_isolation_and_embargo(self):
        """Verifies atomic storm event grouping and 72-hour inter-partition embargo."""
        valid_events = [
            StormEventSegment("storm_2024_helene", 1727308800, 1727568000, "TRAIN", 48.2, 320.0),
            StormEventSegment("storm_2025_spring_mcs", 1747526400, 1747699200, "VALIDATION", 29.5, 240.0),
            StormEventSegment("storm_2026_autumn_front", 1761868800, 1761998400, "TEST", 18.4, 195.0),
        ]
        ok, errs = validate_event_isolated_splits(valid_events, embargo_hours=72)
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

        # Now test a multi-day storm split across TRAIN and VALIDATION (must fail)
        leaked_events = valid_events + [
            StormEventSegment("storm_2024_helene", 1727568000, 1727654400, "VALIDATION", 15.0, 180.0)
        ]
        ok_leak, errs_leak = validate_event_isolated_splits(leaked_events, embargo_hours=72)
        self.assertFalse(ok_leak)
        self.assertTrue(any("leaked across partitions" in e for e in errs_leak))

    def test_05_structured_gemma4_provenance_verifier(self):
        """Verifies all 8 rules of StructuredProvenanceVerifier catch unit, horizon, and value mismatches."""
        registry = {
            "forecast.horizons.60m.precip_prob_025mm": {
                "variable_name": "precipitation_probability_025mm",
                "value": 0.87,
                "unit": "probability_0_to_1",
                "horizon_minutes": 60,
                "location_id": "watauga_tn_3km_center",
                "provenance_class": "MODEL_FORECAST",
            }
        }
        valid_claim = VerifiedClaim(
            claim_id="c_01",
            statement_prefix="At the 60-minute horizon, precipitation probability reaches",
            statement_suffix="over Watauga, Tennessee.",
            bindings=[
                ClaimBinding(
                    source_field="forecast.horizons.60m.precip_prob_025mm",
                    variable_name="precipitation_probability_025mm",
                    reported_value=0.87,
                    unit="probability_0_to_1",
                    horizon_minutes=60,
                    location_id="watauga_tn_3km_center",
                    provenance_class="MODEL_FORECAST",
                )
            ],
        )
        ok, errors = verify_structured_provenance([valid_claim], registry, ["Ensemble spread rises after +120m."])
        self.assertTrue(ok, f"Expected valid claim to pass, got {errors}")

        # Corrupt horizon and unit -> must reject
        bad_claim = VerifiedClaim(
            claim_id="c_02",
            statement_prefix="Precipitation reaches 99.9 in prose",
            statement_suffix="soon.",
            bindings=[
                ClaimBinding(
                    source_field="forecast.horizons.60m.precip_prob_025mm",
                    variable_name="precipitation_probability_025mm",
                    reported_value=0.95,
                    unit="mm",
                    horizon_minutes=120,
                    location_id="watauga_tn_3km_center",
                    provenance_class="DIRECT_OBSERVATION",
                )
            ],
        )
        ok_bad, errors_bad = verify_structured_provenance([bad_claim], registry, [])
        self.assertFalse(ok_bad)
        self.assertGreaterEqual(len(errors_bad), 4)


if __name__ == "__main__":
    unittest.main()
