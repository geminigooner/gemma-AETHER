import React, { useState } from 'react';
import { AtmosphericTensorWindow, VerticalLevelSlice } from '../shared/aetherTypes.ts';
import { PurpleCloudMascot } from './PurpleCloudMascot.tsx';

interface AtmosphericCrossSectionCardProps {
  tensorWindow: AtmosphericTensorWindow;
  compact?: boolean;
}

export const AtmosphericCrossSectionCard: React.FC<AtmosphericCrossSectionCardProps> = ({
  tensorWindow,
  compact = false,
}) => {
  const [selectedSlice, setSelectedSlice] = useState<VerticalLevelSlice | null>(null);
  const slices = tensorWindow.verticalSlicesAtT0;
  const isSaturated =
    tensorWindow.diagnostics.meanLayerRh850To700Pct >= 85.0 &&
    tensorWindow.diagnostics.omega700PaS <= -0.2;

  return (
    <div className="kawaii-panel p-3.5 flex flex-col justify-between relative">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5">
          <span className="text-base">⛰️</span>
          <div>
            <h3 className="font-display text-sm md:text-base font-bold text-[#2D1948] leading-tight">
              Atmospheric Layers — Cross-Section
            </h3>
            <p className="text-[10px] font-mono text-[#593E7A]">
              L=7 Isobaric Levels (1000–200 hPa · 0–12 km MSL) over Watauga & Blue Ridge
            </p>
          </div>
        </div>
        <span className="text-xs font-bold text-[#FFB703]">✦</span>
      </div>

      {/* Main Cross-Section Diagram + Right RH(%) Colorbar */}
      <div className="flex items-stretch gap-2.5 flex-1">
        <div className="relative flex-1 h-44 rounded-xl bg-[#241158] border-2 border-[#2D1948] overflow-hidden p-1">
          <svg viewBox="0 0 260 165" className="w-full h-full overflow-visible">
            <defs>
              <linearGradient id="layer200" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3B1D82" />
                <stop offset="100%" stopColor="#5D3FD3" />
              </linearGradient>
              <linearGradient id="layer500" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4895EF" />
                <stop offset="100%" stopColor="#4CC9F0" />
              </linearGradient>
              <linearGradient id="layer700" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#72EFDD" />
                <stop offset="100%" stopColor="#B565F7" />
              </linearGradient>
              <linearGradient id="layer850" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#FF5CA8" />
                <stop offset="100%" stopColor="#FFD166" />
              </linearGradient>
              <linearGradient id="terrainFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4A2574" />
                <stop offset="100%" stopColor="#261242" />
              </linearGradient>
            </defs>

            {/* Stratified Isobaric Moisture Wave Bands (z = 0 to 12 km / 1000 to 200 hPa) */}
            {/* 9-12 km (300-200 hPa) */}
            <path
              d="M 28 12 Q 75 6, 120 16 T 210 10 L 252 14 L 252 55 L 28 55 Z"
              fill="url(#layer200)"
              opacity="0.95"
            />
            {/* 6-9 km (500-300 hPa) */}
            <path
              d="M 28 42 Q 70 28, 122 44 T 212 30 L 252 38 L 252 90 L 28 90 Z"
              fill="url(#layer500)"
              opacity="0.9"
            />
            {/* 3-6 km (700-500 hPa) */}
            <path
              d="M 28 74 Q 78 55, 125 76 T 208 56 L 252 68 L 252 125 L 28 125 Z"
              fill="url(#layer700)"
              opacity="0.92"
            />
            {/* 0-3 km (1000-850 hPa Saturated Moisture Inflow Core) */}
            <path
              d="M 28 102 Q 82 82, 128 104 T 206 78 L 252 92 L 252 152 L 28 152 Z"
              fill="url(#layer850)"
              opacity="0.95"
            />

            {/* Interactive Isobaric Level Reference Lines (L=7) */}
            {slices.map((s) => {
              const y = 148 - (s.altitudeKmMsl / 12.0) * 134;
              return (
                <g
                  key={s.levelHpa}
                  className="cursor-pointer group"
                  onMouseEnter={() => setSelectedSlice(s)}
                  onMouseLeave={() => setSelectedSlice(null)}
                >
                  <line
                    x1="28"
                    y1={y}
                    x2="252"
                    y2={y}
                    stroke="rgba(255,255,255,0.35)"
                    strokeWidth="1"
                    strokeDasharray="3 3"
                  />
                  <text x="248" y={y - 2} textAnchor="end" fontSize="7.5" fontFamily="JetBrains Mono" fill="#FFFDF9">
                    {s.levelHpa}hPa ({s.domainMeanRhPct}%)
                  </text>
                </g>
              );
            })}

            {/* Appalachian Topographic Ridge Silhouette (Watauga Valley 451m -> Holston & Roan Mountain 1.9km) */}
            <path
              d="M 28 148
                 L 28 136
                 Q 55 128, 78 116
                 Q 102 104, 126 128
                 Q 152 142, 178 118
                 Q 206 94, 232 114
                 L 252 122
                 L 252 148 Z"
              fill="url(#terrainFill)"
              stroke="#FFFDF9"
              strokeWidth="2.2"
            />

            {/* Vertical Altitude Axis z (km) */}
            <line x1="28" y1="10" x2="28" y2="148" stroke="#FFFDF9" strokeWidth="2" />
            {[
              { km: 12, y: 14 },
              { km: 9, y: 47 },
              { km: 6, y: 81 },
              { km: 3, y: 114 },
              { km: 0, y: 148 },
            ].map((tick) => (
              <g key={tick.km}>
                <line x1="24" y1={tick.y} x2="28" y2={tick.y} stroke="#FFFDF9" strokeWidth="1.8" />
                <text x="20" y={tick.y + 3.5} textAnchor="end" fontSize="10" fontWeight="bold" fill="#FFFDF9">
                  {tick.km}
                </text>
              </g>
            ))}
            <text
              x="7"
              y="80"
              transform="rotate(-90 7 80)"
              textAnchor="middle"
              fontSize="10"
              fontWeight="bold"
              fill="#FFFDF9"
            >
              z (km)
            </text>
          </svg>

          {selectedSlice && (
            <div className="pointer-events-none absolute top-1.5 left-9 px-2.5 py-1 rounded-lg bg-[#FFFDF9]/95 border border-[#2D1948] text-[10px] font-mono text-[#2D1948]">
              <b>{selectedSlice.levelHpa} hPa</b> ({selectedSlice.altitudeKmMsl} km): T={selectedSlice.domainMeanTempC}°C · RH={selectedSlice.domainMeanRhPct}% · ω={selectedSlice.domainMeanOmegaPaS} Pa/s
            </div>
          )}
        </div>

        {/* Right RH (%) Colorbar */}
        <div className="w-16 flex flex-col items-center justify-between py-0.5 shrink-0 select-none">
          <div className="text-[11px] font-bold font-mono text-[#2D1948]">RH (%)</div>
          <div className="flex items-stretch gap-1.5 flex-1 my-1">
            <div
              className="w-4 rounded-full border-2 border-[#2D1948]"
              style={{
                background:
                  'linear-gradient(180deg, #FF5CA8 0%, #FFB703 28%, #4CC9F0 58%, #5D3FD3 82%, #241158 100%)',
              }}
            />
            <div className="flex flex-col justify-between text-[10px] font-mono font-bold text-[#2D1948]">
              <span>100</span>
              <span>80</span>
              <span>60</span>
              <span>40</span>
              <span>20</span>
              <span>0</span>
            </div>
          </div>
        </div>
      </div>

      {/* Diagnostic Trigger Footer + Teary/Droplet Mascot */}
      <div className="mt-2 flex items-center justify-between gap-2 text-[10px] font-mono text-[#593E7A]">
        <span>
          850–700hPa RH={tensorWindow.diagnostics.meanLayerRh850To700Pct}% · ω₇₀₀={tensorWindow.diagnostics.omega700PaS} Pa/s
        </span>
        {compact && (
          <div className="-mt-6 -mb-2 shrink-0" title={isSaturated ? '850-700 hPa Saturation & Upward Motion Active' : 'Sub-saturation column'}>
            <PurpleCloudMascot pose={isSaturated ? 'teary_saturation' : 'sun_peek'} size="sm" />
          </div>
        )}
      </div>
    </div>
  );
};
