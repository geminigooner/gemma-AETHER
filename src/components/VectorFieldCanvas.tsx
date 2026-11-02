import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, Maximize2 } from 'lucide-react';
import {
  AtmosphericTensorWindow,
  CANONICAL_PRESSURE_LEVELS_HPA,
  PressureLevelHpa,
  VerticalLevelSlice,
} from '../shared/aetherTypes.ts';
import { PurpleCloudMascot } from './PurpleCloudMascot.tsx';

export type FieldMode = 'wind' | 'temp' | 'pressure' | 'precip';

interface VectorFieldCanvasProps {
  tensorWindow: AtmosphericTensorWindow;
  selectedLevel: PressureLevelHpa;
  onSelectLevel: (lvl: PressureLevelHpa) => void;
  compact?: boolean;
  onExpandToExplorer?: () => void;
}

const TIMELINE_LABELS = [
  { idx: 0, label: '-180m' },
  { idx: 4, label: '-120m' },
  { idx: 8, label: '-60m' },
  { idx: 12, label: '+0h (t₀)' },
  { idx: 13, label: '+15m' },
  { idx: 14, label: '+30m' },
  { idx: 15, label: '+60m' },
  { idx: 16, label: '+120m' },
  { idx: 17, label: '+180m' },
];

export const VectorFieldCanvas: React.FC<VectorFieldCanvasProps> = ({
  tensorWindow,
  selectedLevel,
  onSelectLevel,
  compact = false,
  onExpandToExplorer,
}) => {
  const [fieldMode, setFieldMode] = useState<FieldMode>('wind');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [timeStepIdx, setTimeStepIdx] = useState<number>(12); // Default at t_0 (index 12 of T=13)
  const [hoverProbe, setHoverProbe] = useState<{
    gridRow: number;
    gridCol: number;
    lat: number;
    lon: number;
    u: number;
    v: number;
    speed: number;
    tempC: number;
    rhPct: number;
    precipMmHr: number;
  } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animPhaseRef = useRef<number>(0);

  const activeSlice: VerticalLevelSlice =
    tensorWindow.verticalSlicesAtT0.find((s) => s.levelHpa === selectedLevel) ||
    tensorWindow.verticalSlicesAtT0[2];

  useEffect(() => {
    if (!isPlaying) return;
    let rafId = 0;
    let lastTime = performance.now();

    const renderLoop = (now: number) => {
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      animPhaseRef.current = (animPhaseRef.current + dt * 1.4) % 100;
      drawCanvas(animPhaseRef.current);
      rafId = requestAnimationFrame(renderLoop);
    };

    rafId = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, fieldMode, selectedLevel, timeStepIdx, tensorWindow]);

  useEffect(() => {
    if (!isPlaying) {
      drawCanvas(animPhaseRef.current);
    }
  }, [fieldMode, selectedLevel, timeStepIdx, tensorWindow, isPlaying]);

  const drawCanvas = (phase: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // 1. Vibrant Kawaii Observatory Field Backdrop Gradient matching mockup
    const timeOffset = (timeStepIdx - 12) * 0.08;
    const vortexCenterX = width * (0.64 + timeOffset * 0.25);
    const vortexCenterY = height * (0.52 - timeOffset * 0.12);

    const bgGrad = ctx.createLinearGradient(0, 0, width, height);
    if (fieldMode === 'wind') {
      bgGrad.addColorStop(0, '#1C0D52');
      bgGrad.addColorStop(0.35, '#2E3FE8');
      bgGrad.addColorStop(0.65, '#8A2BE2');
      bgGrad.addColorStop(1, '#FF3EA5');
    } else if (fieldMode === 'temp') {
      bgGrad.addColorStop(0, '#2B2D8F');
      bgGrad.addColorStop(0.45, '#9A3FE8');
      bgGrad.addColorStop(0.75, '#FF6B9D');
      bgGrad.addColorStop(1, '#FFD166');
    } else if (fieldMode === 'pressure') {
      bgGrad.addColorStop(0, '#1D184A');
      bgGrad.addColorStop(0.5, '#5936B4');
      bgGrad.addColorStop(1, '#D668CD');
    } else {
      bgGrad.addColorStop(0, '#1A1446');
      bgGrad.addColorStop(0.4, '#189AB4');
      bgGrad.addColorStop(0.75, '#8B38E8');
      bgGrad.addColorStop(1, '#FF5CA8');
    }
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // 2. Mesoscale Jet / Vortex Radial Luminance Core (matching cyan & magenta glow in mockup)
    const cyanGlow = ctx.createRadialGradient(
      width * 0.32,
      height * 0.34,
      10,
      width * 0.32,
      height * 0.34,
      width * 0.42,
    );
    cyanGlow.addColorStop(0, 'rgba(72, 245, 255, 0.72)');
    cyanGlow.addColorStop(0.5, 'rgba(56, 168, 255, 0.28)');
    cyanGlow.addColorStop(1, 'rgba(28, 13, 82, 0)');
    ctx.fillStyle = cyanGlow;
    ctx.fillRect(0, 0, width, height);

    const vortexGlow = ctx.createRadialGradient(
      vortexCenterX,
      vortexCenterY,
      8,
      vortexCenterX,
      vortexCenterY,
      width * 0.38,
    );
    vortexGlow.addColorStop(0, 'rgba(64, 235, 255, 0.65)');
    vortexGlow.addColorStop(0.42, 'rgba(255, 72, 182, 0.58)');
    vortexGlow.addColorStop(0.78, 'rgba(255, 198, 82, 0.32)');
    vortexGlow.addColorStop(1, 'rgba(46, 26, 71, 0)');
    ctx.fillStyle = vortexGlow;
    ctx.fillRect(0, 0, width, height);

    // 3. Subtle 32x32 Tensor Grid Overlay
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 16; i++) {
      const gx = (width / 16) * i;
      const gy = (height / 16) * i;
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(width, gy);
      ctx.stroke();
    }

    // 4. Regional State / Appalachian Boundary Lines (TN / NC / VA border geometry)
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.88)';
    ctx.lineWidth = 2.6;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // VA / TN border line across top-middle
    ctx.beginPath();
    ctx.moveTo(width * 0.18, height * 0.22);
    ctx.lineTo(width * 0.42, height * 0.21);
    ctx.lineTo(width * 0.46, height * 0.16);
    ctx.lineTo(width * 0.72, height * 0.15);
    ctx.stroke();

    // TN / NC Blue Ridge Escarpment border line winding past Watauga
    ctx.beginPath();
    ctx.moveTo(width * 0.30, height * 0.96);
    ctx.lineTo(width * 0.33, height * 0.78);
    ctx.lineTo(width * 0.39, height * 0.64);
    ctx.lineTo(width * 0.37, height * 0.51);
    ctx.lineTo(width * 0.43, height * 0.37);
    ctx.lineTo(width * 0.41, height * 0.21);
    ctx.stroke();

    // Watauga / Holston basin contour
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.48)';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(width * 0.43, height * 0.37);
    ctx.quadraticCurveTo(width * 0.56, height * 0.48, width * 0.54, height * 0.76);
    ctx.stroke();
    ctx.restore();

    // 5. Animated Flowing Streamlines & Curvilinear Wind Trajectories
    ctx.save();
    const numStreamlines = 26;
    for (let s = 0; s < numStreamlines; s++) {
      const startY = ((s + 0.5) / numStreamlines) * height * 1.2 - height * 0.1;
      ctx.beginPath();
      ctx.strokeStyle = s % 3 === 0 ? 'rgba(255, 255, 255, 0.55)' : 'rgba(235, 215, 255, 0.32)';
      ctx.lineWidth = s % 3 === 0 ? 1.8 : 1.1;

      let px = (s % 2 === 0 ? 0 : width * 0.08);
      let py = startY;
      ctx.moveTo(px, py);

      for (let step = 0; step < 34; step++) {
        const dx = px - vortexCenterX;
        const dy = py - vortexCenterY;
        const r = Math.sqrt(dx * dx + dy * dy) + 18;
        // Combine westerly/southwesterly jet with cyclonic curvature around the trough
        const swirlFactor = Math.exp(-r / (width * 0.32));
        const vx = 14 * (1 - swirlFactor * 0.65) - (dy / r) * 18 * swirlFactor;
        const vy = -5.5 * (1 - swirlFactor * 0.5) + (dx / r) * 18 * swirlFactor;
        px += vx * 0.9;
        py += vy * 0.9;
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    ctx.restore();

    // 6. Directional Vector Field Arrows (u, v) from the 16x16 Tensor Slice
    const rows = 9;
    const cols = 14;
    ctx.save();
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const gRow = Math.min(15, Math.floor((r / rows) * 16));
        const gCol = Math.min(15, Math.floor((c / cols) * 16));
        const u = activeSlice.uGrid16[gRow]?.[gCol] ?? 8;
        const v = activeSlice.vGrid16[gRow]?.[gCol] ?? 4;

        const baseX = ((c + 0.6) / cols) * width;
        const baseY = ((r + 0.6) / rows) * height;

        // Compute local cyclonic angle matching streamline field + tensor u,v
        const dx = baseX - vortexCenterX;
        const dy = baseY - vortexCenterY;
        const dist = Math.sqrt(dx * dx + dy * dy) + 15;
        const swirl = Math.exp(-dist / (width * 0.34));
        const effU = u * 0.6 + 10 * (1 - swirl * 0.6) - (dy / dist) * 16 * swirl;
        const effV = -v * 0.4 - 4 * (1 - swirl * 0.5) + (dx / dist) * 16 * swirl;

        const mag = Math.sqrt(effU * effU + effV * effV);
        const angle = Math.atan2(effV, effU);
        const shift = ((phase * 3 + r * 5 + c * 7) % 12) - 6;
        const cx = baseX + Math.cos(angle) * shift * 0.45;
        const cy = baseY + Math.sin(angle) * shift * 0.45;

        const arrowLen = Math.min(22, Math.max(10, mag * 1.15));

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle);

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.92)';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
        ctx.lineWidth = 1.75;

        ctx.beginPath();
        ctx.moveTo(-arrowLen * 0.5, 0);
        ctx.lineTo(arrowLen * 0.5, 0);
        ctx.stroke();

        // Arrowhead
        ctx.beginPath();
        ctx.moveTo(arrowLen * 0.5 + 2, 0);
        ctx.lineTo(arrowLen * 0.5 - 5, -3.5);
        ctx.lineTo(arrowLen * 0.5 - 3.5, 0);
        ctx.lineTo(arrowLen * 0.5 - 5, 3.5);
        ctx.closePath();
        ctx.fill();

        ctx.restore();
      }
    }
    ctx.restore();

    // 7. Golden Star Marker & Callout Badge for Watauga, TN (36.3654°N, 82.2923°W)
    const wataugaX = width * 0.46;
    const wataugaY = height * 0.66;

    // Draw 5-pointed Golden Star
    ctx.save();
    ctx.translate(wataugaX, wataugaY);
    ctx.beginPath();
    const spikes = 5;
    const outerR = 14;
    const innerR = 6.5;
    for (let i = 0; i < spikes * 2; i++) {
      const r = i % 2 === 0 ? outerR : innerR;
      const a = (i * Math.PI) / spikes - Math.PI / 2;
      const sx = Math.cos(a) * r;
      const sy = Math.sin(a) * r;
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.closePath();
    ctx.fillStyle = '#FFE055';
    ctx.strokeStyle = '#2D1948';
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const relX = Math.max(0, Math.min(0.999, (e.clientX - rect.left) / rect.width));
    const relY = Math.max(0, Math.min(0.999, (e.clientY - rect.top) / rect.height));
    const gCol = Math.floor(relX * 16);
    const gRow = Math.floor(relY * 16);

    const u = activeSlice.uGrid16[gRow]?.[gCol] ?? 0;
    const v = activeSlice.vGrid16[gRow]?.[gCol] ?? 0;
    const speed = Number(Math.sqrt(u * u + v * v).toFixed(2));
    const tempC = activeSlice.tempGrid16[gRow]?.[gCol] ?? activeSlice.domainMeanTempC;
    const rhPct = activeSlice.rhGrid16[gRow]?.[gCol] ?? activeSlice.domainMeanRhPct;
    const precipMmHr = activeSlice.precipGrid16[gRow]?.[gCol] ?? 0;

    const lat = Number((36.3654 + (0.5 - relY) * 0.86).toFixed(4));
    const lon = Number((-82.2923 + (relX - 0.5) * 1.05).toFixed(4));

    setHoverProbe({
      gridRow: gRow * 2,
      gridCol: gCol * 2,
      lat,
      lon,
      u,
      v,
      speed,
      tempC,
      rhPct,
      precipMmHr,
    });
  };

  const currentStepLabel =
    TIMELINE_LABELS.find((t) => t.idx === timeStepIdx)?.label ||
    (timeStepIdx <= 12 ? `${(timeStepIdx - 12) * tensorWindow.cadenceMinutes}m` : `+${(timeStepIdx - 12) * 30}m`);

  return (
    <div className="kawaii-panel p-4 flex flex-col justify-between relative">
      {/* Top Header Strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-[#F3EAFF] border-2 border-[#2D1948] flex items-center justify-center font-bold text-[#2D1948]">
            ≋
          </div>
          <div>
            <h2 className="font-display text-lg md:text-xl font-bold text-[#2D1948] leading-tight">
              Vector Field — {selectedLevel === 1000 ? '10 m / 1000 hPa' : `${selectedLevel} hPa`} Wind (u, v)
            </h2>
            <p className="text-xs text-[#593E7A] font-mono">
              5D Tensor Slice L={selectedLevel} hPa ({activeSlice.altitudeKmMsl} km MSL) · Grid 32×32 (3.0 km) · Step {currentStepLabel}
            </p>
          </div>
        </div>

        {/* Mode Segmented Buttons (Wind | Temp | Pressure | Precip) */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {(
            [
              { id: 'wind', label: 'Wind' },
              { id: 'temp', label: 'Temp' },
              { id: 'pressure', label: 'Pressure' },
              { id: 'precip', label: 'Precip' },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFieldMode(tab.id)}
              className={`px-3.5 py-1 rounded-full text-xs font-bold transition-all whitespace-nowrap shrink-0 ${
                fieldMode === tab.id ? 'kawaii-btn-pink' : 'kawaii-btn-cream'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Isobaric Vertical Level Selector Bar (L = 7 Levels: 1000, 925, 850, 700, 500, 300, 200 hPa) */}
      <div className="flex items-center justify-between gap-2 mb-2.5 px-2.5 py-1.5 rounded-xl bg-[#F6EEFF] border-2 border-[#2D1948]">
        <span className="text-xs font-bold text-[#2D1948] whitespace-nowrap">
          Vertical Level L (7 Isobaric Surfaces):
        </span>
        <div className="flex items-center gap-1 flex-wrap">
          {CANONICAL_PRESSURE_LEVELS_HPA.map((lvl) => (
            <button
              key={lvl}
              onClick={() => onSelectLevel(lvl)}
              className={`px-2 py-0.5 rounded-lg text-xs font-mono font-semibold border transition-colors whitespace-nowrap shrink-0 ${
                selectedLevel === lvl
                  ? 'bg-[#2D1948] text-[#FFE066] border-[#2D1948]'
                  : 'bg-[#FFFDF9] text-[#2D1948] border-[#2D1948]/40 hover:bg-[#EADBFD]'
              }`}
            >
              {lvl} hPa
            </button>
          ))}
        </div>
      </div>

      {/* Main Canvas + Right Colorbar */}
      <div className="flex items-stretch gap-3">
        <div className="relative flex-1 rounded-2xl overflow-hidden border-[2.5px] border-[#2D1948] shadow-[inset_0_2px_6px_rgba(0,0,0,0.35)] bg-[#1C0D52]">
          <canvas
            ref={canvasRef}
            width={680}
            height={compact ? 270 : 360}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setHoverProbe(null)}
            className="w-full h-full block cursor-crosshair"
          />

          {/* Callout Pill attached to Watauga Star */}
          <div
            className="pointer-events-none absolute px-3 py-1 rounded-full bg-[#FFFDF9] border-2 border-[#2D1948] shadow-[0_2px_0_#2D1948] flex items-center gap-1.5 text-xs font-bold text-[#2D1948]"
            style={{ left: '48.5%', top: '52%' }}
          >
            <span className="text-[#2D1948]">★</span>
            <span>Watauga, TN</span>
          </div>

          {/* Live Crosshair Telemetry Probe HUD */}
          {hoverProbe && (
            <div className="pointer-events-none absolute top-2.5 left-2.5 px-3 py-2 rounded-xl bg-[#FFFDF9]/95 border-2 border-[#2D1948] shadow-[0_2px_0_#2D1948] text-[11px] font-mono text-[#2D1948]">
              <div className="font-bold text-[#6C35DE]">
                Cell ({hoverProbe.gridRow}, {hoverProbe.gridCol}) · {hoverProbe.lat}°N, {Math.abs(hoverProbe.lon)}°W
              </div>
              <div>
                u={hoverProbe.u} m/s · v={hoverProbe.v} m/s · |v|={hoverProbe.speed} m/s
              </div>
              <div>
                T={hoverProbe.tempC}°C · RH={hoverProbe.rhPct}% · MRMS={hoverProbe.precipMmHr} mm/hr
              </div>
            </div>
          )}

          {/* Resolution & Provenance Watermark Badge */}
          <div className="pointer-events-none absolute bottom-2 left-2.5 px-2.5 py-0.5 rounded-md bg-[#2D1948]/85 text-[#FFFDF9] text-[10px] font-mono">
            HRRR 3km + ERA5 (0.25° bicubic co-registered) · Valid Mask: {(activeSlice.validFractionMask * 100).toFixed(1)}%
          </div>

          {/* Surprised / Shear-Alert Mascot in bottom-right corner of map */}
          <div
            className="absolute -bottom-1 right-2 z-10"
            title={`850–500 hPa Bulk Shear = ${tensorWindow.diagnostics.deepLayerShear850To500Ms} m/s`}
          >
            <PurpleCloudMascot pose="surprised_shear" size="sm" />
          </div>
        </div>

        {/* Vertical Scale Colorbar */}
        <div className="w-16 flex flex-col items-center justify-between py-1 select-none shrink-0">
          <div className="text-[11px] font-bold font-mono text-[#2D1948] text-center">
            {fieldMode === 'wind'
              ? '|v| (m/s)'
              : fieldMode === 'temp'
                ? 'T (°C)'
                : fieldMode === 'pressure'
                  ? 'P (hPa)'
                  : 'mm/hr'}
          </div>
          <div className="flex items-stretch gap-1.5 flex-1 my-1.5">
            <div
              className="w-5 rounded-full border-2 border-[#2D1948] shadow-[inset_0_1px_2px_rgba(0,0,0,0.2)]"
              style={{
                background:
                  'linear-gradient(180deg, #FF55B8 0%, #FFB84D 25%, #FFF066 45%, #48F5FF 70%, #3A36E0 90%, #24116A 100%)',
              }}
            />
            <div className="flex flex-col justify-between text-[11px] font-mono font-bold text-[#2D1948] py-0.5">
              <span>{fieldMode === 'wind' ? '20' : fieldMode === 'temp' ? '30' : fieldMode === 'pressure' ? String(selectedLevel + 10) : '25'}</span>
              <span>{fieldMode === 'wind' ? '15' : fieldMode === 'temp' ? '15' : fieldMode === 'pressure' ? String(selectedLevel + 5) : '15'}</span>
              <span>{fieldMode === 'wind' ? '10' : fieldMode === 'temp' ? '0' : fieldMode === 'pressure' ? String(selectedLevel) : '10'}</span>
              <span>{fieldMode === 'wind' ? '5' : fieldMode === 'temp' ? '-15' : fieldMode === 'pressure' ? String(selectedLevel - 5) : '5'}</span>
              <span>{fieldMode === 'wind' ? '0' : fieldMode === 'temp' ? '-35' : fieldMode === 'pressure' ? String(selectedLevel - 10) : '0'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Playback & Temporal Scrubber Controls */}
      <div className="mt-3 flex items-center gap-3">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          aria-label={isPlaying ? 'Pause vector animation' : 'Play vector animation'}
          className="w-10 h-9 rounded-full kawaii-btn-pink flex items-center justify-center shrink-0"
        >
          {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
        </button>

        <div className="flex-1 flex flex-col">
          <input
            type="range"
            min={0}
            max={17}
            value={timeStepIdx}
            onChange={(e) => setTimeStepIdx(Number(e.target.value))}
            className="w-full accent-[#7B42BC] cursor-pointer h-2 rounded-lg bg-[#E5D4FF]"
          />
          <div className="flex justify-between text-[11px] font-bold font-mono text-[#2D1948] mt-1 px-1">
            <span>-180m (t₀-12Δt)</span>
            <span>-60m</span>
            <span className="text-[#7B42BC]">+0h (t₀)</span>
            <span>+1h</span>
            <span>+2h</span>
            <span>+3h (+180m)</span>
          </div>
        </div>

        {onExpandToExplorer && (
          <button
            onClick={onExpandToExplorer}
            title="Open Full Vector Field Explorer"
            className="w-9 h-9 rounded-xl kawaii-btn-cream flex items-center justify-center shrink-0"
          >
            <Maximize2 className="w-4 h-4 text-[#2D1948]" />
          </button>
        )}
      </div>
    </div>
  );
};
