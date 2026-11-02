import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { EmbeddingMetadataRecord } from '../shared/aetherTypes.ts';
import { PurpleCloudMascot } from './PurpleCloudMascot.tsx';

interface EmbeddingSpaceExplorerCardProps {
  records: EmbeddingMetadataRecord[];
  activeRegime: 'Clear' | 'Cloudy' | 'Rain' | 'Storm';
  compact?: boolean;
  onSelectRecord?: (rec: EmbeddingMetadataRecord) => void;
}

const REGIME_COLORS: Record<'Clear' | 'Cloudy' | 'Rain' | 'Storm', string> = {
  Clear: '#22A8FF',
  Cloudy: '#9D4EDD',
  Rain: '#FF2A9D',
  Storm: '#FFB703',
};

export const EmbeddingSpaceExplorerCard: React.FC<EmbeddingSpaceExplorerCardProps> = ({
  records,
  activeRegime,
  compact = false,
  onSelectRecord,
}) => {
  const [projectionMode, setProjectionMode] = useState<'PCA' | 'UMAP'>('PCA');
  const [activeIndexSpace, setActiveIndexSpace] = useState<'embeddinggemma-2' | 'gemini-embedding-2'>('embeddinggemma-2');
  const [viewDimension, setViewDimension] = useState<'2D' | '3D'>('2D');
  const [hoveredRecord, setHoveredRecord] = useState<EmbeddingMetadataRecord | null>(null);

  const mount3dRef = useRef<HTMLDivElement | null>(null);

  // Filter strictly to the selected model's isolated 768-d index space!
  const spaceRecords = records.filter((r) => r.modelId === activeIndexSpace);

  // Generate dense background cluster cloud around the anchored storm records for visual clarity
  const syntheticClusterPoints = React.useMemo(() => {
    const pts: {
      x: number;
      y: number;
      z: number;
      regime: 'Clear' | 'Cloudy' | 'Rain' | 'Storm';
    }[] = [];
    const centers: Record<'Clear' | 'Cloudy' | 'Rain' | 'Storm', [number, number, number]> =
      activeIndexSpace === 'embeddinggemma-2'
        ? {
            Clear: [-2.1, -1.3, -0.9],
            Cloudy: [-0.9, 1.6, 0.7],
            Rain: [0.8, -0.9, 0.8],
            Storm: [2.1, 0.9, -1.0],
          }
        : {
            Clear: [1.5, -1.8, 0.8],
            Cloudy: [-1.7, -0.7, -0.6],
            Rain: [0.6, 1.2, -0.9],
            Storm: [-1.1, 1.9, 1.1],
          };

    (['Clear', 'Cloudy', 'Rain', 'Storm'] as const).forEach((reg, rIdx) => {
      const [cx, cy, cz] = centers[reg];
      const scale = projectionMode === 'UMAP' ? 0.55 : 0.72;
      for (let i = 0; i < 38; i++) {
        const a1 = i * 1.93 + rIdx * 3.1;
        const a2 = i * 2.71 + rIdx * 1.7;
        const rad = ((i % 7) + 1) * 0.14 * scale;
        pts.push({
          x: cx + Math.cos(a1) * rad,
          y: cy + Math.sin(a2) * rad,
          z: cz + Math.sin(a1 - a2) * rad,
          regime: reg,
        });
      }
    });
    return pts;
  }, [activeIndexSpace, projectionMode]);

  useEffect(() => {
    if (viewDimension !== '3D' || !mount3dRef.current) return;
    const container = mount3dRef.current;
    const width = container.clientWidth || 340;
    const height = container.clientHeight || 220;

    const isSoftwareGl = (() => {
      const gl = document.createElement('canvas').getContext('webgl2');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xfaf3eb);

    const camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 100);
    camera.position.set(4.5, 3.4, 5.2);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Coordinate Axes Box & Grid
    const gridHelper = new THREE.GridHelper(6, 12, 0x7b42bc, 0xd8c2f2);
    gridHelper.position.y = -2.0;
    scene.add(gridHelper);

    const group = new THREE.Group();
    scene.add(group);

    const sphereGeo = new THREE.SphereGeometry(0.09, 10, 10);
    syntheticClusterPoints.forEach((pt) => {
      const mat = new THREE.MeshBasicMaterial({ color: REGIME_COLORS[pt.regime] });
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.position.set(pt.x, pt.y * 0.8, pt.z);
      group.add(mesh);
    });

    // Current state star octahedron
    const starGeo = new THREE.OctahedronGeometry(0.26, 0);
    const starMat = new THREE.MeshBasicMaterial({ color: 0xffd166 });
    const starMesh = new THREE.Mesh(starGeo, starMat);
    starMesh.position.set(0.75, -0.7, 0.6);
    group.add(starMesh);

    renderer.render(scene, camera);

    let isDragging = false;
    let prevX = 0;
    let prevY = 0;

    const onPointerDown = (e: PointerEvent) => {
      isDragging = true;
      prevX = e.clientX;
      prevY = e.clientY;
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!isDragging) return;
      const dx = (e.clientX - prevX) * 0.012;
      const dy = (e.clientY - prevY) * 0.012;
      prevX = e.clientX;
      prevY = e.clientY;
      group.rotation.y += dx;
      group.rotation.x = Math.max(-0.8, Math.min(0.8, group.rotation.x + dy));
      renderer.render(scene, camera);
    };
    const onPointerUp = () => {
      isDragging = false;
    };

    const domEl = renderer.domElement;
    domEl.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);

    return () => {
      domEl.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      sphereGeo.dispose();
      starGeo.dispose();
      renderer.dispose();
    };
  }, [viewDimension, syntheticClusterPoints]);

  return (
    <div className="kawaii-panel p-3.5 flex flex-col justify-between relative">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-1.5 mb-2">
        <div className="flex items-center gap-1.5">
          <span className="text-base">☁️</span>
          <h3 className="font-display text-sm md:text-base font-bold text-[#2D1948]">
            Embedding Space — Atmospheric States
          </h3>
        </div>

        {/* PCA | UMAP + 2D | 3D Controls */}
        <div className="flex items-center gap-1">
          {(['PCA', 'UMAP'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setProjectionMode(m)}
              className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold whitespace-nowrap shrink-0 ${
                projectionMode === m ? 'kawaii-btn-pink' : 'kawaii-btn-cream'
              }`}
            >
              {m}
            </button>
          ))}
          <button
            onClick={() => setViewDimension(viewDimension === '2D' ? '3D' : '2D')}
            className="px-2 py-0.5 rounded-full text-[11px] font-mono font-bold bg-[#2D1948] text-[#FFE066] border border-[#2D1948] whitespace-nowrap shrink-0"
          >
            {viewDimension}
          </button>
        </div>
      </div>

      {/* Isolated Index Selector (Enforces Non-Equivalence of 768-d Spaces!) */}
      <div className="flex items-center justify-between gap-1.5 mb-2 px-2 py-1 rounded-lg bg-[#F6EEFF] border border-[#2D1948]/50 text-[11px]">
        <span className="font-bold text-[#2D1948] truncate">Isolated 768-d Index:</span>
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={() => setActiveIndexSpace('embeddinggemma-2')}
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
              activeIndexSpace === 'embeddinggemma-2'
                ? 'bg-[#7B42BC] text-white'
                : 'bg-white text-[#2D1948] border border-[#2D1948]/30'
            }`}
          >
            Index A: EmbeddingGemma 2
          </button>
          <button
            onClick={() => setActiveIndexSpace('gemini-embedding-2')}
            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
              activeIndexSpace === 'gemini-embedding-2'
                ? 'bg-[#FF5CA8] text-white'
                : 'bg-white text-[#2D1948] border border-[#2D1948]/30'
            }`}
          >
            Index B: gemini-embedding-2
          </button>
        </div>
      </div>

      {/* Plot + Legend Container */}
      <div className="flex items-center gap-2.5 flex-1">
        {viewDimension === '2D' ? (
          <div className="relative flex-1 h-44 rounded-xl bg-[#FAF3EB] border-2 border-[#2D1948] p-2 overflow-hidden">
            <svg viewBox="0 0 260 175" className="w-full h-full overflow-visible">
              {/* Subtle Grid Lines */}
              {[45, 85, 125].map((y) => (
                <line key={y} x1="28" y1={y} x2="245" y2={y} stroke="#E5D4FF" strokeWidth="1" />
              ))}
              {[75, 135, 195].map((x) => (
                <line key={x} x1={x} y1="15" x2={x} y2="150" stroke="#E5D4FF" strokeWidth="1" />
              ))}

              {/* Axes with Arrowheads */}
              <line x1="28" y1="150" x2="28" y2="14" stroke="#2D1948" strokeWidth="2.2" />
              <polygon points="28,10 24,18 32,18" fill="#2D1948" />
              <line x1="28" y1="150" x2="248" y2="150" stroke="#2D1948" strokeWidth="2.2" />
              <polygon points="252,150 244,146 244,154" fill="#2D1948" />

              {/* Axis Labels */}
              <text x="14" y="88" transform="rotate(-90 14 88)" textAnchor="middle" fontSize="11" fontWeight="bold" fill="#2D1948">
                {projectionMode === 'PCA' ? 'PC2' : 'UMAP-2'}
              </text>
              <text x="138" y="168" textAnchor="middle" fontSize="11" fontWeight="bold" fill="#2D1948">
                {projectionMode === 'PCA' ? 'PC1' : 'UMAP-1'}
              </text>

              {/* Dense Cluster Points */}
              {syntheticClusterPoints.map((pt, i) => {
                const cx = 138 + pt.x * 34;
                const cy = 84 - pt.y * 28;
                return (
                  <circle
                    key={i}
                    cx={cx}
                    cy={cy}
                    r="3.1"
                    fill={REGIME_COLORS[pt.regime]}
                    opacity="0.88"
                  />
                );
              })}

              {/* Indexed Historical Storm Catalogue Nodes (Interactive) */}
              {spaceRecords.map((rec) => {
                const coords = projectionMode === 'PCA' ? rec.pca3d : rec.umap3d;
                const cx = 138 + coords[0] * 32;
                const cy = 84 - coords[1] * 26;
                return (
                  <circle
                    key={`${rec.modelId}_${rec.recordId}`}
                    cx={cx}
                    cy={cy}
                    r="5.2"
                    fill={REGIME_COLORS[rec.synopticRegime]}
                    stroke="#2D1948"
                    strokeWidth="1.8"
                    className="cursor-pointer hover:scale-125 transition-transform"
                    onMouseEnter={() => setHoveredRecord(rec)}
                    onMouseLeave={() => setHoveredRecord(null)}
                    onClick={() => onSelectRecord?.(rec)}
                  />
                );
              })}

              {/* ★ Current Atmospheric State Star */}
              <g
                transform={`translate(${
                  activeRegime === 'Rain' ? 162 : activeRegime === 'Storm' ? 202 : 76
                }, ${activeRegime === 'Rain' ? 112 : activeRegime === 'Storm' ? 58 : 118})`}
              >
                <polygon
                  points="0,-11 3.2,-3.5 11,-3.2 4.8,2.2 6.8,10 0,5.4 -6.8,10 -4.8,2.2 -11,-3.2 -3.2,-3.5"
                  fill="#FFF3B0"
                  stroke="#2D1948"
                  strokeWidth="2.4"
                />
              </g>
            </svg>

            {hoveredRecord && (
              <div className="pointer-events-none absolute top-1.5 left-8 right-2 px-2 py-1 rounded-lg bg-[#2D1948] text-white text-[10px] font-mono truncate">
                {hoveredRecord.recordId} · {hoveredRecord.synopticRegime} ({hoveredRecord.peakPrecipMmHr} mm/hr)
              </div>
            )}
          </div>
        ) : (
          <div
            ref={mount3dRef}
            className="relative flex-1 h-44 rounded-xl bg-[#FAF3EB] border-2 border-[#2D1948] overflow-hidden cursor-grab active:cursor-grabbing"
            title="Drag to rotate 3D exploratory embedding projection"
          />
        )}

        {/* Legend Column matching Mockup */}
        <div className="w-24 flex flex-col justify-center gap-1.5 text-xs font-bold text-[#2D1948] shrink-0">
          {(['Clear', 'Cloudy', 'Rain', 'Storm'] as const).map((reg) => (
            <div key={reg} className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-full border border-[#2D1948] shrink-0"
                style={{ backgroundColor: REGIME_COLORS[reg] }}
              />
              <span>{reg}</span>
            </div>
          ))}
          <div className="flex items-center gap-1.5 pt-0.5">
            <span className="text-sm leading-none text-[#FFB703] drop-shadow-[0_1px_0_#2D1948]">★</span>
            <span>Current</span>
          </div>
        </div>
      </div>

      {/* Exploratory Disclaimer Caption + Mascot */}
      <div className="mt-2 flex items-center justify-between gap-2 text-[10px] text-[#593E7A] font-mono">
        <span>
          Exploratory {projectionMode} inside {activeIndexSpace} only (spaces never directly compared).
        </span>
        {compact && (
          <div className="-mt-6 -mb-2 shrink-0">
            <PurpleCloudMascot pose="curious_embedding" size="sm" />
          </div>
        )}
      </div>
    </div>
  );
};
