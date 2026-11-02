import React from 'react';
import { MascotDiagnosticState } from '../shared/aetherTypes.ts';

export type CloudMascotPose =
  | 'heart'
  | 'sun_peek'
  | 'surprised_shear'
  | 'determined_storm'
  | 'teary_saturation'
  | 'curious_embedding'
  | 'sleeping_zzz'
  | 'provenance_guard';

interface PurpleCloudMascotProps {
  pose?: CloudMascotPose;
  diagnosticState?: MascotDiagnosticState;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

export function mapDiagnosticStateToPose(state: MascotDiagnosticState): CloudMascotPose {
  switch (state) {
    case MascotDiagnosticState.VERTICAL_WIND_SHEAR_ALERT:
      return 'surprised_shear';
    case MascotDiagnosticState.CONVECTIVE_PRECIP_ACTIVE:
      return 'determined_storm';
    case MascotDiagnosticState.COLUMN_MOISTURE_SATURATION:
      return 'teary_saturation';
    case MascotDiagnosticState.RETRIEVAL_FUSION_RUNNING:
      return 'curious_embedding';
    case MascotDiagnosticState.PROVENANCE_OR_PIPELINE_WARNING:
      return 'provenance_guard';
    case MascotDiagnosticState.QUIESCENT_NOMINAL:
    default:
      return 'heart';
  }
}

export const PurpleCloudMascot: React.FC<PurpleCloudMascotProps> = ({
  pose,
  diagnosticState,
  size = 'md',
  className = '',
}) => {
  const activePose: CloudMascotPose = pose || (diagnosticState ? mapDiagnosticStateToPose(diagnosticState) : 'heart');

  const sizeClasses: Record<'sm' | 'md' | 'lg' | 'xl', string> = {
    sm: 'w-14 h-12',
    md: 'w-20 h-16',
    lg: 'w-28 h-24',
    xl: 'w-32 h-28',
  };

  const uid = React.useId().replace(/:/g, '');

  return (
    <div className={`relative inline-flex items-center justify-center select-none ${sizeClasses[size]} ${className}`}>
      <svg viewBox="0 0 160 130" className="w-full h-full overflow-visible drop-shadow-[0_3px_0_rgba(45,25,72,0.85)]">
        <defs>
          <linearGradient id={`cloudGrad_${uid}`} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#B88CFF" />
            <stop offset="55%" stopColor="#8E5CE6" />
            <stop offset="100%" stopColor="#F59AD6" />
          </linearGradient>
          <linearGradient id={`heartGrad_${uid}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#FF8ECA" />
            <stop offset="100%" stopColor="#FF52A2" />
          </linearGradient>
          <linearGradient id={`sunGrad_${uid}`} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#FFE97D" />
            <stop offset="100%" stopColor="#FFB830" />
          </linearGradient>
        </defs>

        {/* Optional Sun Behind Cloud for Watauga Weather Card */}
        {activePose === 'sun_peek' && (
          <g transform="translate(-8, -10)">
            <circle cx="46" cy="44" r="22" fill={`url(#sunGrad_${uid})`} stroke="#2D1948" strokeWidth="3.5" />
            <line x1="16" y1="24" x2="24" y2="30" stroke="#2D1948" strokeWidth="3.5" strokeLinecap="round" />
            <line x1="28" y1="12" x2="34" y2="20" stroke="#2D1948" strokeWidth="3.5" strokeLinecap="round" />
            <line x1="46" y1="8" x2="46" y2="17" stroke="#2D1948" strokeWidth="3.5" strokeLinecap="round" />
            <line x1="10" y1="42" x2="19" y2="42" stroke="#2D1948" strokeWidth="3.5" strokeLinecap="round" />
          </g>
        )}

        {/* Scalloped Fluffy Cloud Body */}
        <path
          d="M 42 98
             C 24 98, 16 84, 24 70
             C 16 56, 26 38, 44 40
             C 52 24, 76 20, 90 32
             C 106 22, 128 32, 130 50
             C 144 56, 146 78, 134 88
             C 138 98, 124 104, 110 100
             C 96 106, 62 106, 42 98 Z"
          fill={`url(#cloudGrad_${uid})`}
          stroke="#2D1948"
          strokeWidth="4"
          strokeLinejoin="round"
        />

        {/* Soft Inner Top Highlight */}
        <path
          d="M 48 42 C 56 29, 76 26, 88 36 C 102 28, 120 35, 123 48"
          fill="none"
          stroke="rgba(255,255,255,0.55)"
          strokeWidth="3.5"
          strokeLinecap="round"
        />

        {/* Rosy Cheeks */}
        <ellipse cx="52" cy="72" rx="7.5" ry="4.5" fill="#FF75B5" opacity="0.78" />
        <ellipse cx="106" cy="72" rx="7.5" ry="4.5" fill="#FF75B5" opacity="0.78" />

        {/* Sparkle Specks on Cloud Body */}
        <circle cx="38" cy="58" r="2" fill="#FFFDF7" opacity="0.85" />
        <circle cx="118" cy="54" r="2.2" fill="#FFFDF7" opacity="0.85" />
        <circle cx="66" cy="36" r="1.8" fill="#FFFDF7" opacity="0.75" />

        {/* Facial Expressions by Diagnostic Pose */}
        {activePose === 'heart' && (
          <g>
            {/* Happy open eyes */}
            <circle cx="62" cy="64" r="4.5" fill="#2D1948" />
            <circle cx="60.5" cy="62.5" r="1.6" fill="#FFFFFF" />
            <circle cx="96" cy="64" r="4.5" fill="#2D1948" />
            <circle cx="94.5" cy="62.5" r="1.6" fill="#FFFFFF" />
            {/* Happy open smile */}
            <path d="M 73 66 Q 79 75 85 66 Z" fill="#FF5C9E" stroke="#2D1948" strokeWidth="2.8" strokeLinejoin="round" />
            {/* Floating Hearts */}
            <path
              d="M 118 22 C 114 15, 104 17, 106 25 C 108 31, 118 36, 118 36 C 118 36, 128 31, 130 25 C 132 17, 122 15, 118 22 Z"
              fill="#FF78B8"
              stroke="#2D1948"
              strokeWidth="2.8"
            />
            {/* Big Held Pink Heart */}
            <path
              d="M 88 86 C 80 74, 62 78, 66 93 C 70 105, 88 114, 88 114 C 88 114, 106 105, 110 93 C 114 78, 96 74, 88 86 Z"
              fill={`url(#heartGrad_${uid})`}
              stroke="#2D1948"
              strokeWidth="3.5"
            />
            {/* Little cloud nubs holding heart */}
            <circle cx="67" cy="92" r="6" fill="#B589FA" stroke="#2D1948" strokeWidth="3" />
            <circle cx="109" cy="92" r="6" fill="#F094D4" stroke="#2D1948" strokeWidth="3" />
          </g>
        )}

        {activePose === 'sun_peek' && (
          <g>
            <circle cx="64" cy="66" r="4.3" fill="#2D1948" />
            <circle cx="62.5" cy="64.5" r="1.5" fill="#FFFFFF" />
            <circle cx="94" cy="66" r="4.3" fill="#2D1948" />
            <circle cx="92.5" cy="64.5" r="1.5" fill="#FFFFFF" />
            <path d="M 74 69 Q 79 74 84 69" fill="none" stroke="#2D1948" strokeWidth="3.2" strokeLinecap="round" />
          </g>
        )}

        {activePose === 'surprised_shear' && (
          <g>
            {/* Wide surprised eyes */}
            <circle cx="63" cy="64" r="5" fill="#2D1948" />
            <circle cx="61.5" cy="62" r="1.8" fill="#FFFFFF" />
            <circle cx="95" cy="64" r="5" fill="#2D1948" />
            <circle cx="93.5" cy="62" r="1.8" fill="#FFFFFF" />
            {/* 'O' surprised mouth */}
            <ellipse cx="79" cy="72" rx="4.5" ry="6" fill="#FF6BA8" stroke="#2D1948" strokeWidth="3" />
            {/* Double Exclamation Marks '!!' */}
            <g transform="translate(8, 6) rotate(-14)">
              <rect x="10" y="10" width="7" height="20" rx="3" fill="#FFE066" stroke="#2D1948" strokeWidth="2.8" />
              <circle cx="13.5" cy="36" r="3.5" fill="#FFE066" stroke="#2D1948" strokeWidth="2.8" />
              <rect x="22" y="8" width="7" height="20" rx="3" fill="#FFE066" stroke="#2D1948" strokeWidth="2.8" />
              <circle cx="25.5" cy="34" r="3.5" fill="#FFE066" stroke="#2D1948" strokeWidth="2.8" />
            </g>
          </g>
        )}

        {activePose === 'determined_storm' && (
          <g>
            {/* Angled determined eyebrows */}
            <path d="M 54 55 L 68 60" stroke="#2D1948" strokeWidth="3.5" strokeLinecap="round" />
            <path d="M 104 55 L 90 60" stroke="#2D1948" strokeWidth="3.5" strokeLinecap="round" />
            <circle cx="63" cy="65" r="4.3" fill="#2D1948" />
            <circle cx="95" cy="65" r="4.3" fill="#2D1948" />
            {/* Wavy determined mouth */}
            <path d="M 72 72 Q 76 68 79 72 Q 82 75 86 71" fill="none" stroke="#2D1948" strokeWidth="3.2" strokeLinecap="round" />
            {/* Steam puff */}
            <path
              d="M 128 24 C 124 16, 136 12, 140 19 C 146 16, 152 25, 144 30 C 146 36, 134 38, 132 31 Z"
              fill="#F6EEFF"
              stroke="#2D1948"
              strokeWidth="2.8"
            />
          </g>
        )}

        {activePose === 'teary_saturation' && (
          <g>
            {/* Sad/saturated eyes with big cyan raindrops */}
            <circle cx="63" cy="65" r="4.5" fill="#2D1948" />
            <circle cx="95" cy="65" r="4.5" fill="#2D1948" />
            <path d="M 55 56 Q 62 52 67 57" fill="none" stroke="#2D1948" strokeWidth="2.8" strokeLinecap="round" />
            <path d="M 103 56 Q 96 52 91 57" fill="none" stroke="#2D1948" strokeWidth="2.8" strokeLinecap="round" />
            {/* Quivering mouth */}
            <path d="M 73 74 Q 79 69 85 74" fill="none" stroke="#2D1948" strokeWidth="3" strokeLinecap="round" />
            {/* Streaming Cyan Tears */}
            <path d="M 59 70 Q 54 84 60 90 Q 65 84 62 70 Z" fill="#5CE1E6" stroke="#2D1948" strokeWidth="2.4" />
            <path d="M 99 70 Q 104 84 98 90 Q 93 84 96 70 Z" fill="#5CE1E6" stroke="#2D1948" strokeWidth="2.4" />
          </g>
        )}

        {activePose === 'curious_embedding' && (
          <g>
            {/* Squinting analytical "> <" or starry eyes */}
            <circle cx="63" cy="64" r="4.4" fill="#2D1948" />
            <circle cx="61.5" cy="62.5" r="1.6" fill="#FFFFFF" />
            <circle cx="95" cy="64" r="4.4" fill="#2D1948" />
            <circle cx="93.5" cy="62.5" r="1.6" fill="#FFFFFF" />
            <path d="M 74 70 Q 79 74 84 70" fill="none" stroke="#2D1948" strokeWidth="3.2" strokeLinecap="round" />
            {/* Four-point Golden Star badge on top-right */}
            <path
              d="M 128 18 L 132 28 L 142 32 L 132 36 L 128 46 L 124 36 L 114 32 L 124 28 Z"
              fill="#FFD166"
              stroke="#2D1948"
              strokeWidth="2.8"
              strokeLinejoin="round"
            />
          </g>
        )}

        {activePose === 'provenance_guard' && (
          <g>
            <circle cx="63" cy="64" r="4.5" fill="#2D1948" />
            <circle cx="95" cy="64" r="4.5" fill="#2D1948" />
            <line x1="73" y1="72" x2="85" y2="72" stroke="#2D1948" strokeWidth="3.2" strokeLinecap="round" />
            {/* Magnifying glass / shield star */}
            <circle cx="122" cy="34" r="11" fill="#FFE066" stroke="#2D1948" strokeWidth="3" />
            <text x="122" y="39" textAnchor="middle" fontSize="14" fontWeight="bold" fill="#2D1948">
              !
            </text>
          </g>
        )}

        {activePose === 'sleeping_zzz' && (
          <g>
            {/* Closed sleeping eyes */}
            <path d="M 55 65 Q 62 72 69 65" fill="none" stroke="#2D1948" strokeWidth="3.4" strokeLinecap="round" />
            <path d="M 89 65 Q 96 72 103 65" fill="none" stroke="#2D1948" strokeWidth="3.4" strokeLinecap="round" />
            {/* Peaceful 'w' smile */}
            <path d="M 73 71 Q 76 75 79 71 Q 82 75 85 71" fill="none" stroke="#2D1948" strokeWidth="2.8" strokeLinecap="round" />
            {/* Floating Zzz */}
            <text x="112" y="32" fontSize="15" fontWeight="800" fill="#6C35DE" stroke="#2D1948" strokeWidth="0.8">
              z
            </text>
            <text x="124" y="23" fontSize="18" fontWeight="800" fill="#8E5CE6" stroke="#2D1948" strokeWidth="0.8">
              z
            </text>
            <text x="138" y="14" fontSize="20" fontWeight="800" fill="#B88CFF" stroke="#2D1948" strokeWidth="0.8">
              z
            </text>
          </g>
        )}
      </svg>
    </div>
  );
};
