#!/usr/bin/env python3
"""Build Cursor canvas with embedded skeleton_playback.json."""

from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path(
    r"C:\Users\hand\Desktop\双外部双目\0810\1\multiview_3d_results\full\skeleton_playback.json"
)
IMU_OVERLAY_PATH = Path(
    r"C:\Users\hand\Desktop\双外部双目\0810\1\multiview_3d_results\full"
    r"\imu_wrist_overlay.json"
)
CANVAS_PATH = Path(
    r"C:\Users\hand\.cursor\projects\c-Users-hand-Desktop-HearWristCam\canvases"
    r"\skeleton-3d-playback.canvas.tsx"
)

CANVAS_BODY = r'''import {
  Button,
  Card,
  CardBody,
  H1,
  H2,
  Pill,
  Row,
  Spacer,
  Stack,
  Stat,
  Text,
  TextInput,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type ViewName = "iso" | "front" | "side" | "top" | "free";
/** Display-frame up axis. Mocap world is Y-up; viewer draws Z-up. */
type UpAxis = "y_up" | "z_up";

type PlaybackData = {
  source: string;
  frame_count: number;
  joints: string[];
  edges: number[][];
  seqs: number[];
  ground_z_m: number;
  xyz_unit: string;
  missing_sentinel: number;
  xyz_i16_b64: string;
  left_above_mm: Array<number | null>;
  right_above_mm: Array<number | null>;
};

type ImuOverlayData = {
  schema: string;
  method: string;
  imu_source: string;
  axis_length_m: number;
  frame_count: number;
  quat_wxyz: {
    left: Array<Array<number> | null>;
    right: Array<Array<number> | null>;
  };
  valid: { left: boolean[]; right: boolean[] };
  calibration: Record<string, unknown>;
  notes: string;
};

const DATA: PlaybackData = JSON.parse(__EMBEDDED_JSON__) as PlaybackData;
const IMU: ImuOverlayData = JSON.parse(__IMU_EMBEDDED_JSON__) as ImuOverlayData;

const MISS = DATA.missing_sentinel;
const N_JOINTS = DATA.joints.length;
const FRAME_COUNT = DATA.frame_count;
const LEFT_ANKLE = DATA.joints.indexOf("left_ankle");
const RIGHT_ANKLE = DATA.joints.indexOf("right_ankle");
const LEFT_WRIST = DATA.joints.indexOf("left_wrist");
const RIGHT_WRIST = DATA.joints.indexOf("right_wrist");
const IMU_AXIS_LEN = IMU.axis_length_m;
const DEFAULT_UP: UpAxis = "y_up";

function b64ToInt16(b64: string): Int16Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const copy = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(copy).set(bytes);
  return new Int16Array(copy);
}

const XYZ_I16 = b64ToInt16(DATA.xyz_i16_b64);

/** Raw mocap / triangulation world meters (Y-up). */
function worldPoint(frame: number, joint: number): [number, number, number] | null {
  const base = (frame * N_JOINTS + joint) * 3;
  const x = XYZ_I16[base];
  const y = XYZ_I16[base + 1];
  const z = XYZ_I16[base + 2];
  if (x === MISS || y === MISS || z === MISS) return null;
  return [x / 1000, y / 1000, z / 1000];
}

/**
 * Map world → display frame where +Z is up (viewer convention).
 * y_up: (x,y,z)_world → (x,z,y)_display  — standing mocap default
 * z_up: identity — previous incorrect assumption
 */
function toDisplay(
  x: number,
  y: number,
  z: number,
  up: UpAxis,
): [number, number, number] {
  if (up === "y_up") return [x, z, y];
  return [x, y, z];
}

function displayPoint(
  frame: number,
  joint: number,
  up: UpAxis,
): [number, number, number] | null {
  const w = worldPoint(frame, joint);
  if (!w) return null;
  return toDisplay(w[0], w[1], w[2], up);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const k = (sorted.length - 1) * p;
  const f0 = Math.floor(k);
  const f1 = Math.ceil(k);
  if (f0 === f1) return sorted[f0];
  return sorted[f0] * (f1 - k) + sorted[f1] * (k - f0);
}

function computeGround(up: UpAxis): number {
  const samples: number[] = [];
  for (let f = 0; f < FRAME_COUNT; f++) {
    for (const joint of [LEFT_ANKLE, RIGHT_ANKLE]) {
      const p = displayPoint(f, joint, up);
      if (p) samples.push(p[2]);
    }
  }
  samples.sort((a, b) => a - b);
  return percentile(samples, 0.05);
}

function computeBounds(up: UpAxis): { cx: number; cy: number; cz: number; span: number } {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let count = 0;
  const step = Math.max(1, Math.floor(FRAME_COUNT / 200));
  for (let f = 0; f < FRAME_COUNT; f += step) {
    for (let j = 0; j < N_JOINTS; j++) {
      const p = displayPoint(f, j, up);
      if (!p) continue;
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
      minZ = Math.min(minZ, p[2]);
      maxZ = Math.max(maxZ, p[2]);
      count += 1;
    }
  }
  if (count === 0) return { cx: 0, cy: 0, cz: 0, span: 1 };
  return {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    cz: (minZ + maxZ) / 2,
    span: Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.5),
  };
}

const GROUND_BY_UP: Record<UpAxis, number> = {
  y_up: computeGround("y_up"),
  z_up: computeGround("z_up"),
};
const BOUNDS_BY_UP: Record<UpAxis, { cx: number; cy: number; cz: number; span: number }> = {
  y_up: computeBounds("y_up"),
  z_up: computeBounds("z_up"),
};

const VIEW_PRESETS: Record<Exclude<ViewName, "free">, { yaw: number; pitch: number }> = {
  iso: { yaw: 40, pitch: 28 },
  front: { yaw: 0, pitch: 8 },
  side: { yaw: 90, pitch: 8 },
  top: { yaw: 0, pitch: 88 },
};

let playTimer: ReturnType<typeof setInterval> | null = null;

function stopPlayTimer() {
  if (playTimer !== null) {
    clearInterval(playTimer);
    playTimer = null;
  }
}

function project(
  x: number,
  y: number,
  z: number,
  yawDeg: number,
  pitchDeg: number,
  width: number,
  height: number,
  bounds: { cx: number; cy: number; cz: number; span: number },
): [number, number, number] {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const dx = x - bounds.cx;
  const dy = y - bounds.cy;
  const dz = z - bounds.cz;
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const x1 = dx * cosY - dy * sinY;
  const y1 = dx * sinY + dy * cosY;
  const z1 = dz;
  const cosP = Math.cos(pitch);
  const sinP = Math.sin(pitch);
  const y2 = y1 * cosP - z1 * sinP;
  const z2 = y1 * sinP + z1 * cosP;
  const x2 = x1;
  const scale = (Math.min(width, height) * 0.78) / bounds.span;
  const sx = width / 2 + x2 * scale;
  const sy = height / 2 - z2 * scale;
  return [sx, sy, y2];
}

function formatMm(v: number | null | undefined): string {
  if (v === null || v === undefined) return "n/a";
  return `${Math.round(v)} mm`;
}

/** Rotate vector by unit quaternion [w,x,y,z] in mocap world frame. */
function quatRotateVec(q: [number, number, number, number], v: [number, number, number]): [number, number, number] {
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

type ImuAxisDraw = {
  key: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  depth: number;
};

function imuAxesAtWrist(
  frame: number,
  wristJoint: number,
  side: "left" | "right",
  up: UpAxis,
  yawDeg: number,
  pitchDeg: number,
  width: number,
  height: number,
  bounds: { cx: number; cy: number; cz: number; span: number },
): ImuAxisDraw[] {
  const origin = displayPoint(frame, wristJoint, up);
  const quat = IMU.quat_wxyz[side][frame] as [number, number, number, number] | null;
  if (!origin || !quat || !IMU.valid[side][frame]) return [];
  const axisDefs: Array<{ key: string; vec: [number, number, number]; color: string }> = [
    { key: "x", vec: [1, 0, 0], color: "#ef4444" },
    { key: "y", vec: [0, 1, 0], color: "#22c55e" },
    { key: "z", vec: [0, 0, 1], color: "#3b82f6" },
  ];
  const out: ImuAxisDraw[] = [];
  for (const axis of axisDefs) {
    const rotated = quatRotateVec(quat, axis.vec);
    const endWorld: [number, number, number] = [
      origin[0] + rotated[0] * IMU_AXIS_LEN,
      origin[1] + rotated[1] * IMU_AXIS_LEN,
      origin[2] + rotated[2] * IMU_AXIS_LEN,
    ];
    const p0 = project(origin[0], origin[1], origin[2], yawDeg, pitchDeg, width, height, bounds);
    const p1 = project(endWorld[0], endWorld[1], endWorld[2], yawDeg, pitchDeg, width, height, bounds);
    out.push({
      key: `${side}-${axis.key}`,
      x1: p0[0],
      y1: p0[1],
      x2: p1[0],
      y2: p1[1],
      color: axis.color,
      depth: (p0[2] + p1[2]) / 2,
    });
  }
  return out.sort((a, b) => a.depth - b.depth);
}

export default function Skeleton3DPlayback() {
  const theme = useHostTheme();
  const [frame, setFrame] = useCanvasState("frame", 0);
  const [playing, setPlaying] = useCanvasState("playing", false);
  const [view, setView] = useCanvasState<ViewName>("view", "iso");
  const [upAxis, setUpAxis] = useCanvasState<UpAxis>("upAxis", DEFAULT_UP);
  const [heightOffsetMm, setHeightOffsetMm] = useCanvasState("heightOffsetMm", "0");
  const [yaw, setYaw] = useCanvasState("yaw", VIEW_PRESETS.iso.yaw);
  const [pitch, setPitch] = useCanvasState("pitch", VIEW_PRESETS.iso.pitch);
  const [drag, setDrag] = useCanvasState<{ x: number; y: number; yaw: number; pitch: number } | null>(
    "drag",
    null,
  );
  const [showImu, setShowImu] = useCanvasState("showImu", true);

  const width = 920;
  const height = 560;
  const safeFrame = Math.max(0, Math.min(FRAME_COUNT - 1, frame));
  const seq = DATA.seqs[safeFrame] ?? safeFrame;
  const bounds = BOUNDS_BY_UP[upAxis];
  const offsetM = (Number(heightOffsetMm) || 0) / 1000;
  const ground = GROUND_BY_UP[upAxis] + offsetM;

  const points: Array<[number, number, number] | null> = [];
  for (let j = 0; j < N_JOINTS; j++) points.push(displayPoint(safeFrame, j, upAxis));

  const leftFoot = points[LEFT_ANKLE];
  const rightFoot = points[RIGHT_ANKLE];
  const leftMm = leftFoot ? (leftFoot[2] - ground) * 1000 : null;
  const rightMm = rightFoot ? (rightFoot[2] - ground) * 1000 : null;

  const projected = points.map((p) =>
    p ? project(p[0], p[1], p[2], yaw, pitch, width, height, bounds) : null,
  );

  const bones = DATA.edges
    .map(([a, b], i) => {
      const pa = projected[a];
      const pb = projected[b];
      if (!pa || !pb) return null;
      return { key: i, x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1], depth: (pa[2] + pb[2]) / 2 };
    })
    .filter(
      (b): b is { key: number; x1: number; y1: number; x2: number; y2: number; depth: number } =>
        b !== null,
    )
    .sort((a, b) => a.depth - b.depth);

  const jointsDrawn = projected
    .map((p, i) => (p ? { key: i, x: p[0], y: p[1], depth: p[2], name: DATA.joints[i] } : null))
    .filter(
      (j): j is { key: number; x: number; y: number; depth: number; name: string } => j !== null,
    )
    .sort((a, b) => a.depth - b.depth);

  const half = bounds.span * 0.55;
  const corners: Array<[number, number, number]> = [
    [bounds.cx - half, bounds.cy - half, ground],
    [bounds.cx + half, bounds.cy - half, ground],
    [bounds.cx + half, bounds.cy + half, ground],
    [bounds.cx - half, bounds.cy + half, ground],
  ];
  const groundPoly = corners
    .map((c) => project(c[0], c[1], c[2], yaw, pitch, width, height, bounds))
    .map((p) => `${p[0]},${p[1]}`)
    .join(" ");

  const leftFootProj = leftFoot
    ? project(leftFoot[0], leftFoot[1], leftFoot[2], yaw, pitch, width, height, bounds)
    : null;
  const leftGroundProj = leftFoot
    ? project(leftFoot[0], leftFoot[1], ground, yaw, pitch, width, height, bounds)
    : null;
  const rightFootProj = rightFoot
    ? project(rightFoot[0], rightFoot[1], rightFoot[2], yaw, pitch, width, height, bounds)
    : null;
  const rightGroundProj = rightFoot
    ? project(rightFoot[0], rightFoot[1], ground, yaw, pitch, width, height, bounds)
    : null;

  const imuAxes = showImu
    ? [
        ...imuAxesAtWrist(safeFrame, LEFT_WRIST, "left", upAxis, yaw, pitch, width, height, bounds),
        ...imuAxesAtWrist(safeFrame, RIGHT_WRIST, "right", upAxis, yaw, pitch, width, height, bounds),
      ].sort((a, b) => a.depth - b.depth)
    : [];

  function applyView(name: Exclude<ViewName, "free">) {
    setView(name);
    setYaw(VIEW_PRESETS[name].yaw);
    setPitch(VIEW_PRESETS[name].pitch);
  }

  function startPlayback() {
    stopPlayTimer();
    setPlaying(true);
    playTimer = setInterval(() => {
      setFrame((prev) => {
        const next = prev + 1;
        if (next >= FRAME_COUNT) {
          stopPlayTimer();
          setPlaying(false);
          return FRAME_COUNT - 1;
        }
        return next;
      });
    }, 33);
  }

  function pausePlayback() {
    setPlaying(false);
    stopPlayTimer();
  }

  function onScrub(value: string) {
    pausePlayback();
    const n = Number(value);
    if (!Number.isFinite(n)) return;
    setFrame(Math.max(0, Math.min(FRAME_COUNT - 1, Math.round(n))));
  }

  return (
    <Stack gap={16} style={{ padding: 20, minHeight: "100%" }}>
      <Stack gap={6}>
        <H1>0810 line1 dual-external 3D skeleton</H1>
        <Text tone="secondary" size="small">
          Filtered multiview XYZ · {FRAME_COUNT} frames · display Z-up via{" "}
          {upAxis === "y_up" ? "Y→Z remap (mocap Y-up)" : "raw Z-up"} · ground = ankle vertical
          p5 ({ground.toFixed(3)} m)
        </Text>
        <Text tone="tertiary" size="small">
          Source: {DATA.source}
        </Text>
        <Text tone="tertiary" size="small">
          IMU overlay: {IMU.method} · {IMU.imu_source}
        </Text>
      </Stack>

      <Row gap={12} wrap>
        <Stat value={String(seq)} label="seq" />
        <Stat value={formatMm(leftMm)} label="left foot above ground" tone="info" />
        <Stat value={formatMm(rightMm)} label="right foot above ground" tone="info" />
        <Stat value={`${(ground * 1000).toFixed(0)} mm`} label="ground (display up)" />
      </Row>

      <Card>
        <CardBody style={{ padding: 12 }}>
          <Stack gap={12}>
            <Row gap={8} align="center" wrap>
              <Button variant="primary" onClick={playing ? pausePlayback : startPlayback}>
                {playing ? "Pause" : "Play"}
              </Button>
              <Pill active={view === "iso"} onClick={() => applyView("iso")}>
                Iso
              </Pill>
              <Pill active={view === "front"} onClick={() => applyView("front")}>
                Front
              </Pill>
              <Pill active={view === "side"} onClick={() => applyView("side")}>
                Side
              </Pill>
              <Pill active={view === "top"} onClick={() => applyView("top")}>
                Top
              </Pill>
              <Pill active={view === "free"} onClick={() => setView("free")}>
                Free
              </Pill>
              <Pill active={showImu} onClick={() => setShowImu(!showImu)}>
                IMU axes
              </Pill>
              <Spacer />
              <Text tone="secondary" size="small">
                Drag viewport to orbit
              </Text>
            </Row>

            <Row gap={8} align="center" wrap>
              <Text weight="medium" size="small">
                Up axis
              </Text>
              <Pill active={upAxis === "y_up"} onClick={() => setUpAxis("y_up")}>
                Y-up (standing)
              </Pill>
              <Pill active={upAxis === "z_up"} onClick={() => setUpAxis("z_up")}>
                Z-up (raw)
              </Pill>
              <Text tone="secondary" size="small">
                ground offset mm
              </Text>
              <TextInput
                value={heightOffsetMm}
                onChange={setHeightOffsetMm}
                placeholder="0"
                type="number"
                style={{ width: 88 }}
              />
            </Row>

            <div
              style={{
                width: "100%",
                height,
                background: theme.bg.editor,
                border: `1px solid ${theme.stroke.secondary}`,
                borderRadius: 8,
                overflow: "hidden",
                touchAction: "none",
                userSelect: "none",
              }}
              onPointerDown={(e: {
                clientX: number;
                clientY: number;
                pointerId: number;
                currentTarget: HTMLDivElement;
              }) => {
                e.currentTarget.setPointerCapture(e.pointerId);
                setView("free");
                setDrag({ x: e.clientX, y: e.clientY, yaw, pitch });
              }}
              onPointerMove={(e: { clientX: number; clientY: number }) => {
                if (!drag) return;
                const dyaw = (e.clientX - drag.x) * 0.35;
                const dpitch = (e.clientY - drag.y) * 0.35;
                setYaw(drag.yaw + dyaw);
                setPitch(Math.max(-89, Math.min(89, drag.pitch + dpitch)));
              }}
              onPointerUp={() => setDrag(null)}
              onPointerCancel={() => setDrag(null)}
            >
              <svg
                width="100%"
                height="100%"
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="xMidYMid meet"
              >
                <polygon points={groundPoly} fill={theme.fill.tertiary} opacity={0.55} />
                {leftFootProj && leftGroundProj ? (
                  <line
                    x1={leftFootProj[0]}
                    y1={leftFootProj[1]}
                    x2={leftGroundProj[0]}
                    y2={leftGroundProj[1]}
                    stroke={theme.accent.primary}
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                  />
                ) : null}
                {rightFootProj && rightGroundProj ? (
                  <line
                    x1={rightFootProj[0]}
                    y1={rightFootProj[1]}
                    x2={rightGroundProj[0]}
                    y2={rightGroundProj[1]}
                    stroke={theme.text.secondary}
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                  />
                ) : null}
                {bones.map((b) => (
                  <line
                    key={b.key}
                    x1={b.x1}
                    y1={b.y1}
                    x2={b.x2}
                    y2={b.y2}
                    stroke={theme.accent.primary}
                    strokeWidth={3}
                    strokeLinecap="round"
                  />
                ))}
                {jointsDrawn.map((j) => (
                  <circle
                    key={j.key}
                    cx={j.x}
                    cy={j.y}
                    r={j.name.includes("ankle") ? 5 : 3.2}
                    fill={j.name.includes("ankle") ? theme.text.primary : theme.fill.primary}
                    stroke={theme.stroke.primary}
                    strokeWidth={1}
                  />
                ))}
                {imuAxes.map((axis) => (
                  <line
                    key={axis.key}
                    x1={axis.x1}
                    y1={axis.y1}
                    x2={axis.x2}
                    y2={axis.y2}
                    stroke={axis.color}
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    opacity={0.92}
                  />
                ))}
              </svg>
            </div>

            <Stack gap={6}>
              <Row align="center" gap={12}>
                <Text weight="medium">Frame</Text>
                <Text tone="secondary">
                  {safeFrame + 1} / {FRAME_COUNT}
                </Text>
                <Spacer />
                <Text tone="tertiary" size="small">
                  yaw {yaw.toFixed(0)}° · pitch {pitch.toFixed(0)}°
                </Text>
              </Row>
              <input
                type="range"
                min={0}
                max={FRAME_COUNT - 1}
                value={safeFrame}
                onChange={(e: { target: { value: string } }) => onScrub(e.target.value)}
                style={{ width: "100%", accentColor: theme.accent.primary }}
              />
            </Stack>
          </Stack>
        </CardBody>
      </Card>

      <Stack gap={4}>
        <H2>How to play</H2>
        <Text tone="secondary" size="small">
          Default display remaps mocap Y-up → viewer Z-up so the person stands on the ground
          plane. Use Play/Pause or the scrubber; Iso/Front/Side/Top or drag to orbit. Foot
          heights are ankle height above ground on the display vertical axis (ankle p5 +
          optional offset). Toggle IMU axes to show Mahony-estimated wrist orientation (RGB =
          XYZ, 80 mm) anchored at skeleton wrist joints.
        </Text>
      </Stack>
    </Stack>
  );
}
'''


def main() -> None:
    import math
    import base64
    import struct

    payload = DATA_PATH.read_text(encoding="utf-8")
    imu_payload = IMU_OVERLAY_PATH.read_text(encoding="utf-8")
    embedded = json.dumps(payload)
    imu_embedded = json.dumps(imu_payload)
    code = CANVAS_BODY.replace("__EMBEDDED_JSON__", embedded).replace(
        "__IMU_EMBEDDED_JSON__", imu_embedded
    )
    CANVAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANVAS_PATH.write_text(code, encoding="utf-8")
    print(f"wrote {CANVAS_PATH} ({CANVAS_PATH.stat().st_size / 1e6:.2f} MB)")

    d = json.loads(payload)
    joints = d["joints"]
    n = len(joints)
    miss = d["missing_sentinel"]
    raw = base64.b64decode(d["xyz_i16_b64"])
    arr = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    la = joints.index("left_ankle")
    ra = joints.index("right_ankle")
    samples = []
    for f in range(d["frame_count"]):
        for j in (la, ra):
            y = arr[(f * n + j) * 3 + 1]
            if y != miss:
                samples.append(y / 1000.0)
    samples.sort()
    k = (len(samples) - 1) * 0.05
    f0, f1 = math.floor(k), math.ceil(k)
    g = samples[f0] if f0 == f1 else samples[f0] * (f1 - k) + samples[f1] * (k - f0)
    print(f"expected Y-up display ground (ankle Y p5): {g:.4f} m")


if __name__ == "__main__":
    main()
