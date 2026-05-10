import axios from "axios";

const SIMSAT_API_BASE = import.meta.env.VITE_SIMSAT_DASHBOARD_URL ?? "http://localhost:8000/api";
const BALI_API_BASE = import.meta.env.VITE_BALI_SIM_API_URL ?? "http://localhost:8010/api";
const BALI_ORIGIN = BALI_API_BASE.replace(/\/api\/?$/, "");

const simsatApi = axios.create({ baseURL: SIMSAT_API_BASE });
const baliApi = axios.create({ baseURL: BALI_API_BASE });

export interface TelemetryPoint {
  satellite: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  altitude: number | null;
  extra: unknown;
}

export interface Command {
  command: "start" | "pause" | "stop" | "set_start_time" | "set_step_size" | "set_replay_speed";
  parameters: {
    start_time?: string;
    step_size_seconds?: number;
    replay_speed?: number;
  };
}

export interface SimulationRun {
  id: number;
  name: string;
  start_date: string;
  end_date: string;
  max_timesteps: number;
  points_per_location: number;
  seed: number;
  size_km: number;
  created_at: string;
  metadata_json: string;
}

export interface SatellitePass {
  id: number;
  timestep_index: number;
  timestamp: string;
  source: string;
}

export interface SimulationModel {
  model_id: string;
  model_label: string;
}

export interface Area {
  area_id: string;
  area_name: string;
  fallback_lon: number;
  fallback_lat: number;
}

export interface Aggregate {
  run_id: number;
  pass_id: number;
  area_id: string;
  model_id: string;
  observation_count: number;
  valid_prediction_count: number;
  flood_risk_score: number | null;
  flood_risk_level: string | null;
  water_extent_score: number | null;
  water_extent_level: string | null;
  severity_score: number | null;
  severity_level: string | null;
  confidence_score: number | null;
  confidence_level: string | null;
  timestep_index: number;
  timestamp: string;
}

export interface OverlayOption {
  id: "flood_risk" | "water_extent" | "severity" | "confidence";
  label: string;
}

export interface SimulationState {
  run: SimulationRun;
  passes: SatellitePass[];
  models: SimulationModel[];
  areas: Area[];
  aggregates: Aggregate[];
  overlays: OverlayOption[];
}

export interface AreaObservation {
  id: number;
  point_id: string;
  point_index: number;
  timestamp: string;
  lon: number;
  lat: number;
  rgb_path: string | null;
  swir_path: string | null;
  rgb_url: string;
  swir_url: string;
  status: string;
  error: string | null;
}

export interface AreaObservationResponse {
  observations: AreaObservation[];
}

export type GeoJsonFeatureCollection = {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    properties: Record<string, unknown>;
    geometry: {
      type: "Polygon" | "MultiPolygon";
      coordinates: unknown;
    };
  }>;
};

export async function fetchRecentTelemetry(): Promise<TelemetryPoint[]> {
  const res = await simsatApi.get<{ telemetry: TelemetryPoint[] }>("/telemetry/recent/");
  return res.data.telemetry ?? [];
}

export async function sendCommand(
  command: Command["command"],
  parameters?: Command["parameters"],
): Promise<{ id: number; command: string; parameters: Record<string, unknown>; created_at: string }> {
  const res = await simsatApi.post("/commands/", {
    command,
    ...(parameters || {}),
  });
  return res.data;
}

export async function fetchSimulationState(runId?: number): Promise<SimulationState> {
  const res = await baliApi.get<SimulationState>("/state", {
    params: runId ? { run_id: runId } : undefined,
  });
  return res.data;
}

export async function fetchGeoJson(): Promise<GeoJsonFeatureCollection> {
  const res = await baliApi.get<GeoJsonFeatureCollection>("/geojson");
  return res.data;
}

export async function fetchAreaObservations(
  areaId: string,
  passId: number,
  runId?: number,
): Promise<AreaObservation[]> {
  const res = await baliApi.get<AreaObservationResponse>(`/areas/${areaId}/observations`, {
    params: { pass_id: passId, ...(runId ? { run_id: runId } : {}) },
  });
  return res.data.observations ?? [];
}

export function baliAssetUrl(path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  return `${BALI_ORIGIN}${path}`;
}
