import React, { useEffect, useMemo, useState } from "react";
import type { GeoJsonFeatureCollection, OverlayOption, SimulationState, TelemetryPoint } from "./api";
import { fetchGeoJson, fetchRecentTelemetry, fetchSimulationState } from "./api";
import { BaliMap } from "./BaliMap";
import { GlobeView } from "./GlobeView";
import { TelemetryPanel } from "./TelemetryPanel";
import { SimulationControls } from "./SimulationControls";

const PLAY_INTERVAL_MS = 1600;

export const App: React.FC = () => {
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [state, setState] = useState<SimulationState | null>(null);
  const [geojson, setGeojson] = useState<GeoJsonFeatureCollection | null>(null);
  const [modelId, setModelId] = useState("lfm2");
  const [overlay, setOverlay] = useState<OverlayOption["id"]>("flood_risk");
  const [passIndex, setPassIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const data = await fetchRecentTelemetry();
        if (!cancelled) setTelemetry(data);
      } catch {
        if (!cancelled) setTelemetry([]);
      }
    };

    poll();
    const handle = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchSimulationState(), fetchGeoJson()])
      .then(([simState, geo]) => {
        if (cancelled) return;
        setState(simState);
        setGeojson(geo);
        const preferred = simState.models.find((model) => model.model_id === "lfm2") ?? simState.models[0];
        if (preferred) setModelId(preferred.model_id);
        setPassIndex(0);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load simulation data");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!playing || !state?.passes.length) return;
    const handle = setInterval(() => {
      setPassIndex((current) => (current + 1) % state.passes.length);
    }, PLAY_INTERVAL_MS);
    return () => clearInterval(handle);
  }, [playing, state?.passes.length]);

  const latest = telemetry[0] ?? null;
  const selectedPass = state?.passes[passIndex] ?? null;
  const selectedOverlay = useMemo(
    () => state?.overlays.find((item) => item.id === overlay) ?? null,
    [overlay, state?.overlays],
  );

  return (
    <div className="app-shell">
      <aside className="left-rail">
        <header className="brand-bar">
          <h1>Bali Flood Prevention Monitoring</h1>
          <span>{state?.run.name ?? "Simulation DB"}</span>
        </header>
        <section className="globe-panel">
          <GlobeView telemetry={telemetry} />
        </section>
        <TelemetryPanel latest={latest} />
        <SimulationControls />
      </aside>

      <main className="map-panel">
        <header className="map-toolbar">
          <div className="toolbar-group">
            <label>
              <span>Model</span>
              <select value={modelId} onChange={(event) => setModelId(event.target.value)}>
                {state?.models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.model_label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Overlay</span>
              <select value={overlay} onChange={(event) => setOverlay(event.target.value as OverlayOption["id"])}>
                {state?.overlays.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="timeline-control">
            <button type="button" onClick={() => setPlaying((value) => !value)}>
              {playing ? "Pause" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={Math.max(0, (state?.passes.length ?? 1) - 1)}
              value={passIndex}
              onChange={(event) => setPassIndex(Number(event.target.value))}
            />
            <div className="timeline-label">
              <strong>{selectedOverlay?.label ?? "Overlay"}</strong>
              <span>{selectedPass?.timestamp ?? "No timestep"}</span>
            </div>
          </div>
        </header>

        {error ? (
          <div className="empty-state">{error}</div>
        ) : (
          <BaliMap
            geojson={geojson}
            areas={state?.areas ?? []}
            aggregates={state?.aggregates ?? []}
            overlay={overlay}
            modelId={modelId}
            runId={state?.run.id ?? null}
            selectedPass={selectedPass}
          />
        )}
      </main>
    </div>
  );
};
