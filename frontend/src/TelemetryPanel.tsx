import React from "react";
import type { TelemetryPoint } from "./api";

interface TelemetryPanelProps {
  latest: TelemetryPoint | null;
}

const formatTime = (isoString: string | null): string => {
  if (!isoString) return "-";
  try {
    const date = new Date(isoString);
    return `${date.toISOString().replace("T", " ").substring(0, 19)} UTC`;
  } catch {
    return isoString;
  }
};

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({ latest }) => {
  return (
    <section className="telemetry-panel">
      <h2>Telemetry</h2>
      {latest ? (
        <div className="telemetry-grid">
          <div>
            <span className="label">Satellite</span>
            <span className="value">{latest.satellite}</span>
          </div>
          <div>
            <span className="label">Latitude</span>
            <span className="value">{latest.latitude.toFixed(4)} deg</span>
          </div>
          <div>
            <span className="label">Longitude</span>
            <span className="value">{latest.longitude.toFixed(4)} deg</span>
          </div>
          <div>
            <span className="label">Altitude</span>
            <span className="value">{latest.altitude != null ? `${latest.altitude.toFixed(2)} km` : "-"}</span>
          </div>
          <div className="wide">
            <span className="label">Simulation Time</span>
            <span className="value mono">{formatTime(latest.timestamp)}</span>
          </div>
        </div>
      ) : (
        <p>No telemetry received.</p>
      )}
    </section>
  );
};
