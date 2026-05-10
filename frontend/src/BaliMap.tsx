import React, { useMemo, useState } from "react";
import type {
  Aggregate,
  Area,
  AreaObservation,
  GeoJsonFeatureCollection,
  OverlayOption,
  SatellitePass,
} from "./api";
import { baliAssetUrl, fetchAreaObservations } from "./api";

type Feature = GeoJsonFeatureCollection["features"][number];

interface BaliMapProps {
  geojson: GeoJsonFeatureCollection | null;
  areas: Area[];
  aggregates: Aggregate[];
  overlay: OverlayOption["id"];
  modelId: string;
  runId: number | null;
  selectedPass: SatellitePass | null;
}

interface HoverState {
  x: number;
  y: number;
  areaName: string;
  label: string;
  score: number | null;
  valid: number;
  total: number;
}

interface DetailState {
  areaId: string;
  areaName: string;
  observations: AreaObservation[];
  index: number;
  loading: boolean;
}

const WIDTH = 1000;
const HEIGHT = 680;

const COLOR_BY_LEVEL: Record<string, Record<string, string>> = {
  flood_risk: { low: "#16a34a", medium: "#f59e0b", high: "#dc2626" },
  water_extent: { small: "#38bdf8", moderate: "#2563eb", large: "#1e3a8a" },
  severity: { low: "#22c55e", medium: "#eab308", high: "#e11d48" },
  confidence: { low: "#94a3b8", medium: "#a78bfa", high: "#14b8a6" },
};

function collectCoordinates(value: unknown, out: Array<[number, number]>): void {
  if (!Array.isArray(value)) return;
  if (
    value.length >= 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number"
  ) {
    out.push([value[0], value[1]]);
    return;
  }
  value.forEach((item) => collectCoordinates(item, out));
}

function boundsFor(features: Feature[]): [number, number, number, number] {
  const coords: Array<[number, number]> = [];
  features.forEach((feature) => collectCoordinates(feature.geometry.coordinates, coords));
  if (!coords.length) return [114.3, -8.9, 115.8, -8.0];
  return [
    Math.min(...coords.map(([lon]) => lon)),
    Math.min(...coords.map(([, lat]) => lat)),
    Math.max(...coords.map(([lon]) => lon)),
    Math.max(...coords.map(([, lat]) => lat)),
  ];
}

function makeProjector(bounds: [number, number, number, number]) {
  const [minLon, minLat, maxLon, maxLat] = bounds;
  const lonSpan = maxLon - minLon || 1;
  const latSpan = maxLat - minLat || 1;
  return ([lon, lat]: [number, number]): [number, number] => [
    ((lon - minLon) / lonSpan) * WIDTH,
    ((maxLat - lat) / latSpan) * HEIGHT,
  ];
}

function ringPath(ring: unknown, project: (point: [number, number]) => [number, number]): string {
  if (!Array.isArray(ring)) return "";
  return ring
    .map((point, index) => {
      if (!Array.isArray(point) || point.length < 2) return "";
      const [x, y] = project([Number(point[0]), Number(point[1])]);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ")
    .concat(" Z");
}

function polygonPath(poly: unknown, project: (point: [number, number]) => [number, number]): string {
  if (!Array.isArray(poly)) return "";
  return poly.map((ring) => ringPath(ring, project)).join(" ");
}

function featurePath(feature: Feature, project: (point: [number, number]) => [number, number]): string {
  const { type, coordinates } = feature.geometry;
  if (type === "Polygon") return polygonPath(coordinates, project);
  if (type === "MultiPolygon" && Array.isArray(coordinates)) {
    return coordinates.map((poly) => polygonPath(poly, project)).join(" ");
  }
  return "";
}

function overlayValue(aggregate: Aggregate | undefined, overlay: OverlayOption["id"]) {
  if (!aggregate) return { level: "no data", score: null };
  if (overlay === "flood_risk") {
    return { level: aggregate.flood_risk_level ?? "no data", score: aggregate.flood_risk_score };
  }
  if (overlay === "water_extent") {
    return { level: aggregate.water_extent_level ?? "no data", score: aggregate.water_extent_score };
  }
  if (overlay === "severity") {
    return { level: aggregate.severity_level ?? "no data", score: aggregate.severity_score };
  }
  return { level: aggregate.confidence_level ?? "no data", score: aggregate.confidence_score };
}

function fillFor(aggregate: Aggregate | undefined, overlay: OverlayOption["id"]): string {
  if (!aggregate) return "#263241";
  const { level } = overlayValue(aggregate, overlay);
  return COLOR_BY_LEVEL[overlay][level] ?? "#334155";
}

function formatScore(score: number | null): string {
  return score == null ? "n/a" : score.toFixed(2);
}

export const BaliMap: React.FC<BaliMapProps> = ({
  geojson,
  areas,
  aggregates,
  overlay,
  modelId,
  runId,
  selectedPass,
}) => {
  const [hover, setHover] = useState<HoverState | null>(null);
  const [detail, setDetail] = useState<DetailState | null>(null);

  const aggregateByArea = useMemo(() => {
    const map = new Map<string, Aggregate>();
    aggregates
      .filter((item) => item.model_id === modelId && item.pass_id === selectedPass?.id)
      .forEach((item) => map.set(item.area_id, item));
    return map;
  }, [aggregates, modelId, selectedPass?.id]);

  const paths = useMemo(() => {
    if (!geojson) return [];
    const project = makeProjector(boundsFor(geojson.features));
    return geojson.features.map((feature) => ({
      areaId: String(feature.properties.area_id ?? ""),
      areaName: String(feature.properties.area_name ?? feature.properties.shapeName ?? ""),
      path: featurePath(feature, project),
    }));
  }, [geojson]);

  const areaNames = useMemo(() => {
    return new Map(areas.map((area) => [area.area_id, area.area_name]));
  }, [areas]);

  const openDetail = async (areaId: string, areaName: string) => {
    if (!runId || !selectedPass) return;
    setDetail({ areaId, areaName, observations: [], index: 0, loading: true });
    const observations = await fetchAreaObservations(areaId, selectedPass.id, runId);
    setDetail({ areaId, areaName, observations, index: 0, loading: false });
  };

  return (
    <div className="map-surface">
      <svg
        className="bali-map"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Bali area risk map"
      >
        <rect x="0" y="0" width={WIDTH} height={HEIGHT} className="map-water" />
        {paths.map((item) => {
          const aggregate = aggregateByArea.get(item.areaId);
          const value = overlayValue(aggregate, overlay);
          return (
            <path
              key={item.areaId}
              d={item.path}
              fill={fillFor(aggregate, overlay)}
              className="area-shape"
              fillRule="evenodd"
              onMouseMove={(event) => {
                setHover({
                  x: event.clientX,
                  y: event.clientY,
                  areaName: item.areaName || areaNames.get(item.areaId) || item.areaId,
                  label: value.level,
                  score: value.score,
                  valid: aggregate?.valid_prediction_count ?? 0,
                  total: aggregate?.observation_count ?? 0,
                });
              }}
              onMouseLeave={() => setHover(null)}
              onClick={() => openDetail(item.areaId, item.areaName || item.areaId)}
            />
          );
        })}
      </svg>

      <div className="map-legend">
        {Object.entries(COLOR_BY_LEVEL[overlay]).map(([level, color]) => (
          <span key={level}>
            <i style={{ background: color }} />
            {level}
          </span>
        ))}
      </div>

      {hover && (
        <div className="map-tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          <strong>{hover.areaName}</strong>
          <span>{hover.label}</span>
          <span>score {formatScore(hover.score)}</span>
          <span>{hover.valid}/{hover.total} valid</span>
        </div>
      )}

      {detail && (
        <ObservationModal
          detail={detail}
          setDetail={setDetail}
        />
      )}
    </div>
  );
};

interface ObservationModalProps {
  detail: DetailState;
  setDetail: React.Dispatch<React.SetStateAction<DetailState | null>>;
}

const ObservationModal: React.FC<ObservationModalProps> = ({ detail, setDetail }) => {
  const current = detail.observations[detail.index] ?? null;
  const count = detail.observations.length;

  const move = (delta: number) => {
    if (!count) return;
    setDetail((prev) => {
      if (!prev) return prev;
      return { ...prev, index: (prev.index + delta + count) % count };
    });
  };

  return (
    <div className="modal-backdrop" onClick={() => setDetail(null)}>
      <section className="image-modal" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <div>
            <h2>{detail.areaName}</h2>
            <span>{current ? `${current.point_id} - ${detail.index + 1}/${count}` : "No images"}</span>
          </div>
          <button type="button" className="icon-button" onClick={() => setDetail(null)} aria-label="Close">
            x
          </button>
        </header>

        {detail.loading ? (
          <div className="modal-empty">Loading images...</div>
        ) : current ? (
          <>
            <div className="image-grid">
              <figure>
                <img src={baliAssetUrl(current.rgb_url)} alt={`${current.point_id} RGB`} />
                <figcaption>RGB</figcaption>
              </figure>
              <figure>
                <img src={baliAssetUrl(current.swir_url)} alt={`${current.point_id} SWIR`} />
                <figcaption>SWIR</figcaption>
              </figure>
            </div>
            <footer className="modal-actions">
              <button type="button" onClick={() => move(-1)}>Previous</button>
              <button type="button" onClick={() => move(1)}>Next</button>
            </footer>
          </>
        ) : (
          <div className="modal-empty">No RGB/SWIR images stored for this area and time.</div>
        )}
      </section>
    </div>
  );
};
