import React, { useEffect, useRef, useState } from "react";
import type { TelemetryPoint } from "./api";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";

interface GlobeViewProps {
  telemetry: TelemetryPoint[];
}

function validPoint(point: TelemetryPoint): boolean {
  return (
    Number.isFinite(point.longitude) &&
    Number.isFinite(point.latitude) &&
    Math.abs(point.longitude) <= 180 &&
    Math.abs(point.latitude) <= 90
  );
}

function makeMapboxProvider(): Cesium.UrlTemplateImageryProvider | null {
  const token = __MAPBOX_TOKEN__?.trim();
  if (!token) return null;
  return new Cesium.UrlTemplateImageryProvider({
    url: `https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}?access_token=${token}`,
    credit: "Mapbox",
    maximumLevel: 19,
    tileWidth: 256,
    tileHeight: 256,
  });
}

export const GlobeView: React.FC<GlobeViewProps> = ({ telemetry }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const [status, setStatus] = useState("Loading Cesium globe...");

  useEffect(() => {
    if (!containerRef.current) return;

    (window as any).CESIUM_BASE_URL = "/static/cesium/";

    const viewer = new Cesium.Viewer(containerRef.current, {
      animation: false,
      timeline: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      terrainProvider: new Cesium.EllipsoidTerrainProvider(),
      baseLayer: false,
    });

    viewer.scene.skyBox.show = false;
    viewer.scene.skyAtmosphere.show = false;
    viewer.scene.backgroundColor = Cesium.Color.BLACK;
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#17344a");
    viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;

    const provider = makeMapboxProvider();
    if (provider) {
      viewer.imageryLayers.addImageryProvider(provider);
      setStatus("");
    } else {
      setStatus("MAPBOX_TOKEN is missing");
    }

    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(78, 28, 9000000),
      orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-42),
        roll: 0,
      },
    });

    viewerRef.current = viewer;
    viewer.scene.requestRender();

    return () => {
      if (!viewer.isDestroyed()) viewer.destroy();
      viewerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const validTelemetry = telemetry.filter(validPoint);

    validTelemetry.forEach((point) => {
      const position = Cesium.Cartesian3.fromDegrees(
        point.longitude,
        point.latitude,
        (point.altitude ?? 0) * 1000,
      );
      let entity = viewer.entities.getById(point.satellite);

      if (!entity) {
        entity = viewer.entities.add({
          id: point.satellite,
          position,
          point: {
            pixelSize: 11,
            color: Cesium.Color.CYAN,
            outlineColor: Cesium.Color.WHITE,
            outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        });
      } else {
        entity.position = new Cesium.ConstantPositionProperty(position);
      }
    });

    viewer.scene.requestRender();
  }, [telemetry]);

  return (
    <div className="globe-shell">
      <div ref={containerRef} className="globe-view" />
      {status && <div className="globe-status">{status}</div>}
    </div>
  );
};
