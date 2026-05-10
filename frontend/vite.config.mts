import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import cesium from "vite-plugin-cesium";
import fs from "node:fs";
import path from "node:path";

function readRootEnv(name: string): string {
  const rootEnv = path.resolve(__dirname, "../../../../.env");
  if (!fs.existsSync(rootEnv)) return "";
  const line = fs
    .readFileSync(rootEnv, "utf-8")
    .split(/\r?\n/)
    .find((item) => item.trim().startsWith(`${name}=`));
  return line?.slice(name.length + 1).trim() ?? "";
}

export default defineConfig({
  plugins: [
    react(),
    cesium({
      cesiumBaseUrl: "cesium",
    }),
  ],
  define: {
    __MAPBOX_TOKEN__: JSON.stringify(process.env.MAPBOX_TOKEN || readRootEnv("MAPBOX_TOKEN")),
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "assets/main.js",
        assetFileNames: "assets/[name].[ext]",
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
  base: "/static/",
});
