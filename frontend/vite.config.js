import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            "/summarize": {
                target: "http://127.0.0.1:5000",
                changeOrigin: true,
            },
        },
    },
    build: {
        outDir: path.resolve(__dirname, "../static/dist"),
        emptyOutDir: true,
        assetsDir: "assets",
        rollupOptions: {
            output: {
                entryFileNames: "app.js",
                assetFileNames: (assetInfo) => {
                    if (assetInfo.name && assetInfo.name.endsWith(".css")) {
                        return "app.css";
                    }
                    return "assets/[name][extname]";
                },
            },
        },
    },
});
