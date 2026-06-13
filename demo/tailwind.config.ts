import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        slide: {
          bg: "#12161d",
          surface: "#1a212b",
          elevated: "#222b38",
          border: "#2d3a4d",
          muted: "#8b9cb3",
          body: "#c5d0de",
          ink: "#f0f4f8",
        },
        accent: {
          DEFAULT: "#4fd1ed",
          dim: "#38a8c0",
          glow: "rgba(79, 209, 237, 0.15)",
        },
        grounded: {
          ink: "#f0f4f8",
          slate: "#8b9cb3",
          mist: "#12161d",
          accent: "#4fd1ed",
          success: "#4ade80",
          warn: "#fbbf24",
        },
      },
      fontFamily: {
        display: ["var(--font-inter)", "system-ui", "sans-serif"],
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "ui-monospace", "monospace"],
      },
      animation: {
        "fade-up": "fade-up 0.5s ease-out forwards",
        shimmer: "shimmer 2.5s linear infinite",
        "pulse-soft": "pulse-soft 2s ease-in-out infinite",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% center" },
          "100%": { backgroundPosition: "-200% center" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "0.7" },
          "50%": { opacity: "1" },
        },
      },
      boxShadow: {
        card: "0 4px 24px rgba(0, 0, 0, 0.25)",
        "accent-sm": "0 0 0 1px rgba(79, 209, 237, 0.25)",
      },
    },
  },
  plugins: [],
};

export default config;
