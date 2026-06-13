"use client";

export function DemoBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 bg-slide-bg" aria-hidden>
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: `
            radial-gradient(ellipse 70% 45% at 0% 0%, rgba(79, 209, 237, 0.07), transparent),
            radial-gradient(ellipse 50% 35% at 100% 100%, rgba(79, 209, 237, 0.05), transparent)
          `,
        }}
      />
      <div
        className="absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.6) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.6) 1px, transparent 1px)
          `,
          backgroundSize: "48px 48px",
        }}
      />
    </div>
  );
}
