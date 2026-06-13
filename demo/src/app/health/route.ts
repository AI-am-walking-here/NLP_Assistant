import { NextResponse } from "next/server";

const backendUrl = process.env.GROUNDED_API_URL ?? "http://127.0.0.1:8080";

export async function GET() {
  try {
    const res = await fetch(`${backendUrl}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(30_000),
    });
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Backend unreachable";
    return NextResponse.json(
      {
        status: "loading",
        mock_mode: false,
        stack_ready: false,
        error:
          message.includes("ECONNREFUSED") || message.includes("fetch failed")
            ? "Python API is starting (model preload takes 1–3 minutes). Refresh shortly."
            : message,
      },
      { status: 503 },
    );
  }
}
