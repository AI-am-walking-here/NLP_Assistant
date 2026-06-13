import { NextResponse } from "next/server";

const backendUrl = process.env.GROUNDED_API_URL ?? "http://127.0.0.1:8080";

export async function GET() {
  try {
    const res = await fetch(`${backendUrl}/api/systems`, {
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
        detail:
          message.includes("ECONNREFUSED") || message.includes("fetch failed")
            ? "API still loading models. Wait for /health stack_ready, then reload."
            : message,
      },
      { status: 503 },
    );
  }
}
