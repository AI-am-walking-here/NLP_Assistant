import http from "node:http";
import https from "node:https";
import { NextRequest, NextResponse } from "next/server";

const backendUrl = process.env.GROUNDED_API_URL ?? "http://127.0.0.1:8080";
const TIMEOUT_MS = 15 * 60 * 1000;

export const maxDuration = 900;
export const dynamic = "force-dynamic";

function proxyPost(
  target: string,
  body: string,
  timeoutMs: number,
): Promise<{ status: number; text: string }> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(target);
    const transport = parsed.protocol === "https:" ? https : http;
    const req = transport.request(
      {
        hostname: parsed.hostname,
        port: parsed.port || (parsed.protocol === "https:" ? 443 : 80),
        path: `${parsed.pathname}${parsed.search}`,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
        timeout: timeoutMs,
      },
      (res: http.IncomingMessage) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          resolve({
            status: res.statusCode ?? 500,
            text: Buffer.concat(chunks).toString("utf8"),
          });
        });
      },
    );
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Backend request timed out"));
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

export async function POST(req: NextRequest) {
  const body = await req.text();
  try {
    const res = await proxyPost(
      `${backendUrl}/api/generate`,
      body,
      TIMEOUT_MS,
    );
    return new NextResponse(res.text, {
      status: res.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Backend request failed";
    return NextResponse.json(
      {
        detail:
          message.includes("timeout") || message.includes("aborted")
            ? "Generation timed out after ~5–15 minutes. For local dev, set NEXT_PUBLIC_GROUNDED_API_URL=http://127.0.0.1:8080 and restart pnpm dev, or try zero_shot (~30s)."
            : message,
      },
      { status: 504 },
    );
  }
}
