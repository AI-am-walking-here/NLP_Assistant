"use client";

import { buildShareUrl } from "@/lib/url-state";

type Props = {
  compareA?: string;
  compareB?: string;
  example?: string;
  present?: boolean;
};

export function QrCodePanel({ compareA, compareB, example, present }: Props) {
  const url = buildShareUrl({ compareA, compareB, example, present });
  if (!url) return null;

  const qrSrc = `https://api.qrserver.com/v1/create-qr-code/?size=120x120&data=${encodeURIComponent(url)}`;

  return (
    <div className="flex items-center gap-3 rounded-lg border border-slide-border bg-slide-bg px-3 py-2">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={qrSrc} alt="Demo QR code" width={72} height={72} className="rounded" />
      <div className="min-w-0">
        <p className="section-kicker text-[10px]">Share demo</p>
        <p className="truncate font-mono text-[10px] text-slide-muted">{url}</p>
        <button
          type="button"
          className="btn-secondary mt-1 px-2 py-1 text-[10px]"
          onClick={() => navigator.clipboard.writeText(url)}
        >
          Copy URL
        </button>
      </div>
    </div>
  );
}
