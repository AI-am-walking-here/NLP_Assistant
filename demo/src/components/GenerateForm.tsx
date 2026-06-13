"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchHealth, fetchSystems, generateAbstract, verifyAbstract } from "@/lib/api";
import { DEMO_EXAMPLES } from "@/lib/examples";
import type { EvalShowcasePrompt } from "@/lib/eval-showcase";
import { fetchEvalShowcase } from "@/lib/eval-showcase";
import {
  DEFAULT_FEATURE_FLAGS,
  loadFeatureFlags,
  type DemoFeatureFlags,
} from "@/lib/feature-flags";
import { GUIDED_ACTS } from "@/lib/guided-tour";
import { healthChanged } from "@/lib/health-utils";
import { assessOutlineQuality } from "@/lib/outline-quality";
import { passageOverlapTokens } from "@/lib/passage-highlight";
import {
  deleteSnapshot,
  loadSnapshots,
  saveSnapshot,
  type DemoSnapshot,
} from "@/lib/snapshots";
import type {
  GenerateResponse,
  HealthResponse,
  PipelineStage,
  SystemInfo,
  VerifyResponse,
} from "@/lib/types";
import { applyUrlToFlags, readUrlState } from "@/lib/url-state";
import { AbstractPanel } from "./AbstractPanel";
import { ComparePanel } from "./ComparePanel";
import { DemoBackground } from "./DemoBackground";
import { EvalShowcasePicker } from "./EvalShowcasePicker";
import { FeatureFlagsPanel } from "./FeatureFlagsPanel";
import { GoldAbstractReveal } from "./GoldAbstractReveal";
import { GuidedTourBar } from "./GuidedTourBar";
import { LiveFactScorePanel } from "./LiveFactScorePanel";
import { OutlineGuardrail } from "./OutlineGuardrail";
import { PassagesPanel } from "./PassagesPanel";
import { PipelineStrip } from "./PipelineStrip";
import { QrCodePanel } from "./QrCodePanel";
import { QuickSystemSwap } from "./QuickSystemSwap";
import { ReadinessDots } from "./ReadinessDots";
import { RerankViewToggle } from "./RerankViewToggle";
import { SnapshotPanel } from "./SnapshotPanel";
import { StatusIndicator } from "./StatusIndicator";
import { SystemSelector } from "./SystemSelector";
import { TopKSlider } from "./TopKSlider";

const COMPARE_B_DEFAULT = "full";

function findExampleById(id: string) {
  return DEMO_EXAMPLES.find((e) => e.id === id);
}

function animateStages(
  stages: PipelineStage[],
  onTick: (index: number) => void,
  intervalMs = 2200,
): () => void {
  if (stages.length === 0) {
    onTick(0);
    return () => {};
  }
  let idx = 0;
  onTick(idx);
  const timer = setInterval(() => {
    idx = Math.min(idx + 1, stages.length - 1);
    onTick(idx);
  }, intervalMs);
  return () => clearInterval(timer);
}

export function GenerateForm() {
  const [flags, setFlags] = useState<DemoFeatureFlags>(DEFAULT_FEATURE_FLAGS);
  const [flagsOpen, setFlagsOpen] = useState(false);

  const [systems, setSystems] = useState<SystemInfo[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [system, setSystem] = useState("full_minus_sft");
  const [compareB, setCompareB] = useState(COMPARE_B_DEFAULT);
  const [title, setTitle] = useState("");
  const [outline, setOutline] = useState("");
  const [topK, setTopK] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [compareA, setCompareA] = useState<GenerateResponse | null>(null);
  const [compareBResult, setCompareBResult] = useState<GenerateResponse | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);
  const [exampleIndex, setExampleIndex] = useState(0);

  const [evalPrompts, setEvalPrompts] = useState<EvalShowcasePrompt[]>([]);
  const [evalPick, setEvalPick] = useState("");
  const [goldAbstract, setGoldAbstract] = useState("");
  const [goldOpen, setGoldOpen] = useState(false);

  const [guidedAct, setGuidedAct] = useState(0);
  const [guidedRunning, setGuidedRunning] = useState(false);

  const [rerankMode, setRerankMode] = useState<"post" | "pre">("post");
  const [selectedPassage, setSelectedPassage] = useState<number | null>(null);

  const [snapshots, setSnapshots] = useState<DemoSnapshot[]>([]);
  const [verifyResult, setVerifyResult] = useState<VerifyResponse | null>(null);
  const [verifyLoading, setVerifyLoading] = useState(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);

  const [loadingStages, setLoadingStages] = useState<PipelineStage[]>([]);
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const url = readUrlState();
    const stored = loadFeatureFlags();
    const merged = applyUrlToFlags(stored, url);
    setFlags(merged);
    if (url.system) setSystem(url.system);
    if (url.compareB) setCompareB(url.compareB);
    if (url.compareA) setSystem(url.compareA);
    if (url.example) {
      const ex = findExampleById(url.example);
      if (ex) {
        setTitle(ex.title);
        setOutline(ex.outline);
      }
    } else if (merged.presentationMode) {
      const ex = findExampleById("nils-jens-demo");
      if (ex) {
        setTitle(ex.title);
        setOutline(ex.outline);
      }
    }
    setSnapshots(loadSnapshots());
    fetchEvalShowcase().then(setEvalPrompts);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [sys, h] = await Promise.all([fetchSystems(), fetchHealth()]);
        if (cancelled) return;
        setHealth(h);
        if (sys.length > 0) {
          setSystems(sys);
          if (sys.some((s) => s.id === "full_minus_sft")) {
            setSystem((prev) => prev || "full_minus_sft");
          } else if (sys[0]) {
            setSystem(sys[0].id);
          }
        }
        if (h.status === "loading" || h.stack_ready === false) {
          setBootError(null);
        } else if (sys.length === 0) {
          setBootError("No systems returned from API.");
        } else {
          setBootError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setBootError(
            err instanceof Error
              ? err.message
              : "Could not reach the NILS-JENS API. Start the backend first.",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const booting =
      health?.status === "loading" ||
      health?.stack_ready === false ||
      systems.length === 0 ||
      bootError !== null;
    if (!booting) return;

    const timer = setInterval(async () => {
      try {
        const h = await fetchHealth();
        setHealth((prev) => (healthChanged(prev, h) ? h : prev));
        if (h.stack_ready && h.status === "ok") {
          const sys = await fetchSystems();
          if (sys.length > 0) {
            setSystems((prev) =>
              prev.length === sys.length &&
              prev.every((s, i) => s.id === sys[i]?.id)
                ? prev
                : sys,
            );
            setBootError(null);
          }
        } else if (h.status === "loading") {
          setBootError(null);
        }
      } catch {
        /* keep polling */
      }
    }, 5000);
    return () => clearInterval(timer);
  }, [health?.status, health?.stack_ready, systems.length, bootError]);

  const runGenerate = useCallback(
    async (systemId: string): Promise<GenerateResponse> => {
      return generateAbstract({
        title: title.trim(),
        outline: outline.trim(),
        system: systemId,
        top_k: topK,
      });
    },
    [title, outline, topK],
  );

  const startLoadingAnimation = useCallback((stages: PipelineStage[]) => {
    const fallback: PipelineStage[] =
      stages.length > 0
        ? stages
        : [
            { id: "vector", label: "BGE + FAISS", detail: "" },
            { id: "rerank", label: "RankRAG", detail: "" },
            { id: "generate", label: "8B generate", detail: "" },
          ];
    setLoadingStages(fallback);
    setStageIndex(0);
    return animateStages(fallback, setStageIndex);
  }, []);

  async function onGenerate(targetSystem?: string) {
    const sysA = targetSystem ?? system;
    setLoading(true);
    setError(null);
    setVerifyResult(null);
    setVerifyError(null);
    setSelectedPassage(null);
    const stopAnim = startLoadingAnimation([]);

    try {
      if (flags.abCompare) {
        setCompareA(null);
        setCompareBResult(null);
        const a = await runGenerate(sysA);
        setCompareA(a);
        const b = await runGenerate(compareB);
        setCompareBResult(b);
        setResult(a);
      } else {
        setCompareA(null);
        setCompareBResult(null);
        const res = await runGenerate(sysA);
        setResult(res);
        if (res.stages?.length) {
          setLoadingStages(res.stages);
          setStageIndex(res.stages.length - 1);
        }
      }
    } catch (err) {
      setResult(null);
      setCompareA(null);
      setCompareBResult(null);
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      stopAnim();
      setLoading(false);
      setGuidedRunning(false);
    }
  }

  function loadExample() {
    const ex = DEMO_EXAMPLES[exampleIndex];
    setTitle(ex.title);
    setOutline(ex.outline);
    setGoldAbstract("");
    setEvalPick("");
    setExampleIndex((exampleIndex + 1) % DEMO_EXAMPLES.length);
  }

  function onEvalPick(prompt: EvalShowcasePrompt) {
    setEvalPick(prompt.id);
    setTitle(prompt.title);
    setOutline(prompt.outline);
    setGoldAbstract(prompt.gold_abstract);
    setGoldOpen(false);
  }

  async function runGuidedAct(index: number) {
    const act = GUIDED_ACTS[index];
    if (!act) return;
    const ex = findExampleById(act.exampleId);
    if (ex) {
      setTitle(ex.title);
      setOutline(ex.outline);
    }
    setSystem(act.system);
    setGuidedRunning(true);
    setGuidedAct(index);
    await onGenerate(act.system);
  }

  function onGuidedStart() {
    void runGuidedAct(0);
  }

  function onGuidedNext() {
    const next = Math.min(guidedAct + 1, GUIDED_ACTS.length - 1);
    if (next === guidedAct) return;
    void runGuidedAct(next);
  }

  function onGuidedReset() {
    setGuidedAct(0);
    setGuidedRunning(false);
  }

  async function onVerify() {
    if (!result?.abstract) return;
    setVerifyLoading(true);
    setVerifyError(null);
    try {
      const passages = result.passages.map((p) => p.text);
      const res = await verifyAbstract({
        abstract: result.abstract,
        passages,
      });
      setVerifyResult(res);
    } catch (err) {
      setVerifyError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setVerifyLoading(false);
    }
  }

  function onSaveSnapshot() {
    if (!result) return;
    setSnapshots(
      saveSnapshot({
        title: title.trim(),
        outline: outline.trim(),
        system: result.system,
        topK,
        result,
        label: result.system,
      }),
    );
  }

  function onLoadSnapshot(snap: DemoSnapshot) {
    setTitle(snap.title);
    setOutline(snap.outline);
    setSystem(snap.system);
    setTopK(snap.topK);
    setResult(snap.result);
    setCompareA(null);
    setCompareBResult(null);
    setGoldAbstract("");
    setEvalPick("");
  }

  const nextExample = DEMO_EXAMPLES[exampleIndex];
  const titleLen = title.trim().length;
  const outlineLen = outline.trim().length;
  const presentation = flags.presentationMode;

  const pipelineStep = loading
    ? "generate"
    : result
      ? "generate"
      : title.trim() && outline.trim()
        ? "retrieve"
        : "compose";

  const apiReady =
    health?.stack_ready !== false &&
    (health?.status === "ok" || !health?.status);

  const displayPassages = useMemo(() => {
    if (!result) return [];
    if (
      flags.rerankToggle &&
      rerankMode === "pre" &&
      result.passages_pre_rerank &&
      result.passages_pre_rerank.length > 0
    ) {
      return result.passages_pre_rerank;
    }
    return result.passages;
  }, [result, flags.rerankToggle, rerankMode]);

  const highlightTokens = useMemo(() => {
    if (!flags.passageLinking || selectedPassage == null || !result?.abstract) {
      return undefined;
    }
    const passage = displayPassages[selectedPassage];
    if (!passage) return undefined;
    return passageOverlapTokens(result.abstract, passage.text);
  }, [
    flags.passageLinking,
    selectedPassage,
    result?.abstract,
    displayPassages,
  ]);

  const outlineQuality = flags.outlineGuardrails
    ? assessOutlineQuality(outline)
    : null;

  const abstractContent = useMemo(() => {
    if (loading && !flags.abCompare) return null;
    if (health?.stack_ready === false) return null;
    return result?.abstract || null;
  }, [loading, flags.abCompare, health?.stack_ready, result?.abstract]);

  const fastPath = system === "zero_shot" || system === "zero_shot_with_sft";
  const showBootBanner =
    health?.status === "loading" || health?.stack_ready === false;

  const systemLabel =
    systems.find((s) => s.id === system)?.label ?? system;
  const compareBLabel =
    systems.find((s) => s.id === compareB)?.label ?? compareB;

  return (
    <>
      <DemoBackground />
      <div
        className={`flex h-dvh max-h-dvh flex-col overflow-hidden px-3 py-2 sm:px-5 ${
          presentation ? "text-lg" : ""
        }`}
      >
        <header className="mb-2 flex shrink-0 items-center justify-between gap-3 border-b border-slide-border pb-2">
          <div className="min-w-0">
            <h1
              className={`truncate font-display font-semibold tracking-tight text-slide-ink ${
                presentation ? "text-3xl sm:text-4xl" : "text-2xl sm:text-3xl"
              }`}
            >
              NILS-JENS
            </h1>
          </div>
          {!presentation ? <PipelineStrip step={pipelineStep} /> : null}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setFlagsOpen(true)}
              className="btn-secondary px-3 py-2 text-xs"
              title="Toggle demo features"
            >
              Features
            </button>
            <StatusIndicator
              health={health}
              bootError={bootError}
              apiReady={apiReady}
            />
          </div>
        </header>

        {flags.qrCode ? (
          <QrCodePanel
            compareA={flags.abCompare ? system : undefined}
            compareB={flags.abCompare ? compareB : undefined}
            example="nils-jens-demo"
            present={presentation}
          />
        ) : null}

        {flags.snapshots ? (
          <SnapshotPanel
            snapshots={snapshots}
            onLoad={onLoadSnapshot}
            onDelete={(id) => setSnapshots(deleteSnapshot(id))}
            onSave={onSaveSnapshot}
            canSave={!!result && !loading}
          />
        ) : null}

        {flags.guidedTour ? (
          <GuidedTourBar
            actIndex={guidedAct}
            running={guidedRunning || loading}
            onStart={onGuidedStart}
            onNext={onGuidedNext}
            onReset={onGuidedReset}
          />
        ) : null}

        {error ? (
          <div className="mb-2 shrink-0 rounded-lg border border-red-400/30 bg-red-400/10 px-4 py-2 text-sm text-red-300">
            {error}
          </div>
        ) : null}

        {showBootBanner && !error ? (
          <div className="mb-2 h-1.5 shrink-0 overflow-hidden rounded-full bg-slide-elevated">
            <div className="h-full w-1/3 animate-pulse rounded-full bg-accent/70" />
          </div>
        ) : null}

        {presentation ? (
          <div className="mb-2 flex shrink-0 flex-wrap items-center gap-2 rounded-lg border border-slide-border bg-slide-elevated px-3 py-2">
            <button
              type="button"
              onClick={loadExample}
              className="btn-secondary px-3 py-2 text-xs"
            >
              ↻ {nextExample.label}
            </button>
            <button
              type="button"
              onClick={() => void onGenerate()}
              disabled={
                loading ||
                !!bootError ||
                health?.stack_ready === false ||
                titleLen < 3 ||
                outlineLen < 10
              }
              className="btn-primary px-4 py-2 text-sm"
            >
              {loading ? "Running…" : "Run champion →"}
            </button>
            <span className="truncate font-mono text-xs text-slide-muted">
              {title || "Load an example to begin"}
            </span>
          </div>
        ) : null}

        <main
          className={`grid min-h-0 flex-1 gap-3 overflow-hidden ${
            flags.abCompare ? "grid-cols-1" : "grid-cols-12"
          }`}
        >
          {!presentation ? (
            <section className="slide-panel-elevated col-span-4 grid min-h-0 grid-rows-[auto_auto_minmax(11rem,1fr)_auto] gap-2 overflow-hidden p-3 sm:p-4">
              <SystemSelector
                systems={systems}
                value={system}
                onChange={setSystem}
                disabled={loading || !!bootError}
              />

              {flags.abCompare ? (
                <div className="space-y-1">
                  <label className="section-kicker">Compare B</label>
                  <select
                    value={compareB}
                    disabled={loading}
                    onChange={(e) => setCompareB(e.target.value)}
                    className="input-field py-2 text-sm"
                  >
                    {systems.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}

              {flags.evalShowcase ? (
                <EvalShowcasePicker
                  prompts={evalPrompts}
                  value={evalPick}
                  onChange={onEvalPick}
                  disabled={loading}
                />
              ) : null}

              <div className="space-y-2">
                <label className="section-kicker">Title</label>
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Paper title"
                  disabled={loading}
                  className="input-field"
                />
              </div>

              <div className="flex min-h-0 flex-col gap-2 overflow-hidden">
                <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
                  <label className="section-kicker">Outline</label>
                  <div className="flex flex-wrap items-center gap-2">
                    <ReadinessDots
                      titleOk={titleLen >= 3}
                      outlineOk={outlineLen >= 10}
                    />
                    <button
                      type="button"
                      onClick={loadExample}
                      title={`Loads: ${nextExample.label}`}
                      className="btn-secondary shrink-0 px-3 py-2 text-xs"
                    >
                      ↻ Example
                    </button>
                  </div>
                </div>
                {outlineQuality ? (
                  <OutlineGuardrail quality={outlineQuality} />
                ) : null}
                <textarea
                  value={outline}
                  onChange={(e) => setOutline(e.target.value)}
                  placeholder="Contributions, method, results…"
                  disabled={loading}
                  className="scroll-panel input-field min-h-0 flex-1 resize-none py-3 font-mono text-sm leading-relaxed"
                />
              </div>

              <div className="flex shrink-0 items-center gap-3 pt-1">
                {!presentation ? (
                  <TopKSlider
                    value={topK}
                    onChange={setTopK}
                    disabled={loading}
                    compact
                  />
                ) : null}
                <button
                  type="button"
                  onClick={() => void onGenerate()}
                  disabled={
                    loading ||
                    !!bootError ||
                    health?.stack_ready === false ||
                    titleLen < 3 ||
                    outlineLen < 10
                  }
                  className="btn-primary min-w-[7rem] shrink-0"
                >
                  {loading
                    ? "…"
                    : flags.abCompare
                      ? "Compare →"
                      : "Run →"}
                </button>
              </div>
            </section>
          ) : null}

          {flags.abCompare ? (
            <ComparePanel
              resultA={compareA}
              resultB={compareBResult}
              labelA={systemLabel}
              labelB={compareBLabel}
              loading={loading}
            />
          ) : (
            <AbstractPanel
              content={abstractContent}
              passages={displayPassages}
              loading={loading || health?.stack_ready === false}
              system={result?.system ?? system}
              mockGeneration={result?.mock_generation ?? health?.mock_mode}
              fastPath={fastPath}
              stages={loading ? loadingStages : result?.stages}
              stageIndex={
                loading
                  ? stageIndex
                  : (result?.stages?.length ?? loadingStages.length) - 1
              }
              highlightTokens={highlightTokens}
              presentation={presentation}
            />
          )}

          {!flags.abCompare ? (
            <aside
              className={`slide-panel-elevated flex min-h-0 flex-col overflow-hidden p-4 ${
                presentation ? "col-span-5" : "col-span-3"
              }`}
            >
              <div className="mb-2 flex shrink-0 items-center justify-between">
                <h2 className="section-kicker">Retrieval</h2>
                {displayPassages.length > 0 ? (
                  <span className="flex h-8 w-8 items-center justify-center rounded-full border border-accent/40 bg-accent/10 font-mono text-sm font-semibold text-accent">
                    {displayPassages.length}
                  </span>
                ) : null}
              </div>

              {flags.rerankToggle ? (
                <RerankViewToggle
                  mode={rerankMode}
                  hasPre={(result?.passages_pre_rerank?.length ?? 0) > 0}
                  onChange={setRerankMode}
                />
              ) : null}

              <div className="scroll-panel min-h-0 flex-1">
                <PassagesPanel
                  passages={displayPassages}
                  loading={loading && !result}
                  selectedIndex={flags.passageLinking ? selectedPassage : null}
                  onSelectPassage={
                    flags.passageLinking ? setSelectedPassage : undefined
                  }
                />
              </div>

              {flags.goldReveal && goldAbstract ? (
                <GoldAbstractReveal
                  goldAbstract={goldAbstract}
                  open={goldOpen}
                  onToggle={() => setGoldOpen((o) => !o)}
                />
              ) : null}

              {flags.liveFactScore ? (
                <LiveFactScorePanel
                  result={verifyResult}
                  loading={verifyLoading}
                  error={verifyError}
                  onScore={() => void onVerify()}
                  disabled={!result?.abstract || loading}
                />
              ) : null}

              {flags.quickSystemSwap && result ? (
                <QuickSystemSwap
                  current={result.system}
                  disabled={loading}
                  onSwap={(id) => {
                    setSystem(id);
                    void onGenerate(id);
                  }}
                />
              ) : null}
            </aside>
          ) : null}
        </main>
      </div>

      <FeatureFlagsPanel
        flags={flags}
        onChange={setFlags}
        open={flagsOpen}
        onClose={() => setFlagsOpen(false)}
      />
    </>
  );
}
