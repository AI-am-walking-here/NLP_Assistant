export type EvalShowcasePrompt = {
  id: string;
  label: string;
  title: string;
  outline: string;
  gold_abstract: string;
  year?: number;
  source?: string;
};

export type EvalShowcaseFile = {
  prompts: EvalShowcasePrompt[];
};

export async function fetchEvalShowcase(): Promise<EvalShowcasePrompt[]> {
  const res = await fetch("/eval-showcase.json", { cache: "force-cache" });
  if (!res.ok) return [];
  const data = (await res.json()) as EvalShowcaseFile;
  return data.prompts ?? [];
}
