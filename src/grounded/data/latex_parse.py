"""Ingress 2 of 2: arXiv S3 LaTeX source → unified `Paper` schema.

Pure functions only. Takes a paper directory containing many `.tex` files
(plus `.bib`, `.cls`, etc.) and emits a `Paper`. No external dependencies
beyond stdlib + pydantic — regex-based parser is the v1; swap in pylatexenc
later by replacing only this module.

Pipeline:
  1. `_pick_main_tex(paper_dir)` — score and choose the entry .tex
  2. `_flatten(main, paper_dir)` — recursively inline `\input{}` / `\include{}`
  3. `_sanitize(text)` — replace math/figures/tables with placeholders, convert
     `\cite{a,b}` to `{{cite:a}} {{cite:b}}`
  4. `_extract_structure(text)` — title, abstract, sections, citation keys
  5. `_parse_bibliography(paper_dir)` — gather `.bib` entries
"""

from __future__ import annotations

import re
from pathlib import Path

from grounded.data.schema import BibEntry, Paper, Section


# ----------------------------- helpers -----------------------------

_BRACE_OPEN = "{"
_BRACE_CLOSE = "}"


def _match_brace(text: str, open_pos: int) -> int:
    """Return index of the `}` that closes the `{` at `open_pos`.

    Naive balanced-brace scan; raises ValueError if unbalanced. Skips comment
    lines (`%`-to-end-of-line) which can hide stray braces in TeX source.
    """
    if text[open_pos] != _BRACE_OPEN:
        raise ValueError(f"_match_brace expected '{{' at {open_pos}")
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "%":
            nl = text.find("\n", i)
            i = nl + 1 if nl != -1 else n
            continue
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == _BRACE_OPEN:
            depth += 1
        elif ch == _BRACE_CLOSE:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces")


def _brace_arg(text: str, open_pos: int) -> tuple[str, int]:
    """Return (contents, end_pos) for a `{...}` group starting at `open_pos`."""
    close = _match_brace(text, open_pos)
    return text[open_pos + 1 : close], close + 1


_INCLUDE_CMD_RE = re.compile(r"\\(input|include|subfile)\s*\{")


def _flatten(main_path: Path, paper_dir: Path, _seen: set[Path] | None = None) -> str:
    """Recursively inline `\\input{}` / `\\include{}` / `\\subfile{}`.

    Resolves relative paths against `paper_dir`; `.tex` extension optional.
    Cycle-safe via `_seen`. Returns the flattened source as one string.
    """
    seen = _seen if _seen is not None else set()
    resolved = main_path.resolve()
    if resolved in seen:
        return ""
    seen.add(resolved)

    try:
        text = main_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return ""

    out: list[str] = []
    pos = 0
    for match in _INCLUDE_CMD_RE.finditer(text):
        out.append(text[pos : match.start()])
        try:
            arg, end = _brace_arg(text, match.end() - 1)
        except ValueError:
            out.append(text[match.start() : match.end()])
            pos = match.end()
            continue

        target = arg.strip()
        # Try the literal path; then the basename (S3 extraction flattens
        # subdirs, so `\input{sec/0_abstract}` lives at `paper_dir/0_abstract.tex`);
        # `.tex` extension is optional in LaTeX so try both.
        target_base = Path(target).name
        candidates = [
            paper_dir / target,
            paper_dir / f"{target}.tex",
            paper_dir / target_base,
            paper_dir / f"{target_base}.tex",
        ]
        included_path = next((p for p in candidates if p.is_file()), None)
        if included_path:
            out.append(_flatten(included_path, paper_dir, seen))
        pos = end

    out.append(text[pos:])
    return "".join(out)


# ----------------------------- main-tex pick -----------------------------


_MAIN_PRIORITY_NAMES = {
    "main.tex": 10,
    "paper.tex": 9,
    "ms.tex": 7,
    "manuscript.tex": 7,
    "article.tex": 6,
    "neurips.tex": 5,
    "icml.tex": 5,
    "acl.tex": 5,
    "emnlp.tex": 5,
}


def _score_tex(path: Path) -> int:
    """Heuristic score: higher = more likely the entry-point file."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:8000]
    except OSError:
        return -1
    score = 0
    if "\\documentclass" in head:
        score += 8
    if "\\begin{document}" in head:
        score += 6
    if re.search(r"\\title\s*\{", head):
        score += 3
    if re.search(r"\\begin\{abstract\}", head):
        score += 3
    if re.search(r"\\section\s*\{", head):
        score += 2
    if path.stat().st_size > 5000:
        score += 2
    score += _MAIN_PRIORITY_NAMES.get(path.name.lower(), 0)
    lname = path.name.lower()
    if "appendix" in lname or "supplement" in lname or "supp" in lname:
        score -= 5
    if "_arxiv" in lname or "arxiv_" in lname:
        score += 1
    return score


def _pick_main_tex(paper_dir: Path) -> Path | None:
    tex_files = [p for p in paper_dir.iterdir() if p.is_file() and p.suffix == ".tex"]
    if not tex_files:
        return None
    scored = sorted(((p, _score_tex(p)) for p in tex_files), key=lambda x: x[1], reverse=True)
    best, best_score = scored[0]
    return best if best_score >= 8 else None  # require at least \documentclass


# ----------------------------- sanitize -----------------------------

_COMMENT_RE = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


_ENV_REPLACE = [
    ("equation", "[EQUATION]"),
    ("equation*", "[EQUATION]"),
    ("align", "[EQUATION]"),
    ("align*", "[EQUATION]"),
    ("eqnarray", "[EQUATION]"),
    ("gather", "[EQUATION]"),
    ("multline", "[EQUATION]"),
    ("figure", "[FIGURE]"),
    ("figure*", "[FIGURE]"),
    ("table", "[TABLE]"),
    ("table*", "[TABLE]"),
    ("algorithm", "[ALGORITHM]"),
    ("lstlisting", "[CODE]"),
    ("verbatim", "[CODE]"),
]


def _replace_environments(text: str) -> str:
    """Replace `\\begin{env}...\\end{env}` blocks with placeholders.

    Non-greedy, single-pass per env name; nested envs of the same name are not
    handled (rare in NLP papers; corner case for later).
    """
    for env, token in _ENV_REPLACE:
        pattern = re.compile(
            r"\\begin\s*\{" + re.escape(env) + r"\}.+?\\end\s*\{" + re.escape(env) + r"\}",
            re.DOTALL,
        )
        text = pattern.sub(token, text)
    return text


_DISPLAY_MATH_RE = re.compile(r"\\\[[^\\]*?(?:\\(?!\])[^\\]*?)*?\\\]", re.DOTALL)
# Inline math: same line only, no escaped openers/closers, no `$$` adjacency.
# This is intentionally conservative — one unbalanced `$` should NOT eat a
# whole document (the original bug that nuked an entire abstract).
_INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\$)[^\n$]+?(?<!\\)\$")
_CITE_RE = re.compile(r"\\(?:cite|citep|citet|citealp|citealt|citeyear|citeauthor)\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")


def _replace_math(text: str) -> str:
    text = _DISPLAY_MATH_RE.sub("[EQUATION]", text)
    text = _INLINE_MATH_RE.sub("[MATH]", text)
    return text


def _convert_cites(text: str) -> tuple[str, list[str]]:
    """`\\cite{a,b}` → `{{cite:a}} {{cite:b}}`. Returns (text, keys_seen)."""
    keys_seen: list[str] = []

    def repl(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        keys_seen.extend(keys)
        return " ".join(f"{{{{cite:{k}}}}}" for k in keys)

    return _CITE_RE.sub(repl, text), keys_seen


def _sanitize(text: str) -> tuple[str, list[str]]:
    text = _strip_comments(text)
    text = _replace_environments(text)
    text = _replace_math(text)
    text, keys = _convert_cites(text)
    return text, keys


# ----------------------------- extract structure -----------------------------


def _extract_title(text: str) -> str:
    match = re.search(r"\\title\s*(?:\[[^\]]*\])?\s*\{", text)
    if not match:
        return ""
    try:
        title, _ = _brace_arg(text, match.end() - 1)
    except ValueError:
        return ""
    return _clean_inline_latex(title).strip()


def _extract_abstract(text: str) -> str:
    match = re.search(r"\\begin\s*\{abstract\}(.+?)\\end\s*\{abstract\}", text, re.DOTALL)
    if match:
        return _clean_inline_latex(match.group(1)).strip()
    match = re.search(r"\\abstract\s*\{", text)
    if match:
        try:
            content, _ = _brace_arg(text, match.end() - 1)
            return _clean_inline_latex(content).strip()
        except ValueError:
            pass
    return ""


_SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\s*\*?\s*\{")


def _extract_sections(text: str) -> list[Section]:
    """Walk `\\section{}` / `\\subsection{}` markers and slice the body."""
    matches: list[tuple[int, int, int, str]] = []  # (start_pos, after_arg_pos, level, heading)
    for match in _SECTION_RE.finditer(text):
        level = {"section": 1, "subsection": 2, "subsubsection": 3}[match.group(1)]
        try:
            heading, end = _brace_arg(text, match.end() - 1)
        except ValueError:
            continue
        matches.append((match.start(), end, level, heading.strip()))

    sections: list[Section] = []
    for i, (_, body_start, level, heading) in enumerate(matches):
        body_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        sections.append(
            Section(heading=_clean_inline_latex(heading), level=level, paragraphs=paragraphs)
        )
    return sections


_TEX_TEXT_CMDS_RE = re.compile(r"\\(?:emph|textbf|textit|textsf|texttt|underline)\s*\{([^{}]*)\}")
_TEX_BRACE_STRIP_RE = re.compile(r"\\[a-zA-Z@]+\s*\*?\s*(?:\[[^\]]*\])?")
_WHITESPACE_RE = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"\[(?:EQUATION|MATH|FIGURE|TABLE|ALGORITHM|CODE)\]")


def _clean_inline_latex(text: str) -> str:
    """Strip leftover inline commands so chunks are clean for embedding."""
    prev = None
    while prev != text:
        prev = text
        text = _TEX_TEXT_CMDS_RE.sub(lambda m: m.group(1), text)
    text = _PLACEHOLDER_RE.sub(" ", text)
    text = _TEX_BRACE_STRIP_RE.sub(" ", text)
    text = text.replace("~", " ").replace("\\\\", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text


# ----------------------------- bibliography -----------------------------

_BIB_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
_BIB_FIELD_RE = re.compile(r"(\w+)\s*=\s*[\{\"]([^}\"]*)[\}\"]")


def _parse_bib_file(bib_path: Path) -> dict[str, BibEntry]:
    """Lightweight `.bib` parser: extract keys, titles, years per entry."""
    try:
        text = bib_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    entries: dict[str, BibEntry] = {}
    starts = [(m.start(), m.group(2)) for m in _BIB_ENTRY_RE.finditer(text)]
    for i, (start, key) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[start:end]
        fields = {m.group(1).lower(): m.group(2).strip() for m in _BIB_FIELD_RE.finditer(body)}
        year_str = fields.get("year")
        try:
            year = int(year_str) if year_str and year_str.isdigit() else None
        except (TypeError, ValueError):
            year = None
        entries[key] = BibEntry(
            citation_key=key,
            raw_entry=body[:1000],
            title=fields.get("title"),
            year=year,
        )
    return entries


def _gather_bibliography(paper_dir: Path) -> dict[str, BibEntry]:
    out: dict[str, BibEntry] = {}
    for bib in paper_dir.glob("*.bib"):
        out.update(_parse_bib_file(bib))
    return out


# ----------------------------- main entrypoint -----------------------------


_NEW_STYLE_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d+")


def _year_from_arxiv_id(arxiv_id: str) -> int | None:
    match = _NEW_STYLE_ID_RE.match(arxiv_id)
    if not match:
        return None
    yy = int(match.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def _failed(arxiv_id: str, reason: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        source="latex_s3",
        parse_status="failed",
        title="",
        abstract="",
        sections=[],
        body_text="",
        year=_year_from_arxiv_id(arxiv_id),
        notes=[reason],
    )


def parse_latex_dir(paper_dir: Path) -> Paper:
    """Top-level: paper directory → `Paper`. Catches structural failures."""
    arxiv_id = paper_dir.name

    main = _pick_main_tex(paper_dir)
    if main is None:
        return _failed(arxiv_id, "no main .tex with \\documentclass found")

    raw = _flatten(main, paper_dir)
    if not raw:
        return _failed(arxiv_id, f"could not read or flatten {main.name}")

    sanitized, citation_keys = _sanitize(raw)

    title = _extract_title(sanitized)
    abstract = _extract_abstract(sanitized)
    sections = _extract_sections(sanitized)
    bibliography = _gather_bibliography(paper_dir)

    body_text = "\n\n".join(p for s in sections for p in s.paragraphs)
    if not body_text and sanitized:
        body_text = _clean_inline_latex(sanitized)

    notes: list[str] = []
    if not title:
        notes.append("missing title")
    if not abstract:
        notes.append("missing abstract")
    if not sections:
        notes.append("no \\section{} markers found")
    if not bibliography:
        notes.append("no .bib file or unparseable")
    if not citation_keys:
        notes.append("no \\cite{} markers in body")

    if not title or not abstract or not body_text:
        status = "failed" if not body_text else "partial"
    elif notes:
        status = "partial"
    else:
        status = "ok"

    return Paper(
        arxiv_id=arxiv_id,
        source="latex_s3",
        parse_status=status,
        title=title,
        abstract=abstract,
        sections=sections,
        body_text=body_text,
        bibliography=bibliography,
        citation_keys_in_body=sorted(set(citation_keys)),
        year=_year_from_arxiv_id(arxiv_id),
        notes=notes,
    )
