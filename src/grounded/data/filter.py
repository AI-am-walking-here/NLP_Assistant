from __future__ import annotations

import io
import json
import random
import re
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from grounded.data.s3_pull import append_ledger, load_ledger, tarball_local_path
from grounded.progress import CountProgressReporter, StatusReporter

# --- arXiv ID helpers ---

NEW_STYLE_ID = re.compile(r"^(\d{4})\.(\d{1,5})$")


@dataclass(frozen=True, order=True)
class ArxivId:
    yymm: str
    number: int
    raw: str


def parse_arxiv_id(arxiv_id: str) -> ArxivId | None:
    match = NEW_STYLE_ID.match(arxiv_id.strip())
    if not match:
        return None
    return ArxivId(yymm=match.group(1), number=int(match.group(2)), raw=arxiv_id)


def normalize_arxiv_id(value: str) -> str | None:
    normalized = value.strip()
    if normalized.startswith("arXiv:"):
        normalized = normalized[len("arXiv:") :]
    if "v" in normalized:
        base, version = normalized.rsplit("v", 1)
        if version.isdigit():
            normalized = base
    parsed = parse_arxiv_id(normalized)
    return parsed.raw if parsed else None


def _arxiv_id_from_basename(base: str) -> str | None:
    if base.endswith(".tar.gz"):
        base = base[: -len(".tar.gz")]
    elif base.endswith(".tar"):
        base = base[: -len(".tar")]
    elif base.endswith(".gz") and "." in base:
        base = base[: base.rfind(".")]
    parsed = parse_arxiv_id(base)
    return parsed.raw if parsed else None


def arxiv_id_from_member(name: str) -> str | None:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized:
        return None
    for part in reversed(normalized.split("/")):
        arxiv_id = _arxiv_id_from_basename(part)
        if arxiv_id:
            return arxiv_id
    return None


def yymm_from_id(arxiv_id: str) -> str | None:
    parsed = parse_arxiv_id(arxiv_id)
    return parsed.yymm if parsed else None


def in_id_range(arxiv_id: str, first_item: str, last_item: str) -> bool:
    target = parse_arxiv_id(arxiv_id)
    first = parse_arxiv_id(first_item)
    last = parse_arxiv_id(last_item)
    if not target or not first or not last:
        return False
    if target.yymm != first.yymm and target.yymm != last.yymm:
        return False
    low = first if first.yymm == target.yymm else ArxivId(target.yymm, 0, f"{target.yymm}.0")
    high = last if last.yymm == target.yymm else ArxivId(target.yymm, 99999, f"{target.yymm}.99999")
    return low <= target <= high


# --- M-1.1 metadata filter ---


def _parse_year(created: str) -> int:
    return datetime.strptime(created, "%a, %d %b %Y %H:%M:%S %Z").year


def iter_kaggle_records(metadata_path: Path) -> Iterator[dict]:
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def filter_cs_cl_records(
    metadata_path: Path,
    category: str,
    category_match: str,
    year_min: int,
    year_max: int,
    show_progress: bool = True,
) -> list[dict]:
    records: list[dict] = []
    reporter = CountProgressReporter("filter-metadata", unit="records") if show_progress else None
    scanned = 0
    for rec in iter_kaggle_records(metadata_path):
        scanned += 1
        categories = rec.get("categories", "")
        category_parts = categories.split()
        if category_match == "primary":
            matches_category = bool(category_parts) and category_parts[0] == category
        elif category_match == "any":
            matches_category = category in category_parts
        else:
            raise ValueError("category_match must be 'primary' or 'any'")
        if not matches_category:
            if reporter and scanned % 50000 == 0:
                reporter.done_count = scanned
                reporter.update(0, detail=f"kept {len(records)}")
            continue
        versions = rec.get("versions") or []
        if not versions:
            if reporter and scanned % 50000 == 0:
                reporter.done_count = scanned
                reporter.update(0, detail=f"kept {len(records)}")
            continue
        year = _parse_year(versions[0]["created"])
        if year_min <= year <= year_max:
            records.append(
                {
                    "id": rec["id"],
                    "year": year,
                    "categories": categories,
                    "title": rec.get("title", ""),
                }
            )
        if reporter and scanned % 50000 == 0:
            reporter.done_count = scanned
            reporter.update(0, detail=f"kept {len(records)}")
    if reporter:
        reporter.done_count = scanned
        reporter.finish(detail=f"kept {len(records)} matching records")
    return records


def write_cs_cl_ids(output_path: Path, records: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def load_cs_cl_id_set(cs_cl_ids_path: Path) -> set[str]:
    with cs_cl_ids_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    return {rec["id"] for rec in records}


def load_cs_cl_records(cs_cl_ids_path: Path) -> list[dict]:
    with cs_cl_ids_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def download_kaggle_metadata(dataset: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_path.parent), "--unzip"],
        check=True,
    )
    if not output_path.exists():
        candidates = list(output_path.parent.glob("arxiv-metadata*.json"))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(f"Expected metadata at {output_path}")
    return output_path


def select_records_for_year_range(records: list[dict], year_min: int, year_max: int) -> list[dict]:
    return [rec for rec in records if year_min <= int(rec["year"]) <= year_max]


def compute_source_target(overall_target: int | None, source_fraction: float) -> int | None:
    """Step 2: this source's paper count = overall_target × source_fraction."""
    if overall_target is None:
        return None
    return max(0, int(round(overall_target * source_fraction)))


def sample_records_for_source(
    records: list[dict],
    *,
    overall_target: int | None,
    source_fraction: float,
    random_seed: int,
) -> tuple[list[dict], int | None]:
    """Apply two-step allocation: overall target, then this source's fraction."""
    source_target = compute_source_target(overall_target, source_fraction)
    sampled = sample_records_by_fraction(
        records,
        source_fraction,
        random_seed,
        paper_count_target=source_target,
    )
    return sampled, source_target


def sample_records_by_fraction(
    records: list[dict],
    paper_fraction: float,
    random_seed: int,
    paper_count_target: int | None = None,
) -> list[dict]:
    if paper_count_target is not None:
        if paper_count_target <= 0:
            return []
        if paper_count_target >= len(records):
            return list(records)
    elif paper_fraction <= 0:
        return []
    elif paper_fraction >= 1:
        return list(records)
    rng = random.Random(random_seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    if paper_count_target is not None:
        keep = paper_count_target
    else:
        keep = max(1, int(round(len(shuffled) * paper_fraction)))
    sampled = shuffled[:keep]
    sampled_ids = {rec["id"] for rec in sampled}
    return [rec for rec in records if rec["id"] in sampled_ids]


def _get_by_dotted_path(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _flatten_text(item)))
    if isinstance(value, dict):
        preferred = []
        for key in ("text", "body_text", "content", "abstract", "title"):
            if key in value:
                preferred.append(_flatten_text(value[key]))
        if preferred:
            return "\n".join(part for part in preferred if part)
        return "\n".join(part for item in value.values() if (part := _flatten_text(item)))
    return str(value)


def _extract_first_text(payload: dict[str, Any], field_paths: list[str]) -> str:
    for field_path in field_paths:
        value = _get_by_dotted_path(payload, field_path)
        text = _flatten_text(value)
        if text:
            return text
    return ""


def _iter_unarxive_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(payload, dict):
            yield payload


def build_unarxive_manifest(
    root_dir: Path,
    shard_glob: str,
    allowed_ids: set[str],
    id_fields: list[str],
    show_progress: bool = True,
) -> dict:
    if not root_dir.exists():
        raise FileNotFoundError(
            f"unarXive root directory not found: {root_dir}. "
            "Download the unarXive dataset and place JSONL shards under this path."
        )

    shard_paths = [path for path in sorted(root_dir.glob(shard_glob)) if path.is_file()]
    if not shard_paths:
        raise FileNotFoundError(
            f"No unarXive shards found under {root_dir} matching {shard_glob!r}."
        )

    matched_ids: dict[str, str] = {}
    scanned_files = 0
    scanned_records = 0
    reporter = (
        CountProgressReporter("unarxive-manifest", total=len(shard_paths), unit="shards")
        if show_progress
        else None
    )

    for path in shard_paths:
        scanned_files += 1
        for record in _iter_unarxive_records(path):
            scanned_records += 1
            record_id = ""
            for field in id_fields:
                value = _get_by_dotted_path(record, field)
                if isinstance(value, str):
                    normalized = normalize_arxiv_id(value)
                    if normalized:
                        record_id = normalized
                        break
            if record_id and record_id in allowed_ids and record_id not in matched_ids:
                matched_ids[record_id] = str(path)
        if reporter:
            reporter.update(1, detail=f"matched {len(matched_ids)}")

    if reporter:
        reporter.finish(detail=f"{len(matched_ids)} IDs matched from {scanned_records} records")

    return {
        "matched_ids": matched_ids,
        "stats": {
            "num_allowed_ids": len(allowed_ids),
            "num_matched_ids": len(matched_ids),
            "num_unmatched_ids": len(allowed_ids) - len(matched_ids),
            "num_scanned_files": scanned_files,
            "num_scanned_records": scanned_records,
        },
    }


def write_unarxive_manifest(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def materialize_unarxive_records(
    manifest_path: Path,
    output_dir: Path,
    ledger_path: Path,
    id_fields: list[str],
    title_fields: list[str],
    abstract_fields: list[str],
    text_fields: list[str],
    delete_shards_after_materialize: bool = False,
    show_progress: bool = True,
) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    matched_ids: dict[str, str] = payload["matched_ids"]
    done = load_ledger(ledger_path)
    pending_ids = {paper_id for paper_id in matched_ids if paper_id not in done}
    files_to_ids: dict[str, list[str]] = {}
    for paper_id, source_path in matched_ids.items():
        if paper_id in pending_ids:
            files_to_ids.setdefault(source_path, []).append(paper_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = 0
    missing_text = 0
    reporter = (
        CountProgressReporter("unarxive-materialize", total=len(pending_ids), unit="papers")
        if show_progress and pending_ids
        else None
    )
    if show_progress and not pending_ids:
        StatusReporter("unarxive-materialize").done("Nothing to materialize; all papers already processed")

    for source_path, source_ids in sorted(files_to_ids.items()):
        wanted = set(source_ids)
        for record in _iter_unarxive_records(Path(source_path)):
            record_id = ""
            for field in id_fields:
                value = _get_by_dotted_path(record, field)
                if isinstance(value, str):
                    normalized = normalize_arxiv_id(value)
                    if normalized:
                        record_id = normalized
                        break
            if record_id not in wanted:
                continue
            paper_dir = output_dir / record_id
            paper_dir.mkdir(parents=True, exist_ok=True)
            title = _extract_first_text(record, title_fields)
            abstract = _extract_first_text(record, abstract_fields)
            body = _extract_first_text(record, text_fields)
            if not body:
                missing_text += 1
            normalized = {
                "id": record_id,
                "title": title,
                "abstract": abstract,
                "text": body,
                "source": "unarxive",
                "raw": record,
            }
            (paper_dir / "paper.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
            combined_text = "\n\n".join(part for part in [title, abstract, body] if part).strip()
            if combined_text:
                (paper_dir / "paper.txt").write_text(combined_text, encoding="utf-8")
            append_ledger(ledger_path, record_id)
            materialized += 1
            if reporter:
                reporter.update(1)
            wanted.remove(record_id)
            if not wanted:
                break

        if delete_shards_after_materialize and not wanted:
            shard_path = Path(source_path)
            if shard_path.is_file():
                shard_path.unlink()

    if reporter:
        reporter.finish(detail=f"{missing_text} records missing body text")

    existing = {p.name for p in output_dir.iterdir() if p.is_dir()}
    return {
        "materialized_this_run": materialized,
        "total_paper_dirs": len(existing),
        "missing_text_records": missing_text,
    }


# --- M-1.2 manifest cross-reference ---


@dataclass(frozen=True)
class ManifestEntry:
    filename: str
    first_item: str
    last_item: str
    yymm: str
    size_bytes: int


def parse_manifest_xml(manifest_path: Path) -> list[ManifestEntry]:
    tree = ET.parse(manifest_path)
    root = tree.getroot()
    entries: list[ManifestEntry] = []
    for file_elem in root.findall("file"):
        filename = file_elem.findtext("filename", "")
        if not filename:
            continue
        entries.append(
            ManifestEntry(
                filename=filename,
                first_item=file_elem.findtext("first_item", ""),
                last_item=file_elem.findtext("last_item", ""),
                yymm=file_elem.findtext("yymm", ""),
                size_bytes=int(file_elem.findtext("size", "0") or 0),
            )
        )
    return entries


def download_manifest(
    bucket: str,
    manifest_key: str,
    output_path: Path,
    region: str,
    request_payer: str,
    max_retries: int = 5,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part = output_path.with_name(f"{output_path.name}.part")
    client = boto3.client("s3", region_name=region)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if part.exists():
                part.unlink()
            client.download_file(
                bucket,
                manifest_key,
                str(part),
                ExtraArgs={"RequestPayer": request_payer},
            )
            part.replace(output_path)
            return output_path
        except (ClientError, BotoCoreError, OSError) as exc:
            last_error = exc
            if attempt < max_retries:
                delay = min(2**attempt, 30)
                StatusReporter("manifest").status(
                    f"Download interrupted (attempt {attempt}/{max_retries}); retrying in {delay}s",
                    force=True,
                )
                time.sleep(delay)
    raise RuntimeError(f"Failed to download s3://{bucket}/{manifest_key}: {last_error}") from last_error


def _tarball_cost_usd(size_bytes: int, egress_per_gb_usd: float, get_request_usd: float) -> float:
    return (size_bytes / (1024**3)) * egress_per_gb_usd + get_request_usd


def _sample_tarballs_within_budget(
    tarballs: list[str],
    tarball_sizes: dict[str, int],
    hard_cap_usd: float,
    egress_per_gb_usd: float,
    get_request_usd: float,
    random_seed: int,
) -> list[str]:
    rng = random.Random(random_seed)
    shuffled = list(tarballs)
    rng.shuffle(shuffled)

    selected: list[str] = []
    running_cost = 0.0
    for key in shuffled:
        cost = _tarball_cost_usd(tarball_sizes[key], egress_per_gb_usd, get_request_usd)
        if running_cost + cost <= hard_cap_usd:
            selected.append(key)
            running_cost += cost
    return sorted(selected)


def build_filtered_manifest(
    entries: list[ManifestEntry],
    cs_cl_ids: set[str],
    hard_cap_usd: float,
    egress_per_gb_usd: float,
    get_request_usd: float,
    random_seed: int,
) -> dict:
    by_yymm: dict[str, list[ManifestEntry]] = {}
    for entry in entries:
        by_yymm.setdefault(entry.yymm, []).append(entry)

    arxiv_id_to_tarball: dict[str, str] = {}
    needed_tarballs: set[str] = set()
    unmapped_ids: list[str] = []

    for arxiv_id in sorted(cs_cl_ids):
        yymm = yymm_from_id(arxiv_id)
        if not yymm:
            unmapped_ids.append(arxiv_id)
            continue
        matched = None
        for entry in by_yymm.get(yymm, []):
            if in_id_range(arxiv_id, entry.first_item, entry.last_item):
                matched = entry.filename
                break
        if matched:
            arxiv_id_to_tarball[arxiv_id] = matched
            needed_tarballs.add(matched)
        else:
            unmapped_ids.append(arxiv_id)

    tarball_sizes = {
        entry.filename: entry.size_bytes
        for entry in entries
        if entry.filename in needed_tarballs
    }
    candidate_tarballs = sorted(needed_tarballs)
    candidate_total_bytes = sum(tarball_sizes.values())
    candidate_estimated_gb = round(candidate_total_bytes / (1024**3), 3)
    candidate_estimated_cost_usd = round(
        sum(_tarball_cost_usd(size, egress_per_gb_usd, get_request_usd) for size in tarball_sizes.values()),
        2,
    )

    selected_tarballs = candidate_tarballs
    selection_mode = "all"
    if candidate_estimated_cost_usd > hard_cap_usd:
        selected_tarballs = _sample_tarballs_within_budget(
            candidate_tarballs,
            tarball_sizes,
            hard_cap_usd,
            egress_per_gb_usd,
            get_request_usd,
            random_seed,
        )
        selection_mode = "random_budget_sample"

    selected_set = set(selected_tarballs)
    selected_id_to_tarball = {
        arxiv_id: key
        for arxiv_id, key in arxiv_id_to_tarball.items()
        if key in selected_set
    }
    selected_sizes = {
        key: size
        for key, size in tarball_sizes.items()
        if key in selected_set
    }
    selected_total_bytes = sum(selected_sizes.values())
    selected_estimated_gb = round(selected_total_bytes / (1024**3), 3)
    selected_estimated_cost_usd = round(
        sum(_tarball_cost_usd(size, egress_per_gb_usd, get_request_usd) for size in selected_sizes.values()),
        2,
    )

    return {
        "tarballs": selected_tarballs,
        "arxiv_id_to_tarball": selected_id_to_tarball,
        "tarball_sizes_bytes": selected_sizes,
        "candidate_tarballs": candidate_tarballs,
        "stats": {
            "num_cs_cl_ids": len(cs_cl_ids),
            "num_mapped_ids": len(arxiv_id_to_tarball),
            "num_unmapped_ids": len(unmapped_ids),
            "num_candidate_tarballs": len(candidate_tarballs),
            "candidate_estimated_download_gb": candidate_estimated_gb,
            "candidate_estimated_cost_usd": candidate_estimated_cost_usd,
            "selection_mode": selection_mode,
            "random_seed": random_seed,
            "hard_cap_usd": hard_cap_usd,
            "num_selected_ids": len(selected_id_to_tarball),
            "num_tarballs": len(selected_tarballs),
            "estimated_download_gb": selected_estimated_gb,
            "estimated_cost_usd": selected_estimated_cost_usd,
        },
        "unmapped_ids": unmapped_ids[:100],
    }


def write_manifest_filtered(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest_filtered(manifest_path: Path) -> dict:
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


# --- M-1.4 tarball extraction ---


def _should_keep_member(name: str, skip_extensions: set[str]) -> bool:
    suffix = Path(name.replace("\\", "/")).suffix.lower()
    return not suffix or suffix not in skip_extensions


def _safe_extract_tar(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    dest: Path,
    skip_extensions: set[str],
) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    if member.isdir() or member.issym() or member.islnk():
        return False
    if not _should_keep_member(member.name, skip_extensions):
        return False
    extracted = tar.extractfile(member)
    if extracted is None:
        return False
    rel = member.name.replace("\\", "/").lstrip("./")
    target = dest / Path(rel).name
    target.write_bytes(extracted.read())
    return True


def _extract_inner_tar_gz(
    data: bytes,
    dest: Path,
    skip_extensions: set[str],
) -> int:
    kept = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as inner:
        for member in inner.getmembers():
            if member.isfile():
                kept += int(_safe_extract_tar(inner, member, dest, skip_extensions))
    return kept


def _extract_gz_payload(
    payload: bytes,
    member_name: str,
    dest: Path,
    skip_extensions: set[str],
) -> int:
    """Unpack arXiv bulk .gz members (tar.gz bundle or single gzipped source file)."""
    try:
        return _extract_inner_tar_gz(payload, dest, skip_extensions)
    except tarfile.TarError:
        import gzip

        try:
            data = gzip.decompress(payload)
        except OSError:
            return 0
        inner_name = Path(member_name.replace("\\", "/")).name
        if inner_name.endswith(".gz"):
            inner_name = inner_name[:-3]
        if not inner_name:
            inner_name = "source.tex"
        if not _should_keep_member(inner_name, skip_extensions):
            return 0
        dest.mkdir(parents=True, exist_ok=True)
        (dest / Path(inner_name).name).write_bytes(data)
        return 1


def extract_paper_from_member(
    outer: tarfile.TarFile,
    member: tarfile.TarInfo,
    arxiv_id: str,
    output_dir: Path,
    skip_extensions: set[str],
) -> bool:
    paper_dir = output_dir / arxiv_id
    name = member.name.replace("\\", "/")
    basename = Path(name).name
    kept_files = 0

    if member.isdir() and arxiv_id_from_member(name) == arxiv_id:
        for sub in outer.getmembers():
            if sub.name.startswith(member.name.rstrip("/") + "/") and sub.isfile():
                if not _should_keep_member(sub.name, skip_extensions):
                    continue
                rel = sub.name[len(member.name) :].lstrip("/")
                target = paper_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = outer.extractfile(sub)
                if extracted:
                    target.write_bytes(extracted.read())
                    kept_files += 1
        return kept_files > 0

    if name.endswith(f"{arxiv_id}.tar.gz") or name.endswith(f"{arxiv_id}.tar"):
        extracted = outer.extractfile(member)
        if not extracted:
            return False
        payload = extracted.read()
        if name.endswith(".tar.gz"):
            kept_files += _extract_inner_tar_gz(payload, paper_dir, skip_extensions)
        else:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as inner:
                for sub in inner.getmembers():
                    if sub.isfile():
                        kept_files += int(_safe_extract_tar(inner, sub, paper_dir, skip_extensions))
        return kept_files > 0

    if basename == f"{arxiv_id}.gz" or name.endswith(f"/{arxiv_id}.gz"):
        extracted = outer.extractfile(member)
        if not extracted:
            return False
        kept_files += _extract_gz_payload(extracted.read(), name, paper_dir, skip_extensions)
        return kept_files > 0

    if member.isfile() and arxiv_id_from_member(name) == arxiv_id:
        if not _should_keep_member(name, skip_extensions):
            return False
        extracted = outer.extractfile(member)
        if not extracted:
            return False
        normalized = name.strip("/")
        prefix = f"{arxiv_id}/"
        rel = normalized[len(prefix) :] if normalized.startswith(prefix) else Path(normalized).name
        target = paper_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(extracted.read())
        return True
    return False


def process_tarball(
    tarball_path: Path,
    key: str,
    *,
    target_ids: set[str],
    output_dir: Path,
    extracted_ledger_path: Path,
    skip_extensions: set[str],
    delete_tarball: bool,
) -> set[str]:
    """Extract target papers from a tarball and optionally delete the archive."""
    extracted_ids = extract_tarball(
        tarball_path,
        target_ids,
        output_dir,
        skip_extensions,
    )
    append_ledger(extracted_ledger_path, key)
    if delete_tarball and tarball_path.exists():
        tarball_path.unlink()
        part = tarball_path.with_name(f"{tarball_path.name}.part")
        if part.exists():
            part.unlink()
    return extracted_ids


def extract_tarball(
    tarball_path: Path,
    target_ids: set[str],
    output_dir: Path,
    skip_extensions: set[str],
) -> set[str]:
    extracted: set[str] = set()
    with tarfile.open(tarball_path, mode="r:") as outer:
        seen_ids: set[str] = set()
        for member in outer.getmembers():
            arxiv_id = arxiv_id_from_member(member.name)
            if not arxiv_id or arxiv_id not in target_ids or arxiv_id in seen_ids:
                continue
            if extract_paper_from_member(
                outer,
                member,
                arxiv_id,
                output_dir,
                skip_extensions,
            ):
                extracted.add(arxiv_id)
                seen_ids.add(arxiv_id)
    return extracted


def extract_all_tarballs(
    raw_dir: Path,
    tarball_keys: list[str],
    target_ids: set[str],
    output_dir: Path,
    ledger_path: Path,
    skip_extensions: set[str],
    delete_tarball_after_extract: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if ledger_path.exists():
        done = {line.strip() for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    all_extracted: set[str] = set()
    processed_tarballs = 0
    failed_tarballs: list[str] = []
    interrupted = False

    try:
        for key in tarball_keys:
            tarball_path = tarball_local_path(raw_dir, key)
            if not tarball_path.exists() or key in done:
                continue
            try:
                all_extracted.update(
                    process_tarball(
                        tarball_path,
                        key,
                        target_ids=target_ids,
                        output_dir=output_dir,
                        extracted_ledger_path=ledger_path,
                        skip_extensions=skip_extensions,
                        delete_tarball=delete_tarball_after_extract,
                    )
                )
            except (tarfile.TarError, OSError, EOFError) as exc:
                failed_tarballs.append(f"{key}: {exc}")
                continue
            processed_tarballs += 1
    except KeyboardInterrupt:
        interrupted = True

    existing = {p.name for p in output_dir.iterdir() if p.is_dir()}
    return {
        "tarballs_processed": processed_tarballs,
        "papers_extracted_this_run": len(all_extracted),
        "total_paper_dirs": len(existing),
        "failed_tarballs": failed_tarballs,
        "interrupted": interrupted,
    }


# --- M-2.3 parse-quality filter (parsed_manifest.jsonl) ---


DEFAULT_MIN_BODY_LEN = 4000
DEFAULT_MIN_CITATION_KEYS = 0


def load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def passes_quality_filter(
    row: dict[str, Any],
    *,
    min_body_len: int = DEFAULT_MIN_BODY_LEN,
    min_citation_keys: int = DEFAULT_MIN_CITATION_KEYS,
) -> bool:
    if row.get("parse_status") == "failed":
        return False
    if row.get("body_len", 0) < min_body_len:
        return False
    if row.get("num_citation_keys", 0) < min_citation_keys:
        return False
    return True


def apply_quality_filter(
    manifest_path: Path,
    output_path: Path,
    *,
    min_body_len: int = DEFAULT_MIN_BODY_LEN,
    min_citation_keys: int = DEFAULT_MIN_CITATION_KEYS,
) -> dict[str, Any]:
    rows = load_manifest_rows(manifest_path)
    kept = [
        r["arxiv_id"]
        for r in rows
        if passes_quality_filter(
            r,
            min_body_len=min_body_len,
            min_citation_keys=min_citation_keys,
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    return {
        "input_count": len(rows),
        "kept_count": len(kept),
        "dropped_count": len(rows) - len(kept),
        "min_body_len": min_body_len,
        "min_citation_keys": min_citation_keys,
    }
