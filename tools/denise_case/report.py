from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from .config import CaseConfig


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _pre(data: Any) -> str:
    return "<pre>" + html.escape(json.dumps(data, indent=2, sort_keys=True)) + "</pre>"


def _embedded_png(path: Path, alt: str) -> str:
    if not path.is_file():
        return f"<p class='warning'>Missing figure: {html.escape(str(path))}</p>"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"<figure><img alt='{html.escape(alt)}' src='data:image/png;base64,{encoded}'><figcaption>{html.escape(alt)}</figcaption></figure>"


def build_report(config: CaseConfig, run_name: str = "true_forward") -> Path:
    qc = config.root / "qc"
    run = config.root / "generated" / run_name
    inventory = _read_json(config.root / "input" / "inventory.json")
    validation = _read_json(run / "validation.json") or _read_json(qc / "validation.json")
    execution = _read_json(run / "run_metadata.json")
    model = _read_json(qc / "models" / "model_qc.json")
    geometry = _read_json(qc / "geometry" / "geometry_qc.json")
    data = _read_json(qc / "data" / "data_qc.json")
    figures: list[str] = []
    if model:
        for component, details in model.get("figures", {}).items():
            figures.append(_embedded_png(config.root / details["path"], f"{component}: true, 1-D start, smooth2, true-minus-start"))
    if geometry:
        figures.append(_embedded_png(config.root / geometry["figure"], "Benchmark-derived OBC acquisition over true Vp"))
    if data:
        for component, details in data.get("components", {}).items():
            for shot, shot_details in details.get("shots", {}).items() if isinstance(details, dict) else []:
                figures.append(_embedded_png(config.root / shot_details["gather_figure"], f"Observed {component} gather, shot {shot}"))
                figures.append(_embedded_png(config.root / shot_details["trace_overlay_figure"], f"Observed {component} selected traces, shot {shot}"))
        wavelet = data.get("wavelet", {})
        if wavelet.get("status") == "read from DENISE output":
            figures.append(_embedded_png(config.root / wavelet["figure"], "Recovered source wavelet"))
            figures.append(_embedded_png(config.root / wavelet["spectrum_figure"], "Recovered source spectrum"))
    warnings = []
    if validation:
        warnings.extend(validation.get("warnings", []))
        warnings.extend(validation.get("errors", []))
    if execution is None:
        warnings.append("True-forward execution metadata is absent; no production run has been recorded.")
    if data is None or all(details.get("status") == "missing" for details in (data or {}).get("components", {}).values()):
        warnings.append("Observed vx/vy products are absent; data QC will populate after a completed forward run.")
    css = """
body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#18202a}
h1,h2{color:#173b57} pre{background:#f3f5f7;padding:1rem;overflow:auto;font-size:.82rem}
.warning{border-left:4px solid #c77d00;background:#fff6dd;padding:.6rem 1rem} img{max-width:100%;image-rendering:auto}
figure{margin:1.5rem 0} figcaption{font-weight:600;margin-top:.4rem} code{background:#eef2f5;padding:.1rem .25rem}
"""
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>Marmousi-II A1 QC report</title><style>{css}</style></head><body>
<h1>Marmousi-II A0/A1 application report</h1>
<p>This report distinguishes the immutable historical benchmark from the generated current-DENISE forward case.</p>
<h2>Case identity and provenance</h2>{_pre({'case': config.data.get('case'), 'provenance': config.data.get('provenance'), 'manifest_sha256': config.sha256})}
<h2>Grid, physics, boundaries and wavelet</h2>{_pre({key: config.data.get(key) for key in ('grid','time','physics','boundary','wavelet')})}
<h2>Acquisition and imported input identities</h2>{_pre({'acquisition': config.data.get('acquisition'), 'inventory': inventory})}
<h2>Validation</h2>{_pre(validation or {'status': 'not run'})}
<h2>Execution and data completeness</h2>{_pre(execution or {'status': 'not run'})}
<h2>Warnings and incomplete evidence</h2>{''.join(f"<p class='warning'>{html.escape(str(item))}</p>" for item in warnings) or '<p>None.</p>'}
<h2>Model and geometry QC</h2>{''.join(figures) or '<p>No QC figures generated.</p>'}
<h2>Machine-readable QC summaries</h2>{_pre({'model_qc': model, 'geometry_qc': geometry, 'data_qc': data})}
</body></html>"""
    target = qc / "report" / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target
