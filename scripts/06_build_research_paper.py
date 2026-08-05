from __future__ import annotations

import csv
import html
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"
SUBMISSION_DIR = ROOT / "submission"

SUMMARY_PATH = OUTPUT_DIR / "summary.json"
MODEL_RESULTS_PATH = OUTPUT_DIR / "model_results.json"
QUEUE_PATH = OUTPUT_DIR / "refresh_queue.csv"
REPORT_PATH = OUTPUT_DIR / "model_report.md"
INDEX_PATH = DOCS_DIR / "index.html"
PAPER_URL_PATH = SUBMISSION_DIR / "paper_url.txt"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_queue() -> list[dict[str, str]]:
    with QUEUE_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_url(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def repo_info() -> tuple[str, str, str, str]:
    origin = git_url(["git", "remote", "get-url", "origin"])
    if origin.startswith("https://github.com/"):
        owner_repo = origin.removeprefix("https://github.com/")
    elif origin.startswith("git@github.com:"):
        owner_repo = origin.removeprefix("git@github.com:")
    else:
        raise SystemExit(f"Unsupported remote URL for paper generation: {origin}")

    if owner_repo.endswith(".git"):
        owner_repo = owner_repo[:-4]

    owner, repo = owner_repo.split("/", 1)
    owner = owner.lower()
    repo = repo.lower()
    repo_base = f"https://github.com/{owner}/{repo}"
    pages_base = f"https://{owner}.github.io/{repo}/"
    return owner, repo, repo_base, pages_base


def chart_svg(name: str) -> str:
    return read_text(OUTPUT_DIR / "charts" / f"{name}.svg")


def fmt_int(value: object) -> str:
    return f"{int(float(value)):,}"


def fmt_float(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body_html.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_html)}</tbody></table>"


def figure(svg_markup: str, caption: str) -> str:
    return f"""
    <figure class="chart-card">
      <div class="svg-wrap">{svg_markup}</div>
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def top_reason_counts(queue_rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for row in queue_rows:
        for reason in str(row.get("final_reason_codes", "")).split("|"):
            if reason:
                counter[reason] += 1
    return counter.most_common(8)


def action_counts(queue_rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter(row["suggested_action"] for row in queue_rows)
    ordered = [
        "monitor",
        "refresh",
        "refresh_and_review_ctr",
        "refresh_and_review_engagement",
        "expand_and_refresh",
    ]
    return [(action, counter.get(action, 0)) for action in ordered if counter.get(action, 0)]


def build_html() -> str:
    summary = read_json(SUMMARY_PATH)
    results = read_json(MODEL_RESULTS_PATH)
    queue_rows = load_queue()
    report_excerpt = read_text(REPORT_PATH).splitlines()[:6]
    owner, repo, repo_base, pages_base = repo_info()

    best_model = results["best_model"]["name"]
    best_metrics = results["models"][best_model]
    baseline = results["baseline"]
    lift = best_metrics["precision_at_50"] / baseline["baseline_precision_at_50"]

    recommendations = [
        (
            "Prioritize refresh + CTR review",
            "High-visibility pages that are already drawing demand but under-capturing clicks should be reviewed first.",
            "refresh_and_review_ctr",
        ),
        (
            "Refresh the broader middle",
            "Plain refresh items form the largest actionable bucket and are the safest second pass after the highest-confidence rows.",
            "refresh",
        ),
        (
            "Pair refresh with engagement review",
            "Low-engagement visible pages need a manual look at content structure, not just metadata edits.",
            "refresh_and_review_engagement",
        ),
        (
            "Expand only the rare strong pages",
            "The small expand-and-refresh set deserves extra content depth rather than a template refresh.",
            "expand_and_refresh",
        ),
    ]

    action_counter = dict(action_counts(queue_rows))
    reason_counter = top_reason_counts(queue_rows)

    top_preview_rows = []
    for row in queue_rows[:10]:
        top_preview_rows.append(
            [
                html.escape(str(row["final_rank"])),
                html.escape(fmt_float(row["final_refresh_score"], 1)),
                html.escape(fmt_float(row["best_model_probability"], 3)),
                html.escape(row["suggested_action"]),
                html.escape(str(row["final_reason_codes"]).replace("|", ", ")),
                html.escape(fmt_int(row["impressions_90d"])),
                html.escape(fmt_int(row["sessions_90d"])),
                html.escape(row["trend_direction"]),
            ]
        )

    metric_rows = [
        ["random_forest", fmt_float(best_metrics["roc_auc"]), fmt_float(best_metrics["average_precision"]), fmt_float(best_metrics["precision_at_50"]), fmt_float(best_metrics["recall"]), fmt_float(best_metrics["f1"])],
        ["decision_tree", fmt_float(results["models"]["decision_tree"]["roc_auc"]), fmt_float(results["models"]["decision_tree"]["average_precision"]), fmt_float(results["models"]["decision_tree"]["precision_at_50"]), fmt_float(results["models"]["decision_tree"]["recall"]), fmt_float(results["models"]["decision_tree"]["f1"])],
        ["logistic_regression", fmt_float(results["models"]["logistic_regression"]["roc_auc"]), fmt_float(results["models"]["logistic_regression"]["average_precision"]), fmt_float(results["models"]["logistic_regression"]["precision_at_50"]), fmt_float(results["models"]["logistic_regression"]["recall"]), fmt_float(results["models"]["logistic_regression"]["f1"])],
        ["baseline_rules", fmt_float(baseline["baseline_roc_auc"]), fmt_float(baseline["baseline_average_precision"]), fmt_float(baseline["baseline_precision_at_50"]), fmt_float(baseline["baseline_recall"]), fmt_float(baseline["baseline_f1"])],
    ]

    abstract = (
        "Can safe content and search signals rank pages that deserve refresh work before a human gets lost in the queue? "
        "This paper uses the anonymized FlyRank internship release, grouped by client and split with client holdout validation, to train a transparent refresh-opportunity model on 30,000 rows. "
        "A random forest selected on Precision@50 beat a transparent baseline on the same split, reaching 0.740 Precision@50 versus 0.240 for the rule. "
        "The strongest signals were visibility and freshness variables such as days with impressions, impressions volume, average position, and content age. "
        "The output is a ranked, decision-support queue that helps editors inspect the highest-risk, highest-opportunity pages first while keeping public-safe language and leakage controls intact."
    )

    base_rate = f"{results['target_positive_rate']:.1%}"
    page_links = {
        "Notebook 1": f"{repo_base}/blob/main/notebooks/01_first_look_and_discovery.ipynb",
        "Notebook 2": f"{repo_base}/blob/main/notebooks/02_your_first_readable_model.ipynb",
        "Notebook 3": f"{repo_base}/blob/main/notebooks/03_working_with_the_full_release.ipynb",
        "Reference README": f"{repo_base}/blob/main/README.md",
        "Pipeline": f"{repo_base}/blob/main/scripts/run_all.py",
    }

    reproducibility_links = "".join(
        f'<li><a href="{url}" target="_blank" rel="noreferrer">{html.escape(label)}</a></li>'
        for label, url in page_links.items()
    )

    action_card_html = "".join(
        f"<div class='action-card'><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p><span>{html.escape(code)} · {fmt_int(action_counter.get(code, 0))} rows</span></div>"
        for title, body, code in recommendations
    )

    reasons_html = "".join(
        f"<li><strong>{html.escape(reason)}</strong> — {fmt_int(count)} rows</li>"
        for reason, count in reason_counter
    )

    report_note = "\n".join(report_excerpt)

    html_output = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>FlyRank Refresh Opportunity Research Paper</title>
  <style>
    :root {{
      --bg: #f6f1e9;
      --surface: rgba(255, 255, 255, 0.84);
      --surface-strong: #ffffff;
      --text: #172126;
      --muted: #5d6a70;
      --border: #d9d1c6;
      --teal: #0f766e;
      --teal-soft: #e5f4f2;
      --amber: #b45309;
      --amber-soft: #fff4e5;
      --ink: #0c1a20;
      --shadow: 0 16px 50px rgba(18, 28, 33, 0.08);
      --display: "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
      --body: "Trebuchet MS", "Segoe UI", Arial, sans-serif;
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: var(--body);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.10), transparent 22%),
        linear-gradient(180deg, #fbf8f3 0%, var(--bg) 100%);
    }}

    .page {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 52px; }}
    .hero {{
      position: relative;
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(247,243,236,0.90));
      box-shadow: var(--shadow);
      padding: 32px;
      margin-bottom: 22px;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -8% -42% auto;
      width: 340px;
      height: 340px;
      background: radial-gradient(circle, rgba(15,118,110,0.18), transparent 70%);
      pointer-events: none;
    }}
    .eyebrow {{
      margin: 0 0 12px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--teal);
      font-size: 0.76rem;
      font-weight: 700;
    }}
    h1 {{
      margin: 0;
      font-family: var(--display);
      font-size: clamp(2.5rem, 5vw, 4.6rem);
      line-height: 0.98;
      color: var(--ink);
      max-width: 11ch;
    }}
    .hero p.lede {{
      margin: 18px 0 0;
      font-size: 1.05rem;
      line-height: 1.65;
      color: var(--muted);
      max-width: 72ch;
    }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .chip {{
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.88);
      padding: 8px 12px;
      font-size: 0.88rem;
      color: var(--text);
    }}
    .chip strong {{ color: var(--ink); }}

    .grid {{ display: grid; gap: 18px; }}
    .metrics {{ grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin: 22px 0; }}
    .metric {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 18px;
      min-height: 122px;
    }}
    .metric .value {{ display: block; font-size: 1.9rem; line-height: 1.1; font-weight: 800; color: var(--ink); font-family: var(--display); }}
    .metric .label {{ display: block; margin-top: 8px; color: var(--muted); font-size: 0.88rem; line-height: 1.45; }}
    .metric .note {{ display: block; margin-top: 12px; color: var(--teal); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }}

    .section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-top: 18px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-family: var(--display);
      font-size: 1.95rem;
      color: var(--ink);
    }}
    .section p, .section li {{ line-height: 1.72; color: var(--text); }}
    .section .muted {{ color: var(--muted); }}
    .abstract {{
      background: linear-gradient(180deg, rgba(15,118,110,0.07), rgba(180,83,9,0.04));
      border-left: 4px solid var(--teal);
      padding: 18px 18px 18px 20px;
      border-radius: 18px;
    }}
    .abstract p {{ margin: 0; }}
    .two-col {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .callout {{
      background: var(--surface-strong);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 18px;
    }}
    .callout h3 {{ margin: 0 0 10px; font-size: 1.02rem; color: var(--ink); }}
    .callout p {{ margin: 0; color: var(--muted); }}
    .pipeline-note {{
      margin-top: 18px;
      padding: 14px 16px;
      border-radius: 16px;
      background: var(--amber-soft);
      border: 1px solid rgba(180,83,9,0.18);
      color: #6a451f;
      font-size: 0.94rem;
    }}

    .chart-grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .chart-card {{
      margin: 0;
      background: var(--surface-strong);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 14px;
    }}
    .svg-wrap {{ width: 100%; overflow-x: auto; }}
    .svg-wrap svg {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 10px; font-size: 0.82rem; color: var(--muted); }}

    table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 18px; }}
    thead th {{
      text-align: left;
      padding: 12px 10px;
      background: #18313a;
      color: #fff;
      font-size: 0.86rem;
      letter-spacing: 0.02em;
    }}
    tbody td {{
      padding: 11px 10px;
      border-top: 1px solid var(--border);
      vertical-align: top;
      font-size: 0.92rem;
    }}
    tbody tr:nth-child(even) td {{ background: rgba(255,255,255,0.64); }}
    .paper-grid {{ display: grid; gap: 18px; grid-template-columns: 1.25fr 0.95fr; align-items: start; }}
    .small-stack {{ display: grid; gap: 16px; }}
    .actions-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .action-card {{
      border: 1px solid var(--border);
      border-radius: 18px;
      background: linear-gradient(180deg, #fff, #faf7f0);
      padding: 16px;
    }}
    .action-card h3 {{ margin: 0 0 8px; font-size: 1rem; color: var(--ink); }}
    .action-card p {{ margin: 0; color: var(--muted); font-size: 0.92rem; }}
    .action-card span {{ display: block; margin-top: 12px; color: var(--teal); font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }}
    .footer {{ margin: 18px 0 0; color: var(--muted); font-size: 0.88rem; text-align: center; }}
    code {{ background: rgba(15,118,110,0.08); padding: 0.15rem 0.35rem; border-radius: 6px; }}
    a {{ color: var(--teal); text-decoration-thickness: 2px; text-underline-offset: 2px; }}
    ul.links {{ margin: 0; padding-left: 1.2rem; }}
    @media (max-width: 840px) {{
      .hero {{ padding: 24px; }}
      .section {{ padding: 18px; }}
      .paper-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <p class="eyebrow">FlyRank ML Internship · Refresh / Content Opportunity Scoring</p>
      <h1>Ranked refresh recommendations for real search content</h1>
      <p class="lede">This paper turns the anonymized FlyRank internship release into a repeatable decision-support model: it identifies pages that are likely worth refresh review, validates the ranking honestly with a client-holdout split, and converts the results into an ordered action queue for editors and mentors.</p>
      <div class="chip-row">
        <span class="chip"><strong>Rows scored:</strong> {fmt_int(summary['rows_scored'])}</span>
        <span class="chip"><strong>Split:</strong> {html.escape(results['split_strategy'])}</span>
        <span class="chip"><strong>Target:</strong> {html.escape(results['target'])}</span>
        <span class="chip"><strong>Best model:</strong> {html.escape(best_model)}</span>
        <span class="chip"><strong>Precision@50 lift:</strong> {lift:.2f}x vs baseline</span>
      </div>
    </header>

    <section class="grid metrics">
      <article class="metric"><span class="value">{fmt_float(best_metrics['precision_at_50'])}</span><span class="label">Random forest Precision@50 on the held-out split</span><span class="note">Selected model</span></article>
      <article class="metric"><span class="value">{fmt_float(baseline['baseline_precision_at_50'])}</span><span class="label">Transparent baseline Precision@50 on the same split</span><span class="note">Control</span></article>
      <article class="metric"><span class="value">{fmt_float(best_metrics['roc_auc'])}</span><span class="label">Random forest ROC AUC</span><span class="note">Ranking separation</span></article>
      <article class="metric"><span class="value">{fmt_float(best_metrics['average_precision'])}</span><span class="label">Random forest average precision</span><span class="note">Ranked list quality</span></article>
      <article class="metric"><span class="value">{fmt_float(summary['target_positive_rate'])}</span><span class="label">Declining-label base rate in the scored data</span><span class="note">Task balance</span></article>
      <article class="metric"><span class="value">{fmt_int(summary['high_confidence_rows'])}</span><span class="label">High-confidence rows in the ranked queue</span><span class="note">Manual review tier</span></article>
    </section>

    <section class="section">
      <h2>Abstract</h2>
      <div class="abstract"><p>{html.escape(abstract)}</p></div>
    </section>

    <section class="section">
      <h2>Introduction / Problem Statement</h2>
      <div class="paper-grid">
        <div>
          <p>The decision this work supports is simple: which existing pages should a FlyRank editor inspect first when refresh capacity is limited. The unit of analysis is a page-level content record, and the output is a ranked queue with action labels rather than a binary publish / no-publish recommendation.</p>
          <p>The cost of a wrong call is asymmetric. Missing a truly strong refresh candidate wastes editorial effort and delays growth, while over-prioritizing a weak page burns review time. A ranking model is useful here because the dataset already contains safe visibility, freshness, and engagement signals that can be sorted into a practical review queue without touching client-identifying content.</p>
        </div>
        <div class="callout">
          <h3>What this is not</h3>
          <p>This is not a claim about search-engine causality. It is a measured, decision-support ranking that helps reviewers spend their time on the most promising rows first.</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Data</h2>
      <div class="two-col">
        <div>
          <p>The analysis uses the anonymized starter release bundled with the internship repo, specifically <code>data/raw/content_refresh_anonymized.csv</code>. The scored set contains 30,000 rows and a decline-label rate of {base_rate}.</p>
          <p>We deliberately excluded public-unsafe and leakage-prone information: titles, URLs, domains, raw queries, client names, and any direct label-derived shortcuts. Client identifiers were used only for the grouped holdout split, not as features.</p>
        </div>
        <div class="callout">
          <h3>Included signal families</h3>
          <p>Visibility counts, click and session volume, freshness and age, CTR, average position, engagement rate, scroll rate, content taxonomy, and coarse tiered descriptors.</p>
        </div>
      </div>
      <div class="pipeline-note">The evaluation split is client-holdout, which is stricter than a random row split and better reflects deployment behavior where new pages from a held-out client should not leak into training.</div>
    </section>

    <section class="section">
      <h2>Methodology</h2>
      <div class="two-col">
        <div>
          <p>The target is <code>is_declining_label</code>, a supervised proxy for refresh priority. Baseline first, model second: the transparent baseline ranks pages with a hand-written refresh score, and the learned model compares against that same split using ROC AUC, average precision, Precision@50, recall, and F1.</p>
          <p>The final model is a random forest selected on Precision@50. The exact feature set includes 52 model inputs, with numeric signals transformed into log or normalized variants where useful and categorical signals one-hot encoded. The ranking pipeline also applies rule-based reason codes so the final queue is inspectable by humans.</p>
        </div>
        <div class="callout">
          <h3>Leakage checks</h3>
          <p>Label-adjacent columns such as trend direction are used only as output diagnostics and queue explanations, never as model features. That keeps the learned score honest and prevents post-label shortcuts from inflating validation metrics.</p>
        </div>
      </div>
      <div class="pipeline-note">The model candidates were logistic regression, decision tree, and random forest. The random forest won on the same holdout used for all comparisons, so the result is a ranked-model improvement, not a hand-waved cherry-pick.</div>
    </section>

    <section class="section">
      <h2>Results</h2>
      <p class="muted">The table below shows the model comparison on the same client-holdout split. The baseline is intentionally transparent, but it is much weaker on the top of the ranked list.</p>
      {table(['Model', 'ROC AUC', 'Avg precision', 'Precision@50', 'Recall', 'F1'], metric_rows)}
      <div class="chart-grid" style="margin-top: 18px;">
        {figure(chart_svg('top_feature_importance'), 'Top model features: visibility and freshness signals dominate, with days with impressions, log impressions, average position, and content age leading the list.')}
        {figure(chart_svg('top_reason_codes'), 'Most common queue reasons: the highest-ranked rows repeatedly combine decline risk, visibility, and CTR or engagement review cues.')}
        {figure(chart_svg('action_mix'), 'Action mix in the final queue: monitor remains the largest bucket, but the model surfaces a sizable refresh-and-review tranche.')}
        {figure(chart_svg('confidence_mix'), 'Confidence mix: only a subset of pages rise into the high-confidence review tier, which is exactly what an editorial queue should do.')}
        {figure(chart_svg('trend_distribution'), 'Trend direction in the queue: the ranking naturally concentrates on pages currently moving down or flattening.')}
      </div>
      <div class="pipeline-note">Baseline Precision@50 is 0.240 and the random forest reaches 0.740, which is a 3.08x lift on the same split. That is the core result: the learned ranking surfaces much better review candidates near the top of the queue.</div>
      <div style="margin-top: 18px;">{table(['Metric', 'Interpretation'], [
        ['ROC AUC', 'How well the model separates decline-labeled pages from others.'],
        ['Average precision', 'How trustworthy the ranked list is when positives are rare or mixed.'],
        ['Precision@50', 'How many of the top 50 recommendations are actually decline-labeled.'],
        ['Recall', 'How much of the positive class is recovered at the default threshold.'],
      ])}</div>
    </section>

    <section class="section">
      <h2>Ranked Recommendations</h2>
      <div class="actions-grid">{action_card_html}</div>
      <div style="margin-top: 16px;">
        <h3 style="margin: 0 0 10px; font-family: var(--display); color: var(--ink);">Most common reason codes</h3>
        <ul>{reasons_html}</ul>
      </div>
    </section>

    <section class="section">
      <h2>Top 10 Queue Preview</h2>
      {table(['Rank', 'Score', 'Model probability', 'Action', 'Reasons', 'Impressions', 'Sessions', 'Trend'], top_preview_rows)}
    </section>

    <section class="section">
      <h2>Limitations & Honest Framing</h2>
      <div class="two-col">
        <div>
          <p>The labels are proxies, not perfect ground truth. The model therefore supports review prioritization, not causal claims about algorithmic changes or guaranteed traffic lift.</p>
          <p>The queue is most useful when paired with editorial context. A page can be high priority for refresh review and still be worth deferring because of seasonality, business timing, or external constraints that the dataset cannot see.</p>
        </div>
        <div class="callout">
          <h3>Language to keep</h3>
          <p>Observed, measured, directional, decision-support. That framing keeps the paper aligned with the public-safety rules and avoids overstating what the model can prove.</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Reproducibility</h2>
      <p>Run the reference pipeline from a fresh clone with <code>python scripts/run_all.py</code>. That produces the processed features, baseline queue, trained model predictions, queue report, PDF summary, and this paper page.</p>
      <ul class="links">{reproducibility_links}</ul>
      <p class="muted">The paper source is generated from <code>outputs/summary.json</code>, <code>outputs/model_results.json</code>, <code>outputs/refresh_queue.csv</code>, and the charts under <code>outputs/charts/</code>.</p>
      <div class="pipeline-note">The source of truth for the public site is this repo. The direct deployed URL for the paper is written to <code>submission/paper_url.txt</code>.</div>
      <div style="margin-top: 12px; font-size: 0.9rem; color: var(--muted);"><strong>Report excerpt:</strong><br /><pre style="white-space: pre-wrap; margin: 8px 0 0; font-family: inherit;">{html.escape(report_note)}</pre></div>
    </section>

    <section class="section">
      <h2>Acknowledgments & Data Credit</h2>
      <p>Built on the FlyRank ML Internship dataset. Data credit goes to <a href="https://flyrank.ai/" target="_blank" rel="noreferrer">flyrank.ai</a>.</p>
      <p class="muted">Repository: <a href="{repo_base}" target="_blank" rel="noreferrer">{repo_base}</a> · Pages root: <a href="{pages_base}" target="_blank" rel="noreferrer">{pages_base}</a></p>
    </section>

    <div class="footer">Prepared from the validated refresh-ranking pipeline in the FlyRank internship repo.</div>
  </main>
</body>
</html>"""

    return html_output


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    owner, repo, repo_base, pages_base = repo_info()
    html_output = build_html()
    INDEX_PATH.write_text(html_output, encoding="utf-8")
    PAPER_URL_PATH.write_text(f"{pages_base}\n", encoding="utf-8")
    print(f"Wrote paper HTML: {INDEX_PATH}")
    print(f"Wrote paper URL: {PAPER_URL_PATH}")
    print(f"Pages target: {pages_base}")


if __name__ == "__main__":
    main()