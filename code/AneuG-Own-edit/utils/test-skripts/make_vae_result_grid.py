#!/usr/bin/env python3
"""Build comparison grids for a VAE optimization result folder."""

from __future__ import annotations

import argparse
import csv
import html
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("vae_optimization_results/vae_opt_20260428_full"),
        help="Sweep result directory containing comparison.csv and per-run inspect folders.",
    )
    parser.add_argument("--cell-width", type=int, default=720)
    parser.add_argument("--label-width", type=int, default=460)
    parser.add_argument("--case-cell-width", type=int, default=1260)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_rows(comparison_csv: Path) -> list[dict[str, str]]:
    with comparison_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["rank"]))
    return rows


def discover_cases(rows: list[dict[str, str]]) -> list[str]:
    cases: set[str] = set()
    for row in rows:
        config_path = Path(row["config"])
        run_dir = config_path.parent
        inspect_dir = run_dir / "inspect"
        for png in inspect_dir.glob("*_vae_recon_2000.png"):
            cases.add(png.name.removesuffix("_vae_recon_2000.png"))
    return sorted(cases)


def fit_image(path: Path, width: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        height = round(image.height * width / image.width)
        return image.resize((width, height), Image.Resampling.LANCZOS)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width_chars: int,
    line_gap: int = 4,
) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars) or [""]:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def metric(row: dict[str, str], key: str, digits: int = 4) -> str:
    value = row.get(key, "")
    if not value:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def run_label(
    row: dict[str, str],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    width: int,
    height: int,
) -> Image.Image:
    label = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(label)
    rank = row["rank"]
    run_name = row["run"]
    title = f"#{rank} {run_name}"
    y = 8
    y = draw_wrapped(draw, (14, y), title, font, (20, 20, 20), width_chars=max(24, width // 13), line_gap=4)
    y += 12
    lines = [
        f"score: {metric(row, 'balanced_score', 4)}",
        f"test: {metric(row, 'test_total_loss', 4)}",
        f"vert: {metric(row, 'test_vert_loss', 4)}",
        f"rmse: {metric(row, 'inspect_rmse_mean', 5)}",
        f"cond mse: {metric(row, 'infer_reconstruct_target_mse_mean', 4)}",
    ]
    for line in lines:
        draw.text((14, y), line, font=small_font, fill=(45, 45, 45))
        y += 34
    if rank == "1":
        draw.rounded_rectangle((14, height - 50, 170, height - 12), radius=7, fill=(31, 120, 77))
        draw.text((26, height - 42), "best score", font=small_font, fill="white")
    return label


def make_all_cases_grid(rows: list[dict[str, str]], cases: list[str], output_path: Path, cell_width: int, label_width: int) -> None:
    title_font = load_font(28, bold=True)
    header_font = load_font(22, bold=True)
    font = load_font(20, bold=True)
    small_font = load_font(18)

    sample = fit_image(Path(rows[0]["config"]).parent / "inspect" / f"{cases[0]}_vae_recon_2000.png", cell_width)
    cell_height = sample.height
    header_height = 150
    row_gap = 18
    col_gap = 16
    margin = 22
    width = margin * 2 + label_width + col_gap + len(cases) * cell_width + (len(cases) - 1) * col_gap
    height = margin * 2 + header_height + len(rows) * cell_height + (len(rows) - 1) * row_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "VAE result grid by test ranking", font=title_font, fill=(15, 15, 15))
    draw.text(
        (margin, margin + 46),
        "Red line/points: GHD-fitting OPA op_v_indices, rendered on input and reconstruction.",
        font=small_font,
        fill=(150, 38, 48),
    )
    draw.text((margin, margin + 78), "Each cell shows: left=input target, right=VAE reconstruction.", font=small_font, fill=(55, 55, 55))

    case_y = margin + 108
    x0 = margin + label_width + col_gap
    for idx, case in enumerate(cases):
        x = x0 + idx * (cell_width + col_gap)
        short_case = case if len(case) <= 28 else f"{case[:25]}..."
        draw.text((x, case_y), short_case, font=header_font, fill=(25, 25, 25))

    y = margin + header_height
    for row_idx, row in enumerate(rows):
        fill = (245, 248, 250) if row_idx % 2 == 0 else (255, 255, 255)
        draw.rectangle((margin, y - 6, width - margin, y + cell_height + 6), fill=fill)
        canvas.paste(run_label(row, font, small_font, label_width, cell_height), (margin, y))
        run_dir = Path(row["config"]).parent
        for case_idx, case in enumerate(cases):
            img_path = run_dir / "inspect" / f"{case}_vae_recon_2000.png"
            x = x0 + case_idx * (cell_width + col_gap)
            if img_path.exists():
                canvas.paste(fit_image(img_path, cell_width), (x, y))
            else:
                draw.rectangle((x, y, x + cell_width, y + cell_height), outline=(220, 0, 0), width=3)
                draw.text((x + 16, y + 16), "missing image", font=font, fill=(180, 0, 0))
        y += cell_height + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def make_case_grid(rows: list[dict[str, str]], case: str, output_path: Path, cell_width: int) -> None:
    title_font = load_font(28, bold=True)
    font = load_font(21, bold=True)
    small_font = load_font(18)

    sample = fit_image(Path(rows[0]["config"]).parent / "inspect" / f"{case}_vae_recon_2000.png", cell_width)
    cell_height = sample.height
    header_height = 140
    row_gap = 16
    label_width = 470
    margin = 22
    width = margin * 2 + label_width + 18 + cell_width
    height = margin * 2 + header_height + len(rows) * cell_height + (len(rows) - 1) * row_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), f"VAE result grid | {case}", font=title_font, fill=(15, 15, 15))
    draw.text(
        (margin, margin + 46),
        "Red line/points: GHD-fitting OPA op_v_indices; left=input target, right=VAE reconstruction.",
        font=small_font,
        fill=(150, 38, 48),
    )

    y = margin + header_height
    for row_idx, row in enumerate(rows):
        fill = (245, 248, 250) if row_idx % 2 == 0 else (255, 255, 255)
        draw.rectangle((margin, y - 6, width - margin, y + cell_height + 6), fill=fill)
        canvas.paste(run_label(row, font, small_font, label_width, cell_height), (margin, y))
        img_path = Path(row["config"]).parent / "inspect" / f"{case}_vae_recon_2000.png"
        canvas.paste(fit_image(img_path, cell_width), (margin + label_width + 18, y))
        y += cell_height + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def write_index(output_dir: Path, all_grid: Path, case_grids: list[Path], rows: list[dict[str, str]]) -> None:
    rows_html = "\n".join(
        f"<tr><td>{html.escape(row['rank'])}</td><td>{html.escape(row['run'])}</td>"
        f"<td>{html.escape(metric(row, 'balanced_score', 5))}</td>"
        f"<td>{html.escape(metric(row, 'test_total_loss', 4))}</td>"
        f"<td>{html.escape(metric(row, 'inspect_rmse_mean', 5))}</td></tr>"
        for row in rows
    )
    case_links = "\n".join(f'<li><a href="{path.name}">{html.escape(path.stem)}</a></li>' for path in case_grids)
    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VAE Result Grids</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #181818; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; margin: 18px 0; }}
    th, td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; text-align: left; }}
  </style>
</head>
<body>
  <h1>VAE Result Grids</h1>
  <p><strong>Red line/points:</strong> GHD-fitting OPA op_v_indices, rendered on input and reconstruction.</p>
  <p><a href="{all_grid.name}">Open full all-case grid PNG</a></p>
  <img src="{all_grid.name}" alt="All-case VAE result grid">
  <h2>Per-case grids</h2>
  <ul>{case_links}</ul>
  <h2>Ranking</h2>
  <table>
    <thead><tr><th>Rank</th><th>Run</th><th>Score</th><th>Test loss</th><th>Inspect RMSE</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    rows = read_rows(results_root / "comparison.csv")
    cases = discover_cases(rows)
    if not cases:
        raise RuntimeError(f"No inspect PNGs found under {results_root}")

    output_dir = results_root / "grids"
    all_grid = output_dir / "all_cases_by_rank.png"
    make_all_cases_grid(rows, cases, all_grid, args.cell_width, args.label_width)

    case_grids = []
    for case in cases:
        output_path = output_dir / f"{case}_by_rank.png"
        make_case_grid(rows, case, output_path, args.case_cell_width)
        case_grids.append(output_path)

    write_index(output_dir, all_grid, case_grids, rows)
    print(f"Wrote {all_grid}")
    for path in case_grids:
        print(f"Wrote {path}")
    print(f"Wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
