#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "fileutils"

ROOT = "/Users/hongyimac/Desktop/CoCoA_like_2D_Seidel_Experiment"
OUT_ROOT = File.join(ROOT, "outputs", "cocoa_like_2d_mechanism")

DATASETS = [
  {
    title: "RMS0.2",
    target_rms: "0.2",
    dir: File.join(
      OUT_ROOT,
      "pretrain_contrast_fullgrid4d_size256_three_images_rms020_pre400_joint1000_20260609_rcp_stats"
    )
  },
  {
    title: "RMS0.4",
    target_rms: "0.4",
    dir: File.join(
      OUT_ROOT,
      "pretrain_contrast_fullgrid4d_size256_three_images_rms040_pre400_joint1000_20260610_rcp_stats"
    )
  }
].freeze

VIEWS = [
  {
    key: "by_coeff_abs",
    figure_key: "complete_by_coeff_abs",
    csv_base: "full_ranking_by_coeff_abs",
    title: "full ranking by aligned coefficient absolute error",
    primary: "Primary rank: mean aligned coefficient absolute error, lower is better",
    sort_key: "mean_coeff_abs",
    descending: false
  },
  {
    key: "by_object_quality_ssim",
    figure_key: "complete_by_object_quality_ssim",
    csv_base: "full_ranking_by_object_quality_ssim",
    title: "full ranking by object recovery quality SSIM",
    primary: "Primary rank: mean object reconstruction SSIM, higher is better",
    sort_key: "mean_ssim",
    descending: true
  }
].freeze

METRICS = [
  {
    key: "op",
    title: "operator error",
    lower_better: true,
    source_col: "operator_error_calibrated",
    oracle_mode: "object_gt_fixed",
    oracle_col: "operator_error_calibrated"
  },
  {
    key: "coeff_abs",
    title: "aligned coefficient absolute error",
    lower_better: true,
    source_col: "aligned_coeff_absolute_error_physical",
    oracle_mode: "object_gt_fixed",
    oracle_col: "aligned_coeff_absolute_error_physical"
  },
  {
    key: "wavefront",
    title: "aligned wavefront error",
    lower_better: true,
    source_col: "aligned_wavefront_error_physical",
    oracle_mode: "object_gt_fixed",
    oracle_col: "aligned_wavefront_error_physical"
  },
  {
    key: "ssim",
    title: "object SSIM",
    lower_better: false,
    source_col: "ssim",
    oracle_mode: "seidel_gt_fixed",
    oracle_col: "ssim_recon_gain_vs_gt"
  },
  {
    key: "nrmse",
    title: "object NRMSE",
    lower_better: true,
    source_col: "nrmse",
    oracle_mode: "seidel_gt_fixed",
    oracle_col: "nrmse_recon_gain_vs_gt"
  }
].freeze

IMAGES = %w[Iksung_beads dendrites dendrites_dense].freeze
ROWS_PER_PAGE = 60
ORACLE_CONTROL_CSV = File.join(
  OUT_ROOT,
  "seidel_oracle_controls_4D_6D_4imgs_2dirs_rms006_020_040_seed0_noRMS_pre400_joint1000_20260607",
  "oracle_controls_evaluator_combined.csv"
)

HEADERS = [
  "rank",
  "method",
  "label",
  "family",
  "pretrain_scalar",
  "target_transform",
  "contrast_alpha",
  "pretrain_rsd_weight",
  "percentile_lo",
  "percentile_hi",
  "gamma",
  "mean_op",
  "mean_coeff_abs",
  "mean_wavefront",
  "mean_ssim",
  "mean_nrmse",
  "Iksung_beads_op",
  "dendrites_op",
  "dendrites_dense_op",
  "Iksung_beads_coeff_abs",
  "dendrites_coeff_abs",
  "dendrites_dense_coeff_abs",
  "Iksung_beads_wavefront",
  "dendrites_wavefront",
  "dendrites_dense_wavefront",
  "Iksung_beads_ssim",
  "dendrites_ssim",
  "dendrites_dense_ssim",
  "Iksung_beads_nrmse",
  "dendrites_nrmse",
  "dendrites_dense_nrmse"
].freeze

def xml_escape(value)
  value.to_s
       .gsub("&", "&amp;")
       .gsub("<", "&lt;")
       .gsub(">", "&gt;")
       .gsub('"', "&quot;")
end

def parse_float(value)
  Float(value)
rescue ArgumentError, TypeError
  nil
end

def fmt(value)
  return "" if value.nil?

  if value.abs >= 10
    format("%.3f", value)
  elsif value.abs >= 1
    format("%.4f", value)
  else
    format("%.5f", value)
  end
end

def color_for(value, values, lower_better)
  return "#f6f6f6" if value.nil?

  min = values.compact.min
  max = values.compact.max
  return "#d7ece9" if min.nil? || max.nil? || (max - min).abs < 1e-12

  t = (value - min) / (max - min)
  t = 1.0 - t unless lower_better
  t = [[t, 0.0].max, 1.0].min
  good = [128, 199, 190]
  neutral = [248, 248, 248]
  bad = [229, 104, 107]
  rgb =
    if t <= 0.5
      local = t / 0.5
      good.zip(neutral).map { |a, b| (a + (b - a) * local).round }
    else
      local = (t - 0.5) / 0.5
      neutral.zip(bad).map { |a, b| (a + (b - a) * local).round }
    end
  format("#%02x%02x%02x", *rgb)
end

def read_csv(path)
  CSV.read(path, headers: true)
end

def write_csv(path, rows)
  FileUtils.mkdir_p(File.dirname(path))
  CSV.open(path, "w", write_headers: true, headers: HEADERS) do |csv|
    rows.each { |row| csv << HEADERS.map { |header| row[header] } }
  end
end

def settings_by_method(dataset)
  rows = read_csv(File.join(dataset[:dir], "stats", "summary_by_setting.csv"))
  rows.to_h { |row| [row["pretrain_method"], row] }
end

def full_rows(dataset)
  comparisons = read_csv(File.join(dataset[:dir], "stats", "comparison_by_case.csv"))
  settings = settings_by_method(dataset)
  grouped = comparisons.group_by { |row| row["pretrain_method"] }
  grouped.map do |method, rows|
    setting = settings[method]
    raise "Missing setting summary for #{method}" if setting.nil?

    out = {}
    out["method"] = method
    out["label"] = setting["label"] || setting["method_label"] || rows.first["method_label"]
    out["family"] = setting["family"]
    out["pretrain_scalar"] = setting["pretrain_scalar"]
    out["target_transform"] = setting["target_transform"]
    out["contrast_alpha"] = setting["contrast_alpha"]
    out["pretrain_rsd_weight"] = setting["pretrain_rsd_weight"]
    out["percentile_lo"] = setting["percentile_lo"]
    out["percentile_hi"] = setting["percentile_hi"]
    out["gamma"] = setting["gamma"]

    by_image = rows.to_h { |row| [row["image"], row] }
    METRICS.each do |metric|
      values = IMAGES.map do |image|
        source = by_image[image]
        raise "Missing image=#{image} for method=#{method}" if source.nil?

        value = parse_float(source[metric[:source_col]])
        out["#{image}_#{metric[:key]}"] = value.to_s
        value
      end
      out["mean_#{metric[:key]}"] = (values.sum / values.length).to_s
    end
    out
  end
end

def load_oracle_lookup
  lookup = {}
  read_csv(ORACLE_CONTROL_CSV).each do |row|
    next unless row["seidel_convention"] == "classical4d"
    next unless row["direction"] == "signed_balanced"
    next unless IMAGES.include?(row["image"])

    key = [row["oracle_mode"], row["target_wavefront_rms"].to_f.round(6).to_s, row["image"]]
    lookup[key] = row
  end
  lookup
end

def oracle_value(lookup, dataset, metric, image)
  key = [metric[:oracle_mode], dataset[:target_rms].to_f.round(6).to_s, image]
  row = lookup[key]
  raise "Missing oracle control row for #{key.inspect}" if row.nil?

  parse_float(row[metric[:oracle_col]])
end

def oracle_row(dataset, oracle_lookup)
  row = HEADERS.to_h { |header| [header, ""] }
  row["rank"] = "GT"
  row["method"] = "oracle_controls_metric_specific"
  row["label"] = "Oracle controls: GT-object for Seidel, GT-Seidel for object"
  row["family"] = "oracle_control"
  METRICS.each do |metric|
    values = IMAGES.map do |image|
      value = oracle_value(oracle_lookup, dataset, metric, image)
      row["#{image}_#{metric[:key]}"] = value.to_s
      value
    end
    row["mean_#{metric[:key]}"] = (values.sum / values.length).to_s
  end
  row
end

def sort_rows(rows, view)
  sorted = rows.sort_by { |row| parse_float(row[view[:sort_key]]) || Float::INFINITY }
  sorted.reverse! if view[:descending]
  sorted.each_with_index { |row, idx| row["rank"] = (idx + 1).to_s }
  sorted
end

def draw_page(path, dataset_title, view, metric, rows, page_idx, total_pages)
  width = 1520
  top = 132
  row_h = 27
  bottom = 30
  row_y0 = top + 26
  height = row_y0 + rows.length * row_h + bottom
  margin = 28
  rank_x = margin
  method_x = rank_x + 58
  mean_x = 790
  img_xs = [970, 1145, 1320]
  cell_w = 150
  cell_h = 20
  metric_key = metric[:key]

  scale_rows = rows.reject { |row| row["family"].to_s.start_with?("oracle") }
  all_values = scale_rows.flat_map do |row|
    [parse_float(row["mean_#{metric_key}"])] +
      IMAGES.map { |image| parse_float(row["#{image}_#{metric_key}"]) }
  end.compact

  File.open(path, "w") do |f|
    f.puts %(<svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">)
    f.puts %(<rect x="0" y="0" width="#{width}" height="#{height}" fill="#ffffff"/>)
    f.puts %(<style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
      .title { font-size: 26px; font-weight: 800; fill: #0b0b0b; }
      .subtitle { font-size: 16px; fill: #4b4b4b; }
      .header { font-size: 15px; font-weight: 800; fill: #0b0b0b; }
      .method { font-size: 13px; fill: #101010; }
      .method-oracle { font-size: 13px; font-weight: 800; fill: #101010; }
      .value { font-size: 13px; fill: #0b0b0b; }
      .small { font-size: 12px; fill: #666; }
    </style>)
    f.puts %(<text class="title" x="#{margin}" y="36">#{xml_escape(dataset_title)} #{xml_escape(view[:title])}: #{xml_escape(metric[:title])}</text>)
    f.puts %(<text class="subtitle" x="#{margin}" y="68">#{xml_escape(view[:primary])}. Page #{page_idx + 1}/#{total_pages}; complete 420-setting ranking.</text>)
    f.puts %(<text class="subtitle" x="#{margin}" y="96">Oracle-control row is repeated for reference; color scale uses trained rows on this page only.</text>)
    f.puts %(<text class="header" x="#{rank_x}" y="#{top - 8}">rank</text>)
    f.puts %(<text class="header" x="#{method_x}" y="#{top - 8}">method</text>)
    f.puts %(<text class="header" x="#{mean_x + cell_w - 2}" y="#{top - 8}" text-anchor="end">mean</text>)
    IMAGES.each_with_index do |image, i|
      f.puts %(<text class="header" x="#{img_xs[i] + cell_w - 2}" y="#{top - 8}" text-anchor="end">#{xml_escape(image.sub("_", " "))}</text>)
    end

    rows.each_with_index do |row, idx|
      y = row_y0 + idx * row_h
      oracle = row["family"].to_s.start_with?("oracle")
      bg = oracle ? "#fff7d6" : (idx.odd? ? "#f5f5f5" : "#ffffff")
      f.puts %(<rect x="0" y="#{y - 19}" width="#{width}" height="#{row_h}" fill="#{bg}"/>)
      rank_text = oracle ? "GT" : row["rank"].to_s
      f.puts %(<text class="#{oracle ? "method-oracle" : "method"}" x="#{rank_x}" y="#{y}">#{xml_escape(rank_text)}</text>)
      label = oracle ? row["label"] : row["method"]
      f.puts %(<text class="#{oracle ? "method-oracle" : "method"}" x="#{method_x}" y="#{y}">#{xml_escape(label)}</text>)
      ([["mean_#{metric_key}", mean_x]] + IMAGES.each_with_index.map { |image, i| ["#{image}_#{metric_key}", img_xs[i]] }).each do |key, x|
        value = parse_float(row[key])
        color = color_for(value, all_values, metric[:lower_better])
        f.puts %(<rect x="#{x}" y="#{y - 17}" width="#{cell_w}" height="#{cell_h}" rx="5" fill="#{color}" opacity="0.95"/>)
        f.puts %(<text class="value" x="#{x + cell_w - 8}" y="#{y - 2}" text-anchor="end">#{fmt(value)}</text>)
      end
    end
    f.puts %(<text class="small" x="#{margin}" y="#{height - 10}">Full ranking CSV includes all 420 trained settings; existing top15/RCP-sorted folders are unchanged.</text>)
    f.puts "</svg>"
  end
end

def convert_svg(svg_path, png_path)
  FileUtils.mkdir_p(File.dirname(png_path))
  system("sips", "-s", "format", "png", svg_path, "--out", png_path, out: File::NULL, err: File::NULL) ||
    warn("WARN: failed to convert #{svg_path}")
end

def update_readme(path)
  original = File.exist?(path) ? File.read(path).rstrip : ""
  note = "Complete ranking: `ranking_full.csv` and `ranking_full_with_gt_oracle.csv` list all 420 trained settings. Paged full-ranking figures live under `figures/full_ranking/`; existing top15 RCP sorting folders are unchanged."
  return if original.include?("ranking_full_with_gt_oracle.csv")

  File.write(path, "#{original}\n\n#{note}\n")
end

oracle_lookup = load_oracle_lookup
DATASETS.each do |dataset|
  base_rows = full_rows(dataset)
  VIEWS.each do |view|
    sorted = sort_rows(base_rows.map(&:dup), view)
    with_oracle = [oracle_row(dataset, oracle_lookup)] + sorted.map(&:dup)

    stats_dir = File.join(dataset[:dir], "stats")
    ranked_dir = File.join(dataset[:dir], "ranked_views", view[:key])
    figure_root = File.join(ranked_dir, "figures", "full_ranking")
    stats_figure_root = File.join(stats_dir, "figures", view[:figure_key])
    [
      [File.join(stats_dir, "#{view[:csv_base]}.csv"), sorted],
      [File.join(stats_dir, "#{view[:csv_base]}_with_gt_oracle.csv"), with_oracle],
      [File.join(ranked_dir, "ranking_full.csv"), sorted],
      [File.join(ranked_dir, "ranking_full_with_gt_oracle.csv"), with_oracle],
      [File.join(figure_root, "#{view[:csv_base]}_with_gt_oracle.csv"), with_oracle],
      [File.join(stats_figure_root, "#{view[:csv_base]}_with_gt_oracle.csv"), with_oracle]
    ].each { |path, rows| write_csv(path, rows) }

    update_readme(File.join(ranked_dir, "README.md"))

    pages = sorted.each_slice(ROWS_PER_PAGE).to_a
    total_pages = pages.length
    METRICS.each do |metric|
      pages.each_with_index do |page_rows, idx|
        rows_for_page = [oracle_row(dataset, oracle_lookup)] + page_rows
        basename = "#{view[:csv_base]}_#{metric[:key]}_page#{format('%02d', idx + 1)}"
        [
          [File.join(figure_root, "svg", "#{basename}.svg"), File.join(figure_root, "png_preview", "#{basename}.svg.png")],
          [File.join(stats_figure_root, "svg", "#{basename}.svg"), File.join(stats_figure_root, "png_preview", "#{basename}.svg.png")]
        ].each do |svg_path, png_path|
          FileUtils.mkdir_p(File.dirname(svg_path))
          draw_page(svg_path, dataset[:title], view, metric, rows_for_page, idx, total_pages)
          convert_svg(svg_path, png_path)
        end
      end
    end
    puts "#{dataset[:title]} #{view[:key]} full rows=#{sorted.length} pages=#{total_pages}"
  end
end
