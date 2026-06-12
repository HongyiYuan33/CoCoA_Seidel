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
    figure_key: "ranked_by_coeff_abs",
    csv_name: "top15_by_coeff_abs_with_per_image_metrics.csv",
    oracle_csv_name: "top15_by_coeff_abs_with_gt_oracle.csv",
    title: "ranked by aligned coefficient absolute error",
    primary: "Primary rank: mean aligned coefficient absolute error, lower is better"
  },
  {
    key: "by_object_quality_ssim",
    figure_key: "ranked_by_object_quality_ssim",
    csv_name: "top15_by_object_quality_ssim_with_per_image_metrics.csv",
    oracle_csv_name: "top15_by_object_quality_ssim_with_gt_oracle.csv",
    title: "ranked by object recovery quality SSIM",
    primary: "Primary rank: mean object reconstruction SSIM, higher is better"
  }
].freeze

METRICS = [
  {
    key: "op",
    title: "operator error",
    lower_better: true,
    oracle_mode: "object_gt_fixed",
    oracle_col: "operator_error_calibrated"
  },
  {
    key: "coeff_abs",
    title: "aligned coefficient absolute error",
    lower_better: true,
    oracle_mode: "object_gt_fixed",
    oracle_col: "aligned_coeff_absolute_error_physical"
  },
  {
    key: "wavefront",
    title: "aligned wavefront error",
    lower_better: true,
    oracle_mode: "object_gt_fixed",
    oracle_col: "aligned_wavefront_error_physical"
  },
  {
    key: "ssim",
    title: "object SSIM",
    lower_better: false,
    oracle_mode: "seidel_gt_fixed",
    oracle_col: "ssim_recon_gain_vs_gt"
  },
  {
    key: "nrmse",
    title: "object NRMSE",
    lower_better: true,
    oracle_mode: "seidel_gt_fixed",
    oracle_col: "nrmse_recon_gain_vs_gt"
  }
].freeze

IMAGES = %w[Iksung_beads dendrites dendrites_dense].freeze
ORACLE_CONTROL_CSV = File.join(
  OUT_ROOT,
  "seidel_oracle_controls_4D_6D_4imgs_2dirs_rms006_020_040_seed0_noRMS_pre400_joint1000_20260607",
  "oracle_controls_evaluator_combined.csv"
)

def xml_escape(value)
  value.to_s
       .gsub("&", "&amp;")
       .gsub("<", "&lt;")
       .gsub(">", "&gt;")
       .gsub('"', "&quot;")
end

def numeric(row, key)
  text = row[key]
  return nil if text.nil? || text.empty?

  Float(text)
rescue ArgumentError
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

def write_csv(path, headers, rows)
  CSV.open(path, "w", write_headers: true, headers: headers) do |csv|
    rows.each do |row|
      csv << headers.map { |header| row[header] }
    end
  end
end

def load_oracle_control_rows
  rows = CSV.read(ORACLE_CONTROL_CSV, headers: true)
  lookup = {}
  rows.each do |row|
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

  Float(row[metric[:oracle_col]])
end

def oracle_row(headers, dataset, oracle_lookup)
  row = {}
  headers.each { |header| row[header] = "" }
  row["rank"] = "GT"
  row["method"] = "oracle_controls_metric_specific"
  row["label"] = "Oracle controls: GT-object for Seidel, GT-Seidel for object"
  row["family"] = "oracle_control"

  METRICS.each do |metric|
    values = IMAGES.map { |image| oracle_value(oracle_lookup, dataset, metric, image) }
    row["mean_#{metric[:key]}"] = (values.sum / values.length).to_s
    IMAGES.each do |image|
      row["#{image}_#{metric[:key]}"] = oracle_value(oracle_lookup, dataset, metric, image).to_s
    end
  end

  row
end

def rows_as_hashes(csv_rows)
  csv_rows.map do |row|
    row.headers.to_h { |header| [header, row[header]] }
  end
end

def draw_svg(path, dataset_title, view, metric, rows)
  width = 1520
  top = 132
  row_h = 40
  bottom = 30
  row_y0 = top + 28
  height = row_y0 + rows.length * row_h + bottom
  margin = 28
  rank_x = margin
  method_x = rank_x + 58
  mean_x = 790
  img_xs = [970, 1145, 1320]
  cell_w = 150
  cell_h = 28
  metric_key = metric[:key]

  scale_rows = rows.reject { |row| row["family"].to_s.start_with?("oracle") }
  all_values = scale_rows.flat_map do |row|
    [numeric(row, "mean_#{metric_key}")] +
      IMAGES.map { |image| numeric(row, "#{image}_#{metric_key}") }
  end.compact

  subtitle = "Rows show metric-specific oracle controls plus top15 trained methods. " \
             "Seidel metrics use GT-object-fixed recovery; object metrics use GT-Seidel-fixed recovery. " \
             "Color scale uses trained rows only."

  File.open(path, "w") do |f|
    f.puts %(<svg xmlns="http://www.w3.org/2000/svg" width="#{width}" height="#{height}" viewBox="0 0 #{width} #{height}">)
    f.puts %(<rect x="0" y="0" width="#{width}" height="#{height}" fill="#ffffff"/>)
    f.puts %(<style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
      .title { font-size: 28px; font-weight: 800; fill: #0b0b0b; }
      .subtitle { font-size: 17px; fill: #4b4b4b; }
      .header { font-size: 17px; font-weight: 800; fill: #0b0b0b; }
      .method { font-size: 16px; fill: #101010; }
      .method-oracle { font-size: 16px; font-weight: 800; fill: #101010; }
      .value { font-size: 16px; fill: #0b0b0b; }
      .small { font-size: 13px; fill: #666; }
    </style>)
    f.puts %(<text class="title" x="#{margin}" y="38">#{xml_escape(dataset_title)} #{xml_escape(view[:title])}: #{xml_escape(metric[:title])}</text>)
    f.puts %(<text class="subtitle" x="#{margin}" y="70">#{xml_escape(view[:primary])}. Reference row switches oracle by metric.</text>)
    f.puts %(<text class="subtitle" x="#{margin}" y="98">#{xml_escape(subtitle)}</text>)

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
      f.puts %(<rect x="0" y="#{y - 25}" width="#{width}" height="#{row_h}" fill="#{bg}"/>)

      rank_text = oracle ? "GT" : row["rank"].to_s
      f.puts %(<text class="#{oracle ? "method-oracle" : "method"}" x="#{rank_x}" y="#{y}">#{xml_escape(rank_text)}</text>)
      method = oracle ? row["label"] : row["method"]
      f.puts %(<text class="#{oracle ? "method-oracle" : "method"}" x="#{method_x}" y="#{y}">#{xml_escape(method)}</text>)

      cells = [["mean_#{metric_key}", mean_x]] + IMAGES.each_with_index.map { |image, i| ["#{image}_#{metric_key}", img_xs[i]] }
      cells.each do |key, x|
        value = numeric(row, key)
        color = color_for(value, all_values, metric[:lower_better])
        f.puts %(<rect x="#{x}" y="#{y - 23}" width="#{cell_w}" height="#{cell_h}" rx="6" fill="#{color}" opacity="0.95"/>)
        f.puts %(<text class="value" x="#{x + cell_w - 10}" y="#{y - 4}" text-anchor="end">#{fmt(value)}</text>)
      end
    end

    f.puts %(<text class="small" x="#{margin}" y="#{height - 10}">Oracle-control row is not a full-grid trained method; trained-method RCP folders remain unchanged.</text>)
    f.puts "</svg>"
  end
end

def convert_svg(svg_path, png_path)
  FileUtils.mkdir_p(File.dirname(png_path))
  system("sips", "-s", "format", "png", svg_path, "--out", png_path, out: File::NULL, err: File::NULL) ||
    warn("WARN: failed to convert #{svg_path}")
end

def update_readme(path)
  original = File.exist?(path) ? File.read(path) : ""
  original = original.gsub(
    /\nNote: `ranking_top15_with_gt_oracle\.csv` and the figures include a `GT object \+ GT Seidel` oracle row as the theoretical best reference\. RCP folders contain only trained settings, so the oracle row has no RCP image\.\n?/,
    "\n"
  )
  original = original.gsub(
    /\nNote: `ranking_top15_with_gt_oracle\.csv` and the figures include a `GT object` reference row for object-quality upper bounds only\. Seidel metrics in that row are N\/A, and RCP folders contain only trained settings\.\n?/,
    "\n"
  )
  note = "\nNote: `ranking_top15_with_gt_oracle.csv` and the figures include a metric-specific oracle-control row: Seidel metrics come from `object_gt_fixed` (GT object fixed, recover Seidel), while object metrics come from `seidel_gt_fixed` (GT Seidel fixed, recover object). RCP folders contain only trained full-grid settings.\n"
  return if original.include?("metric-specific oracle-control row")

  File.write(path, original.rstrip + "\n" + note)
end

oracle_lookup = load_oracle_control_rows

DATASETS.each do |dataset|
  VIEWS.each do |view|
    source_csv = File.join(dataset[:dir], "stats", view[:csv_name])
    rows = CSV.read(source_csv, headers: true)
    headers = rows.headers
    augmented = [oracle_row(headers, dataset, oracle_lookup)] + rows_as_hashes(rows)

    stats_csv = File.join(dataset[:dir], "stats", view[:oracle_csv_name])
    figure_csv = File.join(dataset[:dir], "stats", "figures", view[:figure_key], view[:oracle_csv_name])
    ranked_csv = File.join(dataset[:dir], "ranked_views", view[:key], "ranking_top15_with_gt_oracle.csv")
    [stats_csv, figure_csv, ranked_csv].each do |path|
      FileUtils.mkdir_p(File.dirname(path))
      write_csv(path, headers, augmented)
    end

    readme_path = File.join(dataset[:dir], "ranked_views", view[:key], "README.md")
    update_readme(readme_path)

    METRICS.each do |metric|
      basename = view[:key] == "by_coeff_abs" ? "top15_by_coeff_abs_#{metric[:key]}" : "top15_by_object_quality_ssim_#{metric[:key]}"
      stats_svg = File.join(dataset[:dir], "stats", "figures", view[:figure_key], "svg", "#{basename}.svg")
      stats_png = File.join(dataset[:dir], "stats", "figures", view[:figure_key], "png_preview", "#{basename}.svg.png")
      ranked_svg = File.join(dataset[:dir], "ranked_views", view[:key], "figures", "svg", "#{basename}.svg")
      ranked_png = File.join(dataset[:dir], "ranked_views", view[:key], "figures", "png_preview", "#{basename}.svg.png")

      [stats_svg, ranked_svg].each do |svg_path|
        FileUtils.mkdir_p(File.dirname(svg_path))
        draw_svg(svg_path, dataset[:title], view, metric, augmented)
      end
      convert_svg(stats_svg, stats_png)
      convert_svg(ranked_svg, ranked_png)
    end
  end
end

puts "Added GT oracle rows and regenerated ranked figures for #{DATASETS.length} datasets."
