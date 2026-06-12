#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "fileutils"
require "pathname"

ROOT = "/Users/hongyimac/Desktop/CoCoA_like_2D_Seidel_Experiment"
OUT_ROOT = File.join(ROOT, "outputs", "cocoa_like_2d_mechanism")

DATASETS = [
  {
    title: "RMS0.2",
    dir: File.join(
      OUT_ROOT,
      "pretrain_contrast_fullgrid4d_size256_three_images_rms020_pre400_joint1000_20260609_rcp_stats"
    )
  },
  {
    title: "RMS0.4",
    dir: File.join(
      OUT_ROOT,
      "pretrain_contrast_fullgrid4d_size256_three_images_rms040_pre400_joint1000_20260610_rcp_stats"
    )
  }
].freeze

VIEWS = [
  {
    key: "by_coeff_abs",
    title: "aligned coefficient absolute error",
    primary: "mean aligned coefficient absolute error, lower is better"
  },
  {
    key: "by_object_quality_ssim",
    title: "object recovery quality SSIM",
    primary: "mean object reconstruction SSIM, higher is better"
  }
].freeze

IMAGES = %w[Iksung_beads dendrites dendrites_dense].freeze
METRIC_COLUMNS = {
  op: "operator_error_calibrated",
  coeff: "aligned_coeff_absolute_error_physical",
  ssim: "ssim",
  nrmse: "nrmse"
}.freeze

def read_csv(path)
  CSV.read(path, headers: true)
end

def parse_float(value)
  Float(value)
rescue ArgumentError, TypeError
  nil
end

def fmt(value, digits = 5)
  numeric = parse_float(value)
  return "na" if numeric.nil?

  format("%.#{digits}f", numeric)
end

def safe_name(value)
  value.to_s.gsub(/[^A-Za-z0-9_.=+-]+/, "_")
end

def absolute_path(path)
  return path if Pathname.new(path).absolute?

  File.join(ROOT, path)
end

def relative_target(source, dest)
  Pathname.new(source).relative_path_from(Pathname.new(File.dirname(dest))).to_s
end

def force_symlink(source, dest)
  FileUtils.mkdir_p(File.dirname(dest))
  FileUtils.rm_f(dest)
  FileUtils.ln_s(relative_target(source, dest), dest)
end

def build_manifest_lookup(dataset_dir)
  rows = read_csv(File.join(dataset_dir, "manifest.csv"))
  rows.each_with_object({}) do |row, lookup|
    key = [row["pretrain_method"], row["image"]]
    path = absolute_path(row["path"])
    raise "Missing source RCP: #{path}" unless File.file?(path)

    lookup[key] = path
  end
end

def build_case_metric_lookup(dataset_dir)
  rows = read_csv(File.join(dataset_dir, "stats", "comparison_by_case.csv"))
  rows.each_with_object({}) do |row, lookup|
    lookup[[row["pretrain_method"], row["image"]]] = row
  end
end

def metric_suffix(row)
  [
    "op#{fmt(row[METRIC_COLUMNS[:op]])}",
    "coeff#{fmt(row[METRIC_COLUMNS[:coeff]])}",
    "ssim#{fmt(row[METRIC_COLUMNS[:ssim]])}",
    "nrmse#{fmt(row[METRIC_COLUMNS[:nrmse]])}"
  ].join("__")
end

def write_root_readme(path, dataset, view, rank_rows)
  File.write(
    File.join(path, "README.md"),
    [
      "# #{dataset[:title]} full RCP sorting by #{view[:title]}",
      "",
      "- Ranking basis: #{view[:primary]}",
      "- Trained settings: #{rank_rows.length}",
      "- RCP files are symlinks into `../../rcp_all`, so this folder is lightweight.",
      "- `RCP_full_sorted_by_rank` groups the three image RCPs under each method rank.",
      "- `RCP_full_sorted_by_image` groups all method-rank RCPs separately for each image.",
      "- `_full_ranking_figures_png_preview` points to the matching full-ranking summary figures.",
      "- Source CSV: `../ranking_full.csv`"
    ].join("\n") + "\n"
  )
end

def build_view(dataset, view)
  view_dir = File.join(dataset[:dir], "ranked_views", view[:key])
  rank_rows = read_csv(File.join(view_dir, "ranking_full.csv"))
  manifest_lookup = build_manifest_lookup(dataset[:dir])
  case_metric_lookup = build_case_metric_lookup(dataset[:dir])

  by_rank = File.join(view_dir, "RCP_full_sorted_by_rank")
  by_image = File.join(view_dir, "RCP_full_sorted_by_image")
  FileUtils.rm_rf(by_rank)
  FileUtils.rm_rf(by_image)
  FileUtils.mkdir_p(by_rank)
  FileUtils.mkdir_p(by_image)
  IMAGES.each { |image| FileUtils.mkdir_p(File.join(by_image, image)) }

  figures = File.join(view_dir, "figures", "full_ranking", "png_preview")
  force_symlink(figures, File.join(by_rank, "_full_ranking_figures_png_preview")) if Dir.exist?(figures)
  force_symlink(figures, File.join(by_image, "_full_ranking_figures_png_preview")) if Dir.exist?(figures)

  rank_rows.each do |rank_row|
    rank = rank_row["rank"].to_i
    method = rank_row["method"]
    rank_label = format("rank%03d__%s", rank, safe_name(method))
    rank_dir = File.join(by_rank, rank_label)
    FileUtils.mkdir_p(rank_dir)

    image_lines = []
    IMAGES.each do |image|
      src = manifest_lookup.fetch([method, image]) do
        raise "Missing RCP source for method=#{method} image=#{image}"
      end
      case_row = case_metric_lookup.fetch([method, image]) do
        raise "Missing comparison metrics for method=#{method} image=#{image}"
      end

      filename = [
        image,
        format("rank%03d", rank),
        safe_name(method),
        metric_suffix(case_row)
      ].join("__") + ".png"

      force_symlink(src, File.join(rank_dir, filename))
      force_symlink(src, File.join(by_image, image, filename))
      image_lines << "- #{image}: #{filename}"
    end

    File.write(
      File.join(rank_dir, "README.md"),
      [
        "# #{rank_label}",
        "",
        "- Method label: #{rank_row["label"]}",
        "- Family: #{rank_row["family"]}",
        "- Mean op: #{fmt(rank_row["mean_op"])}",
        "- Mean coeff abs: #{fmt(rank_row["mean_coeff_abs"])}",
        "- Mean wavefront: #{fmt(rank_row["mean_wavefront"])}",
        "- Mean SSIM: #{fmt(rank_row["mean_ssim"])}",
        "- Mean NRMSE: #{fmt(rank_row["mean_nrmse"])}",
        "",
        "## RCP files",
        *image_lines
      ].join("\n") + "\n"
    )
  end

  write_root_readme(by_rank, dataset, view, rank_rows)
  write_root_readme(by_image, dataset, view, rank_rows)

  {
    view: view[:key],
    ranks: rank_rows.length,
    rank_pngs: Dir.glob(File.join(by_rank, "rank*", "*.png")).length,
    image_pngs: Dir.glob(File.join(by_image, "*", "*.png")).length
  }
end

DATASETS.each do |dataset|
  VIEWS.each do |view|
    result = build_view(dataset, view)
    puts "#{dataset[:title]} #{result[:view]} ranks=#{result[:ranks]} " \
         "rank_pngs=#{result[:rank_pngs]} image_pngs=#{result[:image_pngs]}"
  end
end
