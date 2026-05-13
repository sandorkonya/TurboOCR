#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/grpc_response_mode.h"
#include "turbo_ocr/server/language_paths.h"

#include <CLI11.hpp>

namespace turbo_ocr::server {

enum class Profile { Gpu, Cpu };

/// Build-time default profile derived from USE_CPU_ONLY. Lets the same
/// from_env() call site work in either binary without hard-coding.
constexpr Profile build_profile() noexcept {
#ifdef USE_CPU_ONLY
  return Profile::Cpu;
#else
  return Profile::Gpu;
#endif
}

/// Centralized server configuration. Owns every env-var and CLI-flag read
/// for the HTTP/gRPC servers. Loaded once at startup via load_or_die().
struct ServerConfig {
  // ---- Network ----
  std::string host = "0.0.0.0";  // TURBO_OCR_HOST / --host
  int http_port = 8080;          // PORT / --http-port
  int grpc_port = 50051;         // GRPC_PORT / --grpc-port

  // ---- Body limits ----
  int max_body_mb     = 100;     // MAX_BODY_MB / --max-body-mb
  int max_body_mem_mb = 1024;    // MAX_BODY_MEMORY_MB / --max-body-memory-mb

  // ---- Pipeline / threading ----
  // `pipeline_pool_size` and `http_threads` are deliberately `std::optional`:
  // `nullopt` means "operator did not set this; caller chooses the default."
  // GPU main.cpp picks via VRAM auto-detect when pool size is unset; CPU
  // cpu_main.cpp uses a hard-coded 4. http_threads defaults to
  // `max(pool_size*32, 128)` on GPU and isn't read on CPU.
  std::optional<int> pipeline_pool_size;  // PIPELINE_POOL_SIZE / --pool-size
  std::optional<int> http_threads;        // HTTP_THREADS / --http-threads (GPU only consumer)
  int pdf_daemons            = 16;        // PDF_DAEMONS / --pdf-daemons
  int pdf_workers            = 4;         // PDF_WORKERS / --pdf-workers
  int shutdown_grace_seconds = 30;        // SHUTDOWN_GRACE_SECONDS / --shutdown-grace

  // ---- gRPC tuning ----
  int grpc_cqs           = 10;            // GRPC_CQS / --grpc-cqs
  int grpc_batch_workers = 8;             // GRPC_BATCH_WORKERS / --grpc-batch-workers
  int max_pdf_pages      = 2000;          // MAX_PDF_PAGES / --max-pdf-pages
  GrpcResponseMode grpc_response_mode = GrpcResponseMode::json_bytes;

  // ---- Model paths ----
  std::string det_onnx;
  std::string cls_onnx;
  std::string layout_onnx = "models/layout/layout.onnx";
  std::optional<std::string> layout_trt;
  RecPaths    rec_paths;
  std::string ocr_lang_value;

  // ---- TensorRT / decode tuning (consumed by engine + decode subsystems;
  //      strict-validated here so malformed input fails fast at boot) ----
  int         det_max_side    = 960;   // DET_MAX_SIDE [32, 4096]
  int         trt_opt_level   = 5;     // TRT_OPT_LEVEL [0, 5]
  std::string trt_engine_cache;        // TRT_ENGINE_CACHE (empty = "~/.cache/turbo-ocr")
  int         max_image_dim   = 16384; // MAX_IMAGE_DIM [64, 65535]

  // ---- Logging ----
  std::string log_level  = "info";     // LOG_LEVEL {debug, info, warn, error}
  std::string log_format = "json";     // LOG_FORMAT {json, text}

  // ---- Feature toggles ----
  bool        disable_angle_cls = false;
  bool        layout_disabled   = false;
  pdf::PdfMode default_pdf_mode = pdf::PdfMode::Ocr;
  bool        default_pdf_mode_was_set = false;

  /// Effective profile this config was loaded for. Set by from_env.
  Profile profile = build_profile();

  /// Errors accumulated during parsing or cross-field validation. If
  /// non-empty after load, load_or_die() prints them and exit(2).
  std::vector<std::string> errors;

  /// Warnings — recorded but not fatal. Logged at startup.
  std::vector<std::string> warnings;

  /// Parse env vars + CLI flags. CLI overrides env; both override defaults.
  /// Returns a ServerConfig with .errors populated for any malformed input.
  /// Pass argc=0, argv=nullptr for env-only loading.
  static ServerConfig from_env_and_cli(int argc, char **argv,
                                        Profile p = build_profile());

  /// Same as from_env_and_cli but env-only (for tests and library use).
  static ServerConfig from_env(Profile p = build_profile()) {
    return from_env_and_cli(0, nullptr, p);
  }

  /// Load, validate, and exit(2) with a diagnostic list on any error.
  /// Also handles --print-config / --check-config (those exit(0)).
  static ServerConfig load_or_die(int argc, char **argv,
                                   Profile p = build_profile());

  /// Emit one structured INFO log line with every resolved value, so
  /// operators can grep a single post-mortem source of truth.
  void log_effective() const;

  /// JSON dump used by both --print-config and log_effective().
  [[nodiscard]] std::string to_json() const;
};

namespace detail {

inline std::string_view grpc_mode_str(GrpcResponseMode m) noexcept {
  return m == GrpcResponseMode::structured ? "structured" : "json_bytes";
}

inline std::string_view profile_str(Profile p) noexcept {
  return p == Profile::Cpu ? "cpu" : "gpu";
}

/// Minimal JSON string escape — keeps quotes and backslashes safe.
inline std::string esc(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  out += '"';
  for (char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else out += c;
  }
  out += '"';
  return out;
}

/// Render an optional as a JSON value: `null` when empty, otherwise the
/// underlying value. Strings are escaped through esc() so quotes and
/// backslashes can't corrupt the output.
template <typename T>
std::string opt_json(const std::optional<T> &v) {
  if (!v) return "null";
  if constexpr (std::is_same_v<T, std::string>)
    return esc(*v);
  else
    return std::to_string(*v);
}

/// Cross-field validation shared by both the env-only and CLI-included
/// loader paths. Errors land in `c.errors` (fatal at load_or_die time);
/// warnings land in `c.warnings` (advisory, surfaced by log_effective).
/// The mem-cap clamp also runs here so the field is invariant by the
/// time the caller reads it.
inline void cross_field_validate(ServerConfig &c, bool mem_explicit) {
  if (c.http_port == c.grpc_port)
    c.errors.push_back("PORT and GRPC_PORT must differ (both = " +
                       std::to_string(c.http_port) + ")");
  if (c.pdf_workers > c.pdf_daemons)
    c.warnings.push_back("PDF_WORKERS (" + std::to_string(c.pdf_workers) +
                         ") exceeds PDF_DAEMONS (" + std::to_string(c.pdf_daemons) +
                         "); excess workers will sit idle");
  if (c.max_body_mem_mb > c.max_body_mb) {
    if (mem_explicit)
      c.warnings.push_back("MAX_BODY_MEMORY_MB (" + std::to_string(c.max_body_mem_mb) +
                           ") > MAX_BODY_MB (" + std::to_string(c.max_body_mb) +
                           "); clamping to body cap");
    c.max_body_mem_mb = c.max_body_mb;
  }
}

} // namespace detail

inline std::string ServerConfig::to_json() const {
  using detail::esc;
  using detail::opt_json;
  std::string j = "{";
  j += "\"profile\":"            + esc(detail::profile_str(profile));
  j += ",\"host\":"              + esc(host);
  j += ",\"http_port\":"         + std::to_string(http_port);
  j += ",\"grpc_port\":"         + std::to_string(grpc_port);
  j += ",\"max_body_mb\":"       + std::to_string(max_body_mb);
  j += ",\"max_body_mem_mb\":"   + std::to_string(max_body_mem_mb);
  j += ",\"pipeline_pool_size\":" + opt_json(pipeline_pool_size);
  j += ",\"http_threads\":"      + opt_json(http_threads);
  j += ",\"pdf_daemons\":"       + std::to_string(pdf_daemons);
  j += ",\"pdf_workers\":"       + std::to_string(pdf_workers);
  j += ",\"shutdown_grace_seconds\":" + std::to_string(shutdown_grace_seconds);
  j += ",\"grpc_cqs\":"          + std::to_string(grpc_cqs);
  j += ",\"grpc_batch_workers\":" + std::to_string(grpc_batch_workers);
  j += ",\"max_pdf_pages\":"     + std::to_string(max_pdf_pages);
  j += ",\"grpc_response_mode\":" + esc(detail::grpc_mode_str(grpc_response_mode));
  j += ",\"det_onnx\":"          + esc(det_onnx);
  j += ",\"cls_onnx\":"          + esc(cls_onnx);
  j += ",\"layout_onnx\":"       + esc(layout_onnx);
  j += ",\"layout_trt\":"        + (layout_trt ? esc(*layout_trt) : std::string("null"));
  j += ",\"rec\":"               + esc(rec_paths.rec);
  j += ",\"rec_dict\":"          + esc(rec_paths.dict);
  j += ",\"ocr_lang\":"          + esc(ocr_lang_value);
  j += ",\"disable_angle_cls\":" + std::string(disable_angle_cls ? "true" : "false");
  j += ",\"layout_disabled\":"   + std::string(layout_disabled ? "true" : "false");
  j += ",\"default_pdf_mode\":"  + esc(pdf::mode_name(default_pdf_mode));
  j += ",\"det_max_side\":"      + std::to_string(det_max_side);
  j += ",\"trt_opt_level\":"     + std::to_string(trt_opt_level);
  j += ",\"trt_engine_cache\":"  + esc(trt_engine_cache);
  j += ",\"max_image_dim\":"     + std::to_string(max_image_dim);
  j += ",\"log_level\":"         + esc(log_level);
  j += ",\"log_format\":"        + esc(log_format);
  j += "}";
  return j;
}

inline void ServerConfig::log_effective() const {
  TOCR_LOG_INFO("Effective server config", "config", std::string_view(to_json()));
  for (const auto &w : warnings)
    TOCR_LOG_WARN("Config warning", "detail", std::string_view(w));
}

inline ServerConfig ServerConfig::from_env_and_cli(int argc, char **argv,
                                                    Profile p) {
  ServerConfig c;
  c.profile = p;

  const bool is_gpu = (p == Profile::Gpu);
  const char *rec_env = is_gpu ? "REC_ONNX" : "REC_MODEL";
  const char *det_env = is_gpu ? "DET_ONNX" : "DET_MODEL";
  const char *cls_env = is_gpu ? "CLS_ONNX" : "CLS_MODEL";

  // ---- Pass 1: load from env, accumulating parse errors ----
  c.host      = env_or("TURBO_OCR_HOST", "0.0.0.0");
  c.http_port = env_int_strict("PORT",      8080,  1, 65535, c.errors);
  c.grpc_port = env_int_strict("GRPC_PORT", 50051, 1, 65535, c.errors);

  c.max_body_mb     = env_int_strict("MAX_BODY_MB",        100,  1, 102400, c.errors);
  c.max_body_mem_mb = env_int_strict("MAX_BODY_MEMORY_MB", 1024, 1, 102400, c.errors);
  // Default of 1024 MB is intentionally a soft ceiling — bodies up to
  // MAX_BODY_MB stay in RAM. Only warn when the operator explicitly raised
  // the in-memory cap above the body cap (likely a misconfiguration).
  const bool mem_explicit = env_present("MAX_BODY_MEMORY_MB");

  if (env_present("PIPELINE_POOL_SIZE"))
    c.pipeline_pool_size = env_int_strict("PIPELINE_POOL_SIZE", 1, 1, 4096, c.errors);
  if (env_present("HTTP_THREADS"))
    c.http_threads = env_int_strict("HTTP_THREADS", 1, 1, 4096, c.errors);
  c.pdf_daemons = env_int_strict("PDF_DAEMONS", is_gpu ? 16 : 4, 1, 1024, c.errors);
  c.pdf_workers = env_int_strict("PDF_WORKERS", is_gpu ? 4  : 2, 1, 1024, c.errors);
  c.shutdown_grace_seconds = env_int_strict("SHUTDOWN_GRACE_SECONDS", 30, 0, 600, c.errors);

  c.grpc_cqs           = env_int_strict("GRPC_CQS", 10, 1, 1024, c.errors);
  c.grpc_batch_workers = env_int_strict("GRPC_BATCH_WORKERS", 8, 1, 256, c.errors);
  c.max_pdf_pages      = env_int_strict("MAX_PDF_PAGES", 2000, 1, 100000, c.errors);

  {
    auto mode_s = env_choice_strict("GRPC_RESPONSE_MODE", "json_bytes",
                                     {"json_bytes", "structured"}, c.errors);
    c.grpc_response_mode = (mode_s == "structured")
        ? GrpcResponseMode::structured : GrpcResponseMode::json_bytes;
  }

  c.det_onnx    = env_or(det_env, "models/det.onnx");
  c.cls_onnx    = env_or(cls_env, "models/cls.onnx");
  c.layout_onnx = env_or("LAYOUT_ONNX", "models/layout/layout.onnx");
  if (env_present("LAYOUT_TRT"))
    c.layout_trt = env_or("LAYOUT_TRT", "");
  c.rec_paths      = resolve_rec_paths(rec_env);
  c.ocr_lang_value = ocr_lang();

  // ---- TensorRT / decode / logging knobs (validated here, consumed by
  //      other subsystems via the same env vars they always read) ----
  c.det_max_side       = env_int_strict("DET_MAX_SIDE",  960,   32, 4096,  c.errors);
  c.trt_opt_level      = env_int_strict("TRT_OPT_LEVEL", 5,     0,  5,     c.errors);
  c.trt_engine_cache   = env_or("TRT_ENGINE_CACHE", "");
  c.max_image_dim      = env_int_strict("MAX_IMAGE_DIM", 16384, 64, 65535, c.errors);
  c.log_level  = env_choice_strict("LOG_LEVEL",  "info",
      {"debug", "info", "warn", "error"}, c.errors);
  c.log_format = env_choice_strict("LOG_FORMAT", "json",
      {"json", "text"}, c.errors);

  c.disable_angle_cls = env_bool_strict("DISABLE_ANGLE_CLS", false, c.errors);
  c.layout_disabled   = env_bool_strict("DISABLE_LAYOUT",    false, c.errors);
  // ENABLE_LAYOUT removed — operators must migrate.
  if (env_present("ENABLE_LAYOUT")) {
    c.errors.push_back(
        "ENABLE_LAYOUT is no longer supported. Set DISABLE_LAYOUT=1 to "
        "disable layout, or remove this env var (layout is on by default).");
  }

  if (env_present("ENABLE_PDF_MODE")) {
    auto m = env_choice_strict("ENABLE_PDF_MODE", "ocr",
        {"ocr", "geometric", "auto", "auto_verified"}, c.errors);
    c.default_pdf_mode = pdf::parse_pdf_mode(m);
    c.default_pdf_mode_was_set = true;
  }

  // ---- Pass 2: CLI flags (override env) ----
  if (argc > 0 && argv) {
    CLI::App app{"TurboOCR server — GPU/CPU OCR + layout HTTP/gRPC service"};
    app.set_help_flag("-h,--help", "Print this help message and exit");
    bool flag_print_config = false;
    bool flag_check_config = false;
    app.add_flag("--print-config", flag_print_config,
                 "Print resolved configuration as JSON and exit (zero)");
    app.add_flag("--check-config", flag_check_config,
                 "Validate configuration and exit (zero on valid, 2 on errors)");

    app.add_option("--host",        c.host,        "Bind address for HTTP and gRPC")->capture_default_str();
    app.add_option("--http-port",   c.http_port,   "HTTP port")->capture_default_str()->check(CLI::Range(1, 65535));
    app.add_option("--grpc-port",   c.grpc_port,   "gRPC port")->capture_default_str()->check(CLI::Range(1, 65535));
    app.add_option("--max-body-mb", c.max_body_mb, "Max request body size (MB)")->capture_default_str()->check(CLI::Range(1, 102400));
    app.add_option("--max-body-memory-mb", c.max_body_mem_mb,
        "In-memory body buffer cap (MB); always clamped to --max-body-mb so effective default is min(1024, MAX_BODY_MB)")
        ->check(CLI::Range(1, 102400));

    int pool_size_cli = c.pipeline_pool_size.value_or(0);
    int http_threads_cli = c.http_threads.value_or(0);
    auto *opt_pool = app.add_option("--pool-size", pool_size_cli,
        "Pipeline pool size (0 = auto from VRAM on GPU / 4 on CPU)")->check(CLI::Range(0, 4096));
    auto *opt_http = app.add_option("--http-threads", http_threads_cli,
        "HTTP work pool threads (0 = auto from pool size)")->check(CLI::Range(0, 4096));
    app.add_option("--pdf-daemons",  c.pdf_daemons,  "PDF render daemons")->capture_default_str()->check(CLI::Range(1, 1024));
    app.add_option("--pdf-workers",  c.pdf_workers,  "PDF render workers")->capture_default_str()->check(CLI::Range(1, 1024));
    app.add_option("--shutdown-grace", c.shutdown_grace_seconds, "Graceful drain seconds before exit")->capture_default_str()->check(CLI::Range(0, 600));
    app.add_option("--grpc-cqs",            c.grpc_cqs,            "gRPC completion queue count")->capture_default_str()->check(CLI::Range(1, 1024));
    app.add_option("--grpc-batch-workers",  c.grpc_batch_workers,  "Parallel workers in gRPC RecognizeBatch")->capture_default_str()->check(CLI::Range(1, 256));
    app.add_option("--max-pdf-pages",       c.max_pdf_pages,       "Max pages per PDF request")->capture_default_str()->check(CLI::Range(1, 100000));

    std::string grpc_mode_cli = (c.grpc_response_mode == GrpcResponseMode::structured)
                                  ? "structured" : "json_bytes";
    app.add_option("--grpc-response-mode", grpc_mode_cli, "gRPC response mode")
        ->capture_default_str()
        ->check(CLI::IsMember({"json_bytes", "structured"}));

    app.add_option("--det-onnx",    c.det_onnx,    "Detection model ONNX path")->capture_default_str();
    app.add_option("--cls-onnx",    c.cls_onnx,    "Angle-classification model ONNX path")->capture_default_str();
    app.add_option("--layout-onnx", c.layout_onnx, "Layout-detection model ONNX path")->capture_default_str();
    std::string layout_trt_cli = c.layout_trt.value_or("");
    auto *opt_layout_trt = app.add_option("--layout-trt", layout_trt_cli,
        "Pre-built layout TRT engine (GPU only; overrides --layout-onnx build)");

    app.add_option("--det-max-side",      c.det_max_side,    "Max detection input side (px); changes invalidate cached TRT engine")
        ->capture_default_str()->check(CLI::Range(32, 4096));
    app.add_option("--trt-opt-level",     c.trt_opt_level,   "TensorRT builder optimization level (0=fast build / 5=fast runtime)")
        ->capture_default_str()->check(CLI::Range(0, 5));
    app.add_option("--trt-engine-cache",  c.trt_engine_cache,"Directory for cached TensorRT engines (empty = ~/.cache/turbo-ocr)")
        ->capture_default_str();
    app.add_option("--max-image-dim",     c.max_image_dim,   "Max image width/height (px) accepted on decode routes")
        ->capture_default_str()->check(CLI::Range(64, 65535));
    app.add_option("--log-level",         c.log_level,       "Log level")
        ->capture_default_str()->check(CLI::IsMember({"debug", "info", "warn", "error"}));
    app.add_option("--log-format",        c.log_format,      "Log output format")
        ->capture_default_str()->check(CLI::IsMember({"json", "text"}));

    app.add_flag("--disable-angle-cls", c.disable_angle_cls, "Skip angle classifier");
    app.add_flag("--disable-layout",    c.layout_disabled,   "Skip layout detection");

    std::string pdf_mode_cli = std::string(pdf::mode_name(c.default_pdf_mode));
    auto *opt_pdf_mode = app.add_option("--default-pdf-mode", pdf_mode_cli, "Default PDF extraction mode")
        ->capture_default_str()
        ->check(CLI::IsMember({"ocr", "geometric", "auto", "auto_verified"}));

    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      // --help and --version exit 0 here; anything else is a hard error.
      std::exit(app.exit(e));
    }

    // Reflect CLI overrides back onto the optional / enum fields.
    if (opt_pool->count() > 0)
      c.pipeline_pool_size = pool_size_cli > 0 ? std::optional<int>(pool_size_cli) : std::nullopt;
    if (opt_http->count() > 0)
      c.http_threads = http_threads_cli > 0 ? std::optional<int>(http_threads_cli) : std::nullopt;
    if (opt_layout_trt->count() > 0)
      c.layout_trt = layout_trt_cli.empty() ? std::nullopt : std::optional<std::string>(layout_trt_cli);
    c.grpc_response_mode = (grpc_mode_cli == "structured")
        ? GrpcResponseMode::structured : GrpcResponseMode::json_bytes;
    if (opt_pdf_mode->count() > 0) {
      c.default_pdf_mode = pdf::parse_pdf_mode(pdf_mode_cli);
      c.default_pdf_mode_was_set = true;
    }

    detail::cross_field_validate(c, mem_explicit);

    // ---- Print/check modes ----
    if (flag_print_config) {
      std::cout << c.to_json() << "\n";
      std::exit(0);
    }
    if (flag_check_config) {
      if (c.errors.empty()) {
        std::cerr << "config OK\n";
        std::exit(0);
      }
      for (const auto &e : c.errors) std::cerr << "[config error] " << e << "\n";
      std::exit(2);
    }
  } else {
    detail::cross_field_validate(c, mem_explicit);
  }

  return c;
}

inline ServerConfig ServerConfig::load_or_die(int argc, char **argv,
                                               Profile p) {
  ServerConfig c = from_env_and_cli(argc, argv, p);
  if (!c.errors.empty()) {
    for (const auto &e : c.errors)
      std::cerr << "[config error] " << e << "\n";
    std::cerr << "Refusing to start with invalid configuration. "
                 "Use --check-config to validate without booting the pipeline.\n";
    std::exit(2);
  }
  return c;
}

} // namespace turbo_ocr::server
