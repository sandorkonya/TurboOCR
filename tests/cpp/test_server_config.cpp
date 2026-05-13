#include <catch_amalgamated.hpp>

#include <cstdlib>

#include "turbo_ocr/server/server_config.h"

using turbo_ocr::server::Profile;
using turbo_ocr::server::ServerConfig;

namespace {

// Every env var ServerConfig::from_env touches — wiped between cases so each
// test starts from a known baseline.
const char *const kAllEnvVars[] = {
    "TURBO_OCR_HOST", "PORT", "GRPC_PORT",
    "MAX_BODY_MB", "MAX_BODY_MEMORY_MB",
    "PIPELINE_POOL_SIZE", "HTTP_THREADS",
    "PDF_DAEMONS", "PDF_WORKERS", "SHUTDOWN_GRACE_SECONDS",
    "GRPC_CQS", "GRPC_BATCH_WORKERS", "MAX_PDF_PAGES", "GRPC_RESPONSE_MODE",
    "DET_ONNX", "DET_MODEL", "CLS_ONNX", "CLS_MODEL",
    "LAYOUT_ONNX", "LAYOUT_TRT",
    "REC_ONNX", "REC_MODEL", "REC_DICT", "OCR_LANG",
    "DISABLE_ANGLE_CLS", "DISABLE_LAYOUT", "ENABLE_LAYOUT",
    "ENABLE_PDF_MODE",
    "DET_MAX_SIDE", "TRT_OPT_LEVEL", "TRT_ENGINE_CACHE", "MAX_IMAGE_DIM",
    "LOG_LEVEL", "LOG_FORMAT",
};

void reset_env() {
  for (const char *v : kAllEnvVars) ::unsetenv(v);
}

} // namespace

TEST_CASE("from_env defaults are sane (GPU)", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env(Profile::Gpu);
  CHECK(c.errors.empty());
  CHECK(c.host == "0.0.0.0");
  CHECK(c.http_port == 8080);
  CHECK(c.grpc_port == 50051);
  CHECK(c.max_body_mb == 100);
  // Default mem (1024) is silently clamped to body cap (100); no warning.
  CHECK(c.max_body_mem_mb == 100);
  CHECK(c.warnings.empty());
  CHECK_FALSE(c.pipeline_pool_size.has_value());
  CHECK_FALSE(c.http_threads.has_value());
  CHECK(c.pdf_daemons == 16);
  CHECK(c.pdf_workers == 4);
  CHECK(c.shutdown_grace_seconds == 30);
  CHECK(c.grpc_cqs == 10);
  CHECK(c.grpc_batch_workers == 8);
  CHECK(c.max_pdf_pages == 2000);
  CHECK(c.det_onnx == "models/det.onnx");
  CHECK(c.cls_onnx == "models/cls.onnx");
  CHECK_FALSE(c.disable_angle_cls);
  CHECK_FALSE(c.layout_disabled);
  CHECK_FALSE(c.default_pdf_mode_was_set);
  CHECK(c.profile == Profile::Gpu);
}

TEST_CASE("from_env defaults differ on CPU profile", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env(Profile::Cpu);
  CHECK(c.errors.empty());
  CHECK(c.pdf_daemons == 4);
  CHECK(c.pdf_workers == 2);
  CHECK(c.profile == Profile::Cpu);
}

TEST_CASE("from_env uses profile-specific model env names", "[server_config]") {
  reset_env();
  ::setenv("DET_ONNX",  "trt_det.onnx", 1);
  ::setenv("DET_MODEL", "cpu_det.onnx", 1);
  CHECK(ServerConfig::from_env(Profile::Gpu).det_onnx == "trt_det.onnx");
  CHECK(ServerConfig::from_env(Profile::Cpu).det_onnx == "cpu_det.onnx");
}

TEST_CASE("from_env accepts valid integer values", "[server_config]") {
  reset_env();
  ::setenv("PORT",      "9000",  1);
  ::setenv("GRPC_PORT", "50061", 1);
  ::setenv("MAX_BODY_MB", "250", 1);
  auto c = ServerConfig::from_env();
  CHECK(c.errors.empty());
  CHECK(c.http_port == 9000);
  CHECK(c.grpc_port == 50061);
  CHECK(c.max_body_mb == 250);
}

TEST_CASE("from_env rejects malformed integers", "[server_config]") {
  reset_env();
  ::setenv("PORT", "abc", 1);
  auto c = ServerConfig::from_env();
  REQUIRE(c.errors.size() == 1);
  CHECK(c.errors[0].find("PORT") != std::string::npos);
  CHECK(c.errors[0].find("abc")  != std::string::npos);
  // Falls back to default while reporting the error.
  CHECK(c.http_port == 8080);
}

TEST_CASE("from_env rejects out-of-range integers", "[server_config]") {
  reset_env();
  ::setenv("PORT", "70000", 1);
  auto c = ServerConfig::from_env();
  REQUIRE_FALSE(c.errors.empty());
  CHECK(c.errors[0].find("70000") != std::string::npos);
  CHECK(c.errors[0].find("[1, 65535]") != std::string::npos);
}

TEST_CASE("from_env accumulates multiple errors", "[server_config]") {
  reset_env();
  ::setenv("PORT",         "abc",   1);
  ::setenv("GRPC_PORT",    "xyz",   1);
  ::setenv("MAX_BODY_MB",  "9999999", 1);
  auto c = ServerConfig::from_env();
  CHECK(c.errors.size() == 3);
}

TEST_CASE("from_env optional vars stay nullopt when unset", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env();
  CHECK_FALSE(c.pipeline_pool_size.has_value());
  CHECK_FALSE(c.http_threads.has_value());
  CHECK_FALSE(c.layout_trt.has_value());
}

TEST_CASE("from_env optional vars populate when set", "[server_config]") {
  reset_env();
  ::setenv("PIPELINE_POOL_SIZE", "7",  1);
  ::setenv("HTTP_THREADS",       "64", 1);
  ::setenv("LAYOUT_TRT",         "/opt/layout.trt", 1);
  auto c = ServerConfig::from_env();
  REQUIRE(c.pipeline_pool_size.has_value());
  CHECK(*c.pipeline_pool_size == 7);
  REQUIRE(c.http_threads.has_value());
  CHECK(*c.http_threads == 64);
  REQUIRE(c.layout_trt.has_value());
  CHECK(*c.layout_trt == "/opt/layout.trt");
}

TEST_CASE("from_env strict bool parses many spellings", "[server_config]") {
  reset_env();
  for (const char *t : {"1", "true", "TRUE", "Yes", "on"}) {
    ::setenv("DISABLE_ANGLE_CLS", t, 1);
    auto c = ServerConfig::from_env();
    CHECK(c.disable_angle_cls);
    CHECK(c.errors.empty());
  }
  for (const char *f : {"0", "false", "FALSE", "no", "off"}) {
    ::setenv("DISABLE_ANGLE_CLS", f, 1);
    auto c = ServerConfig::from_env();
    CHECK_FALSE(c.disable_angle_cls);
    CHECK(c.errors.empty());
  }
}

TEST_CASE("from_env rejects malformed bool", "[server_config]") {
  reset_env();
  ::setenv("DISABLE_ANGLE_CLS", "maybe", 1);
  auto c = ServerConfig::from_env();
  REQUIRE_FALSE(c.errors.empty());
  CHECK(c.errors[0].find("DISABLE_ANGLE_CLS") != std::string::npos);
}

TEST_CASE("ENABLE_LAYOUT always errors with a migration message", "[server_config]") {
  // Any value of ENABLE_LAYOUT now fails fast — v2.2.x parsing was buggy
  // (only "1" did what the name said, every other value silently disabled
  // layout), so we force explicit migration rather than try to guess intent.
  for (const char *p : {"1", "0", "true", "false", "yes", "no", "on", "off", "maybe"}) {
    for (auto profile : {Profile::Gpu, Profile::Cpu}) {
      reset_env();
      ::setenv("ENABLE_LAYOUT", p, 1);
      auto c = ServerConfig::from_env(profile);
      REQUIRE_FALSE(c.errors.empty());
      bool found = false;
      for (const auto &e : c.errors)
        if (e.find("ENABLE_LAYOUT") != std::string::npos &&
            e.find("DISABLE_LAYOUT=1") != std::string::npos) found = true;
      CHECK(found);
    }
  }
  // Unset is clean — no error, no warning.
  reset_env();
  auto c = ServerConfig::from_env();
  CHECK(c.errors.empty());
  CHECK(c.warnings.empty());
}

TEST_CASE("ENABLE_PDF_MODE validates against allowed set", "[server_config]") {
  reset_env();
  ::setenv("ENABLE_PDF_MODE", "auto", 1);
  auto ok = ServerConfig::from_env();
  CHECK(ok.errors.empty());
  CHECK(ok.default_pdf_mode_was_set);
  CHECK(ok.default_pdf_mode == turbo_ocr::pdf::PdfMode::Auto);

  reset_env();
  ::setenv("ENABLE_PDF_MODE", "bogus", 1);
  auto bad = ServerConfig::from_env();
  REQUIRE_FALSE(bad.errors.empty());
  CHECK(bad.errors[0].find("ENABLE_PDF_MODE") != std::string::npos);
}

TEST_CASE("GRPC_RESPONSE_MODE validates", "[server_config]") {
  reset_env();
  ::setenv("GRPC_RESPONSE_MODE", "structured", 1);
  CHECK(ServerConfig::from_env().grpc_response_mode ==
        turbo_ocr::server::GrpcResponseMode::structured);

  reset_env();
  ::setenv("GRPC_RESPONSE_MODE", "json_typo", 1);
  auto bad = ServerConfig::from_env();
  REQUIRE_FALSE(bad.errors.empty());
}

TEST_CASE("cross-field: PORT == GRPC_PORT is fatal", "[server_config]") {
  reset_env();
  ::setenv("PORT",      "9000", 1);
  ::setenv("GRPC_PORT", "9000", 1);
  auto c = ServerConfig::from_env();
  REQUIRE_FALSE(c.errors.empty());
  bool found = false;
  for (const auto &e : c.errors)
    if (e.find("must differ") != std::string::npos) found = true;
  CHECK(found);
}

TEST_CASE("cross-field: PDF_WORKERS > PDF_DAEMONS is a warning", "[server_config]") {
  reset_env();
  ::setenv("PDF_DAEMONS", "2",  1);
  ::setenv("PDF_WORKERS", "16", 1);
  auto c = ServerConfig::from_env();
  CHECK(c.errors.empty());
  REQUIRE_FALSE(c.warnings.empty());
}

TEST_CASE("cross-field: MAX_BODY_MEMORY_MB > MAX_BODY_MB clamps + warns",
          "[server_config]") {
  reset_env();
  ::setenv("MAX_BODY_MB",        "50",   1);
  ::setenv("MAX_BODY_MEMORY_MB", "1024", 1);
  auto c = ServerConfig::from_env();
  CHECK(c.errors.empty());
  CHECK(c.max_body_mem_mb == 50);  // clamped to MAX_BODY_MB
  REQUIRE_FALSE(c.warnings.empty());
}

TEST_CASE("from_env strict-validates engine/decode/logging knobs", "[server_config]") {
  reset_env();
  auto def = ServerConfig::from_env();
  CHECK(def.det_max_side == 960);
  CHECK(def.trt_opt_level == 5);
  CHECK(def.max_image_dim == 16384);
  CHECK(def.log_level == "info");
  CHECK(def.log_format == "json");
  CHECK(def.errors.empty());

  reset_env();
  ::setenv("DET_MAX_SIDE", "abc", 1);
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());

  reset_env();
  ::setenv("DET_MAX_SIDE", "10", 1);  // below min 32
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());

  reset_env();
  ::setenv("TRT_OPT_LEVEL", "9", 1);  // above max 5
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());

  reset_env();
  ::setenv("LOG_LEVEL", "trace", 1);
  auto bad_log = ServerConfig::from_env();
  REQUIRE_FALSE(bad_log.errors.empty());

  reset_env();
  ::setenv("LOG_FORMAT", "xml", 1);
  auto bad_fmt = ServerConfig::from_env();
  REQUIRE_FALSE(bad_fmt.errors.empty());

  reset_env();
  ::setenv("MAX_IMAGE_DIM", "10", 1);  // below min 64
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());

  reset_env();
  ::setenv("DET_MAX_SIDE",  "2048", 1);
  ::setenv("TRT_OPT_LEVEL", "3",    1);
  ::setenv("MAX_IMAGE_DIM", "8192", 1);
  ::setenv("LOG_LEVEL",     "warn", 1);
  ::setenv("LOG_FORMAT",    "text", 1);
  auto ok = ServerConfig::from_env();
  CHECK(ok.errors.empty());
  CHECK(ok.det_max_side == 2048);
  CHECK(ok.trt_opt_level == 3);
  CHECK(ok.max_image_dim == 8192);
  CHECK(ok.log_level == "warn");
  CHECK(ok.log_format == "text");
}

TEST_CASE("opt_json escapes optional strings with embedded quotes", "[server_config]") {
  // Latent-bug guard: opt_json must escape strings, not just concatenate.
  reset_env();
  auto c = ServerConfig::from_env();
  c.layout_trt = std::string("/path/with\"quote.trt");
  auto j = c.to_json();
  // Should not contain raw unescaped quote inside the string value.
  CHECK(j.find("with\\\"quote") != std::string::npos);
}

TEST_CASE("to_json produces non-empty JSON object", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env();
  auto j = c.to_json();
  REQUIRE(j.size() > 2);
  CHECK(j.front() == '{');
  CHECK(j.back() == '}');
  CHECK(j.find("\"host\":\"0.0.0.0\"") != std::string::npos);
  CHECK(j.find("\"http_port\":8080") != std::string::npos);
}
