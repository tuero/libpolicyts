# Agent Notes for libpolicyts
# This file is for coding agents working in this repo.

## Scope
- Language: C++23 (library + example executables).
- Build system: CMake with presets, vcpkg toolchain, optional libtorch.
- No Cursor or Copilot rules found in .cursor/rules/, .cursorrules, or .github/copilot-instructions.md.

## Quick Orientation
- Public headers: include/libpolicyts/**
- Sources: src/**
- Examples: examples/** (bfs, levints, lubyts, multits, phs)
- Formatting: .clang-format (Google-based, 4 spaces, 120 columns)

## Build Commands (CMake Presets)
### Configure + Build
- Configure release: `cmake --preset=release`
- Build release: `cmake --build --preset=release -- -j8`
- Configure debug: `cmake --preset=debug`
- Build debug: `cmake --build --preset=debug -- -j8`

### Single Target Build (examples)
- Build a single example target:
  - `cmake --build --preset=release --target bfs`
  - `cmake --build --preset=debug --target levints`

### Environment/Toolchain Notes
- Presets expect vcpkg via `VCPKG_ROOT` and custom triplets/toolchain.
- Optional libtorch uses `LIBTORCH_ROOT` (see readme.md for how to set).
- Presets enable examples + torch by default (LIBPOLICYTS_BUILD_EXAMPLES/TORCH=ON).

## Lint / Format
- Formatting: `clang-format -i <file>` (uses .clang-format).
- No clang-tidy target is defined in CMake; lint is manual.

## Tests
- No automated test targets or CTest configuration in this repo.
- There is a `include/libpolicyts/test_runner.h` utility, but no test executable targets.
- Use example executables for smoke tests if needed.

## Code Style Guide
### File Layout
- Use include guards in headers with the pattern `LIBPTS_<PATH>_H_`.
- Top-of-file comments follow `// File: ...` and `// Description: ...`.
- Keep namespaces explicit, e.g. `namespace libpts::algorithm::bfs { ... }`.

### Formatting Rules (from .clang-format)
- Indent width: 4 spaces; tabs disabled.
- Column limit: 120.
- Braces: custom; no mandatory brace insertion.
- Allow single-line `if` without `else` only; short lambdas allowed.
- Pointer/reference alignment: right (`Type *ptr`, `Type &ref`).
- Include sorting: case-insensitive, grouped by category (libpolicyts, quoted, external, std).

### Includes and Ordering
- Prefer `#include <libpolicyts/...>` for project headers.
- Include order should follow clang-format categories:
  1) `<libpolicyts/...>`
  2) `"..."` local headers
  3) External `<...>` with extensions
  4) External library headers (catch2/boost/pybind/absl/spdlog)
  5) Standard `<...>` without extensions
- Rely on clang-format for regrouping and sorting.

### Types and Const-Correctness
- Use `std::size_t` for sizes and indices where appropriate.
- Use `const` references for non-trivial parameters.
- Use `constexpr` for compile-time constants.
- Prefer `auto` when type is verbose and obvious from RHS, but keep readability.
- Expose ownership explicitly with `std::unique_ptr`/`std::shared_ptr`.

### Naming Conventions
- Types: `PascalCase` (classes/structs/enums, e.g. `SearchInput`, `Status`).
- Functions/methods: `lower_snake_case` (e.g. `set_solution_trajectory`).
- Variables/fields: `lower_snake_case`.
- Constants: `UPPER_SNAKE_CASE` for compile-time constants.
- Namespaces: `lower_snake_case` (e.g. `libpts::clustering`).

### Error Handling and Logging
- Logging uses `spdlog` or `SPDLOG_*` macros.
- Fatal errors sometimes call `std::exit(1)` after logging.
- Some paths throw `std::logic_error` for invalid usage.
- Prefer logging a clear error before returning early or exiting.

### Control Flow and Performance
- Avoid unnecessary copies; use move where it clarifies ownership transfer.
- Use `std::views`/`std::ranges` for iteration when it improves clarity.
- Use `[[nodiscard]]` on accessors that should not be ignored.

### Templates and Concepts
- Concepts are used to constrain templates (`IsEnv`, `IsBFSModel`).
- Keep concept requirements explicit and documented inline with comments.

### Comments
- Add comments only for non-obvious logic or external API constraints.
- Prefer short, direct comments near the relevant code.

## Repository-Specific Tips
- `LIBPOLICYTS_BUILD_EXAMPLES=ON` implies `LIBPOLICYTS_BUILD_ENVIRONMENTS=ON`.
- If you enable environments or torch, missing dependencies are fatal at configure time.
- Use the presets to avoid missing vcpkg/toolchain configuration.

## When Adding New Code
- Keep new headers in `include/libpolicyts/...` and source in `src/...`.
- Add new public headers to `LIBPOLICYTS_PUBLIC_HEADERS` in CMakeLists.txt.
- Ensure any new example target is added under `examples/` and wired into CMake.

## Useful Commands Summary
- Release configure/build: `cmake --preset=release && cmake --build --preset=release -- -j8`
- Debug configure/build: `cmake --preset=debug && cmake --build --preset=debug -- -j8`
- Single example build: `cmake --build --preset=release --target bfs`
- Format: `clang-format -i include/libpolicyts/algorithm/bfs.h`

## Gaps / Unknowns
- No standard test runner or lint pipeline is defined; avoid inventing ones.
- If you add tests, consider adding CTest targets or a dedicated test CMakeLists.
