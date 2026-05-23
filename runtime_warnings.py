"""Runtime warning filters for noisy third-party libraries used by PointBench."""

from __future__ import annotations

import warnings


def suppress_known_runtime_warnings() -> None:
    # 某些 Gemini 兼容服务会返回 google.genai SDK 不认识的枚举值，
    # SDK 会在解析时反复打印 UserWarning，但这不会影响当前评测流程。
    warnings.filterwarnings(
        "ignore",
        message=r".*is not a valid [A-Za-z_][A-Za-z0-9_]*",
        category=UserWarning,
        module=r"google\.genai\._common",
    )
