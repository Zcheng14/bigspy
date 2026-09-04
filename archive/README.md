# archive/ — 优化前代码快照

本目录存放 bigspy 性能优化前的**全包逐字节快照**，供 `benchmarks/benchmark_*.py`
做 old-vs-new 对照（计时 + parity 断言）。长期保留，不随清理删除。

## 版本命名约定

`v<N>_<commit-short-hash>` —— 优化轮次编号 + 快照对应的 git commit 短哈希。

| 目录 | 对应 commit | 说明 |
|---|---|---|
| `v0_6d70457/` | `6d70457`（"Add synthetic-data test suite (106 tests); fix 6 bugs they exposed"） | 第一轮性能优化（P1–P6）之前的基准状态 |

快照为 `src/bigspy` 的完整拷贝（含 `__init__.py`，相对导入在包内自洽），通过
`import archive.v0_6d70457.bigspy...` 使用（需仓库根在 `sys.path`，benchmark 脚本已处理）。
对应的 benchmark 报告见 `benchmarks/benchmark_report.md`。
