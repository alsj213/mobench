# 安全策略（Security Policy）

## 支持的版本

| 版本 | 是否提供安全更新 |
|------|------------------|
| 0.9.x | ✅ 是 |
| < 0.9.0 | ❌ 否 |

## 报告安全漏洞

如果你在 **mobench** 中发现了安全问题（例如命令注入、路径穿越、不安全的配置解析、ADB/子进程权限边界缺陷或敏感信息泄露等），**请勿直接提交公开 Issue**。

请通过 GitHub 的 [Private Vulnerability Reporting](https://github.com/alsj213/mobench/security/advisories/new) 功能私密提交报告，并尽量提供：

- 受影响的版本号及运行环境（OS / Python 版本 / 平台类型）
- 复现步骤或最小 PoC
- 潜在影响范围及（如有）修复建议

维护者会在收到报告后的 72 小时内确认受理，并在修复完成后同步披露致谢。
