# Codex CLI 基础

看到 `zxd@zhou:...$` 时在 Ubuntu 终端：输入 `codex` 启动 Codex，输入 `codex resume` 选择并恢复聊天。进入 Codex 后，才使用 `/new`、`/resume` 等斜杠命令。

| Codex 内命令 | 用途 |
| --- | --- |
| `/new` / `/resume` / `/rename` | 新建、恢复、命名聊天 |
| `/status` / `/permissions` | 查看聊天状态、调整审批方式 |
| `/diff` / `/exit` | 检查改动、退出到终端 |
| `@文件名` | 将工作区文件加入当前问题 |

菜单操作：方向键选择，`Enter` 确认，`Esc` 返回，输入文字筛选。

项目约定：主聊天使用 `robot-learning-main`；完整实验可单开聊天；保持 **Workspace (Ask for approval)**；不使用 Danger Full Access 或 `yolo`；修改后先用 `/diff` 检查。

`/fork`、`/compact`、`/apps`、`/plugins` 暂不需要掌握，实际需要时再学。
