# Git 基础

Git 的路径是：工作区（正在改）→ 暂存区（`git add` 选中）→ 本地仓库（`git commit` 保存）→ 远程仓库（`git push` 上传）。

常用检查顺序：

```text
修改 → git status → git diff → git add → git diff --staged → git commit
```

项目中先用 `git diff` 确认改动范围；只有测试或检查通过后才建议提交。`git add` 只选择本次任务的文件，避免把无关改动带入提交。

`git status --short` 用两列状态区分暂存区和工作区：左侧表示已暂存，右侧表示尚未暂存。提交前应特别检查未跟踪文件，避免遗漏应提交的源码或误加入生成物。
