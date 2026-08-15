# Git 基础

## Git 的四个位置

1. 工作区：正在编辑的实际文件
2. 暂存区：通过 `git add` 选入下一次提交的内容
3. 本地仓库：通过 `git commit` 保存的版本历史
4. 远程仓库：通过 `git push` 上传到 GitHub 等服务的提交

## 基本工作流程

```text
修改文件
→ git status
→ git diff
→ git add
→ git diff --staged
→ git commit
→ git status