# Git 远程仓库基础

- `origin` 是本地给 GitHub 远程仓库起的默认名字。
- `main` 跟踪 `origin/main`，因此之后通常直接使用 `git push` 即可。
- 基本流程：修改文件 → `git add` → `git commit` → `git push`。
- GitHub 的网页仓库是远程备份和展示位置；提交首先发生在本地。
- HTTPS 推送使用 Personal Access Token，不使用 GitHub 登录密码；令牌绝不写进项目或笔记。
