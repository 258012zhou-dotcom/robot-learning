# Git 远程仓库基础

`origin` 是远程仓库的常用别名；本地提交先保存在本机，`git push` 才上传到 GitHub 等远程位置。

项目关系：本地 Git 历史用于回退和检查，远程仓库用于备份、展示和协作。HTTPS 推送使用 Personal Access Token，不使用账户密码；令牌不能写入代码或笔记。

`git fetch` 只获取远程历史，`git pull` 获取后再整合到当前分支，`git push` 上传本地提交。同步前先检查当前分支和工作区，避免把未完成改动混入同步过程。
