# Linux 进程、设备与网络环境

## 快速复习

- **进程**有 PID、环境变量和标准输入输出。优先正常退出、`Ctrl+C` 或 `SIGTERM`；`SIGKILL` 仅作最后手段。
- **设备**常以 `/dev` 下的文件出现。无法访问摄像头或串口时，先检查权限和 `video`、`dialout` 用户组，而不是直接使用 root。
- **网络端点**由协议、IP 和端口组成。`127.0.0.1` 只代表当前环境自身，WSL、Windows、容器和远程机器人不一定共享 localhost。

## 项目中的用法

实验 003 的线程仍属于同一进程；`projects/python_concurrency_demo/` 的多进程示例会产生独立 PID。TCP 传感器演示把服务端限制在 `127.0.0.1:50007`，避免默认暴露到局域网。

网络发送的是字节流而不是天然的“消息”；因此 TCP 演示用换行符划分一条 JSON 位置样本。详见[网络与 TCP 传感器基础](network-tcp-sensor-basics.md)。
