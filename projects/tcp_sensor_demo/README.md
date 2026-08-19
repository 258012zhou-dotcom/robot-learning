# TCP Sensor Demo

使用本机 TCP 连接发送一条 JSON 位置样本，并接收服务端确认。

## 运行

先在第一个终端启动服务端：

```bash
python projects/tcp_sensor_demo/server.py
```

在另一个终端运行客户端：

```bash
python projects/tcp_sensor_demo/client.py
```

客户端应收到包含 `status: ok` 和相同 `sequence_id` 的响应。

## 边界与限制

服务只绑定 `127.0.0.1:50007`，不会直接监听局域网接口。换行符用于划分 JSON 消息边界。
当前示例一次只处理一个客户端和一条消息，没有实现认证、重试、超时恢复或持续数据流。
