# TCP Sensor Demo

使用本机 TCP 连接发送一条 JSON 位置样本，并接收服务端确认。

运行服务端：

`python projects/tcp_sensor_demo/server.py`

在另一个终端运行客户端：

`python projects/tcp_sensor_demo/client.py`

服务只绑定 `127.0.0.1:50007`，不会直接监听局域网接口。换行符用于划分 JSON 消息边界。