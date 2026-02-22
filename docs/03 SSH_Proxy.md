## ssh proxy 配置：

编辑 `~/.ssh/config` 文件，在开发机的 Host 配置中添加 `RemoteForward` 指令：

```text
Host dev-server
    HostName 8.140.242.203
    User root
    Port 45560
    RemoteForward 127.0.0.1:7890 127.0.0.1:7890  
    ExitOnForwardFailure yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

**参数说明：**

-   `RemoteForward 127.0.0.1:7890 127.0.0.1:7890`
    -   第一个 `127.0.0.1:7890`：远程主机上监听的地址和端口
    -   第二个 `127.0.0.1:7890`：本地机器上服务的地址和端口
-   `ExitOnForwardFailure yes`：如果端口转发失败，SSH 连接直接退出，避免静默失败

### 验证端口转发

配置完成后，需要验证远程主机是否能正常访问本地代理服务。

#### Step 1 确保本地代理服务正在运行

在本地机器上确认代理服务已启动并监听在 `127.0.0.1:7897`：

```bash
# 检查本地端口是否在监听
lsof -i :7890
# 或
netstat -an | grep 7890
```

#### Step 2 在远程主机上测试连接

连接到开发机后，在远程终端中测试是否能访问 `localhost:7890`：

```bash
# 测试能否连接Google
curl -x http://127.0.0.1:7890 https://www.google.com 

# 设置环境变量代理加速
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 取消代理
unset http_proxy
unset https_proxy
```
