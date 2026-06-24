# 大模型部署

## 部署模型

下载 vLLM 和 huggingface-hub：

```bash
uv pip install vllm --torch-backend=cu124
uv pip install huggingface-hub
```

配置 huggingface 镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

下载模型：

```bash
hf download Qwen/Qwen3-32B
hf download Qwen/Qwen3-Embedding-4B
```

定期修改本地的 `api_key` 内容以避免被滥用，可以使用 [随机字符串生成器](https://www.random.org/strings/)：

```bash
nano ./api_key
```

格式一般为 `sk-xxxx`，即 `sk-` 开头，后面跟随机字符串

通过 tmux 开启会话后，通过 `./run_vllm.sh` 脚本部署模型：

```bash
# ./run_vllm.sh [model] <port>
# 默认模型: Qwen/Qwen3-32B
# 默认端口: 11800

./run_vllm.sh Qwen/Qwen3-32B 11800
./run_vllm.sh Qwen/Qwen3-Embedding-4B 11801
```

成功运行后通过 Ctrl + B + D 退出 tmux 会话，模型将继续运行

## 端口转发

由于 BDMI52 服务器无法直接访问外网，需要通过 BDMI66 服务器进行端口转发

> 风险：如果端口扫描到 BDMI66 服务器的端口，可能会被 ITS 发现并封禁 BDMI66，建议优先处理 BDMI52 的网络问题

在 `~/.ssh/config` 中添加以下内容：

```bash
Host bdmi66
    HostName 101.6.160.66
    User zyx
```

并将本地的 `~/.ssh/id_rsa.pub` 公钥添加到远程服务器的 `~/.ssh/authorized_keys` 文件中

之后运行 `./forward.sh` 将本地端口转发到远程服务器的**相同端口**上：

```bash
./forward.sh start 11800
./forward.sh start 11801
```

停止转发：

```bash
./forward.sh stop 11800
./forward.sh stop 11801
```

查看转发状态：

```bash
./forward.sh status 11800
./forward.sh status 11801
```

## 测试

进入 `./test` 中

```bash
cd ./test
```

安装依赖：

```bash
uv pip install -r requirements.txt
```

修改 `keyset.yml` 

```bash
candidates:
  - "Qwen/Qwen3-32B" # 待测试的模型
api_base: "http://101.6.160.66:11800/v1" # BDMI66 服务器的转发地址
api_key: "sk-xxxx" # 本地部署使用的 api_key
```

保存后运行测试：

```bash
python ./main.py
```
