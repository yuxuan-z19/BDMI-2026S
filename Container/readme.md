# 部署容器

替换 `id_rsa.pub`

构建容器镜像，配置 SSH 登录

```bash
docker build -t bdmi-ssh .
```

创建容器，通过本地的 `12345` 端口 SSH 登录进容器中

```bash
docker run -d -it \
    -p 12345:22 \
    -e "TERM=xterm-256color" \
    --gpus all --privileged \
    --shm-size=128g \
    --name=bdmi26s bdmi-ssh
```

通过 `root` 登录后使用 `setup_users.sh` 读取 `users.csv` 、批量创建用户并设置 `~/.ssh/authorized_keys`

```bash
cd /root/operate

# 设置用户名和SSH公钥
cp users.csv.template users.csv
nano users.csv

# 批量创建用户
bash setup_users.sh users.csv
```
