#!/bin/bash
set -euo pipefail

CSV_FILE=${1:-users.csv}

if [[ ! -f "$CSV_FILE" ]]; then
    echo "[ERROR] CSV not found: $CSV_FILE"
    exit 1
fi

echo "[INFO] Using CSV: $CSV_FILE"

while IFS=',' read -r username pubkey; do

    # 去掉 Windows 回车符
    username="${username//$'\r'/}"
    pubkey="${pubkey//$'\r'/}"

    # 跳过空行 / 注释行
    [[ -z "$username" ]] && continue
    [[ "$username" =~ ^# ]] && continue

    echo "[INFO] Processing user: $username"

    # 1. 创建用户（幂等）
    if id "$username" &>/dev/null; then
        echo "[WARN] user exists: $username"
    else
        adduser --disabled-password --gecos "" "$username" > /dev/null
        usermod -s /bin/bash "$username"
        echo "[OK] user created: $username"
    fi

    home_dir=$(eval echo "~$username")
    ssh_dir="$home_dir/.ssh"
    auth_file="$ssh_dir/authorized_keys"

    # 2. 创建 ssh 目录
    mkdir -p "$ssh_dir"

    # 3. 写入 key（覆盖式）
    echo "$pubkey" > "$auth_file"

    # 4. 修正权限（SSH 强制要求）
    chown -R "$username:$username" "$home_dir"
    chmod 700 "$ssh_dir"
    chmod 600 "$auth_file"

    echo "[OK] configured: $username"

done < "$CSV_FILE"

echo "[DONE] All users processed."
