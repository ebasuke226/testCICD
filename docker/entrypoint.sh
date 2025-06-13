#!/bin/bash

# 権限を修正
mkdir -p /home/jovyan/.local/share
chown -R jovyan:users /home/jovyan/.local
chmod -R 755 /home/jovyan/.local

# 元のコマンドを実行
exec "$@" 