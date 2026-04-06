#!/bin/bash
# 自动检测变更并提交到 GitHub

REPO_DIR=/mnt/g/code/AI/Claude/watermark-remover

cd "$REPO_DIR" || exit 0

# 有未提交的变更才继续
if git diff --quiet && git diff --cached --quiet; then
  exit 0
fi

# 获取变更文件列表，自动生成 commit message
CHANGED=$(git diff --name-only && git diff --cached --name-only | sort -u)
FILE_COUNT=$(echo "$CHANGED" | grep -c .)

# 根据变更内容自动判断提交类型
if echo "$CHANGED" | grep -q "test"; then
  TYPE="test"
elif echo "$CHANGED" | grep -q "detector"; then
  TYPE="feat(detector)"
elif echo "$CHANGED" | grep -q "inpainter"; then
  TYPE="feat(inpainter)"
elif echo "$CHANGED" | grep -q "video"; then
  TYPE="feat(video)"
elif echo "$CHANGED" | grep -q "api"; then
  TYPE="feat(api)"
else
  TYPE="chore"
fi

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
MSG="${TYPE}: auto-commit ${FILE_COUNT} file(s) at ${TIMESTAMP}"

git add -A
git commit -m "$MSG"
git push origin main --quiet

echo "[AutoGit] Committed and pushed: $MSG"
