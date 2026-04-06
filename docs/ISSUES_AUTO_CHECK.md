# GitHub Issues 自动检测配置

## 自动检测

已配置 **每 10 分钟自动检测** 新的 GitHub Issues。

### 配置说明

**定时任务配置：**
- 文件位置：`.claude/scheduled_tasks.json`
- Cron 表达式：`*/10 * * * *` (每 10 分钟执行)
- 自动过期：7 天
- 任务 ID：`02a8ce04`

**检测逻辑：**
1. 调用 GitHub API 获取未关闭的 Issues
2. 如果有新 Issues，列出并提示修复
3. 如果没有 Issues，输出确认信息

### 手动检查

随时可以使用命令：
```
/check-issues
```

### 续期定时任务

定时任务 7 天后自动过期，如需继续自动检测，运行：
```
/cron-create */10 * * * * 检查 GitHub 仓库的新 Issues...
```

### 取消自动检测

如需停止自动检测：
```
/cron-delete <task-id>
```

## 所需权限

GitHub Token 需要以下权限：
- `repo` 或 `public_repo` + `issues`

## 相关命令

| 命令 | 功能 |
|------|------|
| `/check-issues` | 手动检查 Issues |
| `/cron-list` | 查看所有定时任务 |
