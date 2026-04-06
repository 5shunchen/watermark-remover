---
name: error_handling
description: 项目统一的错误处理和日志规范
---

所有模块统一使用以下错误处理模式：
```python
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def safe_process(func):
    """装饰器：统一捕获处理异常，记录日志"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"文件不存在: {e}")
            raise
        except MemoryError:
            logger.error("内存不足，请减小批量大小")
            raise
        except Exception as e:
            logger.exception(f"{func.__name__} 执行失败: {e}")
            raise
    return wrapper
```

日志格式统一配置在 src/utils/logger.py，不要在各模块重复配置。
