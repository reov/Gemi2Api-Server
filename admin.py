"""
Gemi2Api Server 管理面板后端
提供状态监控、配置管理、日志查看等功能
"""

import os
import time
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import deque

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 创建路由器
router = APIRouter(prefix="/admin", tags=["admin"])

# 全局状态跟踪
_start_time = time.time()
_request_log = deque(maxlen=100)  # 保留最近100条日志
_stats = {
    "total_requests": 0,
    "error_count": 0,
    "total_response_time": 0.0,
}

# 环境变量路径
ENV_FILE = Path(__file__).parent / ".env"


class ConfigUpdate(BaseModel):
    """配置更新请求"""
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    feature: Optional[str] = None
    enabled: Optional[bool] = None


def log_request(method: str, path: str, status: int, response_time: float = 0):
    """记录请求日志"""
    _stats["total_requests"] += 1
    _stats["total_response_time"] += response_time
    
    if status >= 400:
        _stats["error_count"] += 1
    
    _request_log.appendleft({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "path": path,
        "status": status,
        "response_time": round(response_time * 1000, 2),
    })


def format_uptime(seconds: float) -> str:
    """格式化运行时间"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if days > 0:
        return f"{days}天 {hours}小时"
    elif hours > 0:
        return f"{hours}小时 {minutes}分钟"
    else:
        return f"{minutes}分钟"


def read_env() -> dict:
    """读取 .env 文件"""
    env_vars = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def write_env(updates: dict):
    """更新 .env 文件"""
    env_vars = read_env()
    env_vars.update(updates)
    
    with open(ENV_FILE, "w") as f:
        for key, value in env_vars.items():
            f.write(f'{key}="{value}"\n')


@router.get("/", response_class=HTMLResponse)
async def admin_page():
    """返回管理面板页面"""
    html_path = Path(__file__).parent / "templates" / "admin.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>管理面板文件未找到</h1>", status_code=404)


@router.get("/api/status")
async def get_status():
    """获取服务状态"""
    from main import (
        API_KEY, SECURE_1PSID, SECURE_1PSIDTS,
        ENABLE_THINKING, TEMPORARY_CHAT, AUTO_DELETE_CHAT,
        HOST, PORT
    )
    
    # 检查 cookie 是否有效（简单检查是否存在）
    cookie_valid = bool(SECURE_1PSID and SECURE_1PSIDTS)
    
    # 计算平均响应时间
    avg_response_time = 0
    if _stats["total_requests"] > 0:
        avg_response_time = round(_stats["total_response_time"] / _stats["total_requests"] * 1000, 2)
    
    # 计算错误率
    error_rate = 0
    if _stats["total_requests"] > 0:
        error_rate = round(_stats["error_count"] / _stats["total_requests"] * 100, 1)
    
    return {
        "running": True,
        "uptime": format_uptime(time.time() - _start_time),
        "total_requests": _stats["total_requests"],
        "avg_response_time": avg_response_time,
        "error_rate": error_rate,
        "host": HOST,
        "port": PORT,
        "api_key_enabled": bool(API_KEY),
        "cookie_valid": cookie_valid,
        "thinking_enabled": ENABLE_THINKING,
        "temporary_chat": TEMPORARY_CHAT,
        "auto_delete_chat": AUTO_DELETE_CHAT,
        "version": "0.1.3",
        "start_time": datetime.fromtimestamp(_start_time).strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/api/logs")
async def get_logs():
    """获取最近的日志"""
    return {"logs": list(_request_log)}


@router.post("/api/config")
async def update_config(config: ConfigUpdate):
    """更新配置"""
    from main import (
        HOST, PORT, API_KEY, ENABLE_THINKING,
        TEMPORARY_CHAT, AUTO_DELETE_CHAT
    )
    
    # 更新功能开关
    if config.feature and config.enabled is not None:
        env_key = None
        if config.feature == "thinking":
            env_key = "ENABLE_THINKING"
        elif config.feature == "temporary":
            env_key = "TEMPORARY_CHAT"
        elif config.feature == "autoDelete":
            env_key = "AUTO_DELETE_CHAT"
        
        if env_key:
            write_env({env_key: str(config.enabled).lower()})
            # 更新运行时变量
            if config.feature == "thinking":
                import main
                main.ENABLE_THINKING = config.enabled
            elif config.feature == "temporary":
                import main
                main.TEMPORARY_CHAT = config.enabled
            elif config.feature == "autoDelete":
                import main
                main.AUTO_DELETE_CHAT = config.enabled
            return {"success": True, "message": f"功能 {config.feature} 已{'启用' if config.enabled else '禁用'}"}
    
    # 更新网络配置
    if config.host or config.port:
        updates = {}
        if config.host:
            updates["HOST"] = config.host
        if config.port:
            updates["PORT"] = str(config.port)
        if config.api_key is not None:
            updates["API_KEY"] = config.api_key
        write_env(updates)
        return {"success": True, "message": "配置已保存，重启服务后生效"}
    
    # 更新 API_KEY
    if config.api_key is not None:
        write_env({"API_KEY": config.api_key})
        return {"success": True, "message": "API_KEY 已更新"}
    
    raise HTTPException(status_code=400, detail="无效的配置请求")


@router.post("/api/restart")
async def restart_service():
    """重启服务"""
    try:
        # 获取当前进程的命令行参数
        python_path = sys.executable
        script_path = os.path.abspath(__file__).replace("admin.py", "main.py")
        
        # 启动新进程
        subprocess.Popen(
            [python_path, script_path],
            cwd=os.path.dirname(script_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # 终止当前进程
        os._exit(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重启失败: {str(e)}")


class CookieUpdate(BaseModel):
    """Cookie 更新请求"""
    secure_1psid: str
    secure_1psidts: str


@router.post("/api/cookies")
async def update_cookies(cookies: CookieUpdate):
    """更新 Gemini Cookie"""
    if not cookies.secure_1psid or not cookies.secure_1psidts:
        raise HTTPException(status_code=400, detail="Cookie 值不能为空")
    
    # 保存到 .env 文件
    write_env({
        "SECURE_1PSID": cookies.secure_1psid,
        "SECURE_1PSIDTS": cookies.secure_1psidts,
    })
    
    # 更新运行时变量
    import main
    main.SECURE_1PSID = cookies.secure_1psid
    main.SECURE_1PSIDTS = cookies.secure_1psidts
    
    return {"success": True, "message": "Cookie 已保存并生效"}


@router.post("/api/reinit")
async def reinit_client():
    """重新初始化 Gemini 客户端"""
    import main
    
    # 关闭旧客户端
    if main.gemini_client is not None:
        try:
            await main.gemini_client.close()
        except Exception:
            pass
        main.gemini_client = None
    
    # 尝试重新初始化
    try:
        client = await main.get_gemini_client()
        if client:
            return {"success": True, "message": "Gemini 客户端重新连接成功"}
        else:
            raise HTTPException(status_code=500, detail="客户端初始化返回空值")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新连接失败: {str(e)}")


def setup_middleware(app):
    """设置请求日志中间件"""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    
    class RequestLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.time()
            
            # 跳过静态资源和管理面板的请求
            path = request.url.path
            if path.startswith("/admin") or path.startswith("/static"):
                return await call_next(request)
            
            response = await call_next(request)
            
            # 记录 API 请求
            if path.startswith("/v1/"):
                duration = time.time() - start
                log_request(request.method, path, response.status_code, duration)
            
            return response
    
    app.add_middleware(RequestLoggingMiddleware)
