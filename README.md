# Gemi2Api-Server
[HanaokaYuzu / Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 的服务端简单实现

| 原始界面 | 管理面板 |
|:---:|:---:|
| [![pmKWe0A.png](https://s41.ax1x.com/2026/06/11/pmKWe0A.png)](https://imgchr.com/i/pmKWe0A) | [![pmKrjUO.png](https://s41.ax1x.com/2026/06/11/pmKrjUO.png)](https://imgchr.com/i/pmKrjUO) |

## 在线体验

| 平台 | 地址 | 面板密码 |
|------|------|---------|
| Render | [gemi2api-server-1ol5.onrender.com/admin/](https://gemi2api-server-1ol5.onrender.com/admin/) | `Gemi2Api-Server`（demo 专用，请勿用于生产） |
| HuggingFace | [zhiyu1998-gemi2api-server.hf.space/admin/](https://zhiyu1998-gemi2api-server.hf.space/admin/) | `Gemi2Api-Server`（demo 专用，请勿用于生产） |

> [!NOTE]
> 以上体验地址未配置 Gemini Cookie，仅展示管理面板界面。如需完整功能请自行部署。

## 快捷部署

### Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/zhiyu1998/Gemi2Api-Server)

### HuggingFace

[![Follow us on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-lg-dark.svg)](https://huggingface.co/new-space?duplicate=true&repo=zhiyu1998/Gemi2Api-Server)

## 直接运行

0. 填入 `SECURE_1PSID` 和 `SECURE_1PSIDTS`（登录 Gemini 在浏览器开发工具中查找 Cookie），有必要的话可以填写 `API_KEY`

```properties
SECURE_1PSID = "COOKIE VALUE HERE"
SECURE_1PSIDTS = "COOKIE VALUE HERE"
API_KEY= "API_KEY VALUE HERE"
TEMPORARY_CHAT = "false" # 使用临时对话模式，此模式会禁用部分功能如思考、图片生成等，默认关闭。
AUTO_DELETE_CHAT = "true" # 生成结束后自动从web端删除对话记录，默认开启。TEMPORARY_CHAT为true时，此项无效。
PUBLIC_BASE_URL = "https://your-domain.com" # 外部URL，用于生成图片代理链接，不填则会使用内部地址。使用反向代理时必填，否则可能导致图片无法访问。
```

1. 安装依赖

```bash
uv init
uv add fastapi uvicorn gemini-webapi httpx h2
```

> [!NOTE]
> 如果存在 `pyproject.toml` 那么就使用下面的命令：

```bash
uv sync
```

或者 `pip` 也可以：

```bash
pip install fastapi uvicorn gemini-webapi httpx h2
```

2. 激活环境

```bash
source venv/bin/activate
```

3. 启动

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 7860
```

> [!WARNING]
> tips: `.env.example` 中默认设置了 `API_KEY=Gemi2Api-Server`，复制到 `.env` 后即可登录管理面板。**正式部署前请务必更换为强密码。**

## 使用 Docker 运行（推荐）

### 快速开始

1. 克隆本项目
   ```bash
   git clone https://github.com/zhiyu1998/Gemi2Api-Server.git
   ```

2. 创建 `.env` 文件并填入你的 Gemini Cookie 凭据:
   ```bash
   cp .env.example .env
   # 用编辑器打开 .env 文件，填入你的 Cookie 值
   ```

3. 启动服务:
   ```bash
   docker-compose up -d
   ```

4. 服务将在 http://127.0.0.1:7860 上运行

### 其他 Docker 命令

```bash
# 查看日志
docker-compose logs

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 重新构建并启动
docker-compose up -d --build
```

## API 端点

- `GET /`: 服务状态检查
- `GET /v1/models`: 获取可用模型列表
- `POST /v1/chat/completions`: 与模型聊天 (类似 OpenAI 接口)
- `GET /gemini-proxy/image`: 图片代理接口（有生成图片需求时，需要保证此端点可直接访问，如果使用反向代理则需要填写 `PUBLIC_BASE_URL` 环境变量）
- `GET /admin`: 管理面板

## 管理面板

访问 `/admin` 进入管理面板，支持以下功能：

- 查看服务状态和 Gemini 连接状态
- 配置 Gemini Cookie（无需重启）
- 修改 API_KEY、限流等参数
- 查看实时日志
- 在线测试聊天

> [!NOTE]
> 未设置 `API_KEY` 时管理面板不可用。密码通过 `.env` 文件中的 `API_KEY` 配置，`.env.example` 提供了默认值 `Gemi2Api-Server`，**正式部署前请务必更换**。

## 常见问题

### 服务器报 500 问题解决方案

500 的问题一般是 IP 不太行 或者 请求太频繁（后者等待一段时间或者重新新建一个隐身标签登录一下重新给 Secure_1PSID 和 Secure_1PSIDTS 即可），见 issue：
- [__Secure-1PSIDTS · Issue #6 · HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API/issues/6)
- [Failed to initialize client. SECURE_1PSIDTS could get expired frequently · Issue #72 · HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API/issues/72)

解决步骤：
1. 使用隐身标签访问 [Google Gemini](https://gemini.google.com/) 并登录
2. 打开浏览器开发工具 (F12)
3. 切换到 "Application" 或 "应用程序" 标签
4. 在左侧找到 "Cookies" > "gemini.google.com"
5. 复制 `__Secure-1PSID` 和 `__Secure-1PSIDTS` 的值
6. 更新 `.env` 文件
7. 重新构建并启动: `docker-compose up -d --build`

## 致谢

- 图片去水印算法基于 [journey-ad/gemini-watermark-remover](https://github.com/journey-ad/gemini-watermark-remover) 以及 [allenk/GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) 实现，并直接使用了其中的两张 png 图片。

## 贡献

同时感谢以下开发者对 `Gemi2Api-Server` 作出的贡献：

<a href="https://github.com/zhiyu1998/Gemi2Api-Server/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zhiyu1998/Gemi2Api-Server&max=1000" />
</a>
