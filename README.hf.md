---
title: Gemi2Api Server
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /admin
pinned: false
---

# Gemi2Api Server

一键部署到 [Hugging Face Space](https://huggingface.co/spaces/zhiyu1998/Gemi2Api-Server) 的 Gemini 网页版逆向代理。

## 使用方法

1. 点击 **Duplicate this Space** 创建你自己的副本
2. 在 Space 设置中填入以下环境变量：
   - `SECURE_1PSID` — Gemini Cookie 中的 `__Secure-1PSID`
   - `SECURE_1PSIDTS` — Gemini Cookie 中的 `__Secure-1PSIDTS`
   - `API_KEY`（可选）— 自定义 API 访问密钥
3. Space 启动后访问 `/admin` 进入管理面板（默认密码：`Gemi2Api-Server`，**请在生产环境更换**）

## 获取 Gemini Cookie

1. 使用隐身标签访问 [Google Gemini](https://gemini.google.com/) 并登录
2. 打开浏览器开发工具 (F12) → Application → Cookies → `gemini.google.com`
3. 复制 `__Secure-1PSID` 和 `__Secure-1PSIDTS` 的值

## API

- `POST /v1/chat/completions` — OpenAI 兼容聊天接口
- `GET /v1/models` — 可用模型列表
- `GET /admin` — 管理面板

## 文档

完整文档请查看 [GitHub 仓库](https://github.com/zhiyu1998/Gemi2Api-Server)。
