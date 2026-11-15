# pen2ai 云端版

点读笔的 AI 服务，跑在 Render 免费套餐上，域名固定。

## 环境变量
- `PEN_TOKEN`：访问口令（笔的 URL 里带 ?k=）
- `LLM_API_KEY`：智谱 GLM 的 key
- `LLM_MODELS`：可选，默认 glm-4-flash,glm-4.5-air,glm-4.6v-flash

## 页面
- `/` 问答页
- `/cam` 拍照扫描
- `/dl` 装笔问AI
- `/log` 按科目看记录
