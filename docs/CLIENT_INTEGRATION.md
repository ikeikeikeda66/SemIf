# SemIf クライアント連携ガイド（Claude Desktop・Claude Code・REST API）

SemIf 常駐サーバーは、Mac 起動時に自動でバックグラウンド常駐し、Apple Silicon GPU（Metal / MPS）を活用して超低遅延（数十〜数百ミリ秒）で意味的決定（Semantic If / 選択・分類判定）を行います。

---

## 1. サーバーの基本情報

- **URL**: `http://127.0.0.1:8765`
- **バックエンド**: PyTorch MPS (`float16`)
- **標準モデル**: `Qwen/Qwen3.5-4B`
- **macOS 自動起動**: `launchd`（`~/Library/LaunchAgents/com.semif.server.plist`）によりログイン時に自動起動・常駐

### サービスの管理コマンド
```bash
# 状態確認
launchctl list | grep com.semif.server

# ログ確認
tail -f ~/Library/Logs/semif-server.log
tail -f ~/Library/Logs/semif-server-error.log

# サービスの再起動
launchctl unload ~/Library/LaunchAgents/com.semif.server.plist
launchctl load ~/Library/LaunchAgents/com.semif.server.plist

# サービスの停止（自動起動解除）
launchctl unload ~/Library/LaunchAgents/com.semif.server.plist
```

---

## 2. エンドポイント仕様

### ① `GET /health` (ヘルスチェック)
サーバーおよびモデルのロード状態を確認します。

```bash
curl -s http://127.0.0.1:8765/health
```

**レスポンス例（準備完了時）**:
```json
{
  "status": "ok",
  "ready": true,
  "model": {
    "source": "Qwen/Qwen3.5-4B",
    "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "dtype": "float16",
    "device": "mps",
    "torch_version": "2.10.0",
    "transformers_version": "5.17.0"
  }
}
```

---

### ② `POST /v1/decide` (意味決定・分類API)
自然言語のコンテキスト（`state`）と質問（`question`）に対し、指定した選択肢（`options`）の確率と最適な決定を瞬時に返します。

#### リクエスト
```bash
curl -X POST http://127.0.0.1:8765/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The deployment completed at 14:02 UTC. Health checks passed in all three zones. No rollback was initiated.",
    "question": "Is there evidence that the deployment succeeded?",
    "options": [
      {"id": "yes", "description": "The deployment succeeded."},
      {"id": "no", "description": "The deployment did not succeed."},
      {"id": "insufficient", "description": "The evidence is insufficient."}
    ]
  }'
```

※ `options` は文字列配列（`["yes", "no"]`）でも指定可能です。

#### レスポンス
```json
{
  "id": "decide-1727018245123",
  "decision": "yes",
  "confidence": 0.9452,
  "probabilities": {
    "yes": 0.9452,
    "no": 0.0412,
    "insufficient": 0.0136
  },
  "option_logits": {
    "yes": 2.451,
    "no": -0.684,
    "insufficient": -1.821
  },
  "input_tokens": 128,
  "forward_seconds": 0.042,
  "total_seconds": 0.048
}
```

---

## 3. Claude Desktop（クロウ）との連携 (MCP)

Claude Desktop の設定ファイルに MCP サーバーを登録することで、Claude が思考中に直接 SemIf の高速判定ツール（`semif_decide`）を呼び出せるようになります。

### 設定手順
1. 設定ファイルを開きます：
   `~/Library/Application Support/Claude/claude_desktop_config.json`
2. `mcpServers` に以下を追加します：

```json
{
  "mcpServers": {
    "semif": {
      "command": "/Volumes/SSD_USB_1/AntiGravitiRoot/semlf/.venv/bin/python",
      "args": [
        "/Volumes/SSD_USB_1/AntiGravitiRoot/semlf/scripts/semif_mcp.py"
      ]
    }
  }
}
```

3. Claude Desktop を再起動すると、チャット内で `semif_decide` ツールが有効になります。

---

## 4. Claude Code との連携

### 方法 A: Claude Code の MCP サーバーとして追加（推奨）
プロジェクトのルートまたはグローバル設定で以下を実行します：

```bash
claude mcp add semif /Volumes/SSD_USB_1/AntiGravitiRoot/semlf/.venv/bin/python /Volumes/SSD_USB_1/AntiGravitiRoot/semlf/scripts/semif_mcp.py
```

またはプロジェクト内の `.mcp.json` に記載：
```json
{
  "mcpServers": {
    "semif": {
      "command": "/Volumes/SSD_USB_1/AntiGravitiRoot/semlf/.venv/bin/python",
      "args": ["/Volumes/SSD_USB_1/AntiGravitiRoot/semlf/scripts/semif_mcp.py"]
    }
  }
}
```

### 方法 B: Claude Code 内から curl / HTTP 経由で呼び出す
Claude Code のプロンプトやスクリプトから直接ローカル API を呼び出せます：

```bash
curl -s -X POST http://127.0.0.1:8765/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"state": "...", "question": "...", "options": ["yes", "no"]}'
```

---

## 5. Python スクリプト等からの呼び出し例

```python
import requests

payload = {
    "state": "ユーザーからパスワード再設定メールが届かないと問い合わせがあった。",
    "question": "どの部署・キューにルーティングすべきですか？",
    "options": [
        {"id": "auth_support", "description": "認証・アカウントサポート"},
        {"id": "billing", "description": "請求・決済サポート"},
        {"id": "sales", "description": "営業・見積もり窓口"}
    ]
}

res = requests.post("http://127.0.0.1:8765/v1/decide", json=payload).json()
print(f"決定: {res['decision']} (確信度: {res['confidence']:.1%})")
```
