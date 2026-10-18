# Sakura: PC と Colab をつなぐ推論

速度を優先する初期構成は、**モデル全体を Colab GPU に置き、生成中の cache を使う**形です。PC は依頼と結果の受け取りを担当します。1 回の生成を GPU 側で完了するので、token ごとに WAN を往復しません。

[Colab でノートブックを開く](https://colab.research.google.com/github/Summer110622/agora/blob/codex/sakura-community-inference/sakura/notebooks/colab_pc_inference.ipynb) · [ノートブックのソース](notebooks/colab_pc_inference.ipynb)

| 使い方 | 用途 | Colab の対象環境 |
|---|---|---|
| ノートブック内の `local` | 最初に生成を試す | 無料枠も対象。GPU 提供・稼働時間は保証されません |
| PC gateway + Colab worker | PC のコードから Colab GPU に依頼する | 対象の有料プランで正の compute unit 残高あり、Enterprise、または自己管理の計算環境 |
| 任意の stage 分割 | 1 台のメモリに収まらない MoE を分担する | 到達可能な各端末と、分散 worker を動かせる利用環境 |

Colab の managed runtime では、compute unit 残高のない無料利用で分散 worker、SSH などの遠隔操作、ノートブック UI の迂回が禁止されています。また、managed runtime 全般で remote proxy 接続は禁止されています。ここではノートブック単体の経路と、対象環境向けの PC 接続を分けています。トンネル・ポート公開による制限回避は含みません。[Colab 公式 FAQ](https://research.google.com/colaboratory/faq.html)

## まずノートブックだけで動かす

1. 上の Colab リンクを開き、GPU runtime を選びます。
2. `MODE="local"` のまま、セルを順に実行します。
3. 生成セルだけを繰り返すと、読み込んだモデルを再利用できます。

初期モデルは [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)、revision は `2fc06364715b967f1860aea9cf38778875588b17` です。公式モデルの config は `qwen3_5` / `Qwen3_5ForConditionalGeneration` です。35B MoE 用の loader に小型 dense モデルを渡す構成にはしていません。[公式 config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)

ノートブックは Colab が提供する PyTorch を維持し、推論用の Transformers などを固定します。学習用の固定 PyTorch 環境とは別です。実際の PyTorch/CUDA/GPU 名を表示し、CUDA を選んだのに GPU が使えない場合は明示的に停止します。実装のローカル試験は CPU / PyTorch 2.10 が対象で、Colab 提供の GPU / PyTorch の組合せは実機で確認が必要です。モデルの初回ダウンロードが必要です。

## PC から Colab GPU に依頼する

通信は次の向きです。

```text
PC の chat ── HTTPS（または loopback HTTP）── PC の署名付き gateway
                                  ↑
                          HTTPS で仕事を取得・結果を送信
                                  │
                            Colab の GPU worker
```

**利用者が管理する、Colab から到達可能な PC 側の HTTPS 入口が必要です。** 例は `https://your-pc.example:49201/rpc` です。インターネットから到達できない private address や、NAT の内側の `localhost` を Colab に指定しても接続できません。PC gateway 自体を TLS で起動する方法と、既存の HTTPS 終端から転送する方法があります。Colab に着信ポートを開く必要はありません。

### 1. PC に推論用の環境を作る

リポジトリを取得済みなら、そのフォルダから始めます。以下は Windows PowerShell の例です。

```powershell
git clone --branch codex/sakura-community-inference https://github.com/Summer110622/agora.git
cd agora
python -m venv sakura/.venv
.\sakura\.venv\Scripts\Activate.ps1
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ./sakura
```

PC が受付・クライアントだけを担当する場合、PC の GPU は不要です。GPU 推論を PC 上でも使う場合は、対応するドライバと [CUDA 環境指定](environments/cuda.txt)を使ってください。

### 2. PC で manifest と鍵を作る

```powershell
python sakura_inference.py init --directory sakura/runs/inference --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17
```

生成されたファイルを次のように扱います。

| ファイル | 置き場所 |
|---|---|
| `inference-run.json` | PC と Colab で同じものを使う公開 manifest |
| `pc.private.json` | PC に保管。Colab へアップロードしない |
| `colab.private.json` | Colab worker のみに渡す秘密鍵 |

秘密鍵の内容をコードセルに貼り付けたり、Git に追加したりしないでください。`init` は既存の鍵を上書きしません。manifest や公開鍵を変更した場合は、両端で同じ manifest を使い直します。

### 3. PC の受付を起動する

別の PowerShell ターミナルで同じ仮想環境を有効にし、実行します。

利用者の hostname に対する信頼済みの証明書 chain と秘密鍵がある場合、gateway が直接 TLS を扱えます。次の証明書パスを実際のものに置き換えます。証明書の秘密鍵は Git の外で保管してください。

```powershell
python sakura_inference.py gateway --manifest sakura/runs/inference/inference-run.json --identity sakura/runs/inference/pc.private.json --host 0.0.0.0 --allow-external --port 49201 --tls-cert C:/certs/fullchain.pem --tls-key C:/certs/privkey.pem
```

この場合、Colab と PC のクライアントは両方とも `https://your-pc.example:49201/rpc` を使います。利用者の DNS、証明書、PC 側ポートへの到達経路が必要です。native TLS のサーバーに `http://127.0.0.1:49201/rpc` を指定しても接続できません。

すでに HTTPS の入口がある場合は、次のように loopback HTTP で起動し、その入口を `http://127.0.0.1:49201/rpc` に転送する構成も使えます。

```powershell
python sakura_inference.py gateway --manifest sakura/runs/inference/inference-run.json --identity sakura/runs/inference/pc.private.json --host 127.0.0.1 --port 49201
```

こちらの構成では PC の `chat` は loopback HTTP、Colab worker は外側の HTTPS URL を指定できます。HTTPS の終端が別マシンにある場合は、既存の private routing とアクセス制限を用意し、必要に応じて bind を明示的に変更します。HTTP の署名は内容を暗号化しません。

このコードはルーター・DNS・TLS 証明書の設定を自動変更しません。HTTPS 入口が未準備なら、先にノートブック単体の `local` を使えます。

### 4. Colab worker を接続する

ノートブックで以下を変更し、最初から実行します。

```python
MODE = "gateway_worker"
ELIGIBLE_DISTRIBUTED_RUNTIME = True  # 対象環境と残高を確認した場合だけ
```

PC の HTTPS RPC URL を入力し、ファイル選択では **`inference-run.json` と `colab.private.json` の 2 ファイルだけ**を選びます。worker セルは仕事を待つ間実行中になります。Colab の停止ボタンで子プロセスも終了します。秘密鍵の内容を表示するセルはありません。

同じ処理を自分で用意した GPU マシンで起動するコードは次です。

```bash
python sakura_inference.py worker --manifest inference-run.json --identity colab.private.json --endpoint https://your-pc.example:49201/rpc --device cuda --dtype auto
```

`https://your-pc.example:49201/rpc` は実際の URL に置き換えます。証明書検証を無効にしたり、Google のログイン Cookie をスクリプトへ渡したりする必要はありません。Colab の `kernel.proxyPort` は現在のユーザーと開いたノートブック向けの機構で、PC の一般的な HTTP クライアントから使える API として扱っていません。[公式 colabtools 実装](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py)

### 5. PC から生成する

PC の別ターミナルから実行します。

```powershell
python sakura_inference.py chat --manifest sakura/runs/inference/inference-run.json --identity sakura/runs/inference/pc.private.json --endpoint https://your-pc.example:49201/rpc --prompt "Pythonで足し算の関数を書いて" --max-new-tokens 64
```

URL を利用者の hostname に置き換えます。既存の HTTPS 終端から loopback HTTP に転送する構成なら、PC 上のこのコマンドでは `http://127.0.0.1:49201/rpc` を使うこともできます。worker がモデルを読み込んだ後、GPU 側で生成して結果が返ります。PC の待ち時間にはキュー待ち・ネットワーク・生成が含まれます。Colab が切断されたら worker を再起動してください。

## cache の効果を実測する

モデル全体が載る GPU で、同じ入力・同じ生成数を使って cache 有効/無効を測定します。すでにノートブックでモデルを読み込んでいる場合は、同じ GPU に二重ロードせず、モデルを解放するか別の新しい実行環境で実行してください。

```bash
python sakura_inference.py local --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --device cuda --dtype auto --prompt "Write a short Python add function." --max-new-tokens 64 --benchmark
```

表示される数値はその環境での実測です。初回ダウンロード、モデル読込、prefill、decode、PC の通信を区別して比較してください。GPU 型・入力長・生成数が異なる測定から、倍率や token/s を一般化しません。

## stage 分割を使う場合

stage 分割は**モデル全体を 1 台に載せられない場合のメモリ上の選択肢**です。単体 GPU に収まるモデルを PC と Colab に分けても、速くなるとは限りません。WAN 経由では生成 token ごとに stage の通信待ちが加わります。初版の split 経路は batch=1、padding なし、context 上限 4096 です。

split は [学習側と同じ manifest / メンバー鍵の導線](README.md)を使う MoE 用の別経路です。前述の dense 0.8B 用 `inference-run.json` と交換して使うことはできません。無料の managed Colab runtime で分散 worker を動かす経路でもありません。

例えば 2 stage の tiny manifest で動作を確認する場合は、PC、Colab、trainer の公開鍵を作り、次のように manifest を生成します。`--members` にはそれぞれの公開 JSON を渡します。

```bash
python sakura_cli.py keygen --member pc --private pc.private.json --public pc.public.json
python sakura_cli.py keygen --member colab --private colab.private.json --public colab.public.json
python sakura_cli.py keygen --member trainer --private trainer.private.json --public trainer.public.json
python sakura_cli.py init --directory sakura/runs/split-demo --stages 2 --members pc.public.json colab.public.json trainer.public.json
```

それぞれのマシンに同じ manifest を渡し、各マシン上で担当する stage を起動します。次は PC と、利用条件を満たす GPU 計算環境で実行する例です。

```bash
# PC 上
python sakura_inference.py split-serve --manifest sakura/runs/split-demo/run.json --identity pc.private.json --stage 0 --device cpu --dtype auto --host 127.0.0.1 --port 49200

# GPU 側のマシン上。外部 bind は既存の private routing / TLS が必要な場合に明示する
python sakura_inference.py split-serve --manifest sakura/runs/split-demo/run.json --identity colab.private.json --stage 1 --device cuda --dtype auto --host 0.0.0.0 --allow-external --port 49200
```

この分割例では、**各 stage の endpoint がクライアントから到達できる必要があります**。通常の managed Colab にそのまま公開の着信経路があるとは仮定しません。自己管理 GPU / private VPC など、既存の経路がある環境で利用してください。Colab への着信経路がない場合は前述の outbound gateway 方式を使います。

`endpoints.json` は次の配列形式です。URL を利用者の実際の endpoint に置き換えます。

```json
[
  {"stage_index": 0, "url": "https://pc-stage.example/rpc", "member_id": "pc"},
  {"stage_index": 1, "url": "https://gpu-stage.example/rpc", "member_id": "colab"}
]
```

```bash
python sakura_inference.py split-chat --manifest sakura/runs/split-demo/run.json --identity trainer.private.json --endpoints endpoints.json --prompt "Write a Python add function." --max-new-tokens 64
```

tiny の重みはランダム初期化なので、ここでは接続・cache・生成処理を確認します。意味のある回答を得るには、対応する固定 revision の実モデルと、その tokenizer が必要です。

実際の Qwen3.5-35B-A3B-Base では [既存の `qwen35` profile](README.md)で local snapshot を指定した manifest を作り、各 `split-serve` に `--base-path LOCAL_SNAPSHOT`、必要なら `--adapter-path PEFT_DIR` を指定します。`split-chat` にも tokenizer の local snapshot を `--base-path` で渡せます。base snapshot と adapter の revision は一致させてください。全 expert の保持容量、各 stage のメモリ、context/cache、回線を実測してから運転します。35B が小型 Colab GPU に収まる保証はありません。

## 接続・検証の確認

| 症状 | 確認すること |
|---|---|
| Colab で CUDA が見つからない | runtime が GPU か、GPU が割り当てられているか。CPU 利用は明示的に選択する |
| worker が PC に接続できない | Colab から PC の HTTPS URL に到達するか。`localhost` / private IP だけでは別ホストから届かない |
| TLS エラー | 正しい証明書・ホスト名と、利用者の HTTPS 終端の設定。証明書検証を無効にしない |
| 署名・manifest 不一致 | 同じ公開 manifest と対応する各メンバーの鍵か。PC / Colab の時刻が合っているか |
| chat が待ち続ける | gateway と worker が起動しているか。初回ダウンロード中でないか。Colab の切断・残高を確認する |
| VRAM 不足 | 入力・生成長を短くし、GPU 上の別モデルを解放する。35B を A3B 分だけの重みとして計算しない |

ノートブックは出力なしで保存し、Colab の「出力を保存しない」設定も含めています。実行後に共有する際は、鍵・認証情報・機密の prompt が出力に残っていないことを確認してください。

配布時のノートブック検証は、形式、Python セル構文、実装とのコマンド/API 整合までです。この作成環境から実際の Colab GPU と利用者の PC の HTTPS 経路には接続していません。Colab 上で前半の生成を実行し、対象環境で後半の worker と PC の `chat` が返ることを確認すると、利用環境での接続確認になります。
