# Sakura community training

Agoraを基にした、固定メンバー用の独立した分散学習ランタイムです。CPU・CUDA・ROCm・MPSを共通のstage操作で扱い、最初にQwen3.5のLoRA、次に全重み学習、独自MoEの比較実験へ進めます。

**現状は実行可能な基盤と小型モデルでの検証までです。35Bの学習済み成果、GPU実機対応の認定、24時間運転、複数の物理ホストへの保存を完了したものではありません。** 同梱の6件の日英データは手書きの動作確認用fixtureです。

ColabとPCをつなぐ推論専用コードとノートブックは [推論ガイド](INFERENCE.md) にあります。GPU内でのキャッシュ付き生成をPCから依頼する方法と、Qwen3.5 MoEを複数端末へ分割する方法を用意しています。

## セットアップ

Python 3.11の独立した仮想環境を使います。既存のAgora用Python環境とは分けてください。

```sh
python -m venv sakura/.venv
# Activate the environment: Windows sakura\.venv\Scripts\activate
# Linux/macOS: source sakura/.venv/bin/activate
python -m pip install -r sakura/environments/cpu.txt
python -m pip install -e './sakura[test]'
```

GPUでは最初のprofileを `cuda.txt`（CUDA 12.8）、`rocm.txt`（Linux / ROCm 7.1）、`mps.txt`（Apple Silicon）へ変更します。各profileとcommunity packageはPyTorch 2.10.0を使用します。[公式の対応wheel](https://pytorch.org/get-started/previous-versions/#v2100)を確認し、実際の端末で `doctor --certify` を実行してください。

Transformers 5.9.0、Hub 1.5.0などの直接依存をpyprojectで固定しています。初期portable環境にFLA・causal-conv1dなどの追加kernelを入れないでください。Hivemind DHTは追加の任意機能で、元のAgoraと同じcommitを参照します。DHTを使わないWindowsでも、署名付きHTTPで通信できます。

## まずCPUで一巡する

リポジトリのルートから実行します。

```sh
python sakura_cli.py doctor --device cpu --certify --host-id my-pc --output sakura/runs/my-pc.json
python sakura_cli.py init --directory sakura/runs/demo --profile tiny --stages 2
python sakura_cli.py train --manifest sakura/runs/demo/run.json --steps 3
python sakura_cli.py resume --manifest sakura/runs/demo/run.json --steps 2
python sakura_cli.py export --manifest sakura/runs/demo/run.json --output sakura/runs/demo/adapter
python sakura_cli.py infer --manifest sakura/runs/demo/run.json --prompt 'Write a Python add function' --max-new-tokens 16
```

tinyは実際のQwen3.5 hybrid構造を小さくしたランダム初期化モデルです。語彙512とfixture専用UTF-8 tokenizerを使うため、Qwenの事前学習済み重みとの互換性やコード品質を示すものではありません。

独立した2workerプロセス、署名付き通信、worker再起動、再開、adapter exportまで自動で確認できます。

```sh
python sakura/scripts/smoke_network.py --directory sakura/runs/network-demo
python -m pytest -c sakura/pyproject.toml -q
```

## メンバーを接続する

各メンバーは自身の端末で秘密鍵と共有用公開鍵を生成します。秘密鍵は共有しません。

```sh
python sakura_cli.py keygen --member trainer --private trainer.private.json --public trainer.public.json
python sakura_cli.py keygen --member worker-a --private worker-a.private.json --public worker-a.public.json
python sakura_cli.py keygen --member worker-b --private worker-b.private.json --public worker-b.public.json
python sakura_cli.py init --directory sakura/runs/community --members trainer.public.json worker-a.public.json worker-b.public.json
```

同一の `run.json` を各端末に配置します。公開鍵の追加・dataset変更・学習方式の変更はrun hashを変えるため、新しいrunとして扱います。各workerを起動します。

```sh
python sakura_cli.py join --manifest sakura/runs/community/run.json --identity worker-a.private.json --stage 0 --device cpu --host 0.0.0.0 --allow-external --port 49200
python sakura_cli.py join --manifest sakura/runs/community/run.json --identity worker-b.private.json --stage 1 --device cuda --host 0.0.0.0 --allow-external --port 49200
```

AMDは `--device rocm`、Appleは `--device mps` を指定します。指定したbackendが使えない場合はCPUへ黙って切り替えません。CLIは既定でloopbackだけにbindします。オフホスト通信は信頼した専用ネットワーク、またはTLSプロキシ経由で運用してください。署名は送信元・改ざん・replayを検証しますが、HTTP自体に暗号化はありません。

trainer側に接続先を用意します。例示IPは実際の専用ネットワークのアドレスへ置き換えます。

```json
[
  {"stage_index": 0, "member_id": "worker-a", "url": "http://192.0.2.10:49200/rpc"},
  {"stage_index": 1, "member_id": "worker-b", "url": "http://192.0.2.11:49200/rpc"}
]
```

```sh
python sakura_cli.py train --manifest sakura/runs/community/run.json --identity trainer.private.json --endpoints endpoints.json --steps 100 --mirror /separate-host/checkpoint-share
```

Hivemindを別途インストールした環境では、joinの `--dht --initial-peer MULTIADDR --announce-url URL` で署名付きアドレスをDHTへ公開できます。`HivemindDiscovery.resolve()` が検証済み接続先を返します。通信は引き続き署名付きHTTPです。既存のPluralis向け認証やnative integrity binaryは読み込みません。旧SPARTAランタイムへの組み込みは今回の経路とは別です。

## 資源検査とQwen3.5 35B

各端末の `doctor` 結果と、trainerの `--role coordinator` の結果を集めます。`host-id` は実際の物理端末ごとに運営者が確認して割り当てます。異なるディレクトリだけでは2台への保存とは認定しません。

```sh
python sakura_cli.py capacity --manifest run.json --reports worker-a.json worker-b.json --coordinator trainer.json --output admission.json
```

容量は全expertの実際のshapeをmetaモデルで数えます。activation・CPU master・optimizer・保存時のコピーを区別し、30%の余裕を確保します。合格しても「下限容量を満たす」意味で、実学習のpeak memory認定ではありません。MPSの推奨メモリ量も空きメモリとは区別します。

35B重みと元のtokenizerは、固定commitの**ローカルsnapshot**を用意してください。モデルの自動ダウンロード、クラウドGPUの購入、学習結果の自動公開は行いません。symlinkを使うHF cacheは通常ファイルへ実体化してから渡します。

```sh
python sakura_cli.py init --directory sakura/runs/qwen35 --profile qwen35 --base-path /models/qwen35-snapshot --revision COMMIT_SHA40 --data /data/bilingual-sft.jsonl --stages 8 --sequence-length 2048 --members trainer.public.json worker-a.public.json worker-b.public.json
```

initはconfig・tokenizer・全safetensorsのSHA256を逐次計算します。各workerは起動時に内容を照合し、所有stageの重みだけをmetaから実体化して読み込みます。端末ごとの保存場所の違いは `--base-path` で指定できます。同一内容であることの検証は省略しません。

35Bでは `capacity` の結果を `train --admission admission.json` に渡すことが必要です。初期LoRAはr16/alpha32/dropout0、attentionとDeltaNetのprojectionが対象です。router・expert・embedding・LM headは凍結し、router補助損失の勾配は上流adapterへ正しく返します。元のtokenizerは変更しません。損失はtailで計算し、全語彙logitsをネットワークへ送りません。

## データ・評価・独自モデル

SFT JSONLの必須項目は `id, language, repository, problem_id, messages, source` です。sourceは `name, url, revision, license` を持ちます。例は [data-fixtures.jsonl](examples/data-fixtures.jsonl) を参照してください。

```sh
python sakura_cli.py prepare-data --input /data/bilingual-sft.jsonl --output /data/prepared
python sakura_cli.py score --manifest run.json --split validation --output heldout-loss.json
python sakura_cli.py experiments --output candidates.json
python sakura_cli.py init --directory sakura/runs/candidate-0 --profile experiment --candidate 0 --mode full
python sakura_cli.py train --manifest sakura/runs/candidate-0/run.json --steps 10
```

問題・repository・重複内容を連結したgroupを同じsplitへ置き、assistant部分だけを学習します。長い文書は1token重複の独立windowに分け、packingしません。重複として除去した入力を含め、出典とデータ内容をrun revisionに結び付けます。fixture corpusのheld-out splitが空なら、scoreは明示的に失敗します。

`experiments` はattention2種×expert3種×語彙2種の候補を出します。実行用proxyは実際のQwen text moduleで、expert数・top-k・幅の関係を維持します。これだけで独自35Bの構成が選ばれたことにはなりません。

```sh
python sakura_cli.py tokenizer --input /data/train.jsonl --input-format sft_jsonl --provenance /data/sources.json --vocab-size 64000 --output /models/sakura-tokenizer64k
python sakura_cli.py init --directory sakura/runs/scratch --profile scratch --mode full --config selected-config.json --scratch-tokenizer /models/sakura-tokenizer64k --allow-large-random --data /data/sft.jsonl --stages 8
```

独自tokenizerはローカル学習データだけから作り、64k/128kの目標語彙に足りないcorpusを本番候補として受け付けません。scratch初期化は所有stageのみを割り当て、分割数に依存しないseedで初期化します。大規模割り当てには明示的opt-inと事前メモリ検査が必要です。

`--mode full` は全重み・routerを更新し、`export` はHFテキストモデルの分割safetensorsと検証済みtokenizerを書き出します。raw CPTには `init --objective cpt --mode full --data documents.jsonl` を使います。各行は `id,language,repository,problem_id,text,source` を持ちます。CPTはchat templateを挿入せず、文書の生tokenとEOSを予測します。token windowは文書ごとに生成し、全corpusのtoken列を保持しません。source record自体はメモリへ読み込むため、TB規模の入力indexや学習済み独自35Bの選定は残る作業です。

`evaluate` は固定した日英結果JSONLを比較できます。`--execute --image IMAGE@sha256:DIGEST` を明示した場合だけ、インストール済みDocker image内でコードを実行します。ホストで生成コードを実行するfallbackはありません。コード生成結果とtrusted testを `code, test_code, problem_id, language` で渡します。Docker実行そのものはこの開発環境では未検証です。

候補の選定は、同じ評価revision・計算量・日英ペア数を要求し、日英実行正解率、bits/byte、持続スループットの順で決めます。7日間の停止時間も含む連続観測は `capacity --observations observations.json --target-tokens N` で30%余裕付きの所要時間へ換算します。

## 復旧・容量上限・検証範囲

- backwardでは更新せず、全stage成功後に共通のglobal gradient clippingを適用して更新します。途中失敗は全stageを同じ世代へ戻します。
- 保存にはsafetensorsとhash付き構造metadataを使います。全stage・optimizer・scheduler・乱数・データ位置を揃え、成功したCURRENTだけを再開対象とします。
- checkpointは別ディレクトリへ検証付きで複製できます。物理的に別ホストであることは運営者の確認が必要です。世代の自動削除はしません。
- wire messageは64MiB、既定blobは1GiB。大きなsnapshotは署名付きchunkで転送します。必要な場合は両側の `--max-blob-mib` を揃えます。最終復号時とtrainerの全stage保存時はCPU RAMが必要で、35B optimizerの完全なストリーミング保存ではありません。
- 初期計算はFP32で、GPU workerのAdam/masterはCPUへoffloadします。BF16最適化、圧縮、非同期学習、SPARTA再利用は基準実装との検証後に追加する項目です。
- CPUの単体・分割・通信境界のLoRA勾配、実更新、PEFT再読込、全重み更新、障害復旧、独立workerプロセスを検証しています。CUDA/ROCm/MPS、実WAN、DHT、24時間運転、本番corpus、35Bの収束は未認定です。

変更はcommunity packageに分離しています。上流のライセンスと既存の公式参加経路はリポジトリ内に保持しています。
