"""AI-diffusion Studio Guide - Built-in knowledge base for beginners."""

GUIDE_SECTIONS = {
    "prompt_basics": {
        "title": "Prompt の基本（プロンプトの書き方）",
        "content": """
## Prompt の書き方

### 基本ルール
- **英語で書く**（日本語は基本的に使えない）
- **カンマ区切り**で要素を並べる
- **前に書いたものほど影響が強い**

### 構造テンプレート
```
[品質タグ], [被写体], [外見], [服装], [ポーズ/アクション], [背景/場所], [雰囲気/照明]
```

### 例：一般向け
```
masterpiece, best quality, 1girl, beautiful face, long black hair,
blue eyes, white dress, standing in flower garden, sunlight,
detailed background, depth of field
```

### 例：リアル写真風
```
photo of a young woman, professional photography, 25 years old,
natural skin texture, casual outfit, urban street, golden hour,
bokeh, 85mm lens, sharp focus
```

### 品質を上げるキーワード
| 目的 | キーワード |
|------|-----------|
| 高品質 | masterpiece, best quality, highly detailed |
| リアル | photorealistic, raw photo, 8k uhd, film grain |
| アニメ | anime style, illustration, cel shading |
| 照明 | dramatic lighting, golden hour, studio lighting |
| 構図 | close-up, full body, cowboy shot, from above |

### Negative Prompt（除外ワード）
これを入れないと品質が下がる：
```
worst quality, low quality, blurry, deformed, ugly,
bad anatomy, bad hands, extra fingers, missing fingers,
watermark, text, signature
```
"""
    },

    "model_guide": {
        "title": "モデル選択ガイド（どのモデルをいつ使う？）",
        "content": """
## モデル選択ガイド

### SD 1.5 モデル（512x768推奨）
軽い、速い、LoRAが豊富。**まずはこれで始める。**

| モデル | 用途 | おすすめ場面 |
|--------|------|------------|
| **cardosAnime** | アニメ系 | イラスト、キャラクター |
| **Realistic Vision** | リアル写真風 | 人物、風景 |
| **DreamShaper** | 汎用 | 何にでも使える |
| **devlishphotorealism** | 超リアル | 商用写真風 |

### SDXL モデル（1024x1024推奨）
高品質だが重い。Mac MPS では20-60秒/枚。

| モデル | 用途 |
|--------|------|
| **realvisxlV30Turbo** | リアル系SDXL最速 |
| **realismEngineSDXL** | 超高品質リアル |
| **colossusProjectXL** | 汎用SDXL |
| **sdxlYamersRealisticNSFW** | NSFW対応リアル |
| **miamodelSFWNSFW** | SFW/NSFW両対応 |

### 選び方の目安
- **速度重視** → SD1.5モデル（ローカル）
- **品質重視** → SDXLモデル（RunPod推奨）
- **アニメ系** → cardosAnime, MeinaMix系
- **リアル系** → Realistic Vision, realisticVision系
- **NSFW** → pyrosNSFW, eroticVision, miamodel系
"""
    },

    "settings_guide": {
        "title": "設定ガイド（パラメータの意味と推奨値）",
        "content": """
## パラメータ設定ガイド

### Steps（ステップ数）
生成の反復回数。多いほど精細だが遅い。

| 用途 | 推奨値 |
|------|--------|
| テスト・プレビュー | 10-15 |
| 通常生成 | 20-30 |
| 高品質 | 30-50 |
| Turboモデル | 4-8 |

### CFG Scale（プロンプト忠実度）
プロンプトにどれだけ従うか。

| 値 | 効果 |
|----|------|
| 1-4 | 自由度高い、独創的 |
| **5-8** | **バランス良い（推奨）** |
| 9-12 | プロンプトに忠実、やや硬い |
| 13+ | 過剰に忠実、画像が崩れやすい |

### Sampler（サンプラー）
| サンプラー | 特徴 | おすすめ |
|-----------|------|---------|
| **euler_ancestral** | 多様性あり、アニメ向き | 初心者おすすめ |
| **dpmpp_2m** | 安定、高品質 | リアル系 |
| **dpmpp_sde** | ディテール豊か | 高品質生成 |
| **lcm** | 超高速（4-8 steps） | テスト用 |

### Scheduler（スケジューラー）
| スケジューラー | 用途 |
|---------------|------|
| **normal** | 標準（迷ったらこれ） |
| **karras** | ノイズ除去が滑らか |
| **exponential** | SDXL向き |

### 画像サイズ
| モデル | 推奨サイズ |
|--------|-----------|
| SD 1.5 | 512x512, 512x768, 768x512 |
| SDXL | 1024x1024, 832x1216, 1216x832 |

**注意**: 推奨サイズから大きく外れると品質が下がる

### Seed（シード値）
- **-1**: ランダム（毎回違う画像）
- **固定値**: 同じ画像を再現。微調整時に便利。
- いい画像が出たら**Seedをメモ**して、プロンプトだけ変えると効率的。
"""
    },

    "lora_guide": {
        "title": "LoRA の使い方",
        "content": """
## LoRA（Low-Rank Adaptation）ガイド

### LoRA とは？
チェックポイントモデルに**追加で適用する小さなモデル**。
スタイル、キャラクター、ポーズ、服装などを指定できる。

### 使い方
1. LoRA ドロップダウンから選択
2. **Strength を 0.5-0.8** に設定（1.0だと強すぎることが多い）
3. LoRAに対応するキーワードをプロンプトに追加

### Strength の目安
| 値 | 効果 |
|----|------|
| 0.3-0.5 | 軽い影響、自然 |
| **0.6-0.8** | **バランス良い** |
| 0.9-1.0 | 強い影響 |
| 1.0+ | 過剰、画像が崩れることも |

### あなたが持っているLoRAの例
| LoRA | 用途 | 推奨Strength |
|------|------|-------------|
| Kimono系 | 着物スタイル | 0.7 |
| underwater系 | 水中撮影風 | 0.6 |
| pop_art | ポップアート風 | 0.7 |
| samurai系 | サムライスタイル | 0.7 |
| Sci-fi_Environments | SF背景 | 0.6 |

### 収益化のヒント
- **独自LoRAを作成** → CivitAIで公開 → DLされるたびにBuzz獲得
- ニッチなLoRA（日本文化、特定のスタイル等）は競争が少ない
"""
    },

    "monetization_guide": {
        "title": "収益化ガイド（CivitAI + その他）",
        "content": """
## AI画像で収入を得る方法

### CivitAI での収益化

#### 1. Creator Program（月次報酬）
- モデル/LoRAの公開 → DL数に応じてBuzz獲得
- 画像投稿 → チップでBuzz獲得
- **月末にBuzzをBank → 現金化**
- 最低引き出し額: $50

#### 2. Early Access（有料先行公開）
- 新しいLoRA/モデルを**有料で先行公開**
- 1-2週間後に無料公開
- **最も直接的な収入源**

#### 3. バウンティ
- リクエストに応えてBuzz報酬を獲得
- CivitAIサイトで確認可能

### CivitAI以外

#### Patreon / FANBOX / Fantia
- 月額サブスクリプション
- 限定コンテンツ（高解像度、NSFW等）
- **安定収入の柱**

#### ストック素材（Adobe Stock, Shutterstock等）
- AI生成画像を素材として販売
- ※各サイトのAI画像ポリシーを確認

#### プリントオンデマンド（SUZURI, Redbubble等）
- Tシャツ、スマホケース、ポスター等
- デザインをアップするだけ

#### SNSでの集客
- X (Twitter): 作品投稿 → フォロワー獲得 → 上記サービスへ誘導
- Instagram: ビジュアル重視
- Pixiv: 日本のイラストコミュニティ

### 月収目安（現実的）
| 期間 | CivitAI | その他 | 合計 |
|------|---------|--------|------|
| 1ヶ月目 | 0-2,000円 | 0円 | 0-2,000円 |
| 3ヶ月目 | 3,000-10,000円 | 0-5,000円 | 3,000-15,000円 |
| 6ヶ月目 | 10,000-30,000円 | 5,000-20,000円 | 15,000-50,000円 |

### 重要なポイント
1. **継続的な投稿**が最も重要（週3-5投稿）
2. **ニッチを見つける**（日本文化、特定のフェチ、特定のスタイル）
3. **品質 > 量**（Photoshopで仕上げると差別化できる）
4. **コミュニティ参加**（コメント、フォロー、コラボ）
"""
    },

    "workflow_tips": {
        "title": "効率的なワークフロー",
        "content": """
## 効率的な制作ワークフロー

### 基本フロー
```
1. アイデア → プロンプト作成
2. Quick タブで低Steps(15)でテスト生成（数秒）
3. いい構図が出たらSeedを固定
4. Steps を上げて高品質生成（25-30）
5. 必要ならPhotoshopで仕上げ
6. CivitAI タブから投稿
```

### 量産テクニック
- **Seed固定 + プロンプト微調整**: 同じ構図でバリエーション
- **Batch Size 4**: 一度に4枚生成して選ぶ
- **LoRA切り替え**: 同じプロンプトでスタイル違い
- **サイズ違い**: SNS用(1:1), スマホ壁紙(9:16), PC壁紙(16:9)

### CivitAI投稿のコツ
- **タグを適切に付ける**（検索されやすくなる）
- **生成パラメータを公開**（信頼性UP、他のユーザーが再現できる）
- **シリーズ化する**（同じテーマで連続投稿）
- **投稿時間**: 欧米のゴールデンタイム（日本の深夜〜早朝）

### ローカル vs クラウドの使い分け
| 作業 | 環境 |
|------|------|
| プロンプトテスト | ローカル（無料） |
| SD1.5 で通常生成 | ローカル |
| SDXL で高品質生成 | RunPod |
| バッチ大量生成 | RunPod |
| 動画生成 | RunPod |
"""
    },
}

# Prompt templates for quick generation
PROMPT_TEMPLATES = {
    "portrait_realistic": {
        "name": "リアル人物ポートレート",
        "prompt": "photo of a beautiful young woman, professional photography, natural skin texture, soft smile, looking at camera, studio lighting, shallow depth of field, 85mm lens, sharp focus, 8k uhd",
        "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, bad hands, watermark, text",
        "settings": {"steps": 25, "cfg": 7, "width": 512, "height": 768, "sampler": "dpmpp_2m"},
        "recommended_models": ["Realistic Vision", "realisticVision", "devlishphotorealism"],
    },
    "portrait_anime": {
        "name": "アニメキャラクター",
        "prompt": "masterpiece, best quality, 1girl, beautiful detailed eyes, long flowing hair, smile, colorful, detailed background, anime style",
        "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, bad hands, extra fingers",
        "settings": {"steps": 25, "cfg": 7, "width": 512, "height": 768, "sampler": "euler_ancestral"},
        "recommended_models": ["cardosAnime", "DreamShaper", "MeinaMix"],
    },
    "landscape": {
        "name": "風景写真",
        "prompt": "stunning landscape photography, mountains, lake reflection, golden hour, dramatic sky, nature, 8k uhd, professional photography, wide angle lens",
        "negative": "worst quality, low quality, blurry, watermark, text, people, person",
        "settings": {"steps": 30, "cfg": 8, "width": 768, "height": 512, "sampler": "dpmpp_2m"},
        "recommended_models": ["Realistic Vision", "DreamShaper"],
    },
    "japanese_culture": {
        "name": "日本文化・和風",
        "prompt": "beautiful Japanese woman wearing traditional kimono, cherry blossoms, temple garden, spring, elegant, professional photography, soft lighting, cultural",
        "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark",
        "settings": {"steps": 25, "cfg": 7, "width": 512, "height": 768, "sampler": "dpmpp_2m"},
        "recommended_models": ["Realistic Vision", "majicMIX"],
    },
    "cyberpunk": {
        "name": "サイバーパンク",
        "prompt": "cyberpunk city street at night, neon lights, rain, futuristic, holographic signs, dark atmosphere, cinematic, blade runner style, 8k",
        "negative": "worst quality, low quality, blurry, bright, daytime, watermark",
        "settings": {"steps": 30, "cfg": 8, "width": 768, "height": 512, "sampler": "dpmpp_sde"},
        "recommended_models": ["DreamShaper", "colossusProjectXL"],
    },
    "product_photo": {
        "name": "商品撮影風（ストック素材向け）",
        "prompt": "professional product photography, clean white background, studio lighting, commercial, high-end, minimalist, sharp focus, 8k",
        "negative": "worst quality, low quality, blurry, messy background, text, watermark",
        "settings": {"steps": 25, "cfg": 7, "width": 768, "height": 768, "sampler": "dpmpp_2m"},
        "recommended_models": ["Realistic Vision", "realisticStockPhoto"],
    },
    "nsfw_realistic": {
        "name": "NSFW リアル (Adult)",
        "prompt": "beautiful woman, detailed skin texture, natural body, intimate, bedroom, soft lighting, photorealistic, 8k uhd, professional photography",
        "negative": "worst quality, low quality, deformed, ugly, bad anatomy, bad hands, cartoon, anime, watermark",
        "settings": {"steps": 30, "cfg": 7, "width": 512, "height": 768, "sampler": "dpmpp_2m"},
        "recommended_models": ["pyrosNSFW", "eroticVision", "miamodel", "sdxlYamersRealisticNSFW"],
    },
    "nsfw_anime": {
        "name": "NSFW アニメ (Adult)",
        "prompt": "masterpiece, best quality, 1girl, beautiful detailed body, sensual pose, detailed skin, anime style, colorful",
        "negative": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, extra limbs",
        "settings": {"steps": 25, "cfg": 7, "width": 512, "height": 768, "sampler": "euler_ancestral"},
        "recommended_models": ["cardosAnime", "AbyssOrangeMix"],
    },
}
