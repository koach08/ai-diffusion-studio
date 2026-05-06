"""Adult Studio - Character Builder, Scene Categories, Position Selector, Undress/Edit presets."""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Character Builder Options
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAR_ETHNICITY = {
    "日本人": "japanese woman, east asian features, natural beauty",
    "韓国人": "korean woman, east asian features, flawless skin",
    "中国人": "chinese woman, east asian features, elegant",
    "東南アジア": "southeast asian woman, tan skin, exotic beauty",
    "白人 (ヨーロッパ)": "caucasian woman, european features, fair skin",
    "ラテン系": "latina woman, hispanic features, olive skin, curvaceous",
    "黒人 (アフリカン)": "black woman, african features, dark skin, beautiful",
    "インド人": "indian woman, south asian features, dusky complexion",
    "中東 (ペルシャン)": "persian woman, middle eastern features, exotic beauty",
    "ミックス (ハーフ)": "mixed race woman, eurasian features, unique beauty",
}

CHAR_AGE = {
    "20代前半 (若々しい)": "young woman in her early twenties, youthful, fresh",
    "20代後半": "woman in her late twenties, prime beauty",
    "30代 (大人の魅力)": "woman in her thirties, mature beauty, confident",
    "40代 (熟女)": "woman in her forties, elegant mature, milf",
}

CHAR_BODY_TYPE = {
    "スリム": "slim body, slender figure, thin waist",
    "普通": "average body, natural proportions",
    "グラマー": "curvy body, voluptuous figure, hourglass shape",
    "アスレチック": "athletic body, toned muscles, fit, abs",
    "ぽっちゃり": "plump body, chubby, soft curves, thick thighs",
    "小柄 (ペティート)": "petite body, small frame, delicate",
}

CHAR_BREAST = {
    "小さめ (A-B)": "small breasts, flat chest",
    "普通 (C)": "medium breasts",
    "大きめ (D-E)": "large breasts, busty, big boobs",
    "巨乳 (F+)": "huge breasts, very busty, enormous boobs",
}

CHAR_BUTT = {
    "普通": "normal buttocks",
    "小尻": "small tight butt",
    "美尻": "beautiful round butt, perky ass",
    "大尻": "big butt, large round ass, thick",
    "桃尻": "peach-shaped butt, perfect round ass",
}

CHAR_HAIR_COLOR = {
    "黒髪": "black hair",
    "茶髪": "brown hair",
    "金髪": "blonde hair, golden hair",
    "赤毛": "red hair, ginger",
    "ピンク": "pink hair",
    "シルバー": "silver hair, platinum",
    "青": "blue hair",
    "紫": "purple hair",
    "白": "white hair",
}

CHAR_HAIR_STYLE = {
    "ロング (ストレート)": "long straight hair, flowing",
    "ロング (ウェーブ)": "long wavy hair, flowing curls",
    "ミディアム": "medium length hair, shoulder length",
    "ショート": "short hair, pixie cut",
    "ボブ": "bob cut hair",
    "ポニーテール": "ponytail hairstyle",
    "ツインテール": "twintails, pigtails",
    "お団子 (アップ)": "hair bun, updo hairstyle",
    "三つ編み": "braided hair, braid",
    "濡れ髪": "wet hair, damp, dripping",
}

CHAR_SKIN = {
    "色白": "pale skin, fair complexion, porcelain skin",
    "普通": "natural skin tone",
    "小麦肌": "tanned skin, sun-kissed",
    "褐色肌": "dark tan skin, bronze complexion",
    "オイリー (艶)": "oiled skin, glistening, shiny wet skin",
    "汗ばんだ": "sweaty skin, glistening with sweat",
}

CHAR_EXPRESSION = {
    "微笑み": "gentle smile, warm expression, soft eyes",
    "誘惑的": "seductive expression, bedroom eyes, sultry look, parted lips",
    "恥じらい": "shy expression, blushing cheeks, looking away, embarrassed",
    "喘ぎ (快感)": "ecstatic expression, open mouth, moaning, pleasure face, ahegao",
    "無表情 (クール)": "expressionless, cool beauty, neutral face, piercing gaze",
    "ウインク": "winking, playful expression, tongue out",
    "上目遣い": "looking up at viewer, upturned eyes, submissive gaze",
    "キス顔": "kiss face, puckered lips, blowing kiss",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Clothing / State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAR_CLOTHING = {
    "--- 着衣 ---": "",
    "制服 (セーラー)": "school uniform, sailor uniform, pleated skirt",
    "制服 (ブレザー)": "school uniform, blazer, plaid skirt, knee socks",
    "OL (オフィスレディ)": "office lady outfit, white blouse, pencil skirt, stockings",
    "メイド服": "maid outfit, maid costume, frills, apron, headpiece",
    "ナース服": "nurse costume, white uniform, short skirt, stockings",
    "体操着・ブルマ": "gym uniform, bloomers, tight fitting",
    "チアリーダー": "cheerleader outfit, short skirt, crop top, pom poms",
    "巫女服": "shrine maiden outfit, miko, white and red",
    "チャイナドレス": "chinese dress, cheongsam, qipao, high slit",
    "バニーガール": "bunny girl outfit, bunny ears, leotard, fishnet stockings",
    "レースクイーン": "race queen outfit, tight bodysuit, high heels",
    "ウェディングドレス": "wedding dress, bridal veil, white gown",
    "着物・浴衣": "kimono, japanese traditional, obi belt",
    "タンクトップ＋ショーパン": "tank top, short shorts, casual, revealing",
    "Tシャツ＋ジーンズ": "t-shirt, jeans, casual outfit",
    "ドレス (セクシー)": "sexy dress, tight fitting, low cut, short hem",
    "コスプレ (自由入力)": "",
    "--- 水着・下着 ---": "",
    "ビキニ": "bikini, two piece swimsuit, revealing",
    "ワンピース水着": "one piece swimsuit, tight fitting",
    "マイクロビキニ": "micro bikini, string bikini, barely covering",
    "ランジェリー (レース)": "lace lingerie, bra and panties, see through, delicate",
    "ランジェリー (シルク)": "silk lingerie, satin, smooth, elegant",
    "ガーターベルト": "garter belt, stockings, suspenders, sexy lingerie",
    "ベビードール": "babydoll nightgown, sheer, see through, lace trim",
    "ボディストッキング": "bodystocking, full body, fishnet, mesh",
    "--- 露出 ---": "",
    "トップレス": "topless, bare breasts, exposed nipples, wearing only panties",
    "ボトムレス": "bottomless, no pants, no panties, shirt only",
    "全裸": "completely nude, naked, fully exposed body, no clothing",
    "タオル一枚": "wrapped in towel only, bath towel, barely covering",
    "エプロンのみ": "naked apron, wearing only an apron, bare back",
    "シーツに包まれ": "wrapped in bedsheets only, barely covering, peek of skin",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pose (Solo / Non-sexual)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAR_POSE = {
    "--- 基本ポーズ ---": "",
    "立ち (正面)": "standing, facing viewer, front view, full body",
    "立ち (横向き)": "standing, side view, profile, full body",
    "立ち (背面)": "standing, from behind, back view, looking over shoulder",
    "座り (椅子)": "sitting on chair, legs crossed, elegant pose",
    "座り (地面)": "sitting on ground, legs spread casually",
    "しゃがみ": "squatting, crouching down",
    "--- セクシーポーズ ---": "",
    "横たわり (仰向け)": "lying on back, supine, legs slightly apart, looking at viewer",
    "横たわり (横向き)": "lying on side, hand on hip, seductive pose",
    "うつ伏せ": "lying face down, prone, arching back, looking at viewer",
    "四つん這い": "on all fours, hands and knees, arching back",
    "M字開脚": "legs spread wide, M-shape, sitting, exposing",
    "膝立ち": "kneeling upright, on knees, thighs apart",
    "バックポーズ": "bent over, presenting from behind, hands on knees",
    "--- 特殊 ---": "",
    "自撮り風": "selfie angle, POV, looking at camera, close-up, arm extended",
    "鏡越し": "mirror reflection, mirror selfie, multiple angles visible",
    "シャワー中": "in shower, water running, wet body, steam",
    "入浴中": "in bathtub, partially submerged, wet hair, steam",
    "ストレッチ": "stretching, flexible pose, yoga, athletic",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sex Positions (体位)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEX_POSITIONS = {
    "なし (ソロ)": "",
    "--- 基本体位 ---": "",
    "正常位": "missionary position, man on top, legs wrapped around, penetration, face to face, intimate",
    "騎乗位": "cowgirl position, woman on top, riding, straddling, bouncing, hands on chest",
    "背面騎乗位": "reverse cowgirl, woman on top facing away, riding, back view, butt visible",
    "バック (後背位)": "doggy style, from behind, penetration from rear, hands gripping hips, arching back",
    "側位 (スプーン)": "spooning position, side by side, from behind, intimate, cuddling",
    "--- バリエーション ---": "",
    "対面座位": "face to face sitting, lotus position, straddling, arms around neck, intimate",
    "立ちバック": "standing doggy style, bent over, standing sex from behind, against wall",
    "駅弁": "standing carry position, legs wrapped, lifted, face to face, supported",
    "松葉崩し": "scissors position, legs intertwined, side angle penetration",
    "寝バック": "prone bone, lying flat face down, penetration from behind, pressing down",
    "座位 (椅子)": "sitting on chair, woman on lap, face to face, straddling on chair",
    "背面座位": "reverse sitting, woman sitting on lap facing away, back visible",
    "--- オーラル ---": "",
    "フェラチオ": "blowjob, oral sex, kneeling, looking up, sucking",
    "フェラチオ (寝)": "blowjob lying down, sixty-nine position angle, oral",
    "クンニリングス": "cunnilingus, face between thighs, oral on woman, licking",
    "69 (シックスナイン)": "sixty-nine position, mutual oral, head between legs, simultaneous",
    "イラマチオ": "deep throat, hands on head, standing oral, forceful",
    "--- その他 ---": "",
    "パイズリ": "titjob, penis between breasts, breasts pressed together, looking up",
    "手コキ": "handjob, gripping with hand, stroking, looking at viewer",
    "足コキ": "footjob, feet on penis, toes, sole",
    "素股": "thigh sex, between thighs, intercrural, rubbing",
    "アナル": "anal sex, anal penetration, from behind",
    "二穴同時": "double penetration, two insertions, threesome",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Camera / Angle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAR_CAMERA = {
    "全身 (フルショット)": "full body shot, wide angle",
    "上半身": "upper body, medium shot, waist up",
    "バストアップ": "bust shot, close-up face and chest",
    "顔アップ": "face close-up, portrait, detailed facial features",
    "ローアングル (下から)": "low angle shot, looking up, from below",
    "ハイアングル (上から)": "high angle shot, looking down, bird eye view",
    "POV (主観)": "pov, first person view, point of view, looking at viewer",
    "背面ショット": "rear view, from behind, back shot",
    "横からのショット": "side view shot, profile angle",
    "俯瞰 (真上から)": "overhead shot, top-down view, directly above",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Setting / Location
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAR_SETTING = {
    "ベッドルーム": "bedroom, large bed, soft lighting, pillows, cozy, warm atmosphere",
    "ラブホテル": "love hotel room, neon lighting, mirror ceiling, luxury bed",
    "バスルーム": "bathroom, wet tiles, steam, shower, bathtub",
    "温泉・露天風呂": "japanese onsen, hot spring, outdoor bath, steam, rocks, nature",
    "プール・ビーチ": "swimming pool, poolside, beach, ocean, sunshine, wet",
    "教室": "classroom, school desk, blackboard, after school, empty classroom",
    "オフィス": "office, desk, computer, professional setting, after hours",
    "更衣室・ロッカールーム": "locker room, changing room, lockers, bench",
    "車内": "inside car, back seat, car interior, confined space",
    "屋外 (自然)": "outdoor, nature, forest clearing, sunlight through trees, grass",
    "屋外 (都市)": "urban outdoor, rooftop, city lights, night, balcony",
    "スタジオ (白背景)": "photo studio, pure white background, professional lighting, clean",
    "スタジオ (黒背景)": "dark studio, black background, dramatic spotlight, moody",
    "和室": "traditional japanese room, tatami floor, sliding doors, futon",
    "ソファ・リビング": "living room, sofa, couch, home interior, casual",
    "キッチン": "kitchen, kitchen counter, domestic setting",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Style (Art style)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAR_STYLE = {
    "フォトリアル (写真風)": {
        "prompt": "photorealistic, raw photo, 8k uhd, dslr, natural skin texture, detailed skin pores, professional photography",
        "negative": "anime, cartoon, painting, illustration, 3d render, deformed, ugly, blurry, bad anatomy, extra fingers, watermark, text, low quality",
    },
    "グラビア風": {
        "prompt": "gravure photography, magazine quality, professional lighting, sharp focus, glamorous, japanese idol style",
        "negative": "anime, cartoon, deformed, ugly, blurry, bad anatomy, watermark, text, low quality",
    },
    "AV風 (アダルトビデオ)": {
        "prompt": "photorealistic, adult video still, natural lighting, raw candid, realistic skin, high resolution, japanese AV style",
        "negative": "anime, cartoon, painting, deformed, ugly, blurry, bad anatomy, watermark, text, low quality",
    },
    "アニメ・イラスト": {
        "prompt": "masterpiece, best quality, ultra detailed anime art, vibrant colors, cel shading, clean lines, anime style illustration",
        "negative": "photorealistic, 3d, photo, ugly, deformed, blurry, bad anatomy, extra fingers, worst quality, low quality",
    },
    "同人誌風": {
        "prompt": "doujinshi style, manga illustration, detailed lineart, screentone, japanese hentai art style, masterpiece quality",
        "negative": "photorealistic, 3d, blurry, low quality, bad anatomy, western comic",
    },
    "3DCG": {
        "prompt": "3d render, highly detailed 3d model, realistic 3d art, unreal engine, volumetric lighting, subsurface scattering",
        "negative": "2d, flat, painting, sketch, anime, low poly, blurry",
    },
    "油絵風": {
        "prompt": "oil painting, classical art, thick brushstrokes, rich texture, dramatic lighting, fine art nude, museum quality",
        "negative": "photo, digital, anime, flat, low quality",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scene Categories (PornX/Promptchan-like one-click)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENE_CATEGORIES = {
    # ── ソロ系 ──
    "グラビア撮影": {
        "prompt": "beautiful woman, gravure photography, bikini, sexy pose, professional lighting, magazine cover quality, beach, golden hour",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "入浴シーン": {
        "prompt": "beautiful woman in bathtub, wet skin, steam, relaxing, nude, partially submerged, candles, warm lighting, bathroom",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "ベッド自撮り": {
        "prompt": "beautiful woman taking selfie on bed, POV, lingerie, playful expression, soft lighting, bedroom, messy sheets, intimate",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "着替え中": {
        "prompt": "beautiful woman changing clothes, caught undressing, surprised expression, removing top, locker room, natural lighting",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "オイルマッサージ": {
        "prompt": "beautiful woman receiving oil massage, oiled skin, glistening body, lying on massage table, relaxed expression, spa, warm lighting",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "ポールダンス": {
        "prompt": "beautiful woman pole dancing, athletic body, high heels, dramatic lighting, club atmosphere, flexible, acrobatic pose",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    # ── セクシー系 ──
    "ランジェリーモデル": {
        "prompt": "beautiful woman modeling lingerie, lace bra and panties, bedroom, seductive pose, professional photography, soft focus background",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "温泉・露天風呂": {
        "prompt": "beautiful woman in japanese onsen, outdoor hot spring, steam, wet hair, nude, towel, traditional rocks, nature background, peaceful",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "コスプレエロ (メイド)": {
        "prompt": "beautiful woman in maid outfit, sexy maid costume, short skirt, thigh high stockings, bending over, playful, bedroom",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "コスプレエロ (ナース)": {
        "prompt": "beautiful woman in nurse costume, sexy nurse outfit, unbuttoned, stethoscope, hospital room, seductive smile",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    # ── カップル系 ──
    "ラブシーン (ソフト)": {
        "prompt": "couple in bed, romantic, intimate, kissing, embracing, soft lighting, bedroom, passionate, love making, gentle",
        "negative": "ugly, deformed, blurry, low quality, violent",
        "style": "フォトリアル (写真風)",
    },
    "正常位シーン": {
        "prompt": "couple having sex, missionary position, man on top, woman's legs wrapped, passionate, bedroom, intimate lighting, penetration",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "騎乗位シーン": {
        "prompt": "woman on top, cowgirl position, riding, straddling, bouncing, hands on chest, bedroom, passionate expression",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "バックシーン": {
        "prompt": "doggy style, from behind, woman on all fours, man behind, gripping hips, bedroom, arching back, pleasure",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    "フェラチオシーン": {
        "prompt": "woman giving blowjob, kneeling, oral sex, looking up at viewer, intimate, bedroom, detailed face",
        "negative": "ugly, deformed, blurry, low quality",
        "style": "フォトリアル (写真風)",
    },
    # ── アニメ系 ──
    "アニメ美少女 (ソロ)": {
        "prompt": "beautiful anime girl, masterpiece, best quality, nude, shy expression, blushing, bedroom, soft lighting, detailed eyes, perfect body",
        "negative": "worst quality, low quality, deformed, ugly, bad anatomy, extra fingers",
        "style": "アニメ・イラスト",
    },
    "アニメ触手": {
        "prompt": "anime girl, tentacles, wrapped around body, blushing, surprised expression, detailed illustration, fantasy, masterpiece quality",
        "negative": "worst quality, low quality, deformed, ugly, bad anatomy",
        "style": "アニメ・イラスト",
    },
    "アニメカップル": {
        "prompt": "anime couple, intimate scene, passionate, beautiful anime art, detailed eyes, bed, romantic, masterpiece, best quality",
        "negative": "worst quality, low quality, deformed, ugly, bad anatomy",
        "style": "アニメ・イラスト",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Undress / Clothing Edit Presets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNDRESS_MODES = {
    "全脱衣 (ヌードに)": {
        "prompt": "completely nude, naked, bare skin, natural body, no clothing, exposed breasts, nipples, natural skin texture, matching skin tone, seamless skin, photorealistic, same lighting, same person",
        "negative": "clothing, fabric, covered, dressed, ugly, deformed, blurry, different person, different skin color, seam, border, artifact, watermark",
        "strength": 0.85,
    },
    "トップレスに": {
        "prompt": "topless, bare breasts, exposed nipples, no top, no bra, natural breasts, realistic skin texture, matching skin tone, seamless, same lighting, same person",
        "negative": "shirt, top, bra, covered chest, ugly, deformed, blurry, different person, seam, border, artifact",
        "strength": 0.8,
    },
    "ランジェリーに変更": {
        "prompt": "wearing lace lingerie, bra and panties, see through, delicate lace, sexy underwear, matching skin tone, realistic fabric, same lighting, same person",
        "negative": "outerwear, fully dressed, nude, naked, ugly, deformed, different person, seam, artifact",
        "strength": 0.75,
    },
    "ビキニに変更": {
        "prompt": "wearing bikini, two piece swimsuit, string bikini, matching skin tone, realistic fabric, same lighting, same person",
        "negative": "fully dressed, winter clothes, ugly, deformed, different person, seam, artifact",
        "strength": 0.75,
    },
    "服を透けさせる": {
        "prompt": "see through clothing, wet clothes, sheer fabric, visible body through transparent material, wet t-shirt, natural skin visible, same lighting, same person",
        "negative": "opaque, thick clothing, ugly, deformed, blurry, different person, artifact",
        "strength": 0.6,
    },
    "服を破る (ダメージ)": {
        "prompt": "torn clothing, ripped outfit, damaged clothes, exposed skin through tears, battle damage, matching skin tone, same person",
        "negative": "intact clothing, clean, new, ugly, deformed, different person, artifact",
        "strength": 0.65,
    },
    "エプロンのみに": {
        "prompt": "wearing only an apron, naked apron, bare back, bare sides, cooking apron only, natural skin, same lighting, same person",
        "negative": "fully dressed, shirt, pants, ugly, deformed, different person, artifact",
        "strength": 0.8,
    },
    "タオル一枚に": {
        "prompt": "wrapped in bath towel only, fresh from shower, wet hair, bare shoulders, barely covering, natural skin, same lighting, same person",
        "negative": "fully dressed, dry, ugly, deformed, different person, artifact",
        "strength": 0.8,
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adult Video Presets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADULT_VIDEO_SCENES = {
    "セクシーダンス": {
        "prompt": "beautiful woman dancing sensually, smooth flowing movement, hips swaying, revealing outfit, professional lighting, slow motion, music video quality",
        "negative": "blurry, distorted, static, jerky motion, low quality",
        "duration": 5,
    },
    "ストリップティーズ": {
        "prompt": "beautiful woman slowly removing clothing, striptease, seductive, teasing, dramatic lighting, smooth motion, looking at camera",
        "negative": "blurry, distorted, jerky motion, low quality, static",
        "duration": 5,
    },
    "入浴・シャワーシーン": {
        "prompt": "beautiful woman in shower, water flowing over body, wet skin, steam, washing hair, sensual, slow motion, natural movement",
        "negative": "blurry, distorted, jerky, low quality, static",
        "duration": 5,
    },
    "ランジェリーウォーク": {
        "prompt": "beautiful woman walking in lingerie, catwalk, confident stride, bedroom, elegant movement, smooth camera follow, professional lighting",
        "negative": "blurry, distorted, jerky, low quality",
        "duration": 5,
    },
    "ベッドシーン (ソフト)": {
        "prompt": "beautiful woman lying on bed, rolling over, stretching sensually, silk sheets, soft lighting, intimate, bedroom, smooth movement",
        "negative": "blurry, distorted, jerky, low quality, static",
        "duration": 5,
    },
    "マッサージシーン": {
        "prompt": "beautiful woman receiving sensual massage, oiled skin, hands on body, relaxed, spa atmosphere, warm lighting, smooth motion",
        "negative": "blurry, distorted, jerky, low quality",
        "duration": 5,
    },
    "正常位シーン (動画)": {
        "prompt": "couple having sex missionary position, rhythmic movement, intimate, passionate, bedroom, natural motion, face to face",
        "negative": "blurry, distorted, jerky, low quality, static",
        "duration": 5,
    },
    "騎乗位シーン (動画)": {
        "prompt": "woman riding on top, cowgirl position, bouncing motion, hands on chest, passionate expression, bedroom, rhythmic",
        "negative": "blurry, distorted, jerky, low quality, static",
        "duration": 5,
    },
    "バックシーン (動画)": {
        "prompt": "doggy style, from behind, rhythmic motion, gripping hips, bedroom, passionate, natural movement",
        "negative": "blurry, distorted, jerky, low quality, static",
        "duration": 5,
    },
    "フェラチオシーン (動画)": {
        "prompt": "woman giving oral, head bobbing motion, kneeling, looking up, intimate, smooth motion, detailed",
        "negative": "blurry, distorted, jerky, low quality, static",
        "duration": 5,
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Number of people
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHAR_PEOPLE_COUNT = {
    "1人 (ソロ)": "1girl, solo, single woman",
    "2人 (カップル)": "1girl, 1boy, couple, two people, man and woman",
    "2人 (女性同士)": "2girls, lesbian, two women, yuri",
    "3人 (3P)": "threesome, three people, group sex",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt Composition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compose_character_prompt(
    style_key, ethnicity, age, body_type, breast, butt, hair_color,
    hair_style, skin, expression, clothing, pose, position, camera,
    setting, people_count, custom_prompt=""
):
    """Compose a full prompt from character builder selections."""
    parts = []
    style_info = CHAR_STYLE.get(style_key, {})

    # Style base
    if style_info.get("prompt"):
        parts.append(style_info["prompt"])

    # People count
    pc = CHAR_PEOPLE_COUNT.get(people_count, "")
    if pc:
        parts.append(pc)

    # Character attributes
    for val, mapping in [
        (ethnicity, CHAR_ETHNICITY),
        (age, CHAR_AGE),
        (body_type, CHAR_BODY_TYPE),
        (breast, CHAR_BREAST),
        (butt, CHAR_BUTT),
        (hair_color, CHAR_HAIR_COLOR),
        (hair_style, CHAR_HAIR_STYLE),
        (skin, CHAR_SKIN),
        (expression, CHAR_EXPRESSION),
    ]:
        tag = mapping.get(val, "")
        if tag:
            parts.append(tag)

    # Clothing
    cloth_tag = CHAR_CLOTHING.get(clothing, "")
    if cloth_tag:
        parts.append(cloth_tag)

    # Pose
    pose_tag = CHAR_POSE.get(pose, "")
    if pose_tag:
        parts.append(pose_tag)

    # Sex position (overrides pose if set)
    pos_tag = SEX_POSITIONS.get(position, "")
    if pos_tag:
        parts.append(pos_tag)

    # Camera
    cam_tag = CHAR_CAMERA.get(camera, "")
    if cam_tag:
        parts.append(cam_tag)

    # Setting
    set_tag = CHAR_SETTING.get(setting, "")
    if set_tag:
        parts.append(set_tag)

    # Custom prompt addition
    if custom_prompt and custom_prompt.strip():
        parts.append(custom_prompt.strip())

    prompt = ", ".join(p for p in parts if p)
    negative = style_info.get("negative", "ugly, deformed, blurry, low quality, bad anatomy, extra fingers, watermark, text")

    return prompt, negative


def compose_scene_prompt(category_key, custom_addition=""):
    """Compose prompt from a scene category."""
    scene = SCENE_CATEGORIES.get(category_key)
    if not scene:
        return "", ""

    prompt = scene["prompt"]
    if custom_addition and custom_addition.strip():
        prompt += ", " + custom_addition.strip()

    style_info = CHAR_STYLE.get(scene.get("style", ""), {})
    if style_info.get("prompt"):
        prompt = style_info["prompt"] + ", " + prompt

    negative = style_info.get("negative", scene.get("negative", "ugly, deformed, blurry, low quality"))
    return prompt, negative


def compose_video_prompt(scene_key, custom_addition=""):
    """Compose prompt for adult video generation."""
    scene = ADULT_VIDEO_SCENES.get(scene_key)
    if not scene:
        return "", "", 5

    prompt = scene["prompt"]
    if custom_addition and custom_addition.strip():
        prompt += ", " + custom_addition.strip()

    return prompt, scene.get("negative", ""), scene.get("duration", 5)


def get_undress_params(mode_key):
    """Get inpainting parameters for undress/clothing edit."""
    mode = UNDRESS_MODES.get(mode_key)
    if not mode:
        return "", "", 0.7
    return mode["prompt"], mode.get("negative", ""), mode.get("strength", 0.7)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LoRA Categories (for Adult LoRA Browser)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADULT_LORA_CATEGORIES = {
    "全て": [],  # populated dynamically
    "体位 (Position)": [
        "DoggystylePOV", "cowgirl-2.0", "cowgirl_position_nonpov",
        "MissionaryVaginal-v2", "MS_Real_Lite_Carrying_Sex",
        "ridingsexscene_lora", "p0vr3vc0wg1rl", "doggy_style_sex_front_view",
        "properkissing",
    ],
    "表情・オーガズム (Expression)": [
        "edgOrgasm_v2", "orgasm_face_v10", "orgasmface_SDXL",
    ],
    "衣装 (Clothing)": [
        "Bikini_02_lora", "Kimono_v1", "Kimono Dress", "realistic_kimono_clothes",
        "Long Sexy Dress", "Sexy Underwear 5", "Wetshirt",
        "flamenco_dance_dress", "aodaimong", "oversized_coat",
        "Female_Balaclava", "Clothing - Hydro Armor",
        "jkCutOutTriangleBikiniFull",
    ],
    "ボディ (Body)": [
        "jkBigNaturalBreastsT02Lite", "puffies_v1", "Tanlines05",
        "Exposed-BP-4", "FappXL", "SexyConiAIv2p",
        "b1keg1rl", "w3t_SDXL", "sextowels_XL",
        "Barracuda_MaleClimax",
    ],
    "スタイル (Style)": [
        "Film lora", "Japanese_INK", "pop_art_v2",
        "Graphic_Novel_Illustration", "Gaz_Ink",
        "business data flat", "tech_startup_illustration",
        "oldjpposter", "jp70",
    ],
    "水中・シーン (Scene)": [
        "underwater", "underwater-photos", "Water_Nymphs_Swimming",
        "jkUnderwaterShot", "reelunderwater", "snorkel-LDv1",
        "scubagear2023",
    ],
    "サイバー・SF (Cyber/SF)": [
        "Cyber_Background_sdxl", "Sci-fi_Environments_sdxl",
        "MechStyle V1", "Samuraibot", "Metropolis City",
        "Terminate_yiu", "cybermask2023", "reelmech1v2",
        "toxic", "superhero_last", "AnonymousMaskNeon",
    ],
    "日本・和風 (Japanese)": [
        "GeishaXL", "Kamon-NoTags", "kitsunev0.4",
        "female samurai style", "urbansamurai_v0.3", "samurai_punk02",
        "ARWsamurai", "LongTchi", "danchi",
    ],
    "セレブ・タレント (Celebrity)": [
        "AnaDeArmasV2Dogu", "Ana in Blade Runner 2049",
        "hanazawakanaV3", "azusa yamamoto", "shoko takahashi",
        "mai haruna", "momo takai", "ichirofull", "joshiana",
        "DDmiss",
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quality Presets — ワンクリック画質設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUALITY_PRESETS = {
    "Draft (高速)": {
        "steps": 12,
        "cfg": 7.0,
        "hires_fix": False,
        "face_detailer": False,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "description": "素早くプレビュー。低解像度・低ステップ。",
    },
    "Standard (標準)": {
        "steps": 25,
        "cfg": 7.0,
        "hires_fix": False,
        "face_detailer": False,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "description": "バランスの取れた品質。普段使い向け。",
    },
    "High (高画質)": {
        "steps": 30,
        "cfg": 7.0,
        "hires_fix": True,
        "hires_scale": 1.5,
        "hires_denoise": 0.45,
        "hires_steps": 18,
        "upscale_model": "4x-UltraSharp.pth",
        "face_detailer": True,
        "face_denoise": 0.4,
        "face_guide_size": 512,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "description": "Hires Fix + FaceDetailer。顔・ディテール大幅改善。",
    },
    "Ultra (最高画質)": {
        "steps": 35,
        "cfg": 7.0,
        "hires_fix": True,
        "hires_scale": 1.75,
        "hires_denoise": 0.4,
        "hires_steps": 22,
        "upscale_model": "4x-UltraSharp.pth",
        "face_detailer": True,
        "face_denoise": 0.35,
        "face_guide_size": 768,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "description": "最高品質。Hires 1.75x + FaceDetailer 768px。時間かかるが圧倒的高画質。",
    },
}


def get_quality_preset(preset_key):
    """Get quality preset settings. Returns dict or None."""
    return QUALITY_PRESETS.get(preset_key)


def apply_quality_to_params(preset_key, current_params=None):
    """Apply a quality preset, returning a merged dict of generation parameters.

    current_params keys: steps, cfg, sampler, scheduler,
                         hires_fix, hires_scale, hires_denoise, hires_steps, upscale_model,
                         face_detailer, face_denoise, face_guide_size
    """
    preset = QUALITY_PRESETS.get(preset_key)
    if not preset:
        return current_params or {}
    merged = dict(current_params or {})
    for k, v in preset.items():
        if k != "description":
            merged[k] = v
    return merged


def filter_loras_by_category(all_loras, category):
    """Filter available LoRAs by category. Returns matching LoRA filenames."""
    if category == "全て" or category not in ADULT_LORA_CATEGORIES:
        return all_loras

    keywords = ADULT_LORA_CATEGORIES[category]
    if not keywords:
        return all_loras

    matched = ["None"]
    for lora in all_loras:
        if lora == "None":
            continue
        lora_lower = lora.lower()
        for kw in keywords:
            if kw.lower() in lora_lower:
                matched.append(lora)
                break
    return matched


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Optimal Settings (per checkpoint)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_OPTIMAL_SETTINGS = {
    # SD1.5 Models
    "eroticVision_v4": {"steps": 30, "cfg": 7, "sampler": "dpmpp_2m", "scheduler": "karras", "w": 512, "h": 768},
    "uberRealisticPornMerge_urpmv13": {"steps": 28, "cfg": 7, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 512, "h": 768},
    "Realistic_Vision_V5.1": {"steps": 30, "cfg": 7, "sampler": "dpmpp_2m", "scheduler": "karras", "w": 512, "h": 768},
    "realisticVisionV60B1": {"steps": 28, "cfg": 7, "sampler": "dpmpp_2m", "scheduler": "karras", "w": 512, "h": 768},
    "DreamShaper_8": {"steps": 25, "cfg": 7, "sampler": "dpmpp_2m", "scheduler": "karras", "w": 512, "h": 768},
    "endlessreality_v6": {"steps": 30, "cfg": 7, "sampler": "euler_ancestral", "scheduler": "normal", "w": 512, "h": 768},
    "cardosAnime_v20": {"steps": 28, "cfg": 7, "sampler": "euler_ancestral", "scheduler": "normal", "w": 512, "h": 768},
    "icbinpICantBelieveIts_lcm": {"steps": 6, "cfg": 1.8, "sampler": "lcm", "scheduler": "normal", "w": 512, "h": 768},
    "onlyfornsfw118_turboLCM": {"steps": 8, "cfg": 2.0, "sampler": "lcm", "scheduler": "normal", "w": 512, "h": 768},
    "devlishphotorealism_v40": {"steps": 30, "cfg": 7, "sampler": "dpmpp_2m", "scheduler": "karras", "w": 512, "h": 768},
    "fennfoto_ff2": {"steps": 28, "cfg": 7, "sampler": "euler_ancestral", "scheduler": "normal", "w": 512, "h": 768},
    "angraRealflex_v60F": {"steps": 25, "cfg": 7, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 512, "h": 768},
    "0001semirealistic_v54": {"steps": 28, "cfg": 7, "sampler": "dpmpp_2m", "scheduler": "karras", "w": 512, "h": 768},
    "povSkinTexture": {"steps": 30, "cfg": 7, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 512, "h": 768},
    # SDXL Models
    "pyrosNSFWSDXL_v04": {"steps": 25, "cfg": 5, "sampler": "euler_ancestral", "scheduler": "normal", "w": 1024, "h": 1536},
    "sdxlYamersRealisticNSFW_v5TX": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "miamodelSFWNSFWSDXL_v30": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "omnigenxlNSFWSFW_v10": {"steps": 25, "cfg": 5, "sampler": "euler_ancestral", "scheduler": "normal", "w": 1024, "h": 1536},
    "realismEngineSDXL_v20VAE": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "realvisxlV30Turbo": {"steps": 8, "cfg": 2.0, "sampler": "dpmpp_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "realityvisionSDXL_v20": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "colossusProjectXLSFW_v53Trained": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "socarealismXL_vae30": {"steps": 25, "cfg": 5, "sampler": "euler_ancestral", "scheduler": "normal", "w": 1024, "h": 1536},
    "wildcardXL_wildcardXLV2": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "xenogasmNSFWSemiReal_v5": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "ultriumV60SDXLVAE": {"steps": 25, "cfg": 5, "sampler": "euler_ancestral", "scheduler": "normal", "w": 1024, "h": 1536},
    "segmindSSD1B_v10": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1024},
    "tfvSDXLBAKED": {"steps": 25, "cfg": 5, "sampler": "euler_ancestral", "scheduler": "normal", "w": 1024, "h": 1536},
    "pixelwave_06": {"steps": 25, "cfg": 5, "sampler": "euler_ancestral", "scheduler": "normal", "w": 1024, "h": 1024},
    "sd_xl_refiner_1.0": {"steps": 25, "cfg": 5, "sampler": "euler_ancestral", "scheduler": "normal", "w": 1024, "h": 1024},
    "realisticStockPhoto_v10": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
    "M4RV3LSDUNGEONSNEWV40COMICS": {"steps": 25, "cfg": 5, "sampler": "euler_ancestral", "scheduler": "normal", "w": 1024, "h": 1536},
    "universestable_v80": {"steps": 25, "cfg": 5, "sampler": "dpmpp_2m_sde", "scheduler": "karras", "w": 1024, "h": 1536},
}


def get_model_settings(model_filename):
    """Get optimal settings for a given model filename. Returns dict or None."""
    if not model_filename:
        return None
    model_lower = model_filename.lower()
    for key, settings in MODEL_OPTIMAL_SETTINGS.items():
        if key.lower() in model_lower:
            return settings
    return None
