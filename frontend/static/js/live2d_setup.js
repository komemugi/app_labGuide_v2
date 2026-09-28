// ==========================================
// Live2D 初期化と描画制御 (PixiJS + pixi-live2d-display)
// ==========================================

const { Live2DModel } = PIXI.live2d;

let app;             // PixiJSアプリケーション
let live2dModel;     // 読み込んだLive2Dモデルのインスタンス

// 瞬き管理用変数
let isBlinking = false;
let baseEyeOpen = 1.0; // 瞳の基本開き具合（1.0が完全に開いている状態）

// 感情の目標値（滑らかに変化させるための変数）
let targetParameters = {
    'face_emoEffect': 0, // 表情_感情効果 (-2:青線+青い顔, -1:青い顔, 0:なし, 1:頬染)
    'eyebrow_form': 0,   // 眉_変形 (-1:恐/悲, 0:中立/期待/喜/信/驚, 1:怒/嫌)
    'ParamEyeLOpen': 1,  // 瞳_開閉 (0:閉じる, 1:開く) 
    'eye_special': 0,    // 瞳_特殊 (-1:白黒目, 0:中立/怒/悲, 1:><)
    'eye_green': 0,      // 瞳_緑 (-1:瞳縮小, 0:中立/怒/悲, 1:瞳キラキラ)
    'ParamMouthForm': 0,  // 口_変形 (-1:への字, 0:にこり, 1:まがお)
    // 'ahoge': 0,          // アホ毛 (0:通常, 1:揺れる) sin波で自動的に揺らすので、ここでは固定値
    // 'ParamHairFront': 0, // 髪_前 (0:中立, 1:少しふわっと) 同上
};

// ==========================================
// 1. 初期化処理 (画面読み込み時に実行)
// ==========================================
async function initLive2D() {
    const canvas = document.getElementById('live2d-canvas');
    const container = document.getElementById('live2d-canvas-container');

    // PixiJSアプリケーションの立ち上げ
    app = new PIXI.Application({
        view: canvas,
        width: 1000, 
        height: 1000,
        transparent: true,
        backgroundAlpha: 0,
        antialias: true, // 線を綺麗にする
    });

    try {
        // モデルの読み込み（パスは環境に合わせて調整してください）
        const modelPath = "/static/assets/live2d_model/koka_fumi.model3.json";
        
        live2dModel = await Live2DModel.from(modelPath);
        app.stage.addChild(live2dModel);

        // モデルのサイズと位置の調整（数値は表示を見ながら調整してください）
        live2dModel.scale.set(0.45); // 0.525 よさそう モデル全体の縮小率
        
        live2dModel.y = container.clientHeight / 2  //- (live2dModel.height / ); // 少し下げる
        live2dModel.x = app.screen.width / 2 - (live2dModel.width / 2); // 真ん中に設置
        // live2dModel.y = app.screen.height - live2dModel.height; // キャンバスの底に接地


        // 毎フレーム（1秒間に60回）実行されるアップデート処理を登録
        app.ticker.add(updateFrame);

        console.log("Live2Dモデルの読み込みに成功しました！");

        scheduleNextBlink(); // 瞬きのスケジュール開始

        // デバッグ用: モデルが持っている全てのパラメータIDをコンソールに表示
        console.log("モデルが持っている全てのID:", live2dModel.internalModel.coreModel._parameterIds);

    } catch (error) {
        console.error("Live2Dモデルの読み込みに失敗しました:", error);
    }
}

// ==========================================
// 2. 毎フレームの更新処理（滑らかな変形とリップシンク）
// ==========================================
function updateFrame() {
    if (!live2dModel) return;

    const coreModel = live2dModel.internalModel.coreModel;

    // A. 感情パラメータの滑らかな移行（補間）
    // ※ここで ParamMouthForm は常に 0（にこり）に向かって固定されるようになります
    for (const [paramId, targetValue] of Object.entries(targetParameters)) {
        let currentValue = coreModel.getParameterValueById(paramId);
        currentValue += (targetValue - currentValue) * 0.1;
        coreModel.setParameterValueById(paramId, currentValue);
    }

    // B. リップシンク（音声波形からの口パク）
    if (analyser && isAudioPlaying) {
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);

        // 音量の平均値を計算

        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
        }

        const volume = sum / dataArray.length;

        // 音量を 0.0 〜 1.0 の範囲に正規化（volume / xのxは感度調整用のマジックナンバー。声の大きさに合わせて調整）
        const mouthOpenY = Math.min(1.0, volume / 60.0); // 分母を小さくすると口が大きく開きやすくなる

        // 口の開閉パラメータに強制上書き（喋っている時だけ）
        // coreModel.setParameterValueById('ParamMouthOpenClose', mouthOpenY);
        coreModel.setParameterValueById('ParamMouthOpenY', mouthOpenY); // 口の上下の変形も同時に操作

        // // 平均値ではなく「一番大きい音量（最大値）」を取得する
        // let maxVolume = 0;
        // for (let i = 0; i < dataArray.length; i++) {
        //     if (dataArray[i] > maxVolume) {
        //         maxVolume = dataArray[i];
        //     }
        // }

        // // maxVolume は 0〜255 の値を取ります。120付近で全開になるように調整。
        // const mouthOpenY = Math.min(1.0, maxVolume / 120.0);

        // // 口の開閉パラメータのみを操作（変形は上で0に固定されているため操作不要）
        // coreModel.setParameterValueById('ParamMouthOpenClose', mouthOpenY);
    }

    // D.sin波を使ったアホ毛と前髪の自動揺らし（表情変化とは独立して常に揺れる）
    // Draw.now()を使って、常に一定の速度で揺れるようにする
    const time = Date.now() / 1000; // 秒単位の時間

    // アホ毛の揺れ（0.5秒周期でゆらゆら）
    const ahogeSpeed = 2.0; // 揺れの速さ(大きくするほど揺れる)
    const ahogeValue = (Math.sin(time * ahogeSpeed) + 1) / 2; // 0〜1の範囲で変化
    coreModel.setParameterValueById('ahoge', ahogeValue); // アホ毛のパラメータIDに合わせて変更

    // 前髪の揺れ（0.7秒周期でゆらゆら）
    const hairSpeed = 1.2 ; // 揺れの速さ(大きくするほど揺れる)
    const hairFrontValue = Math.sin(time * hairSpeed);
    coreModel.setParameterValueById('ParamHairFront', hairFrontValue); // 髪の前のパラメータIDに合わせて変更
}

// 瞬き制御用の関数
function blink() {
    // 特殊な瞳の時には瞬きをしない
    if (targetParameters['eye_special'] !== 0) {
        scheduleNextBlink();
        return;
    }

    isBlinking = true;
    targetParameters['ParamEyeLOpen'] = 0; // 瞳を閉じる

    setTimeout(() => {
        targetParameters['ParamEyeLOpen'] = baseEyeOpen; // もとの開き具合に戻す
        isBlinking = false;
        scheduleNextBlink();
    }, 150); // 瞳を閉じた状態を150ms間保持
}

function scheduleNextBlink() {
    // 3 ~ 7秒のランダムな間隔で次の瞬きをスケジュール
    const nextTime = Math.random() * 4000 + 3000;
    setTimeout(blink, nextTime);
}

// ==========================================
// 3. 感情の変更インターフェース (index.htmlから呼ばれる)
// ==========================================
// backendの配列順序: [joy, trust, fear, surprise, sadness, disgust, anger, anticipation]


function changeLive2DEmotion(emotionStateArray) {
    if (!live2dModel) return;

    // 最も強い感情のインデックスと値を取得
    const maxVal = Math.max(...emotionStateArray);
    const maxIdx = emotionStateArray.indexOf(maxVal);

    // デフォルト（中立）の目標値にリセット
    targetParameters['face_emoEffect'] = 0;
    targetParameters['eyebrow_form'] = 0;

    baseEyeOpen = 1.0; // デフォルトは完全に開いている状態
    // targetParameters['ParamEyeLOpen'] = 1; // 通常は開いている
    targetParameters['eye_special'] = 0;
    targetParameters['eye_green'] = 0;
    targetParameters['ParamMouthForm'] = 0; // 中立は「にこり」指定
    targetParameters['ahoge'] = 0; // アホ毛は通常
    targetParameters['ParamHairFront'] = 0; // 髪の前は中立

    // // 感情が弱い（中立）場合はデフォルトのままリターン
    // if (maxVal < 5.0) return; 
    // 感情が弱い（中立）場合は、瞬き中じゃなければ目を開けてリターン
    if (maxVal < 5.0) {
        if (!isBlinking) targetParameters['ParamEyeLOpen'] = baseEyeOpen;
        return; 
    }

    // 最大感情ごとのパラメータ割り当て
    // にこり 以外の表情変化を禁止にしたバージョン、表情差分のキー割り当てがおかしかったので暫定的な処置
    switch (maxIdx) {
        case 0: // 喜び (Joy)
            targetParameters['face_emoEffect'] = 1; 
            targetParameters['eye_special'] = 1;    
            break;
        case 1: // 信頼 (Trust)
            targetParameters['face_emoEffect'] = 1; 
            // targetParameters['ParamEyeLOpen'] = 0.8; 
            baseEyeOpen = 0.8; // ★直接上書きせず、基準値を0.8にする
            break;
        case 2: // 恐れ (Fear)
            targetParameters['face_emoEffect'] = -1; 
            targetParameters['eyebrow_form'] = -1;       
            targetParameters['eye_special'] = -1;    
            break;
        case 3: // 驚き (Surprise)
            targetParameters['eye_green'] = -1;      
            break;
        case 4: // 悲しみ (Sadness)
            targetParameters['eyebrow_form'] = -1;       
            break;
        case 5: // 嫌悪 (Disgust)
            targetParameters['face_emoEffect'] = -2; 
            targetParameters['eyebrow_form'] = 1;        
            targetParameters['eye_green'] = -1;      
            break;
        case 6: // 怒り (Anger)
            targetParameters['eyebrow_form'] = 1;        
            break;
        case 7: // 期待 (Anticipation)
            targetParameters['eye_green'] = 1;       
            break;
    }

     // baseEyeOpen を targetParameters に反映 (瞬き中でなければ、目の開き具合を更新    
    if (!isBlinking) {
        targetParameters['ParamEyeLOpen'] = baseEyeOpen;
    }

    // switch (maxIdx) {
    //     case 0: // 喜び (Joy)
    //         targetParameters['face_emoEffect'] = 1; // 頬染
    //         targetParameters['eye_special'] = 1;    // ><
    //         break;
    //     case 1: // 信頼 (Trust)
    //         targetParameters['face_emoEffect'] = 1; // 頬染
    //         targetParameters['ParamEyeLOpen'] = 0.8; // 少し閉じる
    //         break;
    //     case 2: // 恐れ (Fear)
    //         targetParameters['face_emoEffect'] = -1; // 青い顔
    //         targetParameters['eyebrow_form'] = -1;       
    //         targetParameters['eye_special'] = -1;    // 白黒目
    //         targetParameters['ParamMouthForm'] = 1;  // 真顔
    //         break;
    //     case 3: // 驚き (Surprise)
    //         targetParameters['eye_green'] = -1;      // 瞳縮小
    //         targetParameters['ParamMouthForm'] = 0;  // にこり
    //         break;
    //     case 4: // 悲しみ (Sadness)
    //         targetParameters['eyebrow_form'] = -1;       
    //         targetParameters['ParamMouthForm'] = -1;  // への字
    //         break;
    //     case 5: // 嫌今 (Disgust)
    //         targetParameters['face_emoEffect'] = -2; // 青線+青い顔
    //         targetParameters['eyebrow_form'] = 1;        
    //         targetParameters['eye_green'] = -1;      // 瞳縮小
    //         targetParameters['ParamMouthForm'] = 1;  // 真顔
    //         break;
    //     case 6: // 怒り (Anger)
    //         targetParameters['eyebrow_form'] = 1;        
    //         targetParameters['ParamMouthForm'] = -1; // への字
    //         break;
    //     case 7: // 期待 (Anticipation)
    //         targetParameters['eye_green'] = 1;       // 瞳キラキラ
    //         break;
    // }
}

// ==========================================
// 4. 音声解析とリップシンクのセットアップ (index.htmlから呼ばれる)
// ==========================================
let audioContext;
let analyser;
let isAudioPlaying = false;

function setLive2DLipSyncAudio(audioElement) {
    // ユーザーの操作（クリック等）の後にAudioContextを作成・再開する必要がある
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256; // 解析の解像度
    }

    // resumeはPromiseなので await するか .then() で繋ぐ
    audioContext.resume().then(() => {
        console.log("AudioContext resumed:", audioContext.state);
    });

    // Audioタグから音声ストリームを抽出してアナライザーに接続
    // ※ MediaElementAudioSourceNode は1つのAudio要素につき1回しか作れないため
    if (!audioElement.sourceNodeAttached) {
        const source = audioContext.createMediaElementSource(audioElement);
        source.connect(analyser);
        analyser.connect(audioContext.destination); // スピーカーへ出力
        audioElement.sourceNodeAttached = true;
    }

    // 再生状態のフラグ管理
    audioElement.addEventListener('play', () => {
        isAudioPlaying = true;
        console.log("再生開始、リップシンク有効"); // デバッグ用
    });
    audioElement.addEventListener('ended', () => {
        isAudioPlaying = false;
        if (live2dModel) {
            live2dModel.internalModel.coreModel.setParameterValueById('ParamMouthOpenY', 0);
        }
    });
    audioElement.addEventListener('pause', () => isAudioPlaying = false);

    // すでに再生中だった場合の対処
    if (!audioElement.paused) {
        isAudioPlaying = true;
    }
}

// DOM読み込み完了時にLive2Dを初期化
window.addEventListener('DOMContentLoaded', initLive2D);