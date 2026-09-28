# backend/app/api/voicevox_client.py
import os
import requests
import json

class VoicevoxClient:
    def __init__(self, host="127.0.0.1", port=50021):
        self.base_url = f"http://{host}:{port}"
        # 春日部つむぎ（ノーマル）のキャラクターIDは 8 
        self.default_speaker_id = 8 

    def generate_audio(self, text: str, speaker_id: int = None, output_filename: str = "response.wav") -> str:
        """
        テキストから音声を生成し、指定したファイル名で保存する
        """
        # 保存先ディレクトリの確保（frontend/assets/audio）
        # ※Flaskのルートディレクトリからの相対パス
        save_dir = "../frontend/assets/audio"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, output_filename)
        
        target_speaker = speaker_id if speaker_id is not None else self.default_speaker_id

        # 1. audio_query (音声合成用のクエリを作成)
        query_payload = {"text": text, "speaker": target_speaker}
        query_res = requests.post(f"{self.base_url}/audio_query", params=query_payload)
        
        if query_res.status_code != 200:
            print(f"[VOICEVOX Error] audio_query failed: {query_res.text}")
            return None
            
        query_data = query_res.json()

        # 2. synthesis (クエリをもとにWAVデータを生成)
        synth_payload = {"speaker": target_speaker}
        synth_res = requests.post(
            f"{self.base_url}/synthesis", 
            params=synth_payload, 
            json=query_data
        )

        if synth_res.status_code != 200:
            print(f"[VOICEVOX Error] synthesis failed: {synth_res.text}")
            return None

        # 3. WAVファイルとして書き出し（ブラウザの「キャッシュ」を回避するため。何度質問しても最初の返答の音声しか再生されなくなるというバグを回避）
        with open(save_path, "wb") as f:
            f.write(synth_res.content)
            
        # フロントエンドからアクセス可能なURLパスを返す
        return f"/assets/audio/{output_filename}"