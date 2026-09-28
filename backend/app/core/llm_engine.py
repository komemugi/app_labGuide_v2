# backend/app/core/llm_engine.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import BitsAndBytesConfig

class LLMEngine:
  def __init__(self, model_id="LiquidAI/LFM2.5-1.2B-JP", quantization_8bit=False):
    """
    クラスの初期化時にモデルをロードする。
    """
    print(f"Loading model: {model_id}...")
    
    # トークナイザーのロード
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # パディングトークンの設定
    if self.tokenizer.pad_token_id is None:
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # モデルのロード
    if quantization_8bit:
      # 8bit量子化
      quantization_config = BitsAndBytesConfig(
                  load_in_8bit = True,
          )
      self.model = AutoModelForCausalLM.from_pretrained(
          model_id,
          device_map="auto",
          quantization_config=quantization_config, # 量子化設定反映
      )
    else:
      self.model = AutoModelForCausalLM.from_pretrained(
          model_id,
          device_map="auto",
          dtype=torch.bfloat16, # CPUでエラーが出たら torch.float32 に変更
      )
    
    # ストリーマーの準備（1文字ずつ表示するため）
    self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
    print("Model loaded successfully.")

#   def chat(self, prompt, system_prompt="あなたは親切なAIキャラクターです。"):
#   def chat(self, prompt, system_prompt="あなたは大学を案内するアシスタントです。日本語で自然に応答してください。"):
  def chat(self, prompt, system_prompt="あなたは誠実で優秀な日本人のアシスタントです。",
           history=None, do_sample=True, temperature=0.8, max_new_tokens=200):
    """
    history : [{"role": "user"/"assistant", "content": str}, ...] 過去のやり取り
    do_sample / temperature / max_new_tokens : 実験時に外から制御する
      ★実験では do_sample=False（greedy）にして再現性を確保する
    return: (生成テキスト, used_token_info)
    """
    # 1. メッセージの構築（system → 履歴 → 今回の発話）
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # 2. チャットテンプレートを文字列として取得
    prompt_str = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # 3. プロンプト文字列をトークナイズして Tensor に変換
    # これで inputs は確実に「input_ids」と「attention_mask」を持った辞書になる（エラー対策）
    inputs = self.tokenizer(
        prompt_str,
        return_tensors="pt",
        add_special_tokens=False
    ).to(self.model.device)

    print("Generating...")
    
    # 4. 生成実行
    # inputs の中身（input_ids, attention_mask）を ** で展開して渡します
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        streamer=self.streamer,
        pad_token_id=self.tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature   # greedy時は渡さない（警告防止）

    outputs = self.model.generate(**inputs, **gen_kwargs)

    # -----------------------------------------------------------------------------------------
    # トークン数の計算と表示：max_new_tokens を幾つに設定すべきかを確認するためのデバック用処理
    # -----------------------------------------------------------------------------------------
    input_len = len(inputs["input_ids"][0]) # 入力トークン数
    total_len = len(outputs[0])             # 総トークン数 (入力 + 出力)
    generated_len = total_len - input_len   # 生成されたトークン数

    used_token_info = (f"{'='*30}\n [Token Usage Report]\n Input Tokens : {input_len}\n"
                       f" Output Tokens: {generated_len}\n Total Tokens : {total_len}\n{'='*30}")

    # 5. 結果のデコード
    # 入力の長さ分をカットして応答部分だけを取り出す
    generated_tokens = outputs[0][input_len:]
    return self.tokenizer.decode(generated_tokens, skip_special_tokens=True), used_token_info