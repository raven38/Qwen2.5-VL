from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio
from PIL import Image
import io
import logging
import os
from contextlib import asynccontextmanager

# 既存のQwen VLのインポート
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text

# モデルのグローバルインスタンス
model = None
processor = None

class QuestionRequest(BaseModel):
    question: str
    options: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時にモデルをロード
    global model, processor
    checkpoint_path = 'Qwen/Qwen2.5-VL-7B-Instruct'
    
    logging.info("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype='auto',
        attn_implementation='flash_attention_2',
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    logging.info("Model loaded successfully")
    
    yield
    
    # シャットダウン時のクリーンアップ
    logging.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """一時ファイルとして画像を保存"""
    try:
        os.makedirs("tmp", exist_ok=True)
        content = await upload_file.read()
        file_path = f"tmp/{upload_file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_inference(image_path: str):
    """非同期で推論を実行"""
    try:
        # メッセージの作成
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": f"'この見積書の内容をcsvおよびフォーマットが同じ出力をするhtmlファイルにしてください"}
                ]
            }
        ]
        
        # Vision情報の処理
        image_inputs, video_inputs = process_vision_info(messages)
        if not image_inputs:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # テキストの処理
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 入力の準備
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt'
        ).to(model.device)
        
        # 推論の実行
        outputs = model.generate(
            **inputs,
            max_new_tokens=1048576,
            do_sample=True,
            top_p=0.8,
            temperature=0.7
        )
        generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        return {"response": _parse_text(_parse_text(response))}
    
    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 一時ファイルの削除
        if os.path.exists(image_path):
            os.remove(image_path)

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
):
    """
    画像とテキストを受け取って推論を実行するエンドポイント
    """
    try:
        # 画像の一時保存
        image_path = await save_upload_file_tmp(image)
        
        
        # 推論の実行
        result = await process_inference(image_path)
        return result
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 本番環境ではFalse
        workers=1  # GPUを使用する場合は1がおすすめ
    )
