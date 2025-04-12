def _gc():
    """Garbage collection helper with CUDA memory cleanup."""
    import gc
    
    # 通常のガベージコレクション
    gc.collect()
    
    # PyTorchのキャッシュをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 全CUDAデバイスの同期
        torch.cuda.synchronize()

def call_model(model, processor, messages):
    try:
        messages = _transform_messages(messages)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 入力テンソルの作成を with torch.no_grad() で囲む
        with torch.no_grad():
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors='pt'
            )
            inputs = inputs.to(model.device)

            streamer = TextIteratorStreamer(
                processor.tokenizer,
                timeout=20.0,
                skip_prompt=True,
                skip_special_tokens=True
            )

            gen_kwargs = {
                'max_new_tokens': 512,
                'streamer': streamer,
                **inputs
            }
            
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                yield generated_text

    finally:
        # 使用したテンソルの明示的な解放
        if 'inputs' in locals():
            for tensor in inputs.values():
                if hasattr(tensor, 'cpu'):
                    tensor.cpu()
                del tensor
            del inputs
        
        # ガベージコレクションの実行
        _gc()

def create_predict_fn():
    def predict(_chatbot, task_history):
        try:
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
                
            print('User: ' + _parse_text(query))
            history_cp = copy.deepcopy(task_history)
            full_response = ''
            messages = []
            content = []
            
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    if _is_video_file(q[0]):
                        content.append({'video': f'file://{q[0]}'})
                    else:
                        content.append({'image': f'file://{q[0]}'})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            # モデル推論の実行
            for response in call_model(model, processor, messages):
                _chatbot[-1] = (_parse_text(chat_query),
                              _remove_image_special(_parse_text(response)))
                yield _chatbot
                full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print('Qwen-VL-Chat: ' + _parse_text(full_response))
            
        finally:
            # 使用済みの変数を明示的に解放
            del history_cp
            del messages
            del content
            if 'response' in locals():
                del response
            
            # メモリクリーンアップの実行
            _gc()
            
        yield _chatbot

    return predict

def _load_model_processor(args):
    device_map = 'cpu' if args.cpu_only else 'auto'
    
    # モデルの設定
    model_kwargs = {
        'device_map': device_map,
        'torch_dtype': torch.float32 if args.cpu_only else torch.float16,
        # メモリ効率を改善するための追加設定
        'low_cpu_mem_usage': True,
        'offload_folder': 'offload'  # オフロード用のディレクトリ
    }
    
    if args.flash_attn2:
        model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    try:
        # モデルの読み込み
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint_path,
            **model_kwargs
        )
        
        # プロセッサの読み込み
        processor = AutoProcessor.from_pretrained(args.checkpoint_path)
        
        # モデルを評価モードに設定
        model.eval()
        
        return model, processor
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
