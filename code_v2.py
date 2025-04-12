import os
import logging
import pandas as pd

from PIL import Image
import time
import re
from threading import Thread

import re
from argparse import ArgumentParser
from threading import Thread

import tyro

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TextIteratorStreamer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)


def transform_messages(messages):
    """Transform messages to the required format."""
    transformed_messages = []
    for message in messages:
        new_content = []
        for item in message['content']:
            if isinstance(item, dict):
                if 'image' in item:
                    new_item = {'type': 'image', 'image': item['image']}
                elif 'text' in item:
                    new_item = {'type': 'text', 'text': item['text']}
                elif 'video' in item:
                    new_item = {'type': 'video', 'video': item['video']}
                else:
                    continue
                new_content.append(new_item)
        
        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)
    
    return transformed_messages

class Qwen25VLInference:
    def __init__(self, checkpoint_path='Qwen/Qwen2-VL-7B-Instruct', device_map='auto'):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype='auto',
            attn_implementation='flash_attention_2',
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.timeout_count = 5
        self.retry_count = 1
        
    def _create_messages(self, image_path, question, options):
        prompt = (
            "Please answer the question with thinking step by step. "
            "You should output only the answer number, i.e., 1, 2, 3, or 4.\n"
            f"Questions: {question}\n"
            f"Option 1: {options[0]} "
            f"2: {options[1]} "
            f"3: {options[2]} "
            f"4: {options[3]}"
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        return transform_messages(messages)
    
    def _extract_answer(self, response):
        match = re.search(r'(?<!\d)[1-4](?!\d)', response)
        if match:
            return int(match.group())
        return None
    
    def _process_single_sample(self, image_path, question, options):
        logging.info(f"Processing image: {image_path}")
        
        image_full_path = os.path.join('images', image_path)
        for attempt in range(self.retry_count):
            try:
                messages = self._create_messages(image_full_path, question, options)
                
                # Process vision information
                image_inputs, video_inputs = process_vision_info(messages)
                if not image_inputs:
                    logging.error(f"No valid image found for {image_path}")
                    return None
                
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Prepare inputs
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors='pt'
                ).to(self.model.device)
                
                # Set up streamer
                streamer = TextIteratorStreamer(
                    self.processor.tokenizer,
                    timeout=self.timeout_count,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                # Generate with streamer
                gen_kwargs = {
                    'max_new_tokens': 10,
                    'streamer': streamer,
                    **inputs
                }
                
                # Start generation in a separate thread
                self.model.generate(**gen_kwargs)
                
                # Collect generated text
                generated_text = ''
                start_time = time.time()
                try:
                    for new_text in streamer:
                        if time.time() - start_time > self.timeout_count:
                            raise TimeoutError("Generation timed out")
                        generated_text += new_text
                except (TimeoutError, Exception) as e:
                    logging.warning(f"Error during generation on attempt {attempt + 1}: {e}")
                    continue
                print(text, generated_text)
                answer = self._extract_answer(generated_text)
                if answer is not None:
                    logging.info(f"Successfully processed {image_path}, Answer: {answer}")
                    return answer
                else:
                    logging.warning(f"Could not extract valid answer on attempt {attempt + 1}")
                    
            except Exception as e:
                logging.error(f"Error processing {image_path} on attempt {attempt + 1}: {e}")
                
        logging.error(f"Failed to process {image_path} after {self.retry_count} attempts")
        return None

def main(
    checkpoint_path: str = 'Qwen/Qwen2-VL-7B-Instruct',
    eval_train: bool = False,
    eval_val: bool = False,
    eval_test: bool = True,
):
    # Initialize model
    inference = Qwen25VLInference(checkpoint_path=checkpoint_path)

    # Eval test data 
    if eval_test:
        # Load test data
        try:
            test_df = pd.read_csv('test_without_answers.csv')
            logging.info(f"Loaded test data with {len(test_df)} samples")
        except Exception as e:
            logging.error(f"Error loading validation data: {e}")
            return
        
        # Create results dataframe
        results_df = pd.DataFrame(columns=['file_name', 'answer'])
        
        # Process each sample
        for idx, row in test_df.iterrows():
            options = [row['option1'], row['option2'], row['option3'], row['option4']]
            answer = inference._process_single_sample(row['file_name'], row['question'], options)
            
            # Default to 1 if processing fails
            if answer is None:
                answer = 1
                logging.warning(f"Using default answer 1 for {row['file_name']}")
            
            # Add result to dataframe
            new_row = pd.DataFrame({
                'file_name': [row['file_name']],
                'answer': [answer]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            # Save intermediate results
            results_df.to_csv('submission.csv', index=False)
            
            # Log progress
            logging.info(f"Test: Processed {idx + 1}/{len(test_df)} samples")

    # Eval validation data 
    if eval_val:
        # Load validation data
        try:
            val_df = pd.read_csv('validation_without_answers.csv')
            logging.info(f"Loaded validation data with {len(val_df)} samples")
        except Exception as e:
            logging.error(f"Error loading validation data: {e}")
            return
        
        # Create results dataframe
        results_df = pd.DataFrame(columns=['file_name', 'answer'])
        
        # Process each sample
        for idx, row in val_df.iterrows():
            options = [row['option1'], row['option2'], row['option3'], row['option4']]
            answer = inference._process_single_sample(row['file_name'], row['question'], options)
            
            # Default to 1 if processing fails
            if answer is None:
                answer = 1
                logging.warning(f"Using default answer 1 for {row['file_name']}")
            
            # Add result to dataframe
            new_row = pd.DataFrame({
                'file_name': [row['file_name']],
                'answer': [answer]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            # Save intermediate results
            results_df.to_csv('sub_validation.csv', index=False)
            
            # Log progress
            logging.info(f"Val: Processed {idx + 1}/{len(val_df)} samples")

    # Eval train data 
    if eval_train:
        # Load train data
        try:
            train_df = pd.read_csv('train_with_answers.csv')
            logging.info(f"Loaded train data with {len(train_df)} samples")
        except Exception as e:
            logging.error(f"Error loading validation data: {e}")
            return
        
        # Create results dataframe
        results_train_df = pd.DataFrame(columns=['file_name', 'answer', 'gt'])
        
        # Process each sample show current accuracy
        for idx, row in train_df.iterrows():
            options = [row['option1'], row['option2'], row['option3'], row['option4']]
            answer = inference._process_single_sample(row['file_name'], row['question'], options)
            gt = row['answer']
            # Default to 1 if processing fails
            if answer is None:
                answer = 1
                logging.warning(f"Using default answer 1 for {row['file_name']}")
            
            # Add result to dataframe
            new_row = pd.DataFrame({
                'file_name': [row['file_name']],
                'answer': [answer],
                'gt' : [gt]
            })
            results_train_df = pd.concat([results_train_df, new_row], ignore_index=True)
            
            # Save intermediate results
            results_train_df.to_csv('train_answer.csv', index=False)
            
            # Log progress
            logging.info(f"Train: Processed {idx + 1}/{len(train_df)} samples, Current Accuracy: {results_train_df[results_train_df['answer'] == results_train_df['gt']].shape[0]/results_train_df.shape[0]}")

        logging.info("Final accuracy for train data: ", results_train_df[results_train_df['answer'] == results_train_df['gt']].shape[0]/results_train_df.shape[0])
    logging.info("Completed processing all samples")

if __name__ == "__main__":
    tyro.cli(main)