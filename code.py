import os
import logging
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import time
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)

class Qwen25VLInference:
    def __init__(self, checkpoint_path='Qwen/Qwen2.5-VL-7B-Instruct', device_map='auto'):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype='auto',
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.timeout_count = 5
        self.retry_count = 5
        
    def _create_prompt(self, question, options):
        base_prompt = (
            "Please answer the question with thinking step by step. "
            "You should output only the answer number.\n"
            f"Questions: {question}\n"
            f"Option 1: {options[0]} "
            f"2: {options[1]} "
            f"3: {options[2]} "
            f"4: {options[3]}"
        )
        return base_prompt
    
    def _extract_answer(self, response):
        # Try to find a single digit between 1 and 4
        match = re.search(r'(?<!\d)[1-4](?!\d)', response)
        if match:
            return int(match.group())
        return None
    
    def _process_single_sample(self, image_path, question, options):
        logging.info(f"Processing image: {image_path}")
        
        try:
            image = Image.open(os.path.join('images', image_path))
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None
            
        for attempt in range(self.retry_count):
            try:
                prompt = self._create_prompt(question, options)
                
                # Prepare input for the model
                inputs = self.processor(
                    text=[prompt],
                    images=[image],
                    return_tensors='pt'
                ).to(self.model.device)
                
                # Generate with timeout
                start_time = time.time()
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=False
                )
                
                if time.time() - start_time > self.timeout_count:
                    logging.warning(f"Timeout on attempt {attempt + 1}")
                    continue
                
                # Process output
                response = self.processor.decode(output[0], skip_special_tokens=True)
                answer = self._extract_answer(response)
                
                if answer is not None:
                    logging.info(f"Successfully processed {image_path}, Answer: {answer}")
                    return answer
                else:
                    logging.warning(f"Could not extract valid answer on attempt {attempt + 1}")
                    
            except Exception as e:
                logging.error(f"Error processing {image_path} on attempt {attempt + 1}: {e}")
                
        logging.error(f"Failed to process {image_path} after {self.retry_count} attempts")
        return None

def main():
    # Load data
    try:
        val_df = pd.read_csv('validation_without_answers.csv')
        logging.info(f"Loaded validation data with {len(val_df)} samples")
    except Exception as e:
        logging.error(f"Error loading validation data: {e}")
        return

    # Initialize model
    inference = Qwen25VLInference()
    
    # Create results dataframe
    results_df = pd.DataFrame(columns=['file_name', 'answer'])
    
    # Process each sample
    for idx, row in val_df.iterrows():
        options = [row['option1'], row['option2'], row['option3'], row['option4']]
        answer = inference.process_single_sample(row['file_name'], row['question'], options)
        
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
        logging.info(f"Processed {idx + 1}/{len(val_df)} samples")

    logging.info("Completed processing all samples")

if __name__ == "__main__":
    main()
