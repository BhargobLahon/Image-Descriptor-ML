import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
)
import cv2
from gtts import gTTS
import pygame
import io
import tempfile
import warnings
warnings.filterwarnings('ignore')

class MultilingualImageToSpeech:
    def __init__(self):
        """Initialize the multilingual image-to-speech model"""
        print("Initializing Multilingual Image-to-Speech Model...")
        
        # Initialize image captioning model (BLIP)
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Initialize translation models
        self.init_translation_models()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Language codes for gTTS
        self.lang_codes = {
            'english': 'en',
            'hindi': 'hi',
            'bengali': 'bn'
        }
        
        print("Model initialization complete!")
    
    def init_translation_models(self):
        """Initialize translation models for Hindi and Bengali"""
        try:
            # For Hindi translation
            self.hindi_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
            self.hindi_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
            
            # For Bengali translation
            self.bengali_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-bn")
            self.bengali_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-bn")
            
            print("Translation models loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load translation models: {e}")
            self.hindi_tokenizer = self.hindi_model = None
            self.bengali_tokenizer = self.bengali_model = None
    
    def preprocess_image(self, image_path):
        """Preprocess image for the model"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')
            
            # Resize image if too large
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def generate_caption(self, image):
        """Generate English caption for the image"""
        try:
            # Process image
            inputs = self.caption_processor(image, return_tensors="pt")
            
            # Generate caption
            with torch.no_grad():
                out = self.caption_model.generate(**inputs, max_length=50, num_beams=4)
            
            # Decode caption
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Unable to describe the image."
    
    def translate_text(self, text, target_language):
        """Translate text to target language"""
        try:
            if target_language.lower() == 'english':
                return text
            
            elif target_language.lower() == 'hindi' and self.hindi_model:
                inputs = self.hindi_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.hindi_model.generate(**inputs, max_length=128, num_beams=4)
                translated = self.hindi_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translated
            
            elif target_language.lower() == 'bengali' and self.bengali_model:
                inputs = self.bengali_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.bengali_model.generate(**inputs, max_length=128, num_beams=4)
                translated = self.bengali_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translated
            
            else:
                print(f"Translation not available for {target_language}")
                return text
                
        except Exception as e:
            print(f"Error translating to {target_language}: {e}")
            return text
    
    def text_to_speech(self, text, language='english', save_path=None):
        """Convert text to speech"""
        try:
            lang_code = self.lang_codes.get(language.lower(), 'en')
            
            # Create TTS object
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            if save_path:
                tts.save(save_path)
                print(f"Audio saved to {save_path}")
            else:
                # Save to temporary file and play
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tts.save(tmp_file.name)
                    self.play_audio(tmp_file.name)
                    os.unlink(tmp_file.name)  # Delete temporary file
                    
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
    
    def play_audio(self, audio_path):
        """Play audio file"""
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
                
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def process_image_to_speech(self, image_path, save_audio=True):
        """Complete pipeline: image -> text -> speech (always in all three languages)"""
        results = {}
        languages = ['english', 'hindi', 'bengali']  # Always process all three languages
        
        print(f"Processing image: {image_path}")
        print("Generating descriptions in English, Hindi, and Bengali...")
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            return results
        
        # Generate English caption
        english_caption = self.generate_caption(image)
        print(f"English Description: {english_caption}")
        
        # Process for each language
        for language in languages:
            print(f"\nProcessing for {language}...")
            
            if language.lower() == 'english':
                description = english_caption
            else:
                description = self.translate_text(english_caption, language)
            
            print(f"{language.title()} Description: {description}")
            
            # Convert to speech (always save audio)
            audio_path = f"description_{language.lower()}.mp3"
            self.text_to_speech(description, language, audio_path)
            
            results[language] = description
        
        return results
    
    def batch_process(self, image_folder):
        """Process multiple images in a folder (always in all three languages)"""
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(image_folder, filename)
                print(f"\n{'='*50}")
                print(f"Processing: {filename}")
                print('='*50)
                
                results = self.process_image_to_speech(image_path)
                
                # Save results to text file
                result_file = f"results_{filename.split('.')[0]}.txt"
                with open(result_file, 'w', encoding='utf-8') as f:
                    for lang, desc in results.items():
                        f.write(f"{lang.title()}: {desc}\n")
                
                print(f"Results saved to {result_file}")

def main():
    """Main function to demonstrate the model"""
    # Initialize the model
    model = MultilingualImageToSpeech()
    
    # Example usage
    print("\n" + "="*60)
    print("MULTILINGUAL IMAGE-TO-SPEECH MODEL")
    print("="*60)
    
    # Example 1: Process single image
    print("\nExample 1: Single Image Processing")
    print("-" * 40)
    
    # You can replace this with your image path
    image_path = "sample_image.jpg"  # Replace with actual image path
    
    # Check if image exists
    if os.path.exists(image_path):
        results = model.process_image_to_speech(image_path)
        
        print("\nResults:")
        for lang, description in results.items():
            print(f"{lang.title()}: {description}")
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path.")
    
    # Example 2: Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Process an image")
        print("2. Process folder of images")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                print("Processing image in English, Hindi, and Bengali...")
                
                results = model.process_image_to_speech(img_path)
                
                print("\nResults:")
                for lang, description in results.items():
                    print(f"{lang.title()}: {description}")
                    
                print("\nAudio files saved:")
                print("- description_english.mp3")
                print("- description_hindi.mp3") 
                print("- description_bengali.mp3")
            else:
                print("Image not found!")
        
        elif choice == '2':
            folder_path = input("Enter folder path: ").strip()
            if os.path.exists(folder_path):
                print("Processing all images in English, Hindi, and Bengali...")
                model.batch_process(folder_path)
            else:
                print("Folder not found!")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    # Install required packages
    print("Required packages:")
    print("pip install torch torchvision transformers pillow opencv-python gtts pygame numpy")
    print("\nMake sure you have an internet connection for the first run to download models.")
    print("\n" + "="*60)
    
    main()