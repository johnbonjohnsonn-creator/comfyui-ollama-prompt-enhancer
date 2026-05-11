"""
ComfyUI Ollama Prompt Enhancer Node
A custom node for enhancing prompts using local Ollama models
"""

import requests
import json

def get_ollama_models(ollama_url="http://localhost:11434"):
    """Fetch available Ollama models"""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            return model_names if model_names else ["llama3.2:latest"]
        else:
            return ["llama3.2:latest"]
    except:
        return ["llama3.2:latest"]


class OllamaPromptEnhancer:
    """
    A ComfyUI node that uses local Ollama models to enhance text prompts
    """
    
    def __init__(self):
        self.type = "OllamaPromptEnhancer"
        self.output_node = False
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available models
        available_models = get_ollama_models()
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful landscape"
                }),
                "model": (available_models, {
                    "default": available_models[0] if available_models else "llama3.2:latest"
                }),
                "enhancement_instructions": ("STRING", {
                    "multiline": True,
                    "default": "Enhance this image generation prompt by adding vivid details, lighting, and artistic style. Keep it concise and focused on visual elements."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "ollama_url": ("STRING", {
                    "default": "http://localhost:11434"
                }),
            },
            "optional": {
                "max_tokens": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "original_prompt")
    FUNCTION = "enhance_prompt"
    CATEGORY = "conditioning"

    def enhance_prompt(self, prompt, model, enhancement_instructions, 
                      temperature, ollama_url, max_tokens=256):
        """
        Enhance a prompt using Ollama
        
        Args:
            prompt: The original prompt to enhance
            model: The Ollama model to use
            enhancement_instructions: Instructions for how to enhance
            temperature: Sampling temperature
            ollama_url: URL of the Ollama server
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (enhanced_prompt, original_prompt)
        """
        try:
            # Strip whitespace from model name to avoid issues
            model = model.strip()
            
            # Construct the full prompt
            full_prompt = f"{enhancement_instructions}\n\nOriginal prompt: {prompt}\n\nEnhanced prompt:"
            
            # Make request to Ollama API
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced = result.get("response", "").strip()
                
                # Clean up the response
                enhanced = enhanced.replace("\n\n", " ").replace("\n", " ")
                enhanced = enhanced.strip()
                
                print(f"[Ollama] Model: {model}")
                print(f"[Ollama] Original: {prompt}")
                print(f"[Ollama] Enhanced: {enhanced}")
                
                # Unload model from VRAM to free up memory
                try:
                    unload_response = requests.delete(
                        f"{ollama_url}/api/generate",
                        json={"model": model, "keep_alive": 0},
                        timeout=5
                    )
                    if unload_response.status_code == 200:
                        print(f"[Ollama] Unloaded model {model} from VRAM")
                    else:
                        print(f"[Ollama] Note: Model may still be in VRAM")
                except Exception as e:
                    print(f"[Ollama] Could not unload model: {e}")
                
                return (enhanced, prompt)
            else:
                print(f"[Ollama] Error: Status {response.status_code}")
                print(f"[Ollama] Response: {response.text}")
                print(f"[Ollama] Model requested: '{model}'")
                print(f"[Ollama] Available models: {get_ollama_models(ollama_url)}")
                return (prompt, prompt)
                
        except requests.exceptions.RequestException as e:
            print(f"[Ollama] Connection error: {str(e)}")
            print(f"[Ollama] Make sure Ollama is running at {ollama_url}")
            return (prompt, prompt)
        except Exception as e:
            print(f"[Ollama] Error enhancing prompt: {str(e)}")
            return (prompt, prompt)


class OllamaPromptEnhancerWithLoRA:
    """
    Enhanced version with LoRA-aware prompt enhancement
    """
    
    def __init__(self):
        self.type = "OllamaPromptEnhancerWithLoRA"
        self.output_node = False
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available models
        available_models = get_ollama_models()
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful landscape"
                }),
                "model": (available_models, {
                    "default": available_models[0] if available_models else "llama3.2:latest"
                }),
                "style_focus": (["general", "photorealistic", "artistic", "cinematic", "anime"], {
                    "default": "general"
                }),
                "detail_level": (["minimal", "moderate", "detailed", "very_detailed"], {
                    "default": "moderate"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "ollama_url": ("STRING", {
                    "default": "http://localhost:11434"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "original_prompt")
    FUNCTION = "enhance_prompt"
    CATEGORY = "conditioning"

    def enhance_prompt(self, prompt, model, style_focus, detail_level, 
                      temperature, ollama_url):
        """Enhanced prompt with style presets"""
        
        # Strip whitespace from model name
        model = model.strip()
        
        style_instructions = {
            "general": "Add vivid visual details and improve clarity.",
            "photorealistic": "Add photographic details like camera settings, lighting, and realistic textures. Use professional photography terminology.",
            "artistic": "Add artistic style, medium, techniques, and aesthetic qualities. Reference art movements or famous artists if appropriate.",
            "cinematic": "Add cinematic elements like camera angles, lighting mood, atmosphere, and film-like qualities.",
            "anime": "Add anime-specific details like art style, character features, and aesthetic elements common in anime."
        }
        
        detail_instructions = {
            "minimal": "Keep enhancements brief and focused.",
            "moderate": "Add a moderate amount of detail.",
            "detailed": "Add substantial descriptive detail.",
            "very_detailed": "Add extensive descriptive detail with specific elements."
        }
        
        enhancement_instructions = f"""You are a prompt enhancement assistant for image generation.

Style focus: {style_focus}
Detail level: {detail_level}

Instructions:
{style_instructions[style_focus]}
{detail_instructions[detail_level]}

Enhance the following prompt for image generation. Return ONLY the enhanced prompt, no explanations or preamble."""

        try:
            full_prompt = f"{enhancement_instructions}\n\nOriginal prompt: {prompt}\n\nEnhanced prompt:"
            
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "num_predict": 256
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced = result.get("response", "").strip()
                enhanced = enhanced.replace("\n\n", " ").replace("\n", " ").strip()
                
                print(f"[Ollama LoRA] Model: {model}")
                print(f"[Ollama LoRA] Original: {prompt}")
                print(f"[Ollama LoRA] Enhanced: {enhanced}")
                
                # Unload model from VRAM to free up memory
                try:
                    unload_response = requests.delete(
                        f"{ollama_url}/api/generate",
                        json={"model": model, "keep_alive": 0},
                        timeout=5
                    )
                    if unload_response.status_code == 200:
                        print(f"[Ollama LoRA] Unloaded model {model} from VRAM")
                except Exception as e:
                    print(f"[Ollama LoRA] Could not unload model: {e}")
                
                return (enhanced, prompt)
            else:
                print(f"[Ollama LoRA] Error: Status {response.status_code}")
                print(f"[Ollama LoRA] Response: {response.text}")
                return (prompt, prompt)
                
        except Exception as e:
            print(f"[Ollama LoRA] Error: {str(e)}")
            return (prompt, prompt)


# Node registration
NODE_CLASS_MAPPINGS = {
    "OllamaPromptEnhancer": OllamaPromptEnhancer,
    "OllamaPromptEnhancerWithLoRA": OllamaPromptEnhancerWithLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptEnhancer": "Ollama Prompt Enhancer",
    "OllamaPromptEnhancerWithLoRA": "Ollama Prompt Enhancer (Style Presets)",
}
