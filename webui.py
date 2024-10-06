# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
from functools import lru_cache
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import threading
from faster_whisper import WhisperModel
import gc

# Global variables to control audio generation
stop_generation_flag = threading.Event()
audio_chunks = []

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# Constants
PROMPT_SR = 16000
TARGET_SR = 22050
MAX_VAL = 0.8
DEFAULT_DATA = np.zeros(TARGET_SR)

inference_mode_list = ['Pre-trained Voice', '3s Rapid Cloning', 'Cross-lingual Cloning', 'Natural Language Control']
instruct_dict = {
    'Pre-trained Voice': '1. Select pre-trained voice\n2. Click generate audio button',
    '3s Rapid Cloning': '1. Select prompt audio file, or record prompt audio, note not to exceed 30s, if both are provided, prompt audio file takes priority\n2. Input prompt text\n3. Click generate audio button',
    'Cross-lingual Cloning': '1. Select prompt audio file, or record prompt audio, note not to exceed 30s, if both are provided, prompt audio file takes priority\n2. Click generate audio button',
    'Natural Language Control': '1. Select pre-trained voice\n2. Input instruct text\n3. Click generate audio button'
}
stream_mode_list = [('No', False), ('Yes', True)]

@lru_cache(maxsize=32)
def load_cached_wav(file_path: str, sample_rate: int) -> torch.Tensor:
    return load_wav(file_path, sample_rate)

def generate_seed() -> dict:
    return {
        "__type__": "update",
        "value": random.randint(1, 100000000)
    }

# orginal top_db: int = 60
def postprocess(speech: torch.Tensor, top_db: int = 40, hop_length: int = 220, win_length: int = 440) -> torch.Tensor:
    with ProcessPoolExecutor() as executor:
        future = executor.submit(librosa.effects.trim, speech.numpy(), top_db=top_db, frame_length=win_length, hop_length=hop_length)
        speech, _ = future.result()
    print(f"Postprocessed speech length: {len(speech)}")  # Debug print
    speech = torch.from_numpy(speech)
    if speech.abs().max() > MAX_VAL:
        speech = speech / speech.abs().max() * MAX_VAL
    speech = torch.cat([speech, torch.zeros(1, int(TARGET_SR * 0.2))], dim=1)
    return speech

def change_instruction(mode_checkbox_group: str) -> str:
    return instruct_dict[mode_checkbox_group]

def check_input_validity(mode: str, prompt_wav: Optional[str], prompt_text: str, instruct_text: str) -> Tuple[bool, str]:
    if mode in ['3s Rapid Cloning', 'Cross-lingual Cloning']:
        if prompt_wav is None:
            return False, 'Prompt audio is empty, did you forget to input prompt audio?'
        if prompt_text == '' and mode == '3s Rapid Cloning':
            return False, 'Prompt text is empty, did you forget to input prompt text?'
    elif mode == 'Natural Language Control':
        if instruct_text == '':
            return False, 'You are using Natural Language Control mode, please input instruct text'
    return True, ''



def generate_audio(tts_text: str, mode_checkbox_group: str, sft_dropdown: str, prompt_text: str, 
                   prompt_wav_upload: Optional[str], prompt_wav_record: Optional[str], instruct_text: str,
                   seed: int, stream: bool, speed: float, stop_generation_flag: threading.Event):
    prompt_wav = prompt_wav_upload or prompt_wav_record

    is_valid, error_message = check_input_validity(mode_checkbox_group, prompt_wav, prompt_text, instruct_text)
    if not is_valid:
        gr.Warning(error_message)
        yield (TARGET_SR, DEFAULT_DATA)
        return

    set_all_random_seed(seed)

    # Clean VRAM before processing new tts_text
    torch.cuda.empty_cache()

    # Split the input text into paragraphs
    paragraphs = re.split(r'\n\s*\n', tts_text.strip())
    
    for i, paragraph in enumerate(paragraphs):
        if stop_generation_flag.is_set():
            break

        paragraph = paragraph.strip()
        print(f"Processing paragraph {i+1}/{len(paragraphs)}: {paragraph[:50]}...")


        if not paragraph:
            continue

        if i > 0:
            yield (TARGET_SR, np.zeros(int(TARGET_SR * 0.05))) # adds 0.5 seconds silent before each paragraph (not first paragraph)

        if mode_checkbox_group == 'Pre-trained Voice':
            generator = cosyvoice.inference_sft(paragraph, sft_dropdown, stream=stream, speed=speed)
        elif mode_checkbox_group == '3s Rapid Cloning':
            # prompt_speech_16k = load_cached_wav(prompt_wav, PROMPT_SR)
            prompt_speech_16k = postprocess(load_cached_wav(prompt_wav, PROMPT_SR))
            generator = cosyvoice.inference_zero_shot(paragraph, prompt_text, prompt_speech_16k, stream=stream, speed=speed)
        elif mode_checkbox_group == 'Cross-lingual Cloning':
            prompt_speech_16k = postprocess(load_cached_wav(prompt_wav, PROMPT_SR))
            generator = cosyvoice.inference_cross_lingual(paragraph, prompt_speech_16k, stream=stream, speed=speed)
        else:  # Natural Language Control
            generator = cosyvoice.inference_instruct(paragraph, sft_dropdown, instruct_text, stream=stream, speed=speed)

        for audio_chunk in generator:
            if stop_generation_flag.is_set():
                break
            yield (TARGET_SR, audio_chunk['tts_speech'].numpy().flatten())

        if i < len(paragraphs) - 1:
            yield (TARGET_SR, np.zeros(int(TARGET_SR * 1.0)))

        # Clean VRAM after processing each paragraph
        torch.cuda.empty_cache()

    # Final VRAM cleanup after processing all paragraphs
    torch.cuda.empty_cache()




def transcribe_audio(audio_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    model_size = "large-v3"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(audio_file, beam_size=5)

    # Clean up to free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Return the transcribed text
    transcribed_text = " ".join([segment.text for segment in segments])
    return transcribed_text


def main():
    def start_generation():
        global audio_chunks
        audio_chunks = []
        stop_generation_flag.clear()
        return gr.update(visible=False), gr.update(visible=True)

    def stop_generation():
        stop_generation_flag.set()
        return gr.update(visible=True), gr.update(visible=False)

    def generate_audio_wrapper(*args):
        global audio_chunks
        audio_chunks = []
        
        # Show "Stop Generating" button immediately
        yield (TARGET_SR, np.zeros(1)), gr.update(visible=False), gr.update(visible=True)
        
        for chunk in generate_audio(*args, stop_generation_flag):
            audio_chunks.append(chunk[1])  # Append only the audio data
            yield (TARGET_SR, np.concatenate(audio_chunks)), gr.update(visible=False), gr.update(visible=True)
        
        # Final yield with the complete audio and button updates
        if audio_chunks:
            yield (TARGET_SR, np.concatenate(audio_chunks)), gr.update(visible=True), gr.update(visible=False)
        else:
            # If no audio was generated, revert to initial state
            yield (TARGET_SR, np.zeros(1)), gr.update(visible=True), gr.update(visible=False)

    def start_transcription(audio_file):
        if audio_file is None:
            return "No audio file selected.", ""
        
        try:
            transcribed_text = transcribe_audio(audio_file)
            return "Transcription complete!", transcribed_text
        except Exception as e:
            return f"Error during transcription: {str(e)}", ""

    with gr.Blocks() as demo:
        gr.Markdown("### Repository [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    Pre-trained Models [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### Please enter the text to be synthesized, select the inference mode, and follow the prompts")

        tts_text = gr.Textbox(label="Input synthesis text", lines=1, value="I am a generative speech large model newly launched by the speech team of Tongyi Lab, providing comfortable and natural speech synthesis capabilities.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select inference mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Operation steps", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Select pre-trained voice', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='Stream inference?', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed adjustment (only supports non-streaming inference)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random inference seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Select prompt audio file, note that sampling rate should not be lower than 16khz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio file')
        
        with gr.Row():
            transcribe_button = gr.Button("Transcribe Audio")
            transcription_status = gr.Textbox(label="Transcription Status")
        
        prompt_text = gr.Textbox(label="Input prompt text", lines=3, placeholder="Transcribed text will appear here. You can also manually edit this field.", value='')
        instruct_text = gr.Textbox(label="Input instruct text", lines=1, placeholder="Please enter instruct text.", value='')


        generate_button = gr.Button("Generate audio")
        stop_button = gr.Button("Stop Generating", visible=False)

        audio_output = gr.Audio(label="Synthesized audio", autoplay=True)

        transcribe_button.click(
            start_transcription,
            inputs=[prompt_wav_upload],
            outputs=[transcription_status, prompt_text]
        )

        # generate_button.click(start_generation, inputs=[], outputs=[generate_button, stop_button])
        generate_button.click(generate_audio_wrapper,
                      inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                              seed, stream, speed],
                      outputs=[audio_output, generate_button, stop_button])
        stop_button.click(stop_generation, inputs=[], outputs=[generate_button, stop_button])
        
        seed_button.click(generate_seed, inputs=[], outputs=seed)

        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])



    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice-300M-SFT',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    main()
