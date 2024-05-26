import streamlit as st
import cv2
import base64
import os
import requests
import numpy as np

from pydub import AudioSegment
from pydub.playback import play
import tempfile
from openai import OpenAI
import ffmpeg

# Initialize the OpenAI client with your API key
openai_api_key = os.getenv("OPENAI_API_KEY", "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")
client = OpenAI(api_key=openai_api_key)

def process_video(file_path):
    video = cv2.VideoCapture(file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_skip_interval = int(0.1 * fps)

    base64Frames = []
    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_skip_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1
    video.release()
    return base64Frames

def generate_openai_response(base64Frames):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::30]),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 200,
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def generate_voiceover_script(base64Frames):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                "These are frames of a video. Commentate in the style of a security guard who's watching for any person or vehicle. No one is allowed to be in this area. Pay attention to details like their clothes and appearance. Only mention timestamps if the video includes timestamps. Only include narration.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::60]),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 500,
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def generate_voiceover(audio_script):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {openai_api_key}"},
        json={"model": "tts-1-1106", "input": audio_script, "voice": "onyx"},
    )
    audio = b""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio += chunk

    audio_path = "voiceover-s.mp3"
    with open(audio_path, "wb") as f:
        f.write(audio)

    wav_path = "voiceover.wav"
    ffmpeg.input(audio_path).output(wav_path).run(overwrite_output=True)

    return wav_path

def main():
    st.title("Video Processing and Voiceover App")

    uploaded_file = st.file_uploader("Upload an MP4 file", type=["mp4"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.write("File uploaded successfully.")

        base64Frames = process_video(temp_file_path)
        st.write(f"{len(base64Frames)} frames read at 1-second intervals.")

        for img in base64Frames[:10]:  # Display first 10 frames as an example
            decoded_img = base64.b64decode(img.encode("utf-8"))
            np_img = np.frombuffer(decoded_img, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            st.image(frame, channels="BGR")

        description = generate_openai_response(base64Frames)
        st.write("Video Description:")
        st.write(description)

        voiceover_script = generate_voiceover_script(base64Frames)
        st.write("Voiceover Script:")
        st.write(voiceover_script)

        wav_path = generate_voiceover(voiceover_script)
        audio_segment = AudioSegment.from_wav(wav_path)
        audio_bytes = audio_segment.export(format="wav").read()
        st.audio(audio_bytes)

        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
