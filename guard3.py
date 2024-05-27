import streamlit as st
import cv2
import base64
import os
import requests
import numpy as np

from pydub.playback import play
import tempfile
from openai import OpenAI
import ffmpeg


from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

from pydub import AudioSegment

# Initialize the OpenAI client with your API key from environment variables
# Load the API key from Streamlit secrets
openai_api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

#openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

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
                "These are frames from a video that I want to upload. Generate a security guard description that I can upload along with the video.",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::60]),
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
                "These are frames of a video. Commentate in the style of a security guard who's watching for any person or vehicle. No one is allowed to be in this area. Keep it detailed and concise. Each description should be around 10 words. Only mention timestamps if the video includes timestamps. Only include narration.",
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

def generate_voiceover(audio_script, video_length):
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

    # Convert MP3 to WAV using ffmpeg-python
    wav_path = "voiceover.wav"
    ffmpeg.input(audio_path).output(wav_path).run(overwrite_output=True)

    # Load the audio and adjust speed to match video length
    audio_segment = AudioSegment.from_wav(wav_path)
    audio_length = len(audio_segment) / 1000.0  # Length in seconds
    st.write(f"Original Audio Length: {audio_length} seconds")  # Debug: Print audio length

    # Calculate the speed change required
    speed_change = audio_length / video_length
    st.write(f"Speed Change Factor: {speed_change}")  # Debug: Print speed change factor
    adjusted_audio = change_audio_speed(audio_segment, speed_change)

    # Debug: Print adjusted audio length
    adjusted_audio_length = len(adjusted_audio) / 1000.0
    st.write(f"Adjusted Audio Length: {adjusted_audio_length} seconds")

    # Export the adjusted audio
    adjusted_audio_path = "adjusted_voiceover.wav"
    adjusted_audio.export(adjusted_audio_path, format="wav")

    return adjusted_audio_path

def change_audio_speed(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)



def adjust_audio_speed(audio_path, target_duration):
    audio = AudioFileClip(audio_path)
    audio = audio.set_duration(target_duration)
    return audio

def combine_video_audio(video_path, audio_path, output_path):
    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(audio_path)
    ffmpeg.output(input_video, input_audio, output_path, vcodec='copy', acodec='aac', strict='experimental').run(overwrite_output=True)

def remove_original_audio(video_path, output_path):
    ffmpeg.input(video_path).output(output_path, an=None, vcodec='copy').run(overwrite_output=True)



def main():
    st.title("Video Processing and Voiceover App")

    uploaded_file = st.file_uploader("Upload an MP4 file", type=["mp4"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.write("File uploaded successfully.")

        # Display the video using Streamlit's video player
        st.video(temp_file_path)

        # Get the video length
        video = cv2.VideoCapture(temp_file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video_length = frame_count / fps
        st.write(f"Video Length: {video_length} seconds")  # Debug: Print video length
        video.release()

        base64Frames = process_video(temp_file_path)
        st.write(f"{len(base64Frames)} frames read at 1-second intervals.")

        description = generate_openai_response(base64Frames)
        st.write("Video Description:")
        st.write(description)

        voiceover_script = generate_voiceover_script(base64Frames)
        st.write("Voiceover Script:")
        st.write(voiceover_script)

        adjusted_audio_path = generate_voiceover(voiceover_script, video_length)

        # Remove original audio from video
        video_no_audio_path = "video_no_audio.mp4"
        remove_original_audio(temp_file_path, video_no_audio_path)

        # Combine video without original audio and new audio
        output_path = "output_video_with_voiceover.mp4"
        combine_video_audio(video_no_audio_path, adjusted_audio_path, output_path)

        # Display the final video with voiceover
        st.video(output_path)

        os.remove(temp_file_path)
        os.remove(video_no_audio_path)
        os.remove(adjusted_audio_path)

if __name__ == "__main__":
    main()
