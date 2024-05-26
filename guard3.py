from IPython.display import display, Image, Audio
import cv2
import base64
import time
from openai import OpenAI
import os
import requests
import numpy as np  # Import numpy
import simpleaudio as sa
import ffmpeg


# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

video = cv2.VideoCapture("test10.mp4")

fps = video.get(cv2.CAP_PROP_FPS)
# Calculate the number of frames to skip to capture 1-second intervals
frame_skip_interval = int(0.1*fps)

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
print(len(base64Frames), "frames read at 1-second intervals.")

for img in base64Frames:
    # Decode the base64 string
    decoded_img = base64.b64decode(img.encode("utf-8"))
    np_img = np.frombuffer(decoded_img, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Display the frame using OpenCV
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


PROMPT_MESSAGES = [
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
    "messages": PROMPT_MESSAGES,
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)


PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames of a video. Commentate in style of a security guard who's watching for any person or vehicle . Noone is allowed to be in this area..Pay attention to details like their clothes and appearance. Only mention timestamps if video includes timestamps. Only include narration. ",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::60]),
        ],
    },
]
params = {
    "model": "gpt-4o",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 500,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)



# Use the API key from the client object or directly specify your API key here
openai_api_key = client.api_key
voiceover_script = result.choices[0].message.content.strip()
print(voiceover_script)

# Generate voiceover
response = requests.post(
    "https://api.openai.com/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {client.api_key}",
    },
    json={
        "model": "tts-1-1106",
        "input": voiceover_script,  # Use the voiceover script
        "voice": "onyx",
    },
)
audio = b""
for chunk in response.iter_content(chunk_size=1024 * 1024):
    audio += chunk

# Save the audio to a file
audio_path = "voiceover-s.mp3"
with open(audio_path, "wb") as f:
    f.write(audio)

# Convert MP3 to WAV using ffmpeg-python
wav_path = "voiceover.wav"
(
    ffmpeg
    .input(audio_path)
    .output(wav_path)
    .run(overwrite_output=True)
)

# Play the audio using simpleaudio
wave_obj = sa.WaveObject.from_wave_file(wav_path)
play_obj = wave_obj.play()
play_obj.wait_done()