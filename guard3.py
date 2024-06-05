import streamlit as st
import cv2
import base64
import requests
import numpy as np
from threading import Thread, Event, Lock
from queue import Queue
from openai import OpenAI
import streamlink
import time
import logging

# Initialize the OpenAI client with your API key from environment variables
openai_api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=openai_api_key)

# Set up logging
logging.basicConfig(filename='error_log.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s:%(message)s')

def detect_person(frame, body_cascade):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
        return len(bodies) > 0
    except Exception as e:
        error_message = f"Error in detect_person: {e}"
        st.error(error_message)
        logging.error(error_message)
        return False

def read_frames_from_stream(youtube_url, frame_queue, stop_event, frame_skip):
    try:
        stream_url = get_youtube_stream_url(youtube_url)
        if not stream_url:
            frame_queue.put((None, "Failed to retrieve the stream URL."))
            return

        cap = cv2.VideoCapture(stream_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 FPS if the value is not available
        frame_duration = 1 / fps
        frame_count = 0

        while not stop_event.is_set():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                frame_queue.put((None, "Error retrieving frame."))
                break
            if frame_count % frame_skip == 0:
                frame_queue.put((frame, None))
            frame_count += 1
            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_duration - elapsed_time)
            time.sleep(sleep_time)

        cap.release()
        frame_queue.put((None, "Stream stopped."))
    except Exception as e:
        error_message = f"Error retrieving frame: {e}"
        frame_queue.put((None, error_message))
        st.error(error_message)
        logging.error(error_message)

def get_youtube_stream_url(youtube_url):
    try:
        streams = streamlink.streams(youtube_url)
        if "best" in streams:
            return streams["best"].to_url()
        else:
            error_message = "No suitable stream found."
            st.error(error_message)
            logging.error(error_message)
            return None
    except Exception as e:
        error_message = f"Error retrieving stream URL: {e}"
        st.error(error_message)
        logging.error(error_message)
        return None

def generate_openai_response(base64Frames):
    try:
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    "These are frames of a video. Commentate in the style of a security guard who's watching for any person or vehicle. No one is allowed to be in this area. Keep it detailed and concise. Each description should be around 15-20 words. Only mention timestamps if the video includes timestamps. Only include narration.",
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames),
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
    except Exception as e:
        error_message = f"Error in generate_openai_response: {e}"
        st.error(error_message)
        logging.error(error_message)
        return "Error generating response."

def resize_frame(frame, width=640, height=360):
    try:
        return cv2.resize(frame, (width, height))
    except Exception as e:
        error_message = f"Error resizing frame: {e}"
        st.error(error_message)
        logging.error(error_message)
        return frame

def analyze_frames(buffer_queue, analysis_queue, stop_event, analysis_lock):
    frame_counter = 0
    batch_size = 10  # Send every 10 frames

    while not stop_event.is_set():
        try:
            analysis_lock.acquire()
            if len(buffer_queue) >= batch_size:
                frames_to_analyze = [buffer_queue.pop(0) for _ in range(batch_size)]
                analysis_lock.release()

                # Convert frames to base64
                base64Frames = []
                for frame in frames_to_analyze:
                    _, buffer = cv2.imencode(".jpg", frame)
                    base64_frame = base64.b64encode(buffer).decode("utf-8")
                    base64Frames.append(base64_frame)

                # Prepare thumbnails HTML
                thumbnails_html = '<div style="white-space: nowrap; overflow-x: auto;">'
                for base64_frame in base64Frames:
                    thumbnails_html += f'<img src="data:image/jpeg;base64,{base64_frame}" style="display: inline-block; margin-right: 5px;" width="100"/>'
                thumbnails_html += '</div>'

                # Send frames to OpenAI for analysis
                st.write(f"Sending frames to OpenAI for analysis: {len(base64Frames)} frames")
                response = generate_openai_response(base64Frames)

                # Send results to analysis_queue
                analysis_queue.put(("thumbnails", thumbnails_html))
                analysis_queue.put(("incident_report", f"OpenAI Response for batch {frame_counter // batch_size + 1}: {response}"))
                frame_counter += batch_size
            else:
                analysis_lock.release()
        except Exception as e:
            error_message = f"Error in analyze_frames: {e}"
            st.error(error_message)
            logging.error(error_message)
        time.sleep(1)  # Adjust this value to control analysis frequency

def main():
    st.title("Live Stream Viewer and AI Incident Reporter")

    youtube_url = st.text_input("Enter Live Stream URL")
    frame_skip = 10 #st.number_input("Enter number of frames to skip", min_value=1, max_value=30, value=5)
    if youtube_url:
        frame_queue = Queue(maxsize=100)  # Increased buffer size
        buffer_queue = []
        analysis_queue = Queue()
        stop_event = Event()
        analysis_lock = Lock()
        display_placeholder = st.empty()
        frame_counter_placeholder = st.empty()
        base64Frames = []
        person_detected = False
        body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

        def fetch_frames():
            read_frames_from_stream(youtube_url, frame_queue, stop_event, frame_skip)

        def process_frames():
            nonlocal person_detected, base64Frames, buffer_queue
            frame_counter = 0

            while not stop_event.is_set():
                try:
                    frame, error_message = frame_queue.get()
                    if error_message:
                        st.error(error_message)
                        logging.error(error_message)
                        break
                    if frame is None:
                        break

                    # Resize the frame to a larger size for display
                    resized_frame = resize_frame(frame, width=640, height=360)
                    
                    # Update the live stream frame
                    display_placeholder.image(resized_frame, channels="BGR", caption="Live Stream")

                    if detect_person(resized_frame, body_cascade):
                        person_detected = True
                        st.write("Person detected. Collecting frames for analysis.")
                        frame_counter = 0  # Reset frame counter when a person is detected

                    if person_detected:
                        analysis_lock.acquire()
                        buffer_queue.append(resized_frame)
                        analysis_lock.release()
                        frame_counter += 1
                        frame_counter_placeholder.write(f"Frames collected: {frame_counter}")

                    # Process analysis results from the queue
                    while not analysis_queue.empty():
                        item_type, content = analysis_queue.get()
                        if item_type == "thumbnails":
                            thumbnails_container = st.empty()
                            thumbnails_container.markdown(content, unsafe_allow_html=True)
                        elif item_type == "incident_report":
                            incident_report_placeholder = st.empty()
                            incident_report_placeholder.write(content)

                    if stop_event.is_set():
                        break
                except Exception as e:
                    error_message = f"Error processing frame: {e}"
                    st.error(error_message)
                    logging.error(error_message)

        fetch_thread = Thread(target=fetch_frames)
        fetch_thread.start()

        analysis_thread = Thread(target=analyze_frames, args=(buffer_queue, analysis_queue, stop_event, analysis_lock))
        analysis_thread.start()

        st.write("Live stream is running...")

        process_frames()

        if st.button("Stop Stream", key="stop_button_end"):
            stop_event.set()
            fetch_thread.join()
            analysis_thread.join()
            st.write("Live stream stopped.")

if __name__ == "__main__":
    main()
