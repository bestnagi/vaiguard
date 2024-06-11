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
import json
import os

# Initialize the OpenAI client with your API key from environment variables
openai_api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=openai_api_key)

# Set up logging
logging.basicConfig(filename='error_log.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s:%(message)s')

# Directory to store logs and video clips
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

# Function to log OpenAI responses
def log_openai_response(video_clip_path, openai_response, log_file="log_history.json"):
    log_entry = {
        "video_clip": video_clip_path,
        "openai_response": openai_response,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    log_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_history = json.load(f)
    else:
        log_history = []
    
    log_history.append(log_entry)
    
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=4)

# Function to detect a person in a frame
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

# Function to read frames from a YouTube stream
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

# Function to get the YouTube stream URL
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

# Function to generate OpenAI response
def generate_openai_response(base64Frames):
    try:
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    "These are frames of a video. Commentate in the style of a security guard who's watching for any person or vehicle. No one is allowed to be in this area. Pay attention to details and the behavior and outerwear. Only mention timestamps if the video includes timestamps. Only include narration.",
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

# Function to resize frame
def resize_frame(frame, width=640, height=360):
    try:
        return cv2.resize(frame, (width, height))
    except Exception as e:
        error_message = f"Error resizing frame: {e}"
        st.error(error_message)
        logging.error(error_message)
        return frame

# Function to create video clip
def create_video_clip(frames, output_path, fps=30):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for frame in frames:
        out.write(frame)
    out.release()

# Function to analyze frames and log responses
def analyze_frames(buffer_queue, analysis_queue, stop_event, analysis_lock, pedestrian_detected_flag):
    frame_counter = 0
    batch_size = 10  # Send every 10 frames
    clip_counter = 0

    while not stop_event.is_set():
        try:
            analysis_lock.acquire()
            if len(buffer_queue) >= batch_size and pedestrian_detected_flag[0]:
                frames_to_analyze = [buffer_queue.pop(0) for _ in range(batch_size)]
                analysis_lock.release()

                # Create a video clip from frames
                clip_path = f"{log_dir}/clip_{clip_counter}.mp4"
                create_video_clip(frames_to_analyze, clip_path)
                clip_counter += 1

                # Convert video clip to base64
                with open(clip_path, "rb") as video_file:
                    video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

                # Prepare video clip HTML with looping effect
                video_html = f'''
                <style>
                .video-thumbnail {{
                    display: inline-block;
                    margin-right: 5px;
                    width: 200px;
                }}
                </style>
                <div style="white-space: nowrap; overflow-x: auto;">
                    <video class="video-thumbnail" controls loop>
                        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    </video>
                </div>
                '''

                # Send frames to OpenAI for analysis
                st.write(f"Sending frames to OpenAI for analysis: {len(frames_to_analyze)} frames")
                response = generate_openai_response([base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8') for frame in frames_to_analyze])

                # Log the response along with the video clip
                log_openai_response(clip_path, response)

                # Send results to analysis_queue
                analysis_queue.put(("video_clip", video_html))
                analysis_queue.put(("incident_report", f"OpenAI Response for batch {frame_counter // batch_size + 1}: {response}"))
                frame_counter += batch_size
            else:
                analysis_lock.release()
        except Exception as e:
            error_message = f"Error in analyze_frames: {e}"
            st.error(error_message)
            logging.error(error_message)
        time.sleep(1)  # Adjust this value to control analysis frequency

# Function to display log history
def display_log_history():
    log_path = os.path.join(log_dir, "log_history.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_history = json.load(f)
        
        for entry in log_history:
            st.write(f"Timestamp: {entry['timestamp']}")
            st.write(f"OpenAI Response: {entry['openai_response']}")
            video_path = entry["video_clip"]
            with open(video_path, "rb") as video_file:
                video_base64 = base64.b64encode(video_file.read()).decode("utf-8")
                video_html = f'''
                <video width="320" height="240" controls loop>
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                </video>
                '''
                st.markdown(video_html, unsafe_allow_html=True)
    else:
        st.write("No log history available.")

def query_logs(query, log_file="log_history.json"):
    log_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_history = json.load(f)
        
        # Format log history as context for the chatbot
        context = "\n".join([f"Timestamp: {entry['timestamp']}\nResponse: {entry['openai_response']}" for entry in log_history])
        
        prompt_messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on the log history of OpenAI responses."},
            {"role": "user", "content": f"Given the following log history:\n{context}\n\nAnswer the following query:\n{query}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt_messages,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    else:
        return "No log history available to query."


# Main function to run the Streamlit app
def main():
    st.title("Live Stream Viewer and AI Incident Reporter")

    youtube_url = st.text_input("Enter Live Stream URL", key="youtube_url")
    frame_skip = 10  # st.number_input("Enter number of frames to skip", min_value=1, max_value=30, value=5, key="frame_skip")

    col1, col2 = st.columns([2, 1])

    with col1:
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
            pedestrian_detected_flag = [False]  # Use list to make it mutable in threads
            body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

            def fetch_frames():
                read_frames_from_stream(youtube_url, frame_queue, stop_event, frame_skip)

            def process_frames():
                nonlocal person_detected, base64Frames, buffer_queue, pedestrian_detected_flag
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

                        # Detect person in the frame
                        if detect_person(resized_frame, body_cascade):
                            person_detected = True
                            pedestrian_detected_flag[0] = True
                            st.write("Person detected. Collecting frames for analysis.")
                            frame_counter = 0  # Reset frame counter when a person is detected
                        else:
                            if person_detected:
                                st.write("No person detected in the frame.")
                            person_detected = False
                            pedestrian_detected_flag[0] = False

                        if person_detected:
                            analysis_lock.acquire()
                            buffer_queue.append(resized_frame)
                            analysis_lock.release()
                            frame_counter += 1
                            frame_counter_placeholder.write(f"Frames collected: {frame_counter}")

                        # Process analysis results from the queue
                        while not analysis_queue.empty():
                            item_type, content = analysis_queue.get()
                            if item_type == "video_clip":
                                video_clip_container = st.empty()
                                video_clip_container.markdown(content, unsafe_allow_html=True)
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

            analysis_thread = Thread(target=analyze_frames, args=(buffer_queue, analysis_queue, stop_event, analysis_lock, pedestrian_detected_flag))
            analysis_thread.start()

            st.write("Live stream is running...")

            process_frames()

            if st.button("Stop Stream", key="stop_button_end"):
                stop_event.set()
                fetch_thread.join()
                analysis_thread.join()
                st.write("Live stream stopped.")

    with col2:
        st.header("Log History")
        display_log_history()

        st.header("Chatbot")
        user_query = st.text_input("Ask a question about the log history:", key="user_query")
        if user_query:
            response = query_logs(user_query)
            st.write(response)

if __name__ == "__main__":
    main()

