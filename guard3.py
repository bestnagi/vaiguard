import streamlit as st
import cv2
import base64
import requests
import numpy as np
import tempfile
from threading import Thread, Event, Lock
from queue import Queue
from openai import OpenAI
import streamlink
import time
import logging
import json
import os
from datetime import datetime
import asyncio
import aiohttp

# Initialize the OpenAI client with your API key from environment variables
openai_api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

if not openai_api_key:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=openai_api_key)

# Set up logging
logging.basicConfig(filename='error_log.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

# Temporary directory for logs and video clips
log_dir = tempfile.gettempdir()

# Function to log OpenAI responses
def log_openai_response(frames, openai_response, log_file="log_history.json"):
    log_entry = {
        "frames": frames,
        "openai_response": openai_response,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    log_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_history = json.load(f)
        except json.JSONDecodeError:
            log_history = []
    else:
        log_history = []
    
    log_history.append(log_entry)
    
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=4)
    logging.debug(f"Logged OpenAI response: {openai_response}")

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

        while not stop_event.is_set():
            start_time = time.time()
            frames = []
            for _ in range(int(fps * 10)):  # Capture 10 seconds of frames
                ret, frame = cap.read()
                if not ret:
                    frame_queue.put((None, "Error retrieving frame."))
                    break
                if len(frames) % frame_skip == 0:
                    frames.append(resize_frame(frame))
                time.sleep(frame_duration)
            frame_queue.put((frames, None))
            elapsed_time = time.time() - start_time
            sleep_time = max(0, 10 - elapsed_time)  # Ensure a 10-second interval between batches
            logging.debug(f"Captured and queued frames. Sleeping for {sleep_time} seconds.")
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
        logging.info(f"Attempting to retrieve streams for URL: {youtube_url}")
        streams = streamlink.streams(youtube_url)
        logging.info(f"Streams found: {streams.keys()}")
        if "best" in streams:
            return streams["best"].to_url()
        else:
            error_message = "No suitable stream found."
            st.error(error_message)
            logging.error(error_message)
            return None
    except streamlink.exceptions.PluginError as e:
        error_message = f"Streamlink PluginError: {e}"
        st.error(error_message)
        logging.error(error_message)
        return None
    except streamlink.exceptions.NoPluginError as e:
        error_message = f"No Streamlink plugin can handle URL: {e}"
        st.error(error_message)
        logging.error(error_message)
        return None
    except Exception as e:
        error_message = f"Unexpected error retrieving stream URL: {e}"
        st.error(error_message)
        logging.error(error_message)
        return None

async def generate_openai_response(session, base64Frames):
    try:
        logging.info(f"Sending {len(base64Frames)} frames to OpenAI for analysis.")
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    "These are frames of a video. You are the best security guard in the world. You pay attention to little details and you're super vigilant. Commentate in the style of a security guard who's watching for any person or vehicle or anything that stands out. Focus also on clothing and behavior and anything you deem relevant. Only mention timestamps if the video includes timestamps. Only include narration.",
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames),
                ],
            },
        ]
        params = {
            "model": "gpt-4o",
            "messages": prompt_messages,
            "max_tokens": 500,
        }
        headers = {
            'Authorization': f'Bearer {openai_api_key}',
            'Content-Type': 'application/json'
        }
        async with session.post('https://api.openai.com/v1/chat/completions', json=params, headers=headers) as response:
            result = await response.json()
            if 'choices' not in result:
                raise ValueError("Invalid response structure")
            logging.debug(f"OpenAI response: {result['choices'][0]['message']['content']}")
            return result['choices'][0]['message']['content']
    except Exception as e:
        error_message = f"Error in generate_openai_response: {e}"
        st.error(error_message)
        logging.error(error_message)
        return "Error generating response."

def resize_frame(frame, width=320, height=180):
    try:
        return cv2.resize(frame, (width, height))
    except Exception as e:
        error_message = f"Error resizing frame: {e}"
        st.error(error_message)
        logging.error(error_message)
        return frame

async def analyze_frames(buffer_queue, analysis_queue, stop_event, analysis_lock, display_placeholder):
    frame_counter = 0
    batch_size = 5  # Send every 5 frames to reduce memory usage

    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                analysis_lock.acquire()
                if len(buffer_queue) >= batch_size:
                    frames_to_analyze = [buffer_queue.pop(0) for _ in range(batch_size)]
                    analysis_lock.release()

                    # Convert frames to base64
                    base64Frames = [base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8') for frame in frames_to_analyze]

                    # Send frames to OpenAI for analysis
                    st.write(f"Sending frames to OpenAI for analysis: {len(base64Frames)} frames")
                    logging.debug(f"Sending batch {frame_counter // batch_size + 1} to OpenAI for analysis.")
                    response = await generate_openai_response(session, base64Frames)
                    log_openai_response(base64Frames, response)
                    
                    # Update the Streamlit display with the OpenAI response
                    analysis_queue.put(("incident_report", f"OpenAI Response for batch {frame_counter // batch_size + 1}: {response}"))
                    frame_counter += batch_size
                else:
                    analysis_lock.release()
            except Exception as e:
                error_message = f"Error in analyze_frames: {e}"
                st.error(error_message)
                logging.error(error_message)
            await asyncio.sleep(1)  # Adjust this value to control analysis frequency

def display_log_history(selected_date=None):
    log_path = os.path.join(log_dir, "log_history.json")
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_history = json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error reading log history: {e}")
            logging.error(f"Error reading log history: {e}")
            return
        
        if selected_date:
            log_history = [entry for entry in log_history if entry["timestamp"].startswith(selected_date)]
        
        for entry in log_history:
            st.write(f"Timestamp: {entry['timestamp']}")
            st.write(f"OpenAI Response: {entry['openai_response']}")
            frames = entry.get("frames", [])
            thumbnails_html = '<div style="white-space: nowrap; overflow-x: auto;">'
            for frame in frames:
                thumbnails_html += f'<img src="data:image/jpeg;base64,{frame}" style="display: inline-block; margin-right: 5px;" width="100"/>'
            thumbnails_html += '</div>'
            st.markdown(thumbnails_html, unsafe_allow_html=True)
    else:
        st.write("No log history available.")

def query_logs(query, log_file="log_history.json"):
    log_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                log_history = json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error reading log history: {e}")
            logging.error(f"Error reading log history: {e}")
            return "Error reading log history."
        
        # Format log history as context for the chatbot
        context = "\n".join([f"Timestamp: {entry['timestamp']}\nResponse: {entry['openai_response']}" for entry in log_history])
        
        prompt_messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on the log history of OpenAI responses."},
            {"role": "user", "content": f"Given the following log history:\n{context}\n\nAnswer the following query:\n{query}"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt_messages,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    else:
        return "No log history available to query."

def main():
    st.title("Live Stream Viewer and AI Incident Reporter")

    youtube_url = st.text_input("Enter Live Stream URL", key="youtube_url")
    frame_skip = 30  # Sample every 30 frames

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
                message_displayed = False

                while not stop_event.is_set():
                    try:
                        frames, error_message = frame_queue.get()
                        if error_message:
                            st.error(error_message)
                            logging.error(error_message)
                            break
                        if frames is None:
                            break

                        for i, frame in enumerate(frames):
                            # Resize the frame to a smaller size for display
                            resized_frame = resize_frame(frame, width=320, height=180)
                            
                            # Update the live stream frame
                            display_placeholder.image(resized_frame, channels="BGR", caption="Live Stream")

                            # Detect person in the frame
                            if detect_person(resized_frame, body_cascade):
                                person_detected = True
                                pedestrian_detected_flag[0] = True
                                st.write("Person detected. Collecting frames for analysis.")
                                frame_counter = 0  # Reset frame counter when a person is detected
                                message_displayed = False
                            elif person_detected and not message_displayed:
                                st.write("Continuing to send frames for analysis until the next batch is processed.")
                                message_displayed = True
                            
                            if person_detected and i % frame_skip == 0:  # Sample frames
                                analysis_lock.acquire()
                                buffer_queue.append(resized_frame)
                                analysis_lock.release()
                                frame_counter += 1
                                frame_counter_placeholder.write(f"Frames collected: {frame_counter}")

                        # Process analysis results from the queue
                        while not analysis_queue.empty():
                            item_type, content = analysis_queue.get()
                            if item_type == "incident_report":
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

            analysis_thread = Thread(target=lambda: asyncio.run(analyze_frames(buffer_queue, analysis_queue, stop_event, analysis_lock, display_placeholder)))
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

        selected_date = st.date_input("Select a date to view log history:")
        selected_date_str = selected_date.strftime("%Y-%m-%d")
        display_log_history(selected_date_str)

        st.header("Chatbot")
        user_query = st.text_input("Ask a question about the log history:", key="user_query")
        if user_query:
            response = query_logs(user_query)
            st.write(response)

if __name__ == "__main__":
    main()
