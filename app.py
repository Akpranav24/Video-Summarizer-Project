import streamlit as st
import os
import yt_dlp
import whisper
import subprocess
import re

#https://www.youtube.com/shorts/Eo7D1yesSSE

def download_audio(video_url, output_file):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '192',
        }],
        'outtmpl': output_file + '.%(ext)s',
        'noplaylist': True,
        'ffmpeg_location': '/opt/homebrew/bin/ffmpeg'  # Replace with the actual path to ffmpeg if necessary
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def extract_audio_from_local_video(video_file, output_file):
    command = [
        'ffmpeg',
        '-i', video_file,
        '-vn',  # No video
        '-acodec', 'aac',
        output_file
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        raise

def transcribe_audio(audio_file):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_file, task="translate")
    return result['text']

def main(input_value):
    audio_file = 'temp_audio.m4a'

    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(url_pattern, input_value):
        download_audio(input_value, 'temp_audio')  # Pass the basename without extension
    elif os.path.isfile(input_value):
        extract_audio_from_local_video(input_value, audio_file)
    else:
        raise ValueError("The input must be a valid URL or an existing local file path.")

    transcribed_text = transcribe_audio(audio_file)
    os.remove(audio_file)  # Clean up temporary audio file

    print("Transcribed Text: ", transcribed_text)

    return transcribed_text

# Streamlit UI
st.title("Video to Text Transcription")

input_value = st.text_input("Please provide a YouTube URL or a local file path:")
if st.button("Transcribe"):
    if input_value:
        try:
            text = main(input_value)
            st.write("Transcribed Text:")
            st.write(text)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please provide a valid input.")
