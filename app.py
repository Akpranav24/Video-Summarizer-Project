import streamlit as st
import os
import yt_dlp
import whisper
import subprocess
import re
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.tokenize import sent_tokenize

# Load the BART model and tokenizer for text summarization
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to download audio from a video URL using yt-dlp
def download_audio(video_url, output_file):
    ydl_opts = {
        'format': 'bestaudio/best',  # Get the best quality audio
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',  # Convert audio to m4a format
            'preferredquality': '192',  # Set audio quality
        }],
        'outtmpl': output_file + '.%(ext)s',  # Output filename template
        'noplaylist': True,  # Only download a single video, not playlists
        'ffmpeg_location': "C:/ffmpeg-n7.0-latest-win64-lgpl-7.0/bin"  # Path to ffmpeg
    }

    # Download and extract audio from the video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Function to extract audio from a local video file using ffmpeg
def extract_audio_from_local_video(video_file, output_file):
    command = [
        'ffmpeg',  # Call ffmpeg
        '-i', video_file,  # Input video file
        '-vn',  # Disable video (only extract audio)
        '-acodec', 'aac',  # Audio codec
        output_file  # Output audio file
    ]
    try:
        # Execute the command to extract audio
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # Handle errors in the ffmpeg process
        print(f"Error: {e.stderr}")
        raise

# Function to transcribe audio to text using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("medium").to("cuda")  # Load the Whisper model to GPU
    result = model.transcribe(audio_file, task="translate")  # Transcribe and translate audio to English
    return result['text']

# Function to split text into chunks for summarization
def chunk_text(text, tokenizer, max_tokens=500):
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    chunks = []
    current_chunk = ""
    current_length = 0

    # Group sentences into chunks with a maximum token length
    for sentence in sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if current_length + len(tokens) < max_tokens:
            current_chunk += " " + sentence
            current_length += len(tokens)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = len(tokens)
    
    # Append the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to summarize a single chunk of text using BART
def summarize_chunk(chunk, tokenizer):
    inputs = tokenizer.encode(chunk, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=250, min_length=60, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to extract key points from the summarized text
def extract_key_points(text):
    sentences = sent_tokenize(text)
    key_points = []
    
    # Extract sentences that are longer than 10 words and contain a period or colon
    for sentence in sentences:
        if len(sentence.split()) > 10 and ('.' in sentence or ':' in sentence):
            key_points.append(sentence)
    
    return key_points

# Function to summarize the entire text by chunking and combining summaries
def summarize_text(text, tokenizer):
    chunks = chunk_text(text, tokenizer)  # Chunk the text
    summarized_chunks = [summarize_chunk(chunk, tokenizer) for chunk in chunks]  # Summarize each chunk
    
    combined_summary = "\n\n".join(summarized_chunks)  # Combine all summarized chunks
    key_points = extract_key_points(combined_summary)  # Extract key points from the combined summary
    
    return {
        'combined_summary': combined_summary,
        'key_points': key_points
    }

# Main function to handle input (URL or file), transcribe, and summarize
def main(input_value):
    audio_file = 'temp_audio.m4a'

    # Regex pattern to validate URLs
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  
        r'localhost|'  
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  
        r'(?::\d+)?'  
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(url_pattern, input_value):  # Check if input is a URL
        download_audio(input_value, 'temp_audio')  # Download audio from the URL
    elif os.path.isfile(input_value):  # Check if input is a file path
        extract_audio_from_local_video(input_value, audio_file)  # Extract audio from the local video file
    else:
        raise ValueError("The input must be a valid URL or an existing local file path.")  # Handle invalid input

    transcribed_text = transcribe_audio(audio_file)  # Transcribe the extracted audio
    os.remove(audio_file)  # Remove the temporary audio file

    results = summarize_text(transcribed_text, tokenizer)  # Summarize the transcribed text
    
    return results

# Streamlit UI setup
st.title("Video to Text Transcription and Summarization")

input_value = st.text_input("Please provide a YouTube URL or a local file path:")  # Input field for URL or file path
if st.button("Transcribe and Summarize"):  # Button to trigger transcription and summarization
    if input_value:
        try:
            results = main(input_value)  # Process the input
            st.write("Combined Summary:")  # Display combined summary
            st.write(results['combined_summary'])
            st.write("Key Points:")  # Display key points
            for point in results['key_points']:
                st.write(f"- {point}")
        except Exception as e:
            st.error(f"An error occurred: {e}")  # Display error message if something goes wrong
    else:
        st.error("Please provide a valid input.")  # Prompt user to provide input if none is given
