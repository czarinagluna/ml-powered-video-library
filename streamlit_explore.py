import streamlit as st
import tempfile

import os
import ffprobe
import speech_recognition as sr 
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence

import wordsegment
wordsegment.load()

import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())

import sys
sys.setrecursionlimit(2000)

import cv2
import pytesseract
import shutil
import re
import numpy as np
try:
    from PIL import Image
except ImportError:
    import Image


###### Page Content ######
def show_explore_page():
    st.title('Explore Video Processing')

    video_upload = st.file_uploader('Upload Video', type='mp4')

    if video_upload:

        st.subheader('Video Uploaded')

        st.video(video_upload)

        st.header('Video to Speech')
        
        if st.button('Extract Speech'):
            video_to_speech = transcribe_audio(video_upload)
            st.write(video_to_speech)

        st.header('Video to Text')
        
        if st.button('Extract Visual Text'):
            video_to_text = extract_visual_text(video_upload)
            st.write(video_to_text)

            username = extract_username(video_to_text)
            st.write(f'*Username Extracted:* @{username}')

            processed_text = process_visual_text(video_to_text)
            st.write(f'*Processed Text:* {processed_text}')
            
        st.header('Video to Object')
        
        if st.button('Extract Object'):
            video_to_object = detect_object(video_upload)
            st.write(f'*Object Detected:* @{video_to_object}')

###### Audio Transcription ######
def transcribe_audio(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    r = sr.Recognizer()

    transcribed_audio_file = 'transcribed_audio.wav'
    audioclip = AudioFileClip(tfile.name)
    audioclip.write_audiofile(transcribed_audio_file)

    try:
        sound = AudioSegment.from_file(tfile.name, 'mp3')
    except:
        sound = AudioSegment.from_file(tfile.name, format='mp4')    

    chunks = split_on_silence(sound, min_silence_len = 500,  silence_thresh = sound.dBFS-14, keep_silence=500)

    folder_name = './data/streamlit/audio-chunks'

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    full_text = ''

    for i, audio_chunk in enumerate(chunks, start=1):

        chunk_filename = os.path.join(folder_name, f'chunk{i}.wav')
        audio_chunk.export(chunk_filename, format='wav')

        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)

            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                full_text += text

    return full_text

###### Visual Text Extraction ######
image_frames = './data/streamlit/image_frames'

def save_frames(video_file):
    '''
    Creates image folder and saves video frames in the folder.
    
    Parameters:
    file_path (str): file path of video to be captured as images.
    
    Returns:
    image_frames folder where the video frames are stored.
    '''
    try:
        os.remove(image_frames)
    except OSError:
        pass

    # Step 1:
    if not os.path.exists(image_frames):
        os.makedirs(image_frames)
    
    # Step 2:
    src_vid = cv2.VideoCapture(video_file)

    index = 0
    while src_vid.isOpened():
        ret, frame = src_vid.read()
        if not ret:
            break

        name = './data/streamlit/image_frames/frame' + str(index) + '.png'

        if index % 20 == 0:
            print('Extracting frames...' + name)
            cv2.imwrite(name, frame)
        index = index + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
  
    src_vid.release()
    cv2.destroyAllWindows()

def sorted_alphanumeric(name_list):
    '''
    Sorts names according to alphanumeric characters.
    
    Parameters:
    name_list (list): list of names to be sorted.
    
    Returns:
    sorted_names (list): sorted list using natural sorting e.g. 1, 2, 10 rather than 1, 10, 2
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    sorted_names = sorted(name_list, key=alphanum_key)
    return sorted_names

# Credits to user136036 for this function found on stack overflow
# https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir

def extract_visual_text(video_file):
    '''
    Extracts visual text from images saved of video frames.

    Parameters:
    file_path (str): file path of video from which to extract the visual text.
    
    Returns:
    full_text (str): text as seen in the video taken from every 20th frame.
    '''
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    vid = save_frames(tfile.name)
    print('Folder created.')
    
    text_list = []
    
    image_list = sorted_alphanumeric(os.listdir(image_frames))
    
    for i in image_list:
        print(str(i))
        single_frame = Image.open(image_frames + '/' + i)
        text = pytesseract.image_to_string(single_frame, lang='eng')
        text_list.append(text)

    visual_text = ' '.join([i for i in text_list])
    visual_text = visual_text.replace('\n', '').replace('\x0c', '').replace('TikTok', '')

    shutil.rmtree('./data/streamlit/image_frames/')
    print('Folder removed.')
    
    return visual_text

def most_frequent(username_list):
    '''Takes in a list of strings and return the most frequent word in the list or none.'''
    most_frequent = max(set(username_list), key = username_list.count)
    if most_frequent == '':
        return np.nan
    else:
        return most_frequent

def extract_username(text):
    '''
    Lists possible usernames from visual text and returns the most frequent one that may most likely be the username.
    
    Parameters:
    text (str): full visual text extracted from video.
    
    Returns:
    username (str): most frequent word that starts with @ sign; if none, returns none.
    '''
    text_list = [word for word in text.lower().split()]

    username_list = []
    for word in text_list:
        if re.search(r'[@]', word):
            username_list.extend([word.rsplit('@')[-1]])
    if username_list == []:
        return np.nan

    else:
        username_list = ' '.join([username for username in username_list])
        username_list = [username for username in username_list.strip().split()]
        try:
            return most_frequent(username_list)
        except:
            return ' '.join(username_list)

def process_visual_text(text):
    '''Processes string of text by removing special characters and splitting words with no spaces between them.'''
    text = text.lower()
    text = re.sub(r'([^A-Za-z0-9|\s|[:punct:]]*)', ' ', text)
    text = text.replace('|', '').replace(':', '')
    text = wordsegment.segment(text) 
    text = ' '.join([i for i in text if i in words])
    return text

###### Object Detection ######
def detect_object(video_file):
    '''
    Uses YOLO algorithm to detect objects in video frames.
    
    Parameters:
    file_path (str): file path of video from which to detect objects in the frames.
    
    Returns:
    object_set (list): list of unique objects detected in the video.
    '''
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    net = cv2.dnn.readNetFromDarknet('./data/yolo/yolov3-spp.cfg', './data/yolo/yolov3-spp.weights')

    classes = []

    with open('./data/yolo/coco.names', 'r') as f:
        classes = f.read().splitlines()
  
    try:
        cap = cv2.VideoCapture(tfile.name)
        count = 0
        object_list = []

        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                break

            if ret:
                cv2.imwrite('frame{:d}.jpg'.format(count), img)
                count += 50
                cap.set(cv2.CAP_PROP_POS_FRAMES, count)

                height, width, _ = img.shape
                blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
                net.setInput(blob)

                output_layers_names = net.getUnconnectedOutLayersNames()
                layerOutputs = net.forward(output_layers_names)

                boxes = []
                confidences = []
                class_ids = []

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0]*width)
                            center_y = int(detection[1]*height)
                            w = int(detection[2]*width)
                            h = int(detection[3]*height)
                            x = int(center_x - w/2)
                            y = int(center_y - h/2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                print(len(boxes))
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                if len(indexes) > 0:
                    print(indexes.flatten())
                    for i in indexes.flatten():
                        label = str(classes[class_ids[i]])
                    object_list.append(label)
            else:
                cap.release()
                cv2.destroyAllWindows()
                break
            
        cap.release()
        cv2.destroyAllWindows()
        
        print('Done detecting object in this video.')
        print(f'These are the objects detected: {list(set(object_list))}')
        
        object_set = list(set(object_list))
        return object_set
    
    except:
        print(f'{video_file} did not work.')
