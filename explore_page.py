import streamlit as st
import tempfile

import pytesseract
import shutil
import os
import random
try:
    from PIL import Image
except ImportError:
    import Image
from wand.image import Image as Img  
import numpy as np
import cv2
import re
import wordsegment
import nltk

import speech_recognition as sr 
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence

# net = cv2.dnn.readNet('./yolo/yolov3-spp.weights', '/yolo/yolov3-spp.cfg')

# classes = []

# with open('./yolo/coco.names', 'r') as f:
#   classes = f.read().splitlines()

# def detect_object(video_file):
#   try:
#     cap = cv2.VideoCapture(video_file)
#     count = 0
#     object_list = []

#     while cap.isOpened():
#       ret, img = cap.read()
      
#       if not ret:
#           break

#       if ret:
#         cv2.imwrite('frame{:d}.jpg'.format(count), img)
#         count += 50
#         cap.set(cv2.CAP_PROP_POS_FRAMES, count)

#         height, width, _ = img.shape
#         blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
#         net.setInput(blob)

#         output_layers_names = net.getUnconnectedOutLayersNames()
#         layerOutputs = net.forward(output_layers_names)

#         boxes = []
#         confidences = []
#         class_ids = []

#         for output in layerOutputs:
#           for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#               center_x = int(detection[0]*width)
#               center_y = int(detection[1]*height)
#               w = int(detection[2]*width)
#               h = int(detection[3]*height)

#               x = int(center_x - w/2)
#               y = int(center_y - h/2)

#               boxes.append([x, y, w, h])
#               confidences.append(float(confidence))
#               class_ids.append(class_id)

#         print(len(boxes))
#         font = cv2.FONT_HERSHEY_PLAIN
#         colors = np.random.uniform(0, 255, size=(len(boxes), 3))

#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#         if len(indexes) > 0:
#           print(indexes.flatten())
#           for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = str(round(confidences[i], 2))
#             color = colors[i]
#             cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#             cv2.putText(img, label + ' ' + confidence, (x, y+20), font, 2, (255,255,255), 2)
#             cv2_imshow(img)

#             object_list.append(label)

#           cap.release()
#           cv2.destroyAllWindows()

#       else:
#         cap.release()
#         cv2.destroyAllWindows()
#         break

#     cap.release()
#     cv2.destroyAllWindows()
#     print('Done detecting object in this video.')
#     print(f'These are the objects detected: {list(set(object_list))}')
#     return list(set(object_list))

#   except:
#     print(f'Unable to process file. Upload another video.')

image_frames = 'image_frames'

def files(video_file):
    try:
        os.remove(image_frames)
    except OSError:
        pass
  
    if not os.path.exists(image_frames):
        os.makedirs(image_frames)

    src_vid = cv2.VideoCapture(video_file)
    return(src_vid)

def process(src_vid):
    index = 0
    while src_vid.isOpened():
        ret, frame = src_vid.read()
        if not ret:
            break

        name = './image_frames/frame' + str(index) + '.png'

        if index % 20 == 0:
            print('Extracting frames...' + name)
            cv2.imwrite(name, frame)
        index = index + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
  
    src_vid.release()
    cv2.destroyAllWindows()

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)

def get_text():
    full_text = []

    for i in sorted_alphanumeric(os.listdir(image_frames)):
        print(str(i))
        my_example = Image.open(image_frames + '/' + i)
        text = pytesseract.image_to_string(my_example, lang='eng')
        full_text.append(text)

    single_text = ' '.join([i for i in full_text]).replace('\n', '').replace('\x0c', '').replace('TikTok', '')

    return single_text

def most_frequent(xlist):
    most_frequent = max(set(xlist), key = xlist.count)
    if most_frequent == '':
      return np.nan
    else:
      return most_frequent

def extract_username(screentext):
  screentext = screentext.lower()
  screentext = ''.join([i for i in screentext if not i.isdigit()])
  screentext = ' '.join(screentext.split())

  textlist = [word for word in screentext.lower().split()]

  usernamelist = []

  for text in textlist:
    if re.search(r'[@]', text):
      usernamelist.extend([text.rsplit('@')[-1]])
  
  if usernamelist == []:
    return np.nan

  else:
    usernamelist = ' '.join([name for name in usernamelist])
    usernamelist = [name for name in usernamelist.strip().split()]

    try:
      return most_frequent(usernamelist)

    except:
        return ' '.join(usernamelist)

wordsegment.load()
nltk.download('words')
words = set(nltk.corpus.words.words())

def process_text(text):
  text = text.lower()
  text = re.sub(r'([^A-Za-z0-9|\s|[:punct:]]*)', '', text)
  text = text.replace('|', '').replace(':', '')
  text = wordsegment.segment(text) 
  text = ' '.join([i for i in text if i in words])
  return text

def extract_text(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    vid = files(tfile.name)
    print('Folder created.')

    process(vid)
    onscreen_text = get_text()
    user_name = extract_username(onscreen_text)
    final_text = process_text(onscreen_text)

    shutil.rmtree('./image_frames/')
    print('Folder removed.')
    return user_name, final_text
    

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

    folder_name = 'audio-chunks'

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


def show_explore_page():
    st.title('Explore Video Content')

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
            username, video_to_text = extract_text(video_upload)
            st.write(f'Username: @{username}')
            st.write(video_to_text)

        # st.header('Video to Object')
        
        # if st.button('Extract Object'):
        #     video_to_object = extract_text(video_upload)
        #     st.write(f'Username: @{username}')
        #     st.write(video_to_text)
