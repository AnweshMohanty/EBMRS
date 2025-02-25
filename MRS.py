import random
import json
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from youtube_search import YoutubeSearch
import os
import cv2

model = load_model('affect_2a.keras')
emotion_labels = ['happy', 'sad', 'neutral', 'excited']
music_df = pd.read_csv('Songs2.csv')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Upload preferences to json file
PREF_FILE = "user_preferences.json"
if not os.path.exists(PREF_FILE):
    with open(PREF_FILE, "w") as f:
        json.dump({
            "liked_songs": [], 
            "liked_singers": {}, 
            "disliked_songs": [],
            "emotion":[]
        }, f)

# Load user preferences
with open(PREF_FILE, "r") as f:
    user_preferences = json.load(f)

def get_youtube_url(song_name, artist):
    try:
        results = YoutubeSearch(f"{song_name} {artist}", max_results=1).to_dict()
        return f"https://www.youtube.com{results[0]['url_suffix']}"
    except:
        return None

def capture_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return None
    detected_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            resized = cv2.resize(face_roi, (96, 96))
            normalized = resized / 255.0
            reshaped_rgb = np.repeat(normalized[..., np.newaxis], 3, axis=-1)
            reshaped_rgb = np.reshape(reshaped_rgb, (1, 96, 96, 3))
            prediction = model.predict(reshaped_rgb)
            emotion_idx = np.argmax(prediction)
            detected_emotion = emotion_labels[emotion_idx]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == 13:  #'Enter' key
            break
    cap.release()
    cv2.destroyAllWindows()
    return detected_emotion

def singer_priority(row):
    singers = row["Singer(s)"].split(" & ")
    singer_priorities = [user_preferences["liked_singers"].get(s, 0) for s in singers]
    if random.random() < 0.6:
        return max(singer_priorities)
    else:
        return random.choice(singer_priorities)

def remove_preference(song_name):
    global user_preferences
    song_row = music_df[music_df["Song Name"] == song_name]
    if not song_row.empty:
        singers = song_row.iloc[0]["Singer(s)"].split(" & ")
        for singer in singers:
            if singer in user_preferences["liked_singers"]:
                user_preferences["liked_singers"][singer] -= 1
                if user_preferences["liked_singers"][singer]==0:
                    del user_preferences["liked_singers"][singer]

    # Remove from song blacklist/whitelist
    if song_name in user_preferences["disliked_songs"]:
        user_preferences["disliked_songs"].remove(song_name)
    if song_name in user_preferences["liked_songs"]:
        user_preferences["liked_songs"].remove(song_name)

    # Save updated preferences
    with open(PREF_FILE, "w") as f:
        json.dump(user_preferences, f)

def recommend_songs(emotion, language_filter):
    global music_df

    # Recalculate valid_songs from scratch
    if language_filter != "All":
        valid_songs = music_df[music_df["Language"] == language_filter]
    else:
        # Ensure a balanced mix of Hindi and English songs
        hindi_songs = music_df[music_df["Language"] == "Hindi"]
        english_songs = music_df[music_df["Language"] == "English"]
        num_samples = min(len(hindi_songs), len(english_songs))

        hindi_songs_sample = hindi_songs.sample(n=num_samples)
        english_songs_sample = english_songs.sample(n=num_samples)
        valid_songs = pd.concat([hindi_songs_sample, english_songs_sample])

    # Exclude disliked songs
    valid_songs = valid_songs[~valid_songs["Song Name"].isin(user_preferences["disliked_songs"])]

    # 🟢 Case 1: Neutral → Any song is valid
    if emotion == "neutral":
        recommendations = valid_songs.sample(n=min(5, len(valid_songs)))

    else:
        # 🟢 Case 2: Filter based on emotion
        filtered_songs = valid_songs[valid_songs["Mood"] == emotion]

        if filtered_songs.empty:
            # If no matching songs, pick from the full valid list
            recommendations = valid_songs.sample(n=min(5, len(valid_songs)))
        else:
            # Assign priority based on singer
            filtered_songs = filtered_songs.copy()  # Avoid SettingWithCopyWarning
            filtered_songs["priority"] = filtered_songs.apply(singer_priority, axis=1)

            # Sort and take at least 5 songs
            filtered_songs = filtered_songs.sort_values(by="priority", ascending=False)
            num_songs_to_sample = min(5, len(filtered_songs))
            recommendations = filtered_songs.sample(n=num_songs_to_sample, replace=False)

    # Return only the necessary columns
    return recommendations[["Song Name", "Singer(s)", "Mood", "Language"]]

# Function to update user preferences (increase singer priority)
def increase_singer_priority(song_name):
    global user_preferences
    song_row = music_df[music_df["Song Name"] == song_name]
    if song_row.empty:
        return  # Skip if song not found

    singers = song_row.iloc[0]["Singer(s)"].split(" & ")
    for singer in singers:
        user_preferences["liked_singers"][singer] = user_preferences["liked_singers"].get(singer, 0) + 1

    # Remove from disliked list if exists
    if song_name in user_preferences["disliked_songs"]:
        user_preferences["disliked_songs"].remove(song_name)
    user_preferences["liked_songs"].append(song_name)

    # Save updated preferences
    with open(PREF_FILE, "w") as f:
        json.dump(user_preferences, f)

# Function to update user preferences (add song to blacklist)
def decrease_song_priority(song_name):
    global user_preferences

    if song_name in music_df["Song Name"].values:
        user_preferences["disliked_songs"].append(song_name)

        # Remove from liked list if exists
        if song_name in user_preferences["liked_songs"]:
            user_preferences["liked_songs"].remove(song_name)

        # Save updated preferences
        with open(PREF_FILE, "w") as f:
            json.dump(user_preferences, f)

detected_emotion=''

# Streamlit App UI
st.title("🎵 Emotion-Based Music Recommendation System")
# Add language filter checkbox system
language_filter = st.selectbox("Select Language", options = ["All", "Hindi", "English"], key = "language_filter")

if detected_emotion == '' and not user_preferences["emotion"]:
    st.write("Press **'Capture Emotion'** to detect your mood and get song recommendations.")
    if st.button("Capture Emotion"):
        st.write("Please wait...")
        detected_emotion = capture_emotion()
        user_preferences["emotion"].append(detected_emotion)
        st.success(f"**Last Detected Emotion:** {detected_emotion}")
        with open(PREF_FILE, "w") as f:
            json.dump(user_preferences, f)
        st.rerun()
else:
    detected_emotion = user_preferences["emotion"][-1]
    
    st.write(f"Current Language Filter: {language_filter}")
    
    # Check if language has changed
    if 'previous_language_filter' not in st.session_state:
        st.session_state.previous_language_filter = language_filter

    # If language has changed, reset the song list and fetch new recommendations
    if language_filter != st.session_state.previous_language_filter:
        st.session_state.previous_language_filter = language_filter  # Update to the new language filter
        st.session_state.is_songs_reset = True  # Flag that songs have been reset
        # Get new recommendations based on the detected emotion and language filter
        st.session_state.recommended_songs = recommend_songs(detected_emotion, language_filter)
        st.rerun()

    if detected_emotion:
        st.subheader(f"🎶 Recommended {detected_emotion} Songs:")

    # Get recommendations
    recommended_songs = recommend_songs(detected_emotion, language_filter)
    # Store the recommended songs in session state
    if "recommended_songs" not in st.session_state:
        # Get recommendations only once (when the page is first loaded or reset)
        recommended_songs = recommend_songs(detected_emotion, language_filter)
        # Store only relevant data (song name, singer, mood, and language)
        st.session_state.recommended_songs = recommended_songs

    # Display the recommended songs
    temp_recommended_songs = st.session_state.recommended_songs

    # Display the recommended songs and handle like/dislike buttons
    for _, song in temp_recommended_songs.iterrows():
        song_name = song["Song Name"]
        singer = song["Singer(s)"].split(" & ")[0]  # Only first singer
        language = song["Language"]
        youtube_url = get_youtube_url(song_name, singer)

    # Check if song is already liked/disliked
        liked = song_name in user_preferences["liked_songs"]
        disliked = song_name in user_preferences["disliked_songs"]

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write(f"🎵 **{song_name}** by *{singer}* | 🌍 {language}")
        with col2:
            if youtube_url:
                st.markdown(f"[▶️ Play on YouTube]({youtube_url})", unsafe_allow_html=True)
            else:
                st.write("🔍 No link found.")
        with col3:
            if liked:
                st.button("✅", key=f"like_{song_name}", help="Liked (Click to remove)", on_click=remove_preference, args=(song_name,))
            else:
                st.button("👍", key=f"like_{song_name}", help="Like this song", on_click=increase_singer_priority, args=(song_name,))

        with col4:
            if disliked:
                st.button("❌", key=f"dislike_{song_name}", help="Disliked (Click to remove)", on_click=remove_preference, args=(song_name,))
            else:
                st.button("👎", key=f"dislike_{song_name}", help="Dislike this song", on_click=decrease_song_priority, args=(song_name,))

    # Reset buttons
    st.subheader("What would you like to do next?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Songs"):
            # Get new recommendations
            st.session_state.recommended_songs = recommend_songs(detected_emotion, language_filter)
            st.session_state.is_songs_reset = True
            st.rerun()
    with col2:
        if st.button("🔄 Reset Emotion"):
            st.write("📷 Restarting emotion detection...")
            # Reset or re-enable the emotion detection mechanism as needed
            user_preferences["emotion"].clear()
            st.session_state.recommended_songs = recommend_songs(detected_emotion, language_filter)
            with open(PREF_FILE, "w") as f:
                json.dump(user_preferences, f)
            st.rerun()