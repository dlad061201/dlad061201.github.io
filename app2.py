# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 19:56:25 2023

@author: Dell
"""

import streamlit as st
from PyPDF2 import PdfReader
import pyttsx3
from gtts import gTTS
import time
from googletrans import Translator
import fitz  # PyMuPDF
from transformers import BartTokenizer, BartForConditionalGeneration
import os

translator = Translator()

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def generate_summary(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)

    # Adjust the parameters here for longer and more comprehensive summaries
    summary_ids = model.generate(inputs["input_ids"], max_length=40000, min_length=1000, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def convert_text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)

def main():
    st.title("PDF to Speech Converter")
    st.write("Upload a PDF file and click the download button to download MP3")

    # File upload
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if pdf_file:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        
        with st.spinner(text='Extracting...'):
            time.sleep(3)

        # Show extracted text
        st.subheader("Extracted Text:")
        st.text_area("Text", pdf_text, height=200)
        
        status = st.selectbox("Select your language", ['English', 'Hindi', 'Gujarati'])
        if(status == 'English'):
            final_text = pdf_text
            output_file = 'English_audio.mp3'
        elif(status == 'Hindi'):
            translation = translator.translate(pdf_text, src='en', dest='hi')
            final_text = translation.text
            output_file = 'Hindi_audio.mp3'
        elif(status == 'Gujarati'):
            translation = translator.translate(pdf_text, src='en', dest='gu')
            final_text = translation.text
            output_file = 'Gujarati_audio.mp3'
        
        if(st.button('Download Audio')):
            with st.spinner(text='Downloading...'):
                convert_text_to_speech(final_text, output_file)
            st.success("Downloaded")
            st.balloons()

if __name__ == "__main__":
    main()
