import streamlit as st
import os
from model import generate_summary
from reportlab.pdfgen import canvas
import base64
from fpdf import FPDF
import shutil

def createPdf(text, filename):
    encoded_text = text.encode('utf-8')
    translateText = encoded_text.replace(b"\n", b" ")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 9)
    pdf.write(5, translateText.decode('unicode-escape'))
    pdf.output(filename,"F")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def app():
    pdf_path = "C:\\Users\\Hp\\Desktop\\Project final\\Shane app\\assets\\uploadedfiles"
    upload_folder = 'uploadedfiles'
    output_filename = "assets\output.pdf"
    PDFbyte = bytes()
    
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    st.set_page_config(page_title= 'Summarizer', layout="centered", initial_sidebar_state="auto")
    add_bg_from_local('assets\Background.png')
    container = st.container()
    st.header("Your One-Stop Hub for Multilingual Summaries: Breaking Down Language Barriers")
    col1, col2 = st.columns([3,1])
    
    with col1:
        pdf_files = st.file_uploader(' ', type='pdf', accept_multiple_files=True)


    if pdf_files is not None:
        for pdf_file in pdf_files:
            with open(os.path.join("assets", upload_folder, pdf_file.name), "wb") as f: #change path to assets folder
                f.write(pdf_file.getbuffer())
                # print(pdf_file)
            st.success("File saved!")
         
    summarize_button = st.button('Summarize PDF')
    if summarize_button:
        try: 
            with open(os.path.join(pdf_path, os.listdir(pdf_path)[0]), 'rb') as f:
                if not f.read(1):
                    st.error('File is empty.')
                else:
                    with st.spinner('Summarizing'):
                        summary = generate_summary(pdf_path)

                    if summary is None:
                        st.error("No summary")       
                    else:
                        st.write(f'Generated Summary: ')
                        st.write(summary)
                        # container = st.container()
                        # container.write(
                        #     f"""
                        #         <div style='background-color: white; padding: 10px;'>
                        #         {st.write(summary)}
                        #     """, unsafe_allow_html = True
                        # )
                            
                        createPdf(summary, output_filename)
                        with open(output_filename, "rb") as pdf_file:
                            PDFbyte = pdf_file.read()
        except IndexError:
            st.error('No file uploaded')
        
    for dirpath, dirnames, filenames in os.walk(r"C:\\Users\\Hp\Desktop\\Project final\\Shane app\\assets\\uploadedfiles"):
        for file in filenames:
            os.remove(os.path.join("assets", upload_folder, file))

    
    st.download_button(label="Download PDF",
                    data=PDFbyte,
                    file_name="output.pdf",
                    mime='application/octet-stream')
    

        
if __name__ == '__main__':
    app()