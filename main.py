import streamlit as st
import time
from io import StringIO
import nltk
from nltk.tokenize import word_tokenize
import readtime
import textstat
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
 

st.set_page_config(page_title="RESEARCH ET AL", 
                   page_icon=":books:",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

def p_title(title):
    st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">{title}</h3>', unsafe_allow_html=True)

st.sidebar.header('Try the magic :crystal_ball:')
nav = st.sidebar.radio('',['Summarize text','Analyze text'])
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

model_options = {
    "BART-large-406M": ("facebook/bart-large-cnn", BartForConditionalGeneration, BartTokenizer),
    "T5-large-738M": ("t5-large", T5ForConditionalGeneration, T5Tokenizer),
}

if nav == 'Summarize text':    
    st.markdown("<h4 style='text-align: center; color:grey;'>Accelerate knowledge with et al bot... &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    p_title('Summarize')
    st.text('')

    source = st.radio("How would you like to start? Choose an option below",
                          ("I want to input some text", "I want to upload a file"))
    st.text('')

    model_selection = st.selectbox("Select a model for summarization", list(model_options.keys()))

    if source == 'I want to input some text':
        input_su = st.text_area("Input your text in English (between 1,000 and 10,000 characters)", height=330)
        if st.button('Summarize'):
            if len(input_su) < 1000:
                st.error('Please enter a text in English of minimum 1,000 characters')
            else:
                with st.spinner('Processing...'):
                    start_time = time.time()
                    time.sleep(2)  # Simulating processing time
                    model_name, model_class, tokenizer_class = model_options[model_selection]
                    tokenizer = tokenizer_class.from_pretrained(model_name)
                    model = model_class.from_pretrained(model_name)
                    inputs = tokenizer("summarize: " + input_su, return_tensors="pt", max_length=1024, truncation=True)
                    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=100, max_length=1500, early_stopping=True)
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    end_time = time.time()
                    st.markdown('___')
                    st.write(f'{model_selection} Model')
                    st.success(summary)
                    st.write(f"Time taken: {end_time - start_time:.2f} seconds")
                    st.balloons()

    if source == 'I want to upload a file':
        file = st.file_uploader('Upload your file here',type=['txt'])
        if file is not None:
            with st.spinner('Processing...'):
                    start_time = time.time()
                    time.sleep(2)  # Simulating processing time
                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    if len(string_data) < 1000 or len(string_data) > 10000:
                        st.error('Please upload a file between 1,000 and 10,000 characters')
                    else:
                        model_name, model_class, tokenizer_class = model_options[model_selection]
                        tokenizer = tokenizer_class.from_pretrained(model_name)
                        model = model_class.from_pretrained(model_name)
                        inputs = tokenizer("summarize: " + string_data, return_tensors="pt", max_length=1024, truncation=True)
                        summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=100, max_length=1500, early_stopping=True)
                        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        end_time = time.time()
                        st.markdown('___')
                        st.write(f'{model_selection} Model')
                        st.success(summary)
                        st.write(f"Time taken: {end_time - start_time:.2f} seconds")
                        st.balloons()

if nav == 'Analyze text':
    st.markdown("<h4 style='text-align: center; color:grey;'>Accelerate knowledge with et al bot... &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    p_title('Analyze text')
    st.text('')

    a_example = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans or animals. Leading AI textbooks define the field as the study of 'intelligent agents': any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term 'artificial intelligence' to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving, however this definition is rejected by major AI researchers. AI applications include advanced web search engines, recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri or Alexa), self-driving cars (such as Tesla), and competing at the highest level in strategic game systems (such as chess and Go). As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology."

    source = st.radio("How would you like to start? Choose an option below",
                          ("I want to input some text", "I want to upload a file"))
    st.text('')

    if source == 'I want to input some text':
        input_me = st.text_area("Input your text in English (maximum of 10,000 characters)", value=a_example, height=330)
        if st.button('Analyze'):
            if len(input_me) > 10000:
                st.error('Please enter a text in English of maximum 10,000 characters')
            else:
                with st.spinner('Processing...'):
                    time.sleep(2)
                    nltk.download('punkt')
                    rt = readtime.of_text(input_me)
                    tc = textstat.flesch_reading_ease(input_me)
                    tokenized_words = word_tokenize(input_me)
                    lr = len(set(tokenized_words)) / len(tokenized_words)
                    lr = round(lr,2)
                    n_s = textstat.sentence_count(input_me)
                    st.markdown('___')
                    st.text('Reading Time')
                    st.write(rt)
                    st.markdown('___')
                    st.text('Text Complexity: from 0 or negative (hard to read), to 100 or more (easy to read)')
                    st.write(tc)
                    st.markdown('___')
                    st.text('Lexical Richness (distinct words over total number of words)')
                    st.write(lr)
                    st.markdown('___')
                    st.text('Number of sentences')
                    st.write(n_s)
                    st.balloons()

    if source == 'I want to upload a file':
        file = st.file_uploader('Upload your file here',type=['txt'])
        if file is not None:
            with st.spinner('Processing...'):
                    time.sleep(2)
                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    if len(string_data) > 10000:
                        st.error('Please upload a file of maximum 10,000 characters')
                    else:
                        nltk.download('punkt')
                        rt = readtime.of_text(string_data)
                        tc = textstat.flesch_reading_ease(string_data)
                        tokenized_words = word_tokenize(string_data)
                        lr = len(set(tokenized_words)) / len(tokenized_words)
                        lr = round(lr,2)
                        n_s = textstat.sentence_count(string_data)
                        st.markdown('___')
                        st.text('Reading Time')
                        st.write(rt)
                        st.markdown('___')
                        st.text('Text Complexity: from 0 or negative (hard to read), to 100 or more (easy to read)')
                        st.write(tc)
                        st.markdown('___')
                        st.text('Lexical Richness (distinct words over total number of words)')
                        st.write(lr)
                        st.markdown('___')
                        st.text('Number of sentences')
                        st.write(n_s)
                        st.balloons()
