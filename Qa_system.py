import os
import streamlit as st
import torch
from transformers import BertTokenizer

st.markdown("<h1 style='text-align: center; color: black;'>Question Answering System</h1>", unsafe_allow_html=True)
st.write("")

def load_model():
	path='E:/sem7 project/implementation/models/bert_prediction.pt'
	model = torch.load(path)
	return model

def gui_input():
	Paragraph = st.text_area("Enter the Paragraph")
	Question = st.text_area("Enter the Query")

	return Paragraph,Question




def gui(model,paragraph,question):
	tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
	encoding = tokenizer.encode_plus(text=question,text_pair=paragraph, add_special=True)
	inputs = encoding['input_ids']  #Token embeddings
	sentence_embedding = encoding['token_type_ids']  #Segment embeddings
	tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens 
	scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
	
	start_index = torch.argmax(scores[0])
	end_index = torch.argmax(scores[1])
	
	answer = ' '.join(tokens[start_index:end_index+1])

	corrected_answer = ''
	for word in answer.split():
		if word[0:2]=='##':
			corrected_answer+=word[2:]
		else:
			corrected_answer+=' '+word
	st.write("Answer Is:",corrected_answer)

model = load_model()
para,ques = gui_input()
gui(model,para,ques)