import os
from pathlib import Path
import sys
import streamlit as st
import json
import pandas as pd
from vncorenlp import VnCoreNLP
from fastai.learner import load_learner

# Ghi đè PosixPath để tránh lỗi trên Windows
if sys.platform == "win32":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# Đường dẫn các file
current_dir = Path.cwd()
vncorenlp_jar_path = os.path.join(current_dir, "vncorenlp", "VnCoreNLP-1.1.1.jar")
model_path = os.path.join(current_dir, "models", "model-vihealthbert-stage-2.pkl")
question_labels_path = os.path.join(current_dir, "dataset", "question_labels.json")
answer_dataset_path = os.path.join(current_dir, "dataset", "healifyLLM_answer_dataset.csv")

# Khởi tạo VnCoreNLP
rdrsegmenter = VnCoreNLP(vncorenlp_jar_path, annotators="wseg", max_heap_size='-Xmx500m')

def vncorenlp_word_segment(text):
    sentences = rdrsegmenter.tokenize(text)
    return " ".join([word for sentence in sentences for word in sentence])

# Tải mô hình FastAI
blurr_model = load_learner(fname=model_path)

# Tải nhãn và dữ liệu câu trả lời
with open(question_labels_path, 'r', encoding='utf-8') as f:
    question_dictionary = json.load(f)

answers_df = pd.read_csv(answer_dataset_path, encoding='utf-8', engine='python')

# Detect question and predict label
def detect_question(text):
    segmented_text = vncorenlp_word_segment(text)
    prediction = blurr_model.blurr_predict(segmented_text)[0]
    probs = prediction['probs']
    class_labels = prediction['class_labels']
    max_prob_index = probs.index(max(probs))
    predicted_label = class_labels[max_prob_index]
    return predicted_label, probs, class_labels

# Get the answer from predicted label
def get_answer_from_label(predicted_label):
    answer_row = answers_df[answers_df['label'] == predicted_label]
    if not answer_row.empty:
        return answer_row['answer'].values[0]
    else:
        return "Xin lỗi! Chúng tôi không tìm thấy câu trả lời cho câu hỏi này 😓"

# Predict answer from a question
def predict_answer_from_question(question):
    predicted_label, probs, class_labels = detect_question(question)
    answer = get_answer_from_label(predicted_label)
    return predicted_label, answer, probs, class_labels

# Streamlit app
apptitle = 'HealifyAI'
st.set_page_config(page_title=apptitle, page_icon="🤖")

st.title("HealifyAI - Hệ thống trả lời câu hỏi sức khỏe 🤖 🧬")
st.write("Chào mừng bạn đến với HealifyAI 🥰 🚀")
st.write("🔎 💬 Bạn có thể hỏi tổng quan về một căn bệnh, nguyên nhân, triệu chứng, đối tượng nguy cơ, phòng ngừa hay các biện pháp chuẩn đoán, điều trị nhé 🫶")
st.caption("Dữ liệu được lấy từ [VINMEC](https://www.vinmec.com/vie/benh/)")
st.write("Hãy nhập câu hỏi của bạn về sức khỏe bên dưới 👇")

# Initialize chat history
with st.chat_message("assistant"):
    st.write("Hello 👋")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to add a message to history
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hãy nhập câu hỏi của bạn về sức khỏe"):
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    predicted_label, answer, probs, class_labels = predict_answer_from_question(prompt)
    sorted_indices = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)
    top_related_questions = [class_labels[i] for i in sorted_indices[1:4]]
    response = (
        "**HealifyAI:**\n" + answer.replace("\n", "  \n") + "\n\n" +
        "**Độ chính xác:** {:.2f}\n\n".format(max(probs)*100)
    )

    add_message("assistant", response)
    with st.chat_message("assistant"):
        st.markdown(response)

        st.markdown("**Các câu hỏi liên quan bạn có thể quan tâm:**\n")
        for i, related_question in enumerate(top_related_questions):
            st.write(f"{related_question}")
                

