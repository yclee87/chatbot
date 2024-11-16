import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load LLama 1B model and tokenizer
@st.cache_resource
def load_model():
    model_name = "decapoda-research/llama-1b-hf"  # LLama 1B 모델 이름
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# 모델 로드
tokenizer, model = load_model()

# Streamlit 앱 인터페이스
st.title("🌸 일본 문화 및 관광 전문가 챗봇")
st.write(
    "이 챗봇은 일본 문화, 관광, 전통, 현대 생활에 대한 전문적인 정보를 제공합니다. "
    "LLama 1B 모델을 활용하며, 일본에 대해 궁금한 모든 것을 물어보세요!"
)

# 사용자 입력 받기
user_input = st.text_input("일본에 대해 무엇이든 물어보세요!")

if user_input:
    # 입력 토큰화
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # 모델로부터 응답 생성
    with st.spinner("응답 생성 중..."):
        outputs = model.generate(
            input_ids, 
            max_length=500,  # 최대 출력 길이
            num_return_sequences=1,
            temperature=0.7  # 출력의 다양성을 조정
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 응답 출력
    st.markdown(f"**챗봇:** {response}")
