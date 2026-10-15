
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

st.set_page_config(
    page_title="Manufacturing SLM Intelligence",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Manufacturing SLM Intelligence")
st.write("A CPU-based language model for manufacturing maintenance assistance.")

st.warning(
    "Prototype demonstration using example knowledge. "
    "Always follow approved machine safety procedures."
)

DATA_PATH = Path("/content/manufacturing_slm/manufacturing_knowledge.csv")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

@st.cache_resource
def load_assistant():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )
    model.to("cpu")
    model.eval()
    knowledge = pd.read_csv(DATA_PATH)
    return tokenizer, model, knowledge

def retrieve_context(question, knowledge):
    q = question.lower()
    rules = {
        "High vibration": ["vibration", "bearing", "imbalance", "misalignment"],
        "High temperature": ["temperature", "overheat", "hot", "cooling"],
        "Low pressure": ["pressure", "leak", "pump"],
        "Voltage fluctuation": ["voltage", "electrical", "power supply"],
        "Machine failure": ["failure", "stoppage", "stopped", "machine fault"]
    }

    topics = [
        topic for topic, words in rules.items()
        if any(word in q for word in words)
    ]
    matches = knowledge[knowledge["topic"].isin(topics)]

    if matches.empty:
        return "No matching record found in the example knowledge base."

    return "\n".join(
        f"Topic: {r['topic']}; Possible causes: {r['possible_cause']}; "
        f"Suggested checks: {r['recommended_action']}"
        for _, r in matches.iterrows()
    )

def generate_answer(question, context, tokenizer, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise manufacturing maintenance assistant. "
                "Use the supplied context. Do not invent readings or claim "
                "a definitive fault diagnosis. If information is insufficient, "
                "say so. Follow approved site safety procedures."
            )
        },
        {
            "role": "user",
            "content": f"Knowledge:\n{context}\n\nQuestion: {question}"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

with st.spinner("Loading the SLM on CPU..."):
    try:
        tokenizer, model, knowledge = load_assistant()
        st.success("SLM loaded on CPU.")
    except Exception as e:
        st.error(f"Could not load the model or knowledge file: {e}")
        st.stop()

st.subheader("Ask the Manufacturing Assistant")
question = st.text_area(
    "Enter an operator question",
    placeholder="Example: What should I check if machine vibration increases?"
)

if st.button("Get answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        context = retrieve_context(question, knowledge)
        with st.spinner("Generating answer on CPU..."):
            answer = generate_answer(question, context, tokenizer, model)

        st.markdown("### Assistant answer")
        st.write(answer)

        with st.expander("View retrieved manufacturing context"):
            st.write(context)

st.divider()
st.caption("CPU inference • Example manufacturing knowledge • Human review required")
