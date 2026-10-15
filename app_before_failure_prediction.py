
import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT = Path("/content/manufacturing_slm")
CSV_FILE = PROJECT / "manufacturing_knowledge.csv"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

st.set_page_config(page_title="Manufacturing SLM", page_icon="🏭", layout="wide")
st.title("🏭 Manufacturing SLM Intelligence")
st.caption("CPU-based manufacturing assistant with image upload and sensor monitoring")

st.warning(
    "Prototype only. Sensor limits below are configurable examples, "
    "not certified safety limits. Follow approved plant procedures."
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model

@st.cache_data
def load_knowledge():
    if CSV_FILE.exists():
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=[
        "topic", "symptom", "possible_cause", "recommended_action"
    ])

knowledge_df = load_knowledge()

def find_context(question):
    q = question.lower()
    keyword_map = {
        "High vibration": ["vibration", "bearing", "imbalance", "misalignment"],
        "High temperature": ["temperature", "overheat", "overheating", "hot", "cooling"],
        "Low pressure": ["pressure", "leak", "leakage", "pump"],
        "Voltage fluctuation": ["voltage", "electrical", "power supply"],
        "Machine failure": ["failure", "stoppage", "stopped", "machine fault"]
    }

    if knowledge_df.empty:
        return "No knowledge-base records are available."

    matched = [
        topic for topic, words in keyword_map.items()
        if any(word in q for word in words)
    ]

    if "topic" not in knowledge_df.columns:
        return "Knowledge-base CSV does not contain a topic column."

    rows = knowledge_df[knowledge_df["topic"].isin(matched)]
    if rows.empty:
        return "No matching record found in the example knowledge base."

    return "\n\n".join(
        "\n".join(f"{col}: {row[col]}" for col in rows.columns)
        for _, row in rows.iterrows()
    )

def generate_answer(question, sensor_summary, image_note):
    tokenizer, model = load_model()
    context = find_context(question)

    prompt = f"""
You are a manufacturing maintenance assistant.
Give concise, cautious guidance to a factory operator.
Use the supplied knowledge and sensor information.
Do not invent measurements or claim a definite diagnosis.
Treat uploaded images as unanalysed unless image observations are supplied.
Recommend following approved site procedures and escalating unsafe conditions.

Manufacturing knowledge:
{context}

Sensor findings:
{sensor_summary}

Image status:
{image_note}

Operator question:
{question}
"""
    messages = [
        {"role": "system", "content": "You are a careful manufacturing assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=220, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

left, right = st.columns([1, 1])

with left:
    st.subheader("📷 Machine image")
    uploaded_image = st.file_uploader(
        "Upload a machine photo", type=["png", "jpg", "jpeg"]
    )
    image_note = "No image uploaded."
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded machine image", use_container_width=True)
            image_note = (
                "Image uploaded and displayed. No automated visual defect "
                "analysis has been performed."
            )
        except Exception as exc:
            st.error(f"Could not open image: {exc}")
            image_note = "Image could not be opened."

with right:
    st.subheader("📈 Sensor readings")
    st.caption("Enter your measurements and adjust the example thresholds to match approved limits.")
    c1, c2 = st.columns(2)

    with c1:
        vibration = st.number_input("Vibration (mm/s)", min_value=0.0, value=2.0, step=0.1)
        vibration_limit = st.number_input("Vibration example limit", min_value=0.1, value=4.5, step=0.1)
        temperature = st.number_input("Temperature (°C)", value=60.0, step=1.0)
        temperature_limit = st.number_input("Temperature example limit", value=80.0, step=1.0)

    with c2:
        pressure = st.number_input("Pressure (bar)", min_value=0.0, value=5.0, step=0.1)
        pressure_min = st.number_input("Minimum pressure example", min_value=0.0, value=3.0, step=0.1)
        voltage = st.number_input("Voltage (V)", min_value=0.0, value=230.0, step=1.0)
        voltage_min = st.number_input("Minimum voltage example", min_value=0.0, value=210.0, step=1.0)
        voltage_max = st.number_input("Maximum voltage example", min_value=0.1, value=250.0, step=1.0)

sensor_values = {
    "Vibration": (vibration, f"mm/s; example upper limit {vibration_limit}", vibration > vibration_limit),
    "Temperature": (temperature, f"°C; example upper limit {temperature_limit}", temperature > temperature_limit),
    "Pressure": (pressure, f"bar; example minimum {pressure_min}", pressure < pressure_min),
    "Voltage": (voltage, f"V; example range {voltage_min}–{voltage_max}", voltage < voltage_min or voltage > voltage_max)
}

findings = []
for name, (value, description, flagged) in sensor_values.items():
    if flagged:
        findings.append(f"{name}: {value} ({description}) — OUTSIDE EXAMPLE LIMIT")
    else:
        findings.append(f"{name}: {value} ({description}) — within example limits")

sensor_summary = "\n".join(findings)

st.subheader("Sensor status")
for name, (value, description, flagged) in sensor_values.items():
    if flagged:
        st.warning(f"{name}: {value} — outside the configured example limit")
    else:
        st.success(f"{name}: {value} — within the configured example limit")

st.subheader("💬 Ask the manufacturing assistant")
question = st.text_area(
    "Your question",
    value="Review these sensor readings and suggest what I should check.",
    height=100
)

if st.button("Analyze readings and answer", type="primary"):
    if not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Generating answer on CPU..."):
            try:
                answer = generate_answer(question, sensor_summary, image_note)
                st.subheader("Assistant answer")
                st.write(answer)
                with st.expander("Show sensor findings and retrieved knowledge"):
                    st.text(sensor_summary)
                    st.text(find_context(question))
                if uploaded_image is not None:
                    st.info(
                        "The image is displayed for operator review only. "
                        "Automated image defect detection is not yet connected."
                    )
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")


# --------------------------------------------------
# AI4I 2020 DATASET ANALYTICS
# --------------------------------------------------
st.divider()
st.header("📊 AI4I Dataset Analytics")

DATASET_PATH = PROJECT / "ai4i2020.csv"

if DATASET_PATH.exists():
    @st.cache_data
    def load_ai4i_data(path):
        return pd.read_csv(path)

    ai4i_df = load_ai4i_data(str(DATASET_PATH))

    total = len(ai4i_df)
    failures = int(ai4i_df["Machine failure"].sum())
    normal = total - failures

    col1, col2, col3 = st.columns(3)
    col1.metric("Total records", f"{total:,}")
    col2.metric("Normal records", f"{normal:,}")
    col3.metric("Failure records", f"{failures:,}")

    st.metric("Dataset failure rate", f"{failures / total * 100:.2f}%")

    st.subheader("Failure modes")
    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    st.bar_chart(ai4i_df[modes].sum())

    st.subheader("Sensor data exploration")
    sensor_columns = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear"
    ]
    selected_sensor = st.selectbox(
        "Choose a sensor to explore",
        sensor_columns
    )
    st.line_chart(ai4i_df[selected_sensor].head(200))

    with st.expander("View dataset sample"):
        st.dataframe(ai4i_df.head(20), use_container_width=True)

    st.caption(
        "This public dataset supports learning and exploration. "
        "Its failure labels are not validated real-world risk probabilities."
    )
else:
    st.error(f"Dataset not found: {DATASET_PATH}")
