
import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)


# ---------------- MODELS ----------------
MODELS = {
    "DistilBART (faster, lighter)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (better quality)": "facebook/bart-large-cnn",
}


# ---------------- SUMMARY LENGTHS ----------------
LENGTHS = {
    "Short": (30, 80),
    "Medium": (60, 150),
    "Long": (100, 250),
}


CHUNK_WORDS = 600


# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner="Loading model (first run downloads it)...")
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    return tokenizer, model


# ---------------- CHUNK TEXT ----------------
def chunk_text(text: str, size: int = CHUNK_WORDS):
    words = text.split()

    chunks = []

    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i + size]))

    return chunks


# ---------------- SUMMARIZE ONE CHUNK ----------------
def summarize_chunk(text, tokenizer, model, min_len, max_len):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    ids = model.generate(
        **inputs,
        min_length=min_len,
        max_length=max_len,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(
        ids[0],
        skip_special_tokens=True
    )


# ---------------- SUMMARIZE COMPLETE TEXT ----------------
def summarize(text, tokenizer, model, min_len, max_len):

    chunks = chunk_text(text)

    progress = st.progress(
        0.0,
        text="Summarizing..."
    )

    parts = []

    for i, chunk in enumerate(chunks, 1):

        words = len(chunk.split())

        # Avoid min_length being too large for short chunks
        safe_min_len = min(min_len, max(5, words // 2))

        parts.append(
            summarize_chunk(
                chunk,
                tokenizer,
                model,
                safe_min_len,
                max_len
            )
        )

        progress.progress(
            i / len(chunks),
            text=f"Summarized chunk {i}/{len(chunks)}"
        )

    progress.empty()

    combined = " ".join(parts)

    # If multiple chunks create a very long summary,
    # summarize the combined result once more.
    if len(chunks) > 1 and len(combined.split()) > CHUNK_WORDS:

        combined = summarize_chunk(
            combined,
            tokenizer,
            model,
            min_len,
            max_len
        )

    return combined


# ---------------- READ PDF ----------------
def read_pdf(file):

    reader = PdfReader(file)

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    return text


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Model",
    list(MODELS.keys())
)

length = st.sidebar.radio(
    "Summary length",
    list(LENGTHS.keys()),
    index=1
)


# ---------------- MAIN UI ----------------
st.title("📝 Text Summarizer")

st.write(
    "Paste text or upload a TXT/PDF file to get a concise summary."
)


source = st.radio(
    "Input type",
    ["Paste text", "Upload file"],
    horizontal=True
)


text = ""


# ---------------- PASTE TEXT ----------------
if source == "Paste text":

    text = st.text_area(
        "Your text",
        height=300,
        placeholder="Paste an article, notes, or report here..."
    )


# ---------------- UPLOAD FILE ----------------
else:

    file = st.file_uploader(
        "Upload .txt or .pdf",
        type=["txt", "pdf"]
    )

    if file:

        if file.name.lower().endswith(".pdf"):

            text = read_pdf(file)

        else:

            text = file.read().decode(
                "utf-8",
                errors="ignore"
            )

        with st.expander("📄 Preview extracted text"):

            st.write(
                text[:3000]
                + ("..." if len(text) > 3000 else "")
            )


# ---------------- SUMMARIZE BUTTON ----------------
if st.button("✨ Summarize", type="primary"):

    word_count = len(text.split())

    if word_count < 50:

        st.warning(
            "Please provide at least 50 words of text."
        )

    else:

        # Load selected model
        tokenizer, model = load_model(
            MODELS[model_name]
        )

        min_len, max_len = LENGTHS[length]

        # Generate summary
        summary = summarize(
            text,
            tokenizer,
            model,
            min_len,
            max_len
        )

        # ---------------- RESULTS ----------------
        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📄 Original")

            st.caption(
                f"{word_count} words"
            )

            st.write(
                text[:5000]
                + ("..." if len(text) > 5000 else "")
            )

        with col2:

            st.subheader("📝 Summary")

            summary_count = len(summary.split())

            reduction = max(
                0,
                100 - (summary_count * 100 // word_count)
            )

            st.caption(
                f"{summary_count} words • {reduction}% shorter"
            )

            st.success(summary)

            st.download_button(
                "⬇️ Download Summary",
                summary,
                "summary.txt",
                "text/plain"
            )

