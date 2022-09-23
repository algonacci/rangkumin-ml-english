from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

kwargs = {
    "max_length": 130,
    "min_length": 30,
    "do_sample": False
}


def summarize(text):
    summary_text = summarizer(text, **kwargs)[0].get("summary_text")
    return summary_text
