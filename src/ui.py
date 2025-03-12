import gradio as gr
import os
import plotly.express as px
import random

from src.workflow import run_workflow_with_tracing


def add_message(history: list, message: str):
    if message is not None or message != "":
        history.append({"role": "user", "content": message})
    return history, gr.Textbox()


async def bot(history: list):
    msg = {"role": "assistant", "content": ""}
    
    query = history[-1]['content']

    content = await run_workflow_with_tracing(query)
    msg["content"] = content

    history.append(msg)
    return history


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        type="messages",
        show_copy_button=True,
    )

    chat_input = gr.Textbox(
        interactive=True,
        placeholder="че нахуй...",
        show_label=False,
    )

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(bot, [chatbot], chatbot, api_name="bot_response")

    chatbot.like(print_like_dislike, None, None)

if __name__ == "__main__":
    demo.launch()
