# Third-party libraries
import gradio as gr
from langfuse import Langfuse

# Local imports
from src.settings import settings
from src.workflow_with_tracing import run_workflow_with_tracing

langfuse = Langfuse(
    public_key=settings.langfuse.PUBLIC_KEY,
    secret_key=settings.langfuse.SECRET_KEY,
    host=settings.langfuse.HOST,
)

def add_message(history: list, message: str):
    if message is not None or message != "":
        history.append({"role": "user", "content": message})
    return history, gr.Textbox(value=None)

async def bot(history: list):
    msg = {"role": "assistant", "content": ""}
    query = history[-1]["content"]

    try:
        content = await run_workflow_with_tracing(query)
        msg["content"] = content
    except Exception as e:
        error_message = (
            "Произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте позже или обратитесь в службу поддержки."
        )
        msg["content"] = error_message
        # Log the actual error for debugging
        print(f"Error processing query '{query}': {str(e)}")

    history.append(msg)
    return history

def print_like_dislike(history: gr.Chatbot, x: gr.LikeData):
    q = history[x.index - 1]["content"]
    a = history[x.index]["content"]

    langfuse.create_dataset_item(
        dataset_name="qa",
        input=q,
        expected_output=a,
        metadata={"positive": x.liked},
    )

with gr.Blocks(title="X5", fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        type="messages",
        autoscroll=True,
        show_copy_button=True,
        show_label=False,
    )

    chat_input = gr.Textbox(
        placeholder="Введите вопрос",
        interactive=True,
        show_label=False,
    )

    clear = gr.ClearButton([chat_input, chatbot], value="Очистить")

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input], [chatbot, chat_input], show_api=False
    )
    bot_msg = chat_msg.then(bot, [chatbot], chatbot, api_name="bot_response")

    chatbot.like(print_like_dislike, chatbot, None, show_api=False)

demo.launch(server_name='0.0.0.0')
