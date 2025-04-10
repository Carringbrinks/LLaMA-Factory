import json
import gradio as gr
from openai import OpenAI
from format_output import format_response


SYSTEM_PROMPT="""请记住你的身份：你是九郡绿建训练的专注于双碳减排领域研究的AI。你熟知能源转型、节能减排技术、碳足迹核算等多方面知识。
无论是工业、交通还是建筑领域的减排策略，或是碳交易机制等相关问题，你都能为用户提供专业的分析和有效的建议，助力推动双碳目标的实现。

当用户问你你是谁或者让你介绍关于自己的时候，请按照上面的要求回答"""

def openai_post(
    usr_message: str,
    history: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    single_usr_messages_log: dict,
    count: int,
    system_prompt: str=SYSTEM_PROMPT,
):

    model_name = llm.models.list().data[0].id
    try:
        message_log = []
   
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": usr_message})
        response = llm.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
            
        )
        response = response.choices[0].message.content
        print(response)
        response = format_response(response)
        history.append({"role": "user", "content": usr_message})
        history.append({"role": "assistant", "content": response})

        message_log.append({"role": "system", "content": system_prompt})
        message_log.append({"role": "user", "content": usr_message})
        message_log.append({"role": "assistant", "content": response})
        single_usr_messages_log.update({str(count): message_log})
        count += 1
        ## 保存当前页面
        save_log(current_log_path, [{message_log[1]["content"]:single_usr_messages_log}])
        return history, single_usr_messages_log, count
    except Exception as e:
        print(e)
        return f"An error occurred: {str(e)}"


def clear_chat(messages_log, all_usr_message_log):
    chatbot = []
    history = []
    count = 0
    if messages_log:
        ## 保存clear之前的会话
        all_usr_message_log.append({messages_log["0"][1]["content"]:messages_log})
        save_log(history_log_path, all_usr_message_log)
    messages_log = {}
    return chatbot, history, messages_log, all_usr_message_log, count


def save_log(json_path, messages):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                messages,
                indent=4,
                ensure_ascii=False,
            )
        )

def create_chat_box():
    with gr.Blocks() as demo:
        with gr.Column():
            chatbot = gr.Chatbot(type="messages", show_copy_button=True)
            history = gr.State([])
            messages_log = gr.State({})
            all_messages_log = gr.State([])
            count = gr.State(0)
            with gr.Row():
                with gr.Column(scale=4):
                    # system_prompt = gr.Textbox("", label="System Prompt")
                    query = gr.Textbox(show_label=False, lines=8)
                    submit_btn = gr.Button(variant="primary")

                with gr.Column(scale=1):
                    max_new_tokens = gr.Slider(
                        minimum=8,
                        maximum=4096,
                        value=2048,
                        step=1,
                        label="Max_new_tokens",
                    )
                    top_p = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.7, step=0.01, label="Top_p"
                    )
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.5,
                        value=0.95,
                        step=0.01,
                        label="Temperature",
                    )
                    clear_btn = gr.Button(variant="primary", value="clear")
        submit_btn.click(
            openai_post,
            [
                query,
                history,
                # system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                messages_log,
                count,
            ],
            [chatbot, messages_log, count],
        )
        clear_btn.click(
            clear_chat,
            inputs = [messages_log, all_messages_log],
            outputs=[chatbot, history, messages_log, all_messages_log, count],
        )
    return demo


if __name__ == "__main__":
    
    openai_api_key = "zjuici"
    openai_base_url = "http://127.0.0.1:8000/v1"
    llm = OpenAI(
        api_key=openai_api_key,
        base_url=openai_base_url,
    )

    current_log_path = "./current_log_72b.json"
    history_log_path = "./history_log_72b.json"

    create_chat_box().launch(share=True)

# CUDA_VISIBLE_DEVICES=0 vllm serve /home/scb123/HuggingfaceWeight/Qwen2.5-1.5B --api-key zjuici --served-model-name qwen2.5-pt --port 23335 --uvicorn-log-level info --dtype float16  --max-model-len 8192

