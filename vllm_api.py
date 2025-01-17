from openai import OpenAI
import requests


def openai_post(
    usr_message: str,
    system_prompt: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    api_type: str = "chat",
):
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
    model_name = client.models.list().data[0].id

    try:
        if api_type == "chat":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_message},
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            prompt = usr_message
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return response.choices[0].text
    except Exception as e:
        return f"An error occurred: {str(e)}"


def request_post(
    usr_message: str,
    url: str,
    model_name: str,
    system_prompt: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    api_type: str = "chat",
):

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if api_type == "chat":

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_message},
        ]
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        messages = usr_message
        payload = {
            "model": model_name,
            "prompt": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(response.json()["usage"])
        return response.json()["choices"][0]["text"]
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"



if __name__ == "__main__":

    # Set OpenAI's API key and API base to use vLLM's API server.
    OPENAI_API_KEY = "zjuici"
    OPENAI_API_BASE = "http://localhost:23335/v1"

    url_chat = "http://localhost:23335/v1/chat/completions"
    url_complete = "http://localhost:23335/v1/completions"

    model_id = "qwen2.5-pt"
    system_messages = ""
    user_message = "郭啸天带着张十五来到村头一家小酒店中，在张饭桌旁坐小酒店的主人是个跛子，撑着两根拐杖，慢慢烫了两壶黄酒，"
    new_tokens = 1000
    temperature = 0.5

    api_type = ""

    # 使用 openai 库
    # result1 = openai_post(
    #     usr_message=user_message,
    #     system_prompt=system_messages,
    #     max_tokens=new_tokens,
    #     temperature=temperature,
    #     api_type=api_type,
    # )
    # print("Response from openai library:", result1)

    # 使用 requests 库
    result1 = request_post(
        url=url_chat if api_type else url_complete,
        model_name=model_id,
        usr_message=user_message,
        system_prompt=system_messages,
        max_tokens=new_tokens,
        temperature=temperature,
        api_type=api_type,
    )
    print("Response from requests library:", result1)

    # print("="*100)

    # url_chat = "http://localhost:23336/v1/chat/completions"
    # url_complete = "http://localhost:23336/v1/completions"
    # result2 = request_post(
    # url=url_chat if api_type else url_complete,
    # model_name=model_id,
    # usr_message=user_message,
    # system_prompt=system_messages,
    # max_tokens=new_tokens,
    # temperature=temperature,
    # api_type=api_type,
    # )
    # print("Response from requests library:", result2)