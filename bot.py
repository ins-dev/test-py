threads = {}

import json
import os
import time
import requests
from dotenv import load_dotenv
import openai
from openai import OpenAI
import requests
import re
import logging
import subprocess
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ContextTypes,
    CallbackContext,
    MessageFilter,
)

import asyncio

from concurrent.futures import ThreadPoolExecutor

from pydub import AudioSegment

load_dotenv()
key = os.getenv("OPENAI_API_KEY")
TOKEN = os.getenv("TELEGRAM_TOKEN")
SERP_API_KEY = os.getenv("SERP_API_KEY")


client = OpenAI(api_key=key)
assistantId = os.getenv("ASSISTANT_ID")
threads_path = "threads.json"


def generate_image_with_dalle(
    prompt, quality="standard", size="1024x1024", style="natural"
):
    api_key = key 
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "style": style,
        "response_format": "url",
    }
    response = requests.post(
        "https://api.openai.com/v1/images/generations", headers=headers, json=data
    )
    if response.status_code == 200:
        result = response.json()
        return result["data"][0]["url"]
    else:
        raise ValueError(f"Error generating image: {response.text}")


async def generate_voice_response(text):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/FCcj7yrONI0InGe1yyYr"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "c83c27f3af64fbf3525a82d5ce9bd964",
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        voice_message_path = "output.mp3"
        with open(voice_message_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return voice_message_path
    else:
        print(f"Error generating voice response: {response.status_code}")
        return None


def fetch_google_results(query):
    params = {"q": query, "api_key": SERP_API_KEY}

    url = "https://serpapi.com/search"
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def transcribe_audio(wav_file_path):
    try:
        with open(wav_file_path, "rb") as audio_file:
            print(f"Transcribing file: {wav_file_path}")
            transcription_result = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            print("Transcription result:", transcription_result)

            return transcription_result

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


executor = ThreadPoolExecutor()


async def transcribe_and_process_audio(file_path, update, context):
    global threads  

    print(f"Starting transcription of {file_path}")
    transcribed_text = transcribe_audio(file_path)

    if transcribed_text is None:
        print("Transcription failed. No result returned.")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Sorry, I couldn't understand the voice message.",
        )
        return

    transcribed_text_str = (
        transcribed_text.text
    )  
    print(f"Transcribed text: {transcribed_text_str}")

    thread = None

    try:
        user_thread_id = threads.get(str(update.effective_user.id))
        if not user_thread_id:
            thread = client.beta.threads.create()
            threads[str(update.effective_user.id)] = thread.id
            with open(threads_path, "w") as file:
                json.dump(threads, file)
            print(f"Created new thread with ID: {thread.id}")
        else:
            thread = client.beta.threads.retrieve(thread_id=user_thread_id)
            print(f"Retrieved existing thread with ID: {thread.id}")
    except Exception as e:
        print(f"Error during thread creation/retrieval: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="An error occurred. Please try again later.",
        )
        return

    if thread is not None:
        try:
            answer = answerResponse(thread, transcribed_text_str)
            print(f"Answer received: {answer}")
        except Exception as e:
            print(f"Error in answerResponse function: {e}")
            answer = "Server Error, please try again."
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text=answer
            )
            return

        print("Generating voice message...")  
        voice_message_path = await generate_voice_response(answer) 

        if voice_message_path:  
            with open(voice_message_path, "rb") as voice: 
                await context.bot.send_voice(
                    chat_id=update.effective_chat.id, voice=voice
                ) 
            os.remove(voice_message_path) 
        else:  
            print("Error: voice message generation failed.")
    else:
        print("No thread object to proceed with answering.")


def clean_api_response(data):
    regex = r"【\d+†source】"
    return re.sub(regex, "", data)


def convert_ogg_to_wav(ogg_file_path):
    wav_file_path = ogg_file_path.replace(".ogg", ".wav")
    command = ["ffmpeg", "-i", ogg_file_path, wav_file_path]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("FFmpeg Output:", process.stdout.decode())
    print("FFmpeg Error:", process.stderr.decode())
    return wav_file_path


def answerResponse(thread, message):
    print(f"Preparing to answer: {message}")
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistantId,
    )
    old_status = ""
    print(f"Thread run status: {run.status}")

    while True:
        run = client.beta.threads.runs.retrieve(
            run_id=run.id,
            thread_id=thread.id,
        )
        current_status = run.status
        if current_status != old_status:
            print(f"Run status: {run.status} For Thread: {thread.id}")
            old_status = current_status

        if run.status not in ["queued", "in_progress", "cancelling", "requires_action"]:
            break
        elif run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                try:
                    if tool_call.function.arguments:
                        arguments = json.loads(tool_call.function.arguments)
                        print(f"Should call '{function_name}' with args {arguments}")
                        if function_name == "get_token_price":
                            token_name = arguments["token_name"]
                            price = get_token_price(token_name)
                            tool_outputs.append(
                                {"tool_call_id": tool_call.id, "output": price}
                            )

                            print(f"Called '{function_name}' with args {arguments}")
                        elif function_name == "fetch_google_results":
                            query = arguments["query"]
                            results = fetch_google_results(query)
                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "output": json.dumps(results),
                                }
                            )
                        elif function_name == "generate_image_with_dalle":
                            prompt = arguments["prompt"]

                            quality = arguments.get("quality", "standard")
                            size = arguments.get("size", "1024x1024")
                            style = arguments.get("style", "natural")

                            try:
                                image_url = generate_image_with_dalle(
                                    prompt, quality=quality, size=size, style=style
                                )
                                tool_outputs.append(
                                    {"tool_call_id": tool_call.id, "output": image_url}
                                )
                                print(
                                    f"Generated image with DALL-E for prompt: {prompt}"
                                )
                            except ValueError as e:
                                print(f"Error: {str(e)}")
                                tool_outputs.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "output": "Error generating image",
                                    }
                                )

                            print(f"Called '{function_name}' with args {arguments}")
                    else:
                        print("No arguments provided for the function call")
                except json.JSONDecodeError as e:
                    print(f"An error occurred while decoding JSON: {e}")
            client.beta.threads.runs.submit_tool_outputs(
                run_id=run.id,
                thread_id=thread.id,
                tool_outputs=tool_outputs,
            )

        else:
            time.sleep(0.1)

    if run.status == "completed":
        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            limit=1,
        )
        response = clean_api_response(messages.data[0].content[0].text.value)
        print(f"Generated response: {response}")
        return response
    else:
        print(f"Run status issue: {run.status}")
        return "Server Error, please try again."


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.ERROR,
)


def start(update: Update, context: CallbackContext):
    threads_path = "./telegram/threads.json"

    with open(threads_path, "r") as file:
        threads = json.load(file)
    user_thread = threads.get(str(update.effective_user.id))
    print("user_thread: " + str(user_thread))
    if user_thread is None:
        thread = client.beta.threads.create()
        threads[str(update.effective_user.id)] = thread.id
        with open(threads_path, "w") as file:
            json.dump(threads, file)
    else:
        thread = client.beta.threads.retrieve(thread_id=user_thread)
    print(f"Thread ID: {thread.id}\n")

    print("User : " + str(update.effective_user.id))
    print("Message : " + str(update.message.text))
    print("Chat id :" + str(update.effective_chat.id))
    print("from user: " + str(update.effective_user.id))
    context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me! start!"
    )


async def clear(update: Update, context: CallbackContext):
    threads_path = "./telegram/threads.json"

    with open(threads_path, "r") as file:
        threads = json.load(file)
    user_thread = threads.get(str(update.effective_user.id))
    print("user_thread: " + str(user_thread))
    threads[str(update.effective_user.id)] = None
    with open(threads_path, "w") as file:
        json.dump(threads, file)

    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Conversation cleared! "
    )


def echo(update: Update, context: CallbackContext):
    threads_path = "./telegram/threads.json"
    message_text = update.message.text

    with open(threads_path, "r") as file:
        threads = json.load(file)

    user_thread_id = threads.get(str(update.effective_user.id))

    if user_thread_id:
        try:
            thread = client.beta.threads.retrieve(thread_id=user_thread_id)
        except openai.NotFoundError:
            thread = client.beta.threads.create()
            threads[str(update.effective_user.id)] = thread.id
            with open(threads_path, "w") as file:
                json.dump(threads, file)
    else:
        thread = client.beta.threads.create()
        threads[str(update.effective_user.id)] = thread.id
        with open(threads_path, "w") as file:
            json.dump(threads, file)

    answer = answerResponse(thread, message_text)

    if update.effective_chat.type != "private":
        if answer.startswith("http://") or answer.startswith("https://"):
            context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=answer,
                reply_to_message_id=update.message.message_id,
            )
        else:
            context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=answer,
                reply_to_message_id=update.message.message_id,
            )
    else:
        if answer.startswith("http://") or answer.startswith("https://"):
            context.bot.send_photo(chat_id=update.effective_chat.id, photo=answer)
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


def handle_voice_message(update: Update, context: CallbackContext):
    voice_file_id = update.message.voice.file_id

    voice_file = context.bot.get_file(voice_file_id)

    os.makedirs("voice_messages", exist_ok=True)
    ogg_file_path = f"voice_messages/{voice_file_id}.ogg"

    voice_file.download(ogg_file_path)
    print(f"Voice message saved at {ogg_file_path}")

    wav_file_path = convert_ogg_to_wav(ogg_file_path)
    print(f"Converted to WAV at {wav_file_path}")
    with ThreadPoolExecutor() as executor:
        executor.submit(
            asyncio.run, transcribe_and_process_audio(wav_file_path, update, context)
        )


def caps(update: Update, context: CallbackContext):
    print("args", context.args)
    text_caps = " ".join(context.args).upper()
    context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)


class FilterReplyToBot(MessageFilter):
    def filter(self, message):
        return message.reply_to_message and message.reply_to_message.from_user.is_bot


class FilterBotMentionOrStart(MessageFilter):
    name_keywords = ["bot", "ai", "бот", "ии"]

    def filter(self, message):
        text = message.text.lower()
        return ("@bot_username" in text) or any(
            text.startswith(keyword) for keyword in self.name_keywords
        )


filter_reply_to_bot = FilterReplyToBot()
filter_bot_mention_or_start = FilterBotMentionOrStart()


if __name__ == "__main__":
    updater = Updater(TOKEN, use_context=True)

    echo_handler = MessageHandler(
        Filters.text
        & (~Filters.command)
        & (
            filter_reply_to_bot
            | filter_bot_mention_or_start
            | Filters.chat_type.private
        ),
        echo,
    )
    updater.dispatcher.add_handler(echo_handler)

    updater.dispatcher.add_handler(CommandHandler("start", start))
    updater.dispatcher.add_handler(CommandHandler("clear", clear))
    updater.dispatcher.add_handler(CommandHandler("caps", caps))
    updater.dispatcher.add_handler(MessageHandler(Filters.voice, handle_voice_message))

    print("Bot started")
    updater.start_polling()
    updater.idle()
