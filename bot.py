import os
import logging
import tempfile
import base64
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import anthropic
from groq import Groq
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Claude — Scott Bradley's personal AI assistant, available on his phone via Telegram.

## Who Scott Is
- Wildlife videographer and Director of Storytelling at SOSA (Save Our Species Alliance)
- Nomadic — travels 10-11 months/year (Africa, South America, oceans, jungles)
- Lake District, UK home base
- Fast and scrappy in the field, methodical when planning
- Big dreamer who thinks in story arcs, then refines the details
- Loves his work — this is the dream, not a job

## His Business (SOSA)
- Save Our Species Alliance — conservation media agency + trip operator
- Helps NGOs worldwide tell their story and find funding
- Runs hands-on conservation experiences for everyday guests
- Founder: Blake Moynes (The Bachelor fame), funded personally
- Team of 8. Scott is on $4,500 CAD/month retainer
- Early startup, not yet profitable — this is urgent
- SOSA Instagram: ~17k followers. Scott personal IG: ~3,500 followers
- YouTube: <300 subs. Content quality is good, reach/discovery is the problem

## His Priorities Right Now
1. Help SOSA become profitable (directly grows his own income toward £60-80k/year target)
2. Build a real personal Instagram content strategy — consistent, intentional, purposeful
3. Develop conservation storytelling skills and manage his team better

## How to Work With Scott
- Quick decisions: short, direct, tough-love. Don't overthink it.
- Strategy topics: detailed is fine and helpful
- Default: just do it, take action, get stuff done
- He's on his phone — keep responses conversational and mobile-friendly
- When he sends a voice note, he's thinking on the fly — help him develop the idea, not just transcribe it
- Flag if platform/algorithm advice might be outdated
- Collaborative tone — help him think, not just receive answers

## Context
- When Scott is in the field he's moving fast — be concise
- He values structure but isn't rigid — direction over rules
- Conservation, storytelling, travel, Instagram growth, SOSA profitability are the recurring themes
"""

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "scottb-sosa/sosa-telegram-bot")
LOG_FILE_PATH = "logs/voice-notes.md"


async def log_to_github(entry: str) -> None:
    """Append a log entry to the notes log file in the GitHub repo."""
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set — skipping GitHub log")
        return

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{LOG_FILE_PATH}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    async with httpx.AsyncClient() as client:
        # Get current file content and SHA (needed to update)
        response = await client.get(api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            current_content = base64.b64decode(data["content"]).decode("utf-8")
            sha = data["sha"]
        elif response.status_code == 404:
            current_content = "# Scott's Notes Log\n\n"
            sha = None
        else:
            logger.error(f"GitHub API error fetching log: {response.status_code} {response.text}")
            return

        new_content = current_content + entry
        encoded = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")

        payload = {
            "message": f"Bot log: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "content": encoded,
        }
        if sha:
            payload["sha"] = sha

        put_response = await client.put(api_url, headers=headers, json=payload)
        if put_response.status_code not in (200, 201):
            logger.error(f"GitHub API error writing log: {put_response.status_code} {put_response.text}")


# Store conversation history per user (in memory — resets on bot restart)
conversation_histories: dict[int, list] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hey Scott — ready to go. Send me a voice note or message and let's get into it."
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id

    await update.message.reply_text("Got it, transcribing...")

    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
            await voice_file.download_to_drive(tmp_path)

        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=("voice.ogg", audio_file, "audio/ogg"),
                model="whisper-large-v3",
            )

        os.unlink(tmp_path)
        text = transcription.text

        # Show transcription so Scott can confirm it's right
        await update.message.reply_text(f"_{text}_", parse_mode="Markdown")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        await log_to_github(f"---\n\n**{timestamp} — Voice Note**\n\n{text}\n\n")

        await process_message(update, context, text, user_id)

    except Exception as e:
        logger.error(f"Voice handling error: {e}", exc_info=True)
        await update.message.reply_text(f"Voice error: {e}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    text = update.message.text
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    await log_to_github(f"---\n\n**{timestamp} — Text Note**\n\n{text}\n\n")
    await process_message(update, context, text, user_id)


async def process_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, user_id: int
) -> None:
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []

    conversation_histories[user_id].append({"role": "user", "content": text})

    # Keep last 20 messages to avoid token bloat
    if len(conversation_histories[user_id]) > 20:
        conversation_histories[user_id] = conversation_histories[user_id][-20:]

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=conversation_histories[user_id],
        )

        reply = response.content[0].text
        conversation_histories[user_id].append({"role": "assistant", "content": reply})

        await update.message.reply_text(reply)

    except Exception as e:
        logger.error(f"Claude API error: {e}", exc_info=True)
        await update.message.reply_text(f"Claude error: {e}")


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    conversation_histories[user_id] = []
    await update.message.reply_text("Conversation cleared. Fresh start.")


async def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    async with app:
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Bot is running...")
        await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
