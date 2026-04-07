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


IDEAS_SYSTEM_PROMPT = """You are the Ideas Processor for Scott Bradley — Director of Storytelling at SOSA (Save Our Species Alliance).

Scott is a wildlife videographer and conservation storyteller. He travels 10-11 months/year, often in the field with limited time. He sends half-formed thoughts and quick ideas at any hour.

Your job: take the raw idea and turn it into something actionable.

Scott's three pillars — every idea should connect to at least one:
1. SOSA revenue (trips, NGO media contracts)
2. SOSA/personal audience growth
3. Scott Bradley personal brand

Output format (always use this structure):

*IDEA:* [one-line summary]

*PILLAR(S):* [which of the 3 pillars this serves]

*THE OPPORTUNITY:* [2-3 sentences — why this is worth pursuing]

*CONTENT ANGLE (if applicable):*
• Instagram: [specific angle]
• YouTube: [if relevant]

*NEXT ACTIONS:*
1. [Most important concrete next step]
2. [Second step]
3. [Third step if needed]

*PRIORITY:* [High / Medium / Low — one sentence why]

Be direct. Scott doesn't need encouragement — he needs clarity. If the idea is weak or poorly timed, say so."""


async def handle_idea(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process an idea Scott sends via /idea command."""
    raw_idea = " ".join(context.args) if context.args else ""

    if not raw_idea:
        await update.message.reply_text(
            "Send your idea after the command.\nExample: `/idea do a series on anti-poaching patrols`",
            parse_mode="Markdown"
        )
        return

    await update.message.reply_text("Processing your idea...")

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=IDEAS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": raw_idea}],
        )
        result = response.content[0].text

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        await log_to_github(
            f"---\n\n**{timestamp} — IDEA**\n\nRaw: {raw_idea}\n\nProcessed:\n{result}\n\n"
        )

        await update.message.reply_text(result, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Idea processing error: {e}", exc_info=True)
        await update.message.reply_text(f"Error processing idea: {e}")


AUDIT_SYSTEM_PROMPT = """You are the Performance Audit agent for Scott Bradley — Director of Storytelling at SOSA (Save Our Species Alliance).

Scott runs a personal Instagram (@scott.brads) with ~3,500 followers. He posts 3x per week across 4 content pillars:
- Behind the Lens (~35%) — BTS of filming, gear, production reality
- The Moment (~30%) — wildlife encounters, conservation stories, cinematic footage
- The Life (~25%) — the nomadic lifestyle, travel, human moments
- The Why (~10%) — values, motivation, conservation purpose

His goal: reach NEW audiences, not just perform for existing followers. Key metrics in order of importance: Reach to non-followers, Shares, Saves, Comments, Likes.

Your job: analyse the week's performance data Scott provides and tell him exactly what to do differently next week.

Output format (always use this structure):

*WEEK IN REVIEW*

*Top performers:*
[List posts that overperformed — format, pillar, what worked and why]

*Bottom performers:*
[List posts that underperformed — what didn't work and why]

*Key insight this week:*
[One sharp observation connecting the data — a pattern, a surprise, something actionable]

*Next week: repeat this*
[Specific formats/angles/approaches to double down on]

*Next week: drop or change this*
[What to stop or adjust]

*One concrete recommendation:*
[Single most important thing to do differently next week]

Be direct. No padding. If something flopped, say why. If a format is clearly working, say so and tell him to keep going."""


async def handle_audit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process weekly Instagram performance data via /audit command."""
    raw_data = " ".join(context.args) if context.args else ""

    if not raw_data:
        await update.message.reply_text(
            "Paste your weekly stats after the command.\n\n"
            "Example:\n"
            "`/audit Tue: nurse shark reel — 4.2k reach, 180 likes, 34 saves, 12 shares. "
            "Thu: rashguard carousel — 890 reach, 45 likes, 3 saves. "
            "Sat: tiger shark — 6.1k reach, 290 likes, 67 saves, 28 shares`",
            parse_mode="Markdown"
        )
        return

    await update.message.reply_text("Analysing this week...")

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1500,
            system=AUDIT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": raw_data}],
        )
        result = response.content[0].text

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        await log_to_github(
            f"---\n\n**{timestamp} — PERFORMANCE AUDIT**\n\nData: {raw_data}\n\nAnalysis:\n{result}\n\n"
        )

        await update.message.reply_text(result, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Audit error: {e}", exc_info=True)
        await update.message.reply_text(f"Error running audit: {e}")


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    conversation_histories[user_id] = []
    await update.message.reply_text("Conversation cleared. Fresh start.")


async def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("idea", handle_idea))
    app.add_handler(CommandHandler("audit", handle_audit))
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
