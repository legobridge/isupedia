import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

import backoff
import openai
import requests
import wikipedia
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from environment variables)
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if TELEGRAM_BOT_TOKEN is None or OPENAI_API_KEY is None:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN and OPENAI_API_KEY must be set in environment variables."
    )

# Initialise OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TOPIC_LEN: int = 25
MAX_SUMMARY_LEN: int = 1_000
THUMB_SIZE: int = 500
OPENAI_MODEL: str = "gpt-4.1-nano-2025-04-14"

# ---------------------------------------------------------------------------
# Telegram command handlers
# ---------------------------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start handler: Greets the user."""
    assert update.message is not None
    assert update.effective_user is not None
    logger.info("/start invoked by user_id=%s", update.effective_user.id)
    await update.message.reply_text("Hello, I'm Ishanopedia II! What can I do you for?")


def get_page_summaries_concurrently(options: List[str], max_len: int = 50) -> list[Optional[str]]:
    def fetch_summary(title):
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            return trim_summary(page.summary, max_len=max_len)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_summary, title): title for title in options}
        summaries = []
        for future in as_completed(futures):
            summaries.append(future.result())
    ordered_summaries = [None] * len(options)
    for idx, title in enumerate(options):
        for future, fut_title in futures.items():
            if fut_title == title:
                try:
                    ordered_summaries[idx] = future.result()
                except Exception:
                    ordered_summaries[idx] = None
                break
    return ordered_summaries


@backoff.on_exception(
    backoff.expo,
    (AssertionError, ValueError, openai.APIError),
    max_tries=3,
)
def disambiguate_using_gpt(topic: str, options: List[str]) -> str:
    """Ask GPT‑4.1-nano to pick the best Wikipedia page title from *options*."""
    assert options, "The list of options must have at least one option"

    option_summaries = get_page_summaries_concurrently(options, max_len=200)

    options_str = "\n".join(f"{i}: {opt} - {opt_sum}" for i, (opt, opt_sum) in enumerate(zip(options, option_summaries)))
    dev_prompt = f"""
    You are an assistant that selects the single most relevant Wikipedia page title for a user query from the following list:
    {options_str} 
        
    Respond **only** with the number of the chosen option.
    """

    logger.debug("Disambiguating '%s' from options: %s", topic, options)
    oai_response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        temperature=0.1,
        instructions=dev_prompt,
        input=f"User query: {topic}",
    )
    logger.debug("OpenAI response: %s", oai_response)
    response_words = oai_response.output_text.strip().split()
    assert response_words, "OpenAI response cannot be empty"
    chosen_option = options[int(response_words[0])]
    logger.info(
        "GPT‑4/Disambiguation selected '%s' for query '%s'", chosen_option, topic
    )
    return chosen_option


def get_wiki_page(topic: str) -> Optional[wikipedia.WikipediaPage]:
    """Return a WikipediaPage, resolving disambiguation or returning *None*."""
    try:
        logger.info("Fetching Wikipedia page for '%s' (direct)", topic)
        return wikipedia.page(topic, auto_suggest=False, redirect=True)
    except wikipedia.DisambiguationError as exc:
        logger.info("DisambiguationError for '%s': %s", topic, exc.options[:10])
        title = disambiguate_using_gpt(topic, exc.options[:10])
        return wikipedia.page(title, auto_suggest=False, redirect=True)
    except wikipedia.PageError:
        logger.info("PageError for '%s'; falling back to wikipedia.search", topic)
        search_results = wikipedia.search(topic)
        if not search_results:
            logger.error("No search results for '%s'", topic)
            return None
        title = disambiguate_using_gpt(topic, search_results)
        return wikipedia.page(title, auto_suggest=False, redirect=True)


def get_thumbnail(page: wikipedia.WikipediaPage) -> Optional[str]:
    """Return the thumbnail URL for *page*, if available."""
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page.title,
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": THUMB_SIZE,
    }
    try:
        logger.debug("Requesting thumbnail for '%s'", page.title)
        resp = requests.get(api_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        logger.exception("Thumbnail request failed for '%s'", page.title)
        return None

    pages = data.get("query", {}).get("pages", {})
    for _, info in pages.items():
        thumb = info.get("thumbnail", {}).get("source")
        if thumb:
            logger.debug("Thumbnail found: %s", thumb)
            return thumb
    logger.debug("No thumbnail for '%s'", page.title)
    return None


def trim_summary(summary: str, max_len: int = MAX_SUMMARY_LEN) -> str:
    """Clip *summary* to *max_len*, ending at the last sentence boundary."""
    clipped = summary[:max_len]
    match = re.match(r"^(.*[.!?])", clipped, re.DOTALL)
    return match.group(1) if match else clipped


async def get_wiki(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/isu command handler."""
    assert update.message is not None
    assert update.effective_user is not None

    raw_text = update.message.text or "/isu"
    topic = raw_text.removeprefix("/isu").strip()

    if not topic:
        await update.message.reply_text("Say /isu <topic> and I'll go fetch.")
        return

    user_id = update.effective_user.id
    logger.info("User %s requested topic '%s'", user_id, topic)

    if len(topic) > MAX_TOPIC_LEN:
        await update.message.reply_text(
            f"Topic cannot exceed {MAX_TOPIC_LEN} characters."
        )
        return

    page = get_wiki_page(topic)
    if page is None:
        await update.message.reply_text(
            f"Could not find a Wikipedia page for '{topic}'."
        )
        return

    thumbnail_url = get_thumbnail(page)
    summary = trim_summary(page.summary)

    if thumbnail_url:
        logger.info("Sending photo response for '%s'", page.title)
        await update.message.reply_photo(thumbnail_url, caption=summary)
    else:
        logger.info("Sending text response for '%s'", page.title)
        await update.message.reply_text(summary)


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Starting Telegram bot …")
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("isu", get_wiki))
    app.run_polling()


if __name__ == "__main__":
    main()
