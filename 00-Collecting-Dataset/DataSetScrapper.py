import aiohttp
import asyncio
import random
import time
import re
from tqdm import tqdm

semaphore = asyncio.Semaphore(50)
import sys
instance_id = int(sys.argv[1])
start_id = int(sys.argv[2])
end_id = int(sys.argv[3])

async def fetch_json(session, url):
    async with semaphore:
        try:
            async with session.get(url, timeout=15) as response:
                if response.status == 200:
                    return await response.json()
        except:
            return None

async def get_book_info_by_isbn_async(session, isbn):
    cleaned = re.sub(r'[- ]', '', isbn.strip())
    if not cleaned or len(cleaned) not in [10, 13]:
        return None

    ol_url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{cleaned}&format=json&jscmd=data"
    gb_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{cleaned}"

    async def fetch(url):
        try:
            async with semaphore:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
        except:
            return None

    ol_data, gb_data = await asyncio.gather(fetch(ol_url), fetch(gb_url))

    if ol_data and f"ISBN:{cleaned}" in ol_data:
        book = ol_data[f"ISBN:{cleaned}"]
        title = book.get("title")
        author = ", ".join([a["name"] for a in book.get("authors", [])]) if book.get("authors") else None
        publisher = ", ".join([p["name"] for p in book.get("publishers", [])]) if book.get("publishers") else None
        dewey = book.get("classifications", {}).get("dewey_decimal_class")
        dewey_code = dewey[0] if isinstance(dewey, list) else dewey
        publish_date = book.get("publish_date")
        year_match = re.search(r"\b\d{4}\b", publish_date or "")
        publish_year = year_match.group(0) if year_match else publish_date
        return f"{title} | {dewey_code or 'Sin Dewey'} | {publish_year or 'Sin año'}"

    if gb_data and gb_data.get("totalItems", 0) > 0 and "items" in gb_data:
        info = gb_data["items"][0].get("volumeInfo", {})
        title = info.get("title")
        author = ", ".join(info.get("authors", [])) if info.get("authors") else None
        publisher = info.get("publisher")
        published_date = info.get("publishedDate")
        year_match = re.search(r"^\d{4}", published_date or "")
        publish_year = year_match.group(0) if year_match else published_date
        return f"{title} | Sin Dewey | {publish_year or 'Sin año'}"

    return None

async def get_title_dewey_abstract(session, book_id):
    work_url = f"https://openlibrary.org/works/OL{book_id}W.json"
    work_data = await fetch_json(session, work_url)
    if not work_data or not work_data.get("title"):
        return None

    title = work_data["title"]
    description = work_data.get("description")
    abstract = description.get("value") if isinstance(description, dict) else description if isinstance(description, str) else None

    editions_url = f"https://openlibrary.org/works/OL{book_id}W/editions.json?limit=1"
    editions_data = await fetch_json(session, editions_url)
    if not editions_data or not editions_data.get("entries"):
        return None

    edition = editions_data["entries"][0]
    dewey_code = edition.get("dewey_decimal_class", [None])[0]
    if dewey_code and dewey_code != "Sin Dewey":
        return f"{title} | {dewey_code} | {abstract or 'Sin resumen'}"

    # Fallback por ISBN si no hay Dewey
    isbn_list = edition.get("isbn_13") or edition.get("isbn_10") or []
    for isbn in isbn_list:
        fallback = await get_book_info_by_isbn_async(session, isbn)
        if fallback:
            return fallback

    return None

async def main():
    books = []
    attempts = 0
    batch_size = 100
    progress_bar = tqdm(total=1000, desc="📚 Libros válidos")

    async with aiohttp.ClientSession() as session:
        while len(books) < 100:
            batch_ids = random.sample(range(start_id, end_id), batch_size)

            tasks = [get_title_dewey_abstract(session, book_id) for book_id in batch_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_results = [r for r in results if isinstance(r, str)]
            books.extend(valid_results)
            progress_bar.update(len(valid_results))
            attempts += batch_size

            print(f"🔄 Batch: {len(valid_results)} válidos | Total: {len(books)} | Intentos acumulados: {attempts}")
            await asyncio.sleep(2)

    with open(f"titulos_dewey_resumen_{instance_id}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(books))

    print("✅ Archivo 'titulos_dewey_resumen.txt' creado con 1000 libros válidos.")

if __name__ == "__main__":
    asyncio.run(main())
