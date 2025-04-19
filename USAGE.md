# Usage Instructions for main.py

This script allows you to index and query PDF books using Gemini-Book-Bot.

## Required Arguments

- `pdf_path`: Path to the PDF file you want to index and query.
- `user_id`: Unique identifier for the user (e.g., username, email, or UUID).

## Optional Arguments

- `action`: Action to perform. Can be:
  - `query` (default): Query the indexed book.
  - `delete`: Delete all indexed chunks for the given book and user.
- `--list-books`: List all books already indexed for the given user.

## Examples

### Query a book (default action)

```bash
python main.py path/to/book.pdf your_user_id
```

### Delete all indexed chunks for a book

```bash
python main.py path/to/book.pdf your_user_id delete
```

### List all books indexed for a user

```bash
python main.py path/to/book.pdf your_user_id --list-books
```

## Notes

- Make sure the PDF file exists at the specified path.
- Ensure your `.env` file contains the required API keys:
  - `GEMINI_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_ENVIRONMENT`
  - `PINECONE_CLOUD`
  - `PINECONE_REGION`

## Example

```bash
python main.py SmartLibrary/book.pdf alice@example.com
```

This will index and query the book `book.pdf` for the user `alice@example.com`.
