import os
import json
import time
import argparse

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)


MODEL = "gpt-5"

system_prompt = """
You are a professional assistant specialised in ESRS law and auditing.
You are an expert in sustainability regulations and auditing practices in the European Union.
 
TASK:
Generate exactly {num_questions} clear and distinct audit questions and answers regarding the European Sustainability Reporting Standards (ESRS),
based **only** on the provided CHUNK.
 
REQUIREMENTS:
- Each question must target a single, specific compliance aspect, disclosure requirement, control, or risk explicitly stated in the CHUNK.  
- Phrase the questions so that they are answerable solely from the CHUNK, without external references.  
- Ensure variation in phrasing across the {num_questions} questions (avoid all starting the same way).
- Ensure variation in phrasing and terminology across the {num_questions} questions. Do not copy the exact wording from the CHUNK; instead, use synonyms, paraphrases, or conceptual restatements.  
- Rephrase concepts rather than copying wording directly from the CHUNK. Use synonyms or legal/compliance phrasing to avoid trivial overlap.  
- Questions must be technical and law-focused, designed to test whether the CHUNK adheres to ESRS reporting obligations.  
- Do not combine multiple sub-questions into one.  
- Do not rely on assumptions or knowledge outside the CHUNK.
- Do not use institution/organisation etc, ensure that you name the corresponding bank in every questions.
- Avoid overly narrow or program-specific details
- Phrase questions so they remain broadly applicable within the ESRS framework and the Application Requirements.
 
OUTPUT FORMAT:
- Always return a valid JSON object.
- The JSON must contain the following fields:
  {{
      "chunk": "<leave this empty>",
      "chunk_id": <leave this empty>,
      "bank_name": "<leave this empty>",
      "questions": [
          {{
              "question": "",
              "answer": "",
          }}
      ]
  }}
- Do not include any additional text, explanations, or formatting outside the JSON object.
""".strip()


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}")

    return rows


def build_user_prompt(row):
    bank_name = row.get("bank_name") or row.get("doc_name") or ""
    chunk_id = (
        row.get("retrieval_chunk_id")
        or row.get("canonical_chunk_id")
        or row.get("chunk_id")
    )

    title = row.get("title", "")
    heading = row.get("heading", "")
    text = row.get("text") or row.get("page_content") or ""
    num_questions = int(row.get("num_queries", 0) or 0)

    return f"""
BANK:
{bank_name}

CHUNK_ID:
{chunk_id}

TITLE:
{title}

HEADING:
{heading}

NUMBER_OF_QUESTIONS:
{num_questions}

CHUNK:
{text}
""".strip()


def create_batch_input(rows, batch_input_file, max_test_rows=None):
    os.makedirs(os.path.dirname(batch_input_file) or ".", exist_ok=True)

    metadata_by_custom_id = {}

    written = 0
    skipped = 0

    with open(batch_input_file, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            if not row.get("generate_queries", False):
                skipped += 1
                continue

            num_queries = int(row.get("num_queries", 0) or 0)

            if num_queries <= 0:
                skipped += 1
                continue

            bank_name = row.get("bank_name") or row.get("doc_name") or "unknown_bank"
            doc_name = row.get("doc_name") or bank_name

            chunk_id = (
                row.get("retrieval_chunk_id")
                or row.get("canonical_chunk_id")
                or row.get("chunk_id")
            )

            custom_id = f"row-{idx}"

            metadata_by_custom_id[custom_id] = {
                "doc_name": doc_name,
                "bank_name": bank_name,
                "company": bank_name,
                "chunk_id": str(chunk_id),
                "ground_truth_chunk_id": str(chunk_id),
                "source_chunk_id": str(row.get("chunk_id")),
                "title": row.get("title", ""),
                "heading": row.get("heading", ""),
                "text": row.get("text") or row.get("page_content") or "",
                "num_queries": num_queries,
            }

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt.format(num_questions=num_queries),
                        },
                        {
                            "role": "user",
                            "content": build_user_prompt(row),
                        },
                    ],
                },
            }

            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            written += 1

            if max_test_rows is not None and written >= max_test_rows:
                break

    print("=== BATCH INPUT CREATED ===")
    print(f"Rows written: {written}")
    print(f"Rows skipped: {skipped}")

    return metadata_by_custom_id


def submit_and_wait_for_batch(batch_input_file):
    input_file = client.files.create(
        file=open(batch_input_file, "rb"),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print("Batch started:", batch.id)

    while True:
        batch_status = client.batches.retrieve(batch.id)
        print("Status:", batch_status.status)

        if batch_status.status == "completed":
            return batch_status

        if batch_status.status in {"failed", "expired", "cancelled"}:
            print("Batch did not complete successfully.")
            print("Errors:", batch_status.errors)
            raise SystemExit(1)

        time.sleep(30)


def download_results(batch_status, raw_results_file):
    print("Batch status:", batch_status.status)
    print("Output file id:", batch_status.output_file_id)
    print("Error file id:", batch_status.error_file_id)

    if batch_status.output_file_id:
        result_file = client.files.content(batch_status.output_file_id)

        with open(raw_results_file, "wb") as f:
            f.write(result_file.read())

        print(f"Raw batch results saved to: {raw_results_file}")
        return raw_results_file

    if batch_status.error_file_id:
        error_file_path = raw_results_file.replace(".jsonl", "_errors.jsonl")
        error_file = client.files.content(batch_status.error_file_id)

        with open(error_file_path, "wb") as f:
            f.write(error_file.read())

        print(f"Batch had no output file. Errors saved to: {error_file_path}")
        raise ValueError(
            f"No output_file_id found. Check the batch error file: {error_file_path}"
        )

    raise ValueError("No output_file_id or error_file_id found on completed batch.")


def postprocess_results(raw_results_file, final_output, metadata_by_custom_id):
    os.makedirs(os.path.dirname(final_output) or ".", exist_ok=True)

    written = 0
    failed = 0

    with open(raw_results_file, "r", encoding="utf-8") as infile, \
         open(final_output, "w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue

            data = json.loads(line)
            custom_id = data["custom_id"]

            meta = metadata_by_custom_id.get(custom_id)

            if meta is None:
                failed += 1
                print(f"Missing metadata for {custom_id}")
                continue

            try:
                response_body = data["response"]["body"]
                content = response_body["choices"][0]["message"]["content"]
                result_json = json.loads(content)

                usage = response_body.get("usage", {}) or {}

                input_tokens = int(usage.get("prompt_tokens", 0) or 0)
                output_tokens = int(usage.get("completion_tokens", 0) or 0)
                total_tokens = int(usage.get("total_tokens", 0) or 0)

            except Exception as e:
                failed += 1
                print(f"Could not parse result for {custom_id}: {e}")
                continue

            questions = result_json.get("questions", [])

            for i, qa in enumerate(questions, start=1):
                record = {
                    "financebench_id": None,
                    "query_id": f"{meta['bank_name']}_{meta['chunk_id']}_q{i}",
                    "company": meta.get("company"),
                    "doc_name": meta.get("doc_name"),
                    "bank_name": meta["bank_name"],

                    "query": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "ground_truth_chunk_id": meta["ground_truth_chunk_id"],
                    "source_chunk_id": meta["source_chunk_id"],

                    "chunk_title": meta["title"],
                    "chunk_heading": meta["heading"],
                    "chunk": meta["text"],

                    "query_generation_model": MODEL,
                    "query_generation_method": "openai_batch",
                    "query_generation_input_tokens": input_tokens,
                    "query_generation_output_tokens": output_tokens,
                    "query_generation_total_tokens": total_tokens,
                }

                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print("=== POSTPROCESSING COMPLETE ===")
    print(f"Final queries written: {written}")
    print(f"Failed rows: {failed}")
    print(f"Final output: {final_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to chunk JSONL file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to final validation queries JSONL"
    )

    parser.add_argument(
        "--workdir",
        default="batch_work",
        help="Folder for temporary batch input/results files"
    )

    parser.add_argument(
        "--model",
        default=MODEL,
        help="OpenAI model to use"
    )

    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Maximum number of chunk rows to send to the batch API for testing"
    )

    args = parser.parse_args()

    MODEL = args.model

    os.makedirs(args.workdir, exist_ok=True)

    batch_input_file = os.path.join(args.workdir, "batch_input.jsonl")
    raw_results_file = os.path.join(args.workdir, "batch_results.jsonl")

    rows = load_jsonl(args.input)

    metadata_by_custom_id = create_batch_input(
        rows=rows,
        batch_input_file=batch_input_file,
        max_test_rows=args.max_test_rows,
    )

    batch_status = submit_and_wait_for_batch(batch_input_file)

    download_results(
        batch_status=batch_status,
        raw_results_file=raw_results_file,
    )

    postprocess_results(
        raw_results_file=raw_results_file,
        final_output=args.output,
        metadata_by_custom_id=metadata_by_custom_id,
    )