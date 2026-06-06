#!/usr/bin/env python3

import argparse
import base64
import json
import os
import sys
import anthropic

def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Load an image file and return base64-encoded data and media type."""
    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext)
    if not media_type:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    return image_data, media_type


def parse_image_to_json(schema_path: str, image_path: str, output_path: str, model: str):
    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        raise EnvironmentError("CLAUDE_API_KEY environment variable is not set.")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    image_data, media_type = load_image_as_base64(image_path)

    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "Please parse the data from the attached image into the following JSON schema. "
        "Return only valid JSON with no additional text, markdown formatting, or code blocks.\n\n"
        f"JSON Schema:\n{json.dumps(schema, indent=2)}"
    )

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    response_text = message.content[0].text.strip()

    # Strip markdown code fences if present
    if response_text.startswith("```"):
        lines = response_text.splitlines()
        response_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    parsed = json.loads(response_text)

    with open(output_path, "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"Successfully wrote parsed JSON to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse an image into JSON using the Anthropic API."
    )
    parser.add_argument("schema", help="Path to the JSON schema file")
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("output", help="Path to the output JSON file")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model to use (default: claude-sonnet-4-6)",
    )

    args = parser.parse_args()

    try:
        parse_image_to_json(args.schema, args.image, args.output, args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

