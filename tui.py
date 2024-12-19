import os
import argparse
from dotenv import load_dotenv

from rich.console import Console

from main import Agent

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Understand your files."
    )
    parser.add_argument(
        "--file", type=str, required=True, help="A file to understand."
    )
    parser.add_argument(
        "--query", type=str, required=False, default=None, help="A question about the file."
    )
    parser.add_argument(
        "--summarise", action="store_true", help="Summarise the file."
    )
    console = Console()

    with console.status("Getting file context..."):
        args = parser.parse_args()
        agent = Agent(os.getenv("COHERE_API_KEY"))
        if args.summarise:
            response = agent.summarise(args.file)
        else:
            response = agent.rag(args.file, args.query)

        console.print(response)

if __name__ == "__main__":
    main()
