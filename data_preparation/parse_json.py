import argparse
import csv
import gzip
import json
import logging
from typing import Iterator

# Configure logging for debugging and tracking progress.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

# Setting up a console handler for the logger.
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def parse(path: str) -> Iterator[dict]:
    """
    Generator function to parse a gzipped JSON file.
    Yields each JSON object as a dictionary.

    Args:
        path (str): Path to the gzipped JSON file.

    Yields:
        Iterator[dict]: Iterator of dictionaries parsed from JSON objects.
    """
    with gzip.open(path, 'rb') as g:
        for l in g:
            try:
                # Try to parse each line as a JSON object.
                yield json.loads(l)
            except json.JSONDecodeError:
                # Log an error if JSON parsing fails and skip to the next line.
                logger.error("Invalid JSON encountered")
                continue


def parse_json_to_csv(read_path: str, write_path: str) -> None:
    """
    Function to parse a gzipped JSON file and write its contents to a CSV file.

    Args:
        read_path (str): Path to the gzipped JSON input file.
        write_path (str): Path to the output CSV file.
    """
    with open(write_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = None
        for i, d in enumerate(parse(read_path)):
            if csv_writer is None:
                # Initialize the CSV writer with the headers from the first JSON object.
                header = d.keys()
                csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                csv_writer.writeheader()

            # Convert all string values in the dictionary to lowercase.
            d = {k: v.lower() if isinstance(v, str) else v for k, v in d.items()}
            csv_writer.writerow(d)

            if i % 50000 == 0:  # Log progress every 50,000 rows.
                logger.info(f'Rows processed: {i:,}')

    # Log completion and the output file path.
    logger.info(f'Csv saved to {write_path}')


if __name__ == '__main__':
    # Command line interface setup.
    parser = argparse.ArgumentParser(description='Parsing json (gzipped) to csv')
    parser.add_argument('read_path', type=str, help='Path to input gzipped json')
    parser.add_argument('write_path', type=str, help='Path to output csv')
    args = parser.parse_args()

    # Call the function with the provided CLI arguments.
    parse_json_to_csv(args.read_path, args.write_path)
