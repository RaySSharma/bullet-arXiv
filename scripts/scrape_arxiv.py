"""Scrape the arXiv using the abstract scraper from Mahdi Sadjadi (2017)
arxivscraper: Zenodo. http://doi.org/10.5281/zenodo.889853

Usage: python scrape_arxiv.py physics:astro-ph 2012-09-01 2022-09-01 data/arxiv_abstracts.csv
"""

import argparse
import pathlib

import arxivscraper
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument(
        "category", type=str, help="Single arXiv category tag (e.g, physics:astro-ph)",
    )
    parser.add_argument(
        "date_from", type=str, help="Starting date for scrape in format YYYY-MM-DD",
    )
    parser.add_argument(
        "date_until", type=str, help="Ending date for scrape in format YYYY-MM-DD",
    )
    parser.add_argument("outfile", type=pathlib.Path, help="Path to output CSV file")
    args = parser.parse_args()
    return args


def scrape(category, date_from, date_until):
    """Scrape data using arxivscraper between dates `date_from` and `date_until`, in category `category`.

    Args:
        category (str): ArXiv category tag.
        date_from (str): Date in YYYY-MM-DD format.
        date_until (str): Date in YYYY-MM-DD format.

    Returns:
        dict: Scraped data dictionary.
    """
    scraper = arxivscraper.Scraper(
        category=category, date_from=date_from, date_until=date_until
    )
    output = scraper.scrape()
    return output


def package(output, columns):
    """Generated DataFrame from scraped data.

    Args:
        output (dict): Scraped data from arxivscraper.
        columns (list): List of columns to keep from arxivscraper output.

    Returns:
        pandas.DataFrame: Packaged DataFrame.
    """
    df = pd.DataFrame(output, columns=columns)
    return df


def save(df, outfile, **kwargs):
    df.to_csv(outfile, index=False, **kwargs)


def run(category, date_from, date_until, outfile):
    output = scrape(category, date_from, date_until)
    df = package(
        output,
        columns=("id", "title", "categories", "abstract", "doi", "created", "authors"),
    )
    save(df, outfile)
    return df


if __name__ == "__main__":
    args = parse_args()
    category = args.category
    date_from = args.date_from
    date_until = args.date_until
    outfile = args.outfile
    run(category, date_from, date_until, outfile)
