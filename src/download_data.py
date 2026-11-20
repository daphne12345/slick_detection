"""
Collect the raw data (Sec. III of the paper):

* fino:       download the webcam images of the FINO platforms every ten minutes,
* wind:       download wind speeds measured at the lighthouse Alte Weser from pegelonline,
* spiekeroog: extract the images of the Spiekeroog beach camera and name them by their UTC time.
"""

import os
import time
import urllib.request
import datetime as dt
from datetime import datetime
from os import listdir, path
from shutil import move
from urllib.parse import urljoin, urlparse

import click
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

from src.utils import DATE_FORMAT

# new data at least every 10 minutes
SLEEP = 60.0 * 10

# For Fino2 --> slightly different website
FINO2_DIRS = [f"https://fino.go-sys.de/cam0{num}.jpg" for num in [1, 2, 7, 8, 9]]

DATE_FORMAT_CAMERA = "%Y:%m:%d %H:%M:%S%z"  # %z is the timezone-offset in the format +HHMM


def is_valid(url):
    """
    Checks whether `url` is a valid URL.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_images(url, url_short):
    """
    Returns all image URLs on a single `url`
    """
    soup = bs(requests.get(url).content, "html.parser")

    urls = []

    for img in tqdm(soup.find_all("img"), "Extracting images"):
        img_url = img.attrs.get("src")
        if not img_url:
            # if img does not contain src attribute, just skip
            continue

        # make the URL absolute by joining domain with the URL that is just extracted
        img_url = urljoin(url_short, img_url)

        try:
            pos = img_url.index("?")
            img_url = img_url[:pos]
        except ValueError:
            pass

            # finally, if the url is valid
            if is_valid(img_url):
                urls.append(img_url)
    return urls


def download(url, pathname):
    """
    Downloads a file given an URL and puts it in the folder `pathname`
    """
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))
    # get the file name
    now = datetime.utcnow()
    date_str = now.strftime(DATE_FORMAT)
    folder = url.split("/")[-1].split(".")[0]
    if "-" in folder:
        return
    if not os.path.isdir(os.path.join(pathname, folder)):
        os.mkdir(os.path.join(pathname, folder))
    filename = os.path.join(pathname, folder, date_str + ".jpg")
    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(
        response.iter_content(1024),
        f"Downloading {filename}",
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )
    with open(filename, "wb") as f:
        for data in progress.iterable:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))


@click.group()
def cli():
    pass


@cli.command()
def fino():
    """Download the images posted on the websites of the FINO-project (runs until stopped)."""
    while True:
        for fino_no in [1, 2, 3]:
            url = f"https://www.fino{fino_no}.de/de/aktuelles/webcams.html"
            url_short = f"https://www.fino{fino_no}.de/"
            save_to = f"Fino{fino_no if fino_no > 1 else ''}"

            if fino_no != 2:
                imgs = get_all_images(url, url_short)
            else:
                imgs = FINO2_DIRS
            for img in imgs:
                if "logo" in img:
                    continue
                download(img, save_to)
        time.sleep(SLEEP)


@cli.command()
@click.option("--saveto", type=str, default="WindFinder/pegelonline")
@click.option("--start", type=str, default="2022-08-01")
def wind(saveto, start):
    """Load wind speeds from pegelonline (last 30 days are usually available)."""
    # author: Jörge Schneemann
    days = pd.date_range(
        start, (dt.datetime.today() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    )
    for day in days:
        filename = saveto + "/{}_Windgeschwindigkeit_AlteWeser.txt".format(
            day.strftime("%Y%m%d")
        )
        if os.path.isfile(filename):
            print("Data already downloaded, passing {}".format(day.strftime("%d.%m.%Y")))
        else:
            try:
                url = "https://www.pegelonline.wsv.de/webservices/files/WINDGESCHWINDIGKEIT/WESER/" \
                      "LEUCHTTURM+ALTE+WESER/{}/down.txt".format(day.strftime("%d.%m.%Y"))
                urllib.request.urlretrieve(url, filename)
                print("{} saved!".format(filename))
            except Exception:
                pass


@cli.command()
@click.option("--directory", type=str, default="Spiekeroog",
              help="Directory of the images to be extracted.")
@click.option("--exiftool_path", type=str,
              default="C:/Program Files/Programme/ExifTool/exiftool.exe")
def spiekeroog(directory, exiftool_path):
    """Extract first image from the directory tree from the Spiekeroog dataset."""
    from exiftool import exiftool

    directory_new = directory + "/events"
    dirs = [
        dir_
        for dir_ in listdir(directory)
        if path.isdir(f"{directory}/{dir_}") and f"{directory}/{dir_}" != directory_new
    ]

    # this structure is specific to the given situation
    for dir_ in dirs:
        subdirs = listdir(f"{directory}/{dir_}")
        if "events" not in subdirs:
            continue
        for sub in listdir(f"{directory}/{dir_}/events"):
            if not path.isdir(f"{directory}/{dir_}/events/{sub}"):
                continue
            for subsub in listdir(f"{directory}/{dir_}/events/{sub}"):
                if not path.isdir(f"{directory}/{dir_}/events/{sub}/{subsub}"):
                    continue
                fname = f"{directory}/{dir_}/events/{sub}/{subsub}/E00000.jpg"
                if not path.exists(fname):
                    continue
                with exiftool.ExifTool(exiftool_path) as exift:
                    output = exift.get_metadata(fname)
                create_date_str = output["File:FileModifyDate"]
                # match the timezone offset (remove colon if there is one)
                create_datetime_str, create_offset_str = create_date_str.split("+")
                create_offset_str = create_offset_str.replace(":", "")
                create_date = datetime.strptime(
                    "+".join([create_datetime_str, create_offset_str]), DATE_FORMAT_CAMERA
                )
                # in the correct format and in UTC
                new_date_str = datetime.strftime(
                    create_date - create_date.utcoffset(), DATE_FORMAT
                )
                move(fname, f"{directory_new}/{new_date_str}.jpg")


if __name__ == "__main__":
    cli()
