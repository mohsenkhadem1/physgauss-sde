"""Download licensed PokeFlex files from ETH into the local, git-ignored data directory."""

import argparse
import getpass
import shutil
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

ENDPOINT = "https://pokeflex.ait.ethz.ch/download.php"
ROOT = Path(__file__).resolve().parents[2] / "runs/third_party/PokeFlex-data"
METADATA = {"LICENSE.txt", "readme.md"}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


class Listing(HTMLParser):
    def __init__(self):
        super().__init__()
        self.children = set()

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href")
        if tag != "a" or not href:
            return
        url = urlsplit(urljoin(ENDPOINT, href))
        if url._replace(query="", fragment="").geturl() != ENDPOINT:
            return
        query = parse_qs(url.query)
        for key in ("dir", "file"):
            for value in query.get(key, []):
                path = PurePosixPath(value.lstrip("/"))
                if ".." not in path.parts and "\\" not in value and ":" not in value:
                    self.children.add((key, path))


def readable_zip(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.infolist())
    except zipfile.BadZipFile:
        return False


def download(open_url, token, scope, root):
    def fetch(params):
        return open_url(ENDPOINT + "?" + urlencode({"dt": token, **params}), timeout=90)

    def walk(folder):
        params = {"dir": "/" + folder.as_posix()} if folder.parts else {}
        with fetch(params) as response:
            listing = Listing()
            listing.feed(response.read().decode("utf-8"))
        # Filter breadcrumbs and external links before constructing local paths.
        children = sorted(((k, p) for k, p in listing.children if p.parts and p.parent == folder),
                          key=lambda entry: (entry[0] != "file", entry[1]))
        if not children:
            raise ValueError("No dataset entries found. Check that the ETH access link is valid.")
        matched = False
        for kind, path in children:
            in_scope = scope == path or scope in path.parents
            if kind == "dir" and (in_scope or path in scope.parents):
                matched |= walk(path)
            if kind != "file" or not (path.as_posix() in METADATA or
                                      (in_scope and path.suffix.lower() == ".zip")):
                continue
            dest = root / path
            archival = path.suffix.lower() == ".zip"
            if in_scope and archival:
                matched = True
            if dest.is_file() and (readable_zip(dest) if archival else dest.stat().st_size > 0):
                print(f"Already downloaded {path}", flush=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name + ".part")
            print(f"Downloading {path}", flush=True)
            with fetch({"file": "/" + path.as_posix()}) as response:
                if response.headers.get_content_type() == "text/html":
                    raise ValueError("ETH returned an HTML page in place of a file.")
                length = response.headers.get("Content-Length")
                with partial.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            size = partial.stat().st_size
            if not size or (length is not None and size != int(length)):
                raise ValueError(f"Incomplete download for {path}. Run again to retry.")
            if archival and not readable_zip(partial):
                raise ValueError(f"Invalid ZIP for {path}. Run again to retry.")
            partial.replace(dest)
        return matched

    if not walk(PurePosixPath(".")):
        raise ValueError("No ZIP archives matched the requested dataset path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", nargs="?", default=".", help="e.g. poking/FoamDice or a single ZIP path")
    parser.add_argument("--output", type=Path, default=ROOT, help="local storage directory")
    args = parser.parse_args()
    scope = PurePosixPath(args.scope)
    if scope.is_absolute() or ".." in scope.parts:
        parser.error("Use a relative dataset path.")
    if not sys.stdin.isatty():
        parser.error("Use an interactive terminal for the private access link.")
    link = getpass.getpass("Paste your personal ETH PokeFlex download URL (hidden): ")
    url = urlsplit(link.strip())
    tokens = parse_qs(url.query).get("dt", [])
    if url._replace(query="", fragment="").geturl() != ENDPOINT or len(tokens) != 1:
        parser.error("Expected an ETH PokeFlex download URL containing one dt token.")
    try:
        download(build_opener(NoRedirect()).open, tokens[0], scope, args.output)
    except HTTPError as exc:
        sys.exit(f"ETH download failed (HTTP {exc.code}). Check your link and try again.")
    except URLError:
        sys.exit("Could not reach ETH. Check connectivity and try again.")
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
