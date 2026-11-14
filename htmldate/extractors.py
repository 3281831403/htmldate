# pylint:disable-msg=E0611,I1101
"""
Custom parsers and XPath expressions for date extraction
"""

import logging
import re

from collections.abc import Iterator
from datetime import date, datetime, time
from functools import lru_cache
from itertools import islice
from typing import TYPE_CHECKING

from dateutil.parser import parse as dateutil_parse

from lxml.etree import XPath
from lxml.html import HtmlElement

# own
from .settings import CACHE_SIZE, MAX_SEGMENT_LEN
from .utils import Extractor, remove_if_attached, trim_text
from .validators import correct_year, parse_ymd, validate, validate_ymd

if TYPE_CHECKING:  # pragma: no cover
    from dateparser import DateDataParser  # type: ignore[attr-defined]

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_external_parser(today: date | None = None) -> "DateDataParser":
    "Lazy import, unused in fast mode, dates preferred from before the end of the day."
    from dateparser import DateDataParser  # type: ignore[attr-defined]

    today = today or date.today()
    return DateDataParser(
        languages=None,
        locales=None,
        region=None,
        settings={
            "NORMALIZE": True,  # False may be faster
            "PARSERS": [
                "custom-formats",
                "absolute-time",
            ],
            "PREFER_DATES_FROM": "past",
            "PREFER_LOCALE_DATE_ORDER": True,
            "RELATIVE_BASE": datetime.combine(today, time.max),
            "RETURN_AS_TIMEZONE_AWARE": False,
            "STRICT_PARSING": True,
        },
    )


FAST_TAGS = ("div", "h2", "h3", "h4", "li", "p", "span", "time", "ul")
# further tests needed: b, em, font, i, strong
FAST_PREPEND = ".//*[" + " or ".join(f"self::{t}" for t in FAST_TAGS) + "]"
FREE_TEXT_EXPRESSIONS = XPath(FAST_PREPEND + "/text()")

# discard parts of the webpage
# archive.org banner inserts, navigation menus outside footers (dates of other pages),
# site headers (today's date, recent stories) but not the header of the page's own post
DISCARD_EXPRESSIONS = XPath(
    './/div[@id="wm-ipp-base" or @id="wm-ipp"]|.//nav[not(ancestor::footer)]'
    "|.//header[not(ancestor::article or ancestor::main or ancestor::footer)][not(.//h1)]"
)
# not discarded for consistency (see above):
# .//footer
# .//*[(self::div or self::section)][@id="footer" or @class="footer"]

DAY_RE = "[0-3]?[0-9]"
MONTH_RE = "[0-1]?[0-9]"
# keep the (?:...): interpolated bare, the "|" would split the enclosing group
YEAR_RE = "(?:199[0-9]|20[0-3][0-9])"

# missing parts must not come from today
DATEUTIL_DEFAULT = datetime(1, 1, 1)

# day component of an ISO-like or numeric date expression (before any time part)
DAY_PART_PATTERN = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")


def _has_day_part(string: str) -> bool:
    "Whether the expression states a day, instead of leaving it to default to 1."
    return bool(DAY_PART_PATTERN.match(string))

# regex cache
YMD_NO_SEP_PATTERN = re.compile(r"\b(\d{8})\b")
YMD_PATTERN = re.compile(
    rf"(?:\D|^)(?:(?P<year>{YEAR_RE})[\-/.](?P<month>{MONTH_RE})[\-/.](?P<day>{DAY_RE})|"
    rf"(?P<day2>{DAY_RE})[\-/.](?P<month2>{MONTH_RE})[\-/.](?P<year2>\d{{2,4}}))(?:\D|$)"
)
YM_PATTERN = re.compile(
    rf"(?:\D|^)(?:(?P<year>{YEAR_RE})[\-/.](?P<month>{MONTH_RE})|"
    rf"(?P<month2>{MONTH_RE})[\-/.](?P<year2>{YEAR_RE}))(?:\D|$)"
)
# Chinese/Japanese 年月日 and Korean 년월일, e.g. 2025年10月12日 and 2025년 10월 12일;
# the day suffix is optional since trailing non-digits are cut from element text
CJK_DAY_END = r"(?:\s?[日일]|(?!\d))"
CJK_YMD_PATTERN = re.compile(
    rf"(?:\D|^)(?P<year>{YEAR_RE})\s?[年년]\s?(?P<month>{MONTH_RE})\s?[月월]\s?"
    rf"(?P<day>{DAY_RE}){CJK_DAY_END}"
)

# any word from its start, MONTH_NUMBERS decides
MONTH_WORD = r"(?<![^\W\d_])[^\W\d_]{3,}"
LONG_TEXT_PATTERN = re.compile(
    rf"""(?P<month>{MONTH_WORD})\s
(?P<day>{DAY_RE})(?:st|nd|rd|th)?,? (?P<year>{YEAR_RE})|
(?P<day2>{DAY_RE})(?:st|nd|rd|th|\.)? (?:of )?
(?P<month2>{MONTH_WORD})[,.]? (?P<year2>{YEAR_RE})""".replace("\n", ""),
    re.I,
)

COMPLETE_URL = re.compile(rf"\D({YEAR_RE})[/_-]({MONTH_RE})[/_-]({DAY_RE})(?:\D|$)")

# optional ISO-8601 time-of-day and time zone suffix (e.g. "T08:37:00+05:30")
TIME_TZ_RE = r"[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:?[0-9]{2})?"
# groups: date, time ("?" appended makes the time optional)
YMD_TIME_RE = rf"({YEAR_RE}-{MONTH_RE}-{DAY_RE})({TIME_TZ_RE})"

JSON_MODIFIED = re.compile(rf'"dateModified": ?"{YMD_TIME_RE}?', re.I)
JSON_PUBLISHED = re.compile(rf'"datePublished": ?"{YMD_TIME_RE}?', re.I)

# English, French, German, Indonesian, Polish, Russian and Turkish month names
MONTHS = [
    (
        "jan",
        "januar",
        "jänner",
        "january",
        "januari",
        "janvier",
        "ocak",
        "oca",
        "stycznia",
        "styczeń",
        "январь",
        "января",
    ),
    (
        "feb",
        "februar",
        "feber",
        "february",
        "februari",
        "février",
        "şubat",
        "şub",
        "lutego",
        "luty",
        "февраль",
        "февраля",
    ),
    (
        "mar",
        "mär",
        "märz",
        "march",
        "maret",
        "mart",
        "mars",
        "marca",
        "marzec",
        "март",
        "марта",
    ),
    (
        "apr",
        "april",
        "avril",
        "nisan",
        "nis",
        "kwietnia",
        "kwiecień",
        "апрель",
        "апреля",
    ),
    ("may", "mai", "mei", "mayıs", "maja", "maj", "май", "мая"),
    (
        "jun",
        "juni",
        "june",
        "juin",
        "haziran",
        "haz",
        "czerwca",
        "czerwiec",
        "июнь",
        "июня",
    ),
    (
        "jul",
        "juli",
        "july",
        "juillet",
        "temmuz",
        "tem",
        "lipca",
        "lipiec",
        "июль",
        "июля",
    ),
    (
        "aug",
        "august",
        "agustus",
        "ağustos",
        "ağu",
        "aout",
        "sierpnia",
        "sierpień",
        "август",
        "августа",
    ),
    (
        "sep",
        "september",
        "septembre",
        "eylül",
        "eyl",
        "września",
        "wrzesnia",
        "wrzesień",
        "сентябрь",
        "сентября",
    ),
    (
        "oct",
        "oktober",
        "october",
        "octobre",
        "okt",
        "ekim",
        "eki",
        "października",
        "pazdziernika",
        "październik",
        "октябрь",
        "октября",
    ),
    (
        "nov",
        "november",
        "kasım",
        "kas",
        "novembre",
        "listopada",
        "listopad",
        "ноябрь",
        "ноября",
    ),
    (
        "dec",
        "dez",
        "dezember",
        "december",
        "desember",
        "décembre",
        "aralık",
        "ara",
        "grudnia",
        "grudzień",
        "декабрь",
        "декабря",
    ),
]


def _fold_month_name(token: str) -> str:
    "str.lower() alone mishandles the Turkish dotless i."
    return token.replace("ı", "i").replace("İ", "i").lower()


MONTH_NUMBERS = {
    _fold_month_name(name): i for i, names in enumerate(MONTHS, 1) for name in names
}


# gate for try_date_expr
TEXT_DATE_PATTERN = re.compile(r"[.:,_/ -]")
YEAR_TOKEN = re.compile(r"(?<!\d)\d{4}(?!\d)")
DAY_TOKEN = re.compile(r"(?<!\d)\d{1,2}(?!\d)")
# shorter entries ("de", "I") are common words
WORD = re.compile(r"[^\W\d_]{3,}")
MONTH_KEYS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


@lru_cache(maxsize=None)
def month_words() -> frozenset[str]:
    "Month names in all dateparser locales, plus MONTH_NUMBERS keys."
    from dateparser.languages.loader import default_loader

    return frozenset(MONTH_NUMBERS) | frozenset(
        _fold_month_name(word)
        for locale in default_loader.get_locales()
        for key in MONTH_KEYS
        for word in locale.info.get(key, [])
        if WORD.fullmatch(word)
    )


def has_date_cue(string: str, min_date: datetime, max_date: datetime) -> bool:
    "The first 4-digit year decides, else any dateparser month name."
    match = YEAR_TOKEN.search(string)
    if match is not None:
        return min_date.year <= int(match[0]) <= max_date.year
    return any(_fold_month_name(word) in month_words() for word in WORD.findall(string))


DISCARD_PATTERNS = re.compile(
    r"^\d{2}:\d{2}(?: |:|$)|"
    r"^\D*\d{4}\D*$|"
    r"[$€¥Ұ£¢₽₱฿#₹]|"  # currency symbols and special characters
    r"[A-Z]{3}[^A-Z]|"  # currency codes
    r"(?:^|\D)(?:\+\d{2}|\d{3}|\d{5})\D|"  # tel./IPs/postal codes
    r"ftps?|https?|sftp|"  # protocols
    r"\.(?:com|net|org|info|gov|edu|de|fr|io)\b|"  # TLDs
    r"IBAN|[A-Z]{2}[0-9]{2}|"  # bank accounts
    r"®"  # ©
)

# use of regex module for speed?
# numeric core shared by all TEXT_PATTERNS alternatives, used as prefilter
DATE_CORE_PATTERN = re.compile(r"[0-9]{1,4}[./][0-9]{1,2}[./][0-9]{2,4}")

# gap-runs bounded to {0,9} (ReDoS + keeps matches within the prefilter windows)
TEXT_PATTERNS = re.compile(
    r'(?:date[^0-9"]{,20}|updated|last-modified|published|posted|on)[ :]{0,9}?([0-9]{1,4})[./]([0-9]{1,2})[./]([0-9]{2,4})|'  # EN
    r"(?:Datum|Stand|Veröffentlicht am):? ?([0-9]{1,2})\.([0-9]{1,2})\.([0-9]{2,4})|"  # DE
    r"(?:güncellen?me|yayı(?:m|n)lan?ma) {0,9}?(?:tarihi)? {0,9}?:? {0,9}?([0-9]{1,2})[./]([0-9]{1,2})[./]([0-9]{2,4})|"
    r"([0-9]{1,2})[./]([0-9]{1,2})[./]([0-9]{2,4}) {0,9}?(?:'de|'da|'te|'ta|’de|’da|’te|’ta|tarihinde) {0,9}(?:güncellendi|yayı(?:m|n)landı)",  # TR
    re.I,
)


def discard_unwanted(tree: HtmlElement) -> HtmlElement:
    """Delete unwanted sections of an HTML document."""
    for subtree in DISCARD_EXPRESSIONS(tree):
        remove_if_attached(subtree)
    return tree


def extract_url_date(
    testurl: str | None,
    options: Extractor,
) -> datetime | None:
    """Extract the date out of an URL string complying with the Y-M-D format"""
    if testurl is not None:
        match = COMPLETE_URL.search(testurl)
        if match:
            LOGGER.debug("found date in URL: %s", match[0])
            try:
                dateobject = datetime(int(match[1]), int(match[2]), int(match[3]))
                result = validate(dateobject, options.min, options.max)
                if result is None:
                    options.state.full_date_rejected = True
                else:
                    return result
            except ValueError as err:
                LOGGER.debug("conversion error: %s %s", match[0], err)
    return None


def try_swap_values(day: int, month: int) -> tuple[int, int]:
    """Swap day and month values if it seems feasible."""
    return (month, day) if month > 12 and day <= 12 else (day, month)


def _build_dmy(day: int, month: int, year: int) -> datetime:
    "Build a datetime from day/month/year, fixing 2-digit years and day/month order."
    year = correct_year(year)
    day, month = try_swap_values(day, month)
    return datetime(year, month, day)


def regex_parse(string: str) -> datetime | None:
    """Try full-text parse for date elements using a series of regular expressions
    with particular emphasis on the languages in MONTHS"""
    # https://github.com/vi3k6i5/flashtext ?
    # multilingual day-month-year + American English patterns
    pos = 0
    while match := LONG_TEXT_PATTERN.search(string, pos):
        word = _fold_month_name(match["month"] or match["month2"])
        # month-first: glued words ("SmithMarch"), longest known suffix
        # capped tail: slicing a whole long word is quadratic
        if match["month"]:
            word = word[-12:]
            word = next(
                (word[i:] for i in range(len(word) - 2) if word[i:] in MONTH_NUMBERS),
                word,
            )
        month = MONTH_NUMBERS.get(word)
        if month is None:
            # matches can overlap
            pos = match.start() + 1
            continue
        # process and return
        try:
            dateobject = _build_dmy(
                int(match["day"] or match["day2"]),
                month,
                int(match["year"] or match["year2"]),
            )
        except ValueError:
            pos = match.end()
            continue
        LOGGER.debug("multilingual text found: %s", dateobject)
        return dateobject
    return None


def _parse_yyyymmdd(digits: str) -> datetime:
    "Build a datetime from a leading 8-digit YYYYMMDD run."
    return datetime(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))


# Python 3.10: fromisoformat reads "+00:00" but not "Z" or "+0000"
ISO_OFFSET_PATTERN = re.compile(r"(:\d\d(?:\.\d+)?)(?:Z|([+-]\d\d)(\d\d))$")


def _iso_offset(match: re.Match[str]) -> str:
    return f"{match[1]}{match[2]}:{match[3]}" if match[2] else f"{match[1]}+00:00"


def _date_candidates(string: str) -> Iterator[tuple[datetime | None, bool]]:
    "Yield (datetime candidate, full date?) pairs for custom_parse, in priority order."
    # 1. shortcut
    if string[:4].isdigit():
        candidate = None
        # a. '201709011234' not covered by dateparser, and regex too slow
        if string[4:8].isdigit():
            try:
                candidate = _parse_yyyymmdd(string)
            except ValueError:
                LOGGER.debug("8-digit error: %s", string[:8])
        # b. much faster than extensive parsing
        else:
            try:
                candidate = datetime.fromisoformat(
                    ISO_OFFSET_PATTERN.sub(_iso_offset, string)
                )
            except ValueError:
                LOGGER.debug("not an ISO date string: %s", string)
                try:
                    candidate = dateutil_parse(string, default=DATEUTIL_DEFAULT)
                    # "T17-54-24" gives an offset of -54 hours, which datetime refuses
                    candidate.utcoffset()
                except (OverflowError, TypeError, ValueError):
                    LOGGER.debug("dateutil parsing error: %s", string)
                    candidate = None
        # dateutil/fromisoformat only fill missing parts with the 1s of
        # DATEUTIL_DEFAULT, so the expression's own precision is read from the
        # string rather than from the reconstructed datetime (e.g. "2018-01")
        yield candidate, candidate is not None and _has_day_part(string)

    # 2. Try YYYYMMDD, use regex
    match = YMD_NO_SEP_PATTERN.search(string)
    if match:
        try:
            yield _parse_yyyymmdd(match[1]), True
        except ValueError:
            LOGGER.debug("YYYYMMDD value error: %s", match[0])

    # 3. Try the very common YMD, Y-M-D, and D-M-Y patterns
    match = YMD_PATTERN.search(string)
    if match:
        try:
            if match.lastgroup == "day":
                candidate = datetime(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                )
            else:
                candidate = _build_dmy(
                    int(match.group("day2")),
                    int(match.group("month2")),
                    int(match.group("year2")),
                )
        except ValueError:
            # impossible calendar dates (e.g. Feb. 30) are not full dates:
            # they must keep falling through to year-level patterns
            LOGGER.debug("regex value error: %s", match[0])
        else:
            yield candidate, True

    # 4. Try the CJK year-month-day pattern
    cjk_match = CJK_YMD_PATTERN.search(string)
    if cjk_match:
        try:
            yield (
                datetime(
                    int(cjk_match.group("year")),
                    int(cjk_match.group("month")),
                    int(cjk_match.group("day")),
                ),
                True,
            )
        except ValueError:
            LOGGER.debug("CJK regex value error: %s", cjk_match[0])

    # 5. Try the Y-M and M-Y patterns, not as a fallback for a full date
    if not (match or cjk_match) and (match := YM_PATTERN.search(string)):
        try:
            year = match.group("year") or match.group("year2")
            month = match.group("month") or match.group("month2")
            yield datetime(int(year), int(month), 1), False
        except ValueError:
            LOGGER.debug("Y-M value error: %s", match[0])

    # 6. Try the other regex pattern: None covers both "no match" and
    # impossible calendar dates such as Feb. 30, neither of which is a full
    # date turned down by the window
    word_date = regex_parse(string)
    if word_date is not None:
        yield word_date, True


def parse_with_window(
    string: str, min_date: datetime, max_date: datetime
) -> tuple[datetime | None, bool]:
    """Parse a string and validate it against the date window.

    Full dates (year, month and day) take precedence over partials: an
    in-window full date is returned even if an in-window partial appears first,
    and an out-of-window full date makes the call report a rejection instead
    of falling back to a coarser expression. Partials are only used when the
    string offers no usable full date at all. The second return value is True
    when a full date was rejected for lying outside the window."""
    LOGGER.debug("custom parse test: %s", string)
    partial_fallback: datetime | None = None
    for candidate, is_full in _date_candidates(string):
        if candidate is None:
            continue
        if not is_full:
            # keep the first (highest-priority) partial, but keep scanning:
            # a full date later in the same string must win
            if partial_fallback is None:
                partial_fallback = validate(candidate, min_date, max_date)
            continue
        result = validate(candidate, min_date, max_date)
        if result is not None:
            return result, False
        # a parseable full date outside the window blocks lower granularity
        return None, True
    return partial_fallback, False


def custom_parse(
    string: str, min_date: datetime, max_date: datetime
) -> datetime | None:
    """Try to bypass the slow dateparser"""
    return parse_with_window(string, min_date, max_date)[0]


def external_date_parser(string: str) -> datetime | None:
    "Parse the string with dateparser."
    LOGGER.debug("send to external parser: %s", string)
    try:
        return get_external_parser(date.today()).get_date_data(string)["date_obj"]
    # 2 types of errors possible
    except (OverflowError, ValueError) as err:  # pragma: no cover
        LOGGER.error("external parser error: %s %s", string, err)
        return None


@lru_cache(maxsize=CACHE_SIZE)
def _try_date_expr_cached(
    string: str | None,
    extensive_search: bool,
    min_date: datetime,
    max_date: datetime,
) -> tuple[datetime | None, bool, bool]:
    """Use a series of heuristics and rules to parse a potential date expression.

    Return the parsed datetime, whether a full date was turned down by the
    date window, and whether an accepted result has full (day) granularity."""
    if not string:
        return None, False, False

    # trim
    string = trim_text(string)[:MAX_SEGMENT_LEN]

    # formal constraint: 4 to 18 digits
    if not string or not 4 <= sum(map(str.isdigit, string)) <= 18:
        return None, False, False

    # check if string only contains time/single year or digits and not a date
    if DISCARD_PATTERNS.search(string):
        return None, False, False

    # try to parse using the faster method
    customresult, rejected = parse_with_window(string, min_date, max_date)
    if customresult is not None:
        return customresult, False, _has_day_part(string)

    # use slow but extensive search
    # only plausible dates reach the slow external parser
    if (
        extensive_search
        and TEXT_DATE_PATTERN.search(string)
        and DAY_TOKEN.search(string)
        and has_date_cue(string, min_date, max_date)
    ):
        # the external parser yields None for unparseable/impossible dates,
        # which must keep falling through to year-level patterns; preserve a
        # rejection already established by the faster parser for the same string
        external = external_date_parser(string)
        if external is None:
            return None, rejected, False
        # a day token was required to reach this parser, so it is a full date
        if validate(external, min_date, max_date) is not None:
            return external, False, True
        return None, True, False

    return None, rejected, False


def try_date_expr(
    string: str | None,
    extensive_search: bool,
    min_date: datetime,
    max_date: datetime,
) -> datetime | None:
    "Backward-compatible wrapper around the cached implementation."
    return _try_date_expr_cached(string, extensive_search, min_date, max_date)[0]


# keep the cache management API on the public wrapper (see htmldate.meta)
try_date_expr.cache_clear = _try_date_expr_cached.cache_clear  # type: ignore[attr-defined]
try_date_expr.cache_info = _try_date_expr_cached.cache_info  # type: ignore[attr-defined]


def try_date_expr_opts(string: str | None, options: Extractor) -> datetime | None:
    "Uncached wrapper: the cache key leaves out original_date and the window state."
    result, rejected, _ = _try_date_expr_cached(
        string, options.extensive, options.min, options.max
    )
    if rejected:
        options.state.full_date_rejected = True
    return result


def try_date_expr_tagged(
    string: str | None, options: Extractor
) -> tuple[datetime | None, bool]:
    """Parse one date expression with its granularity.

    Return the accepted datetime and whether it is a partial (month/year-only)
    expression. A full date turned down by the window is recorded on the
    document state so callers can suppress every partial instead of
    downgrading precision, regardless of where the expressions occur."""
    result, rejected, is_full = _try_date_expr_cached(
        string, options.extensive, options.min, options.max
    )
    if rejected:
        options.state.full_date_rejected = True
    return result, result is not None and not is_full


def img_search(
    tree: HtmlElement,
    options: Extractor,
) -> datetime | None:
    """Skim through image elements"""
    element = tree.find('.//meta[@property="og:image"][@content]')
    if element is not None:
        return extract_url_date(
            element.get("content"),
            options,
        )
    return None


def pattern_search(
    text: str,
    date_pattern: re.Pattern[str],
    options: Extractor,
) -> datetime | None:
    "Look for date expressions using a regular expression on a string of text."
    match = date_pattern.search(text)
    if not match:
        return None
    date = validate_ymd(match[1], options.min, options.max)
    if date is None:
        # a parseable full Y-M-D key outside the window must not resurface at
        # lower granularity (e.g. as day 1 of the same month) later in the
        # cascade; impossible calendar dates (Feb. 30) still fall through
        candidate = parse_ymd(match[1])
        if candidate is not None:
            options.state.full_date_rejected = True
        return None
    LOGGER.debug("regex found: %s %s", date_pattern, match[0])
    # group 2: optional time of day and time zone
    if match[2]:
        return custom_parse(match[1] + match[2], options.min, options.max) or date
    return date


def json_search(
    tree: HtmlElement,
    options: Extractor,
) -> datetime | None:
    """Look for JSON time patterns in JSON sections of the tree"""
    # determine pattern
    json_pattern = JSON_PUBLISHED if options.original else JSON_MODIFIED
    # look throughout the HTML tree
    for elem in tree.xpath(
        './/script[@type="application/ld+json" or @type="application/settings+json"]'
    ):
        if not elem.text or '"date' not in elem.text:
            continue
        result = pattern_search(elem.text, json_pattern, options)
        if result is not None:
            return result
    return None


def idiosyncrasies_search(
    htmlstring: str,
    options: Extractor,
) -> datetime | None:
    """Look for author-written dates throughout the web page"""
    # probe ±60-char windows around numeric cores (enough: gap-runs are bounded),
    # then re-search unbounded so the result equals a full slow scan
    match = None
    cores = DATE_CORE_PATTERN.finditer(htmlstring)
    for core in islice(cores, 1000):
        start = max(0, core.start() - 60)
        if TEXT_PATTERNS.search(htmlstring, start, core.end() + 60):  # EN+DE+TR
            match = TEXT_PATTERNS.search(htmlstring, start)
            break
    else:
        if next(cores, None) is not None:  # cap hit on a date-dense document
            match = TEXT_PATTERNS.search(htmlstring)
    if match:
        parts = list(filter(None, match.groups()))

        try:
            if len(parts[0]) == 4:  # year in first position
                candidate = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
            else:  # len(parts[2]) in (2, 4):  # DD/MM/YY
                candidate = _build_dmy(int(parts[0]), int(parts[1]), int(parts[2]))
        except (IndexError, ValueError):
            LOGGER.debug("cannot process idiosyncrasies: %s", match[0])
        else:
            result = validate(candidate, options.min, options.max)
            if result is not None:
                return result
            options.state.full_date_rejected = True

    return None
