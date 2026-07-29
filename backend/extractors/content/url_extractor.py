"""Extract URLs from message text, email addresses and attachments."""

import logging
import re
from email.utils import parseaddr
from io import BytesIO
from typing import List, Set

import docx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger("URLExtractor")


class URLExtractor:
    """Extract validated HTTP(S) URLs while rejecting common false positives."""
    VALID_TLDS = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'pl', 'de', 'uk', 'fr',
        'ru', 'cn', 'jp', 'br', 'au', 'in', 'it', 'es', 'nl', 'se', 'no', 'fi', 'dk',
        'ch', 'at', 'be', 'pt', 'cz', 'hu', 'ro', 'bg', 'gr', 'tr', 'ua', 'info', 'biz',
        'online', 'site', 'xyz', 'app', 'dev', 'tech', 'cloud', 'shop', 'store', 'blog',
        'news', 'email', 'link', 'click', 'support', 'eu', 'ca', 'mx', 'ar', 'cl', 'nz', 'sg', 'hk',
        'tw', 'kr', 'th', 'vn', 'id', 'my', 'ph', 'za', 'eg', 'ng', 'ke', 'ie', 'sk',
        'lt', 'lv', 'ee', 'hr', 'si', 'rs', 'me', 'al', 'mk', 'ba', 'is', 'lu', 'mt',
        'cy', 'by', 'md', 'am', 'ge', 'az', 'kz', 'uz', 'ae', 'il', 'sa', 'qa', 'kw',
        'pro', 'mobi', 'tv', 'cc', 'ws', 'fm'
    }

    BLACKLISTED_DOMAINS = {
        'asp.net', 'iplatforma.net', 'vb.net', 'ado.net', 'wpf.net',
        'linq.net', 'wcf.net', 'mvc.net', 'xamarin.net'
    }

    TRUNCATED_SUBDOMAINS = {
        'odle', 'dle', 'gle', 'oogle', 'le',
        'oft', 'rosoft', 'icrosoft',
        'cle', 'acle', 'racle',
        'pedia', 'ikipedia', 'kipedia',
        'trum', 'ectrum',
        'obe', 'iobe',
        'ob', 'ub', 'oo',
        'pl', 'de', 'uk', 'fr', 'ru',
    }

    KNOWN_SHORT_DOMAINS = {
        'go.com', 'ge.com', 'hp.com', 'lg.com', 'at.com', 'bt.com',
        'bb.com', 'td.com', 'vw.com', 'bbc.com', 'cnn.com', 'nbc.com',
        'abc.com', 'cbs.com', 'ibm.com', 'ups.com', 'dhl.com', 'bmw.com',
    }

    MIN_SLD_LENGTH = 3
    MIN_SUBDOMAIN_LENGTH = 3

    IP_PATTERN = re.compile(
        r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$'
    )
    TLD_PATTERN = re.compile(r'^[a-z]{2,63}$', re.IGNORECASE)

    def __init__(self):
        """Compile URL, IP URL and bare-domain recognition patterns."""
        self.url_pattern = self._compile_url_pattern()
        self.ip_url_pattern = self._compile_ip_url_pattern()
        self.bare_domain_pattern = self._compile_bare_domain_pattern()

    @staticmethod
    def _compile_url_pattern():
        """Build the explicit hostname URL pattern."""
        return re.compile(
            r'https?://'
            r'(?:www\.)?'
            r'[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?'
            r'(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)*'
            r'\.[a-zA-Z]{2,}'
            r'(?::\d{1,5})?'
            r'(?:/[^\s\\\[\]<>\"\']*)?',
            re.IGNORECASE
        )

    @staticmethod
    def _compile_ip_url_pattern():
        """Build the explicit IPv4 URL pattern."""
        return re.compile(
            r'https?://'
            r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
            r'(?::\d{1,5})?'
            r'(?:/[^\s\\\[\]<>\"\']*)?',
            re.IGNORECASE
        )

    @staticmethod
    def _compile_bare_domain_pattern():
        """Build the bare-domain pattern used in message text."""
        return re.compile(
            r'(?<![A-Za-z0-9@/.-])'
            r'(?:www\.)?'
            r'([a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)'
            r'((?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)*)'
            r'\.([a-zA-Z]{2,})\b'
        )

    def extract_from_text(self, text: str) -> Set[str]:
        """Extract validated URLs, bare domains and IP-based URLs from text."""
        if not text:
            return set()
        text = self._clean_text(text)
        return self._extract_explicit_urls(text) | self._extract_bare_domains(text)

    def _extract_explicit_urls(self, text: str) -> Set[str]:
        """Extract explicit HTTP URLs and IP URLs."""
        urls = {
            self._clean_url(match.group(0))
            for match in self.url_pattern.finditer(text)
            if self._is_valid_url(self._clean_url(match.group(0)))
        }
        urls.update(self._clean_url(match.group(0)) for match in self.ip_url_pattern.finditer(text))
        return urls

    def _extract_bare_domains(self, text: str) -> Set[str]:
        """Extract bare domains while rejecting email addresses and mixed-case TLDs."""
        urls = set()
        for match in self.bare_domain_pattern.finditer(text):
            if not self._valid_bare_domain_match(text, match):
                continue
            candidate = f"https://{match.group(0).lower()}"
            if self._is_valid_url(candidate):
                urls.add(candidate)
        return urls

    def _valid_bare_domain_match(self, text: str, match) -> bool:
        """Validate a regex match before URL normalization."""
        tld = match.group(3)
        if not self.TLD_PATTERN.fullmatch(tld):
            return False
        if tld != tld.lower() and tld != tld.upper():
            return False
        return match.start() == 0 or text[match.start() - 1] != "@"

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\\(?:r\\n|n|r|t)', ' ', text)
        return re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)

    def _clean_url(self, url: str) -> str:
        while url and url[-1] in '.,;:!?)>\'"':
            url = url[:-1]

        while url.count('(') < url.count(')') and url.endswith(')'):
            url = url[:-1]

        while url.count('[') < url.count(']') and url.endswith(']'):
            url = url[:-1]

        url = re.sub(r'\[\d+\]$', '', url)
        url = re.sub(r'(\.(html?|php|aspx?|jsp|pdf|txt))([A-Z][a-zA-Z]+)$',
                     r'\1', url, flags=re.IGNORECASE)
        url = re.sub(r'(/[a-z_\-]+)([A-Z][a-z]+)$', r'\1', url)
        url = re.sub(r'(#[\w\-]+)([A-Z][a-z]+.*)$', r'\1', url)
        return url.rstrip('-')

    def _is_valid_url(self, url: str) -> bool:
        """Return whether a URL has a supported and plausible domain."""
        try:
            domain = self._domain_from_url(url)
            return self._validate_domain(domain)
        except Exception:
            return False

    def _domain_from_url(self, url: str) -> str:
        """Extract a lower-case host without path, port or query."""
        value = url.lower()
        if value.startswith("https://"):
            value = value[8:]
        elif value.startswith("http://"):
            value = value[7:]
        else:
            return ""
        value = value[4:] if value.startswith("www.") else value
        return value.split("/")[0].split(":")[0].split("?")[0].split("#")[0]

    def _validate_domain(self, domain: str) -> bool:
        """Validate base domain rules and all domain labels."""
        if domain in self.BLACKLISTED_DOMAINS or "." not in domain or len(domain) < 4:
            return False
        if self.IP_PATTERN.match(domain):
            return True
        parts = domain.split(".")
        if not self._valid_tld_and_sld(parts):
            return False
        return all(self._valid_domain_label(parts, index) for index in range(len(parts) - 1))

    def _valid_tld_and_sld(self, parts: list[str]) -> bool:
        """Validate TLD, second-level name and short-domain exceptions."""
        tld, sld = parts[-1], parts[-2]
        if not self.TLD_PATTERN.fullmatch(tld) or not self._looks_like_real_word(sld):
            return False
        blocked = {"asp", "vb", "ado", "wpf", "linq", "wcf", "mvc", "iplatforma"}
        if tld == "net" and sld in blocked:
            return False
        return len(sld) >= self.MIN_SLD_LENGTH or ".".join(parts[-2:]) in self.KNOWN_SHORT_DOMAINS

    def _valid_domain_label(self, parts: list[str], index: int) -> bool:
        """Validate one subdomain label."""
        if not self._is_valid_domain_part(parts[index]):
            return False
        return index != 0 or len(parts) <= 2 or self._is_valid_subdomain(parts[index])

    def _looks_like_real_word(self, word: str) -> bool:
        """Reject improbable consonant-only or truncated domain labels."""
        if len(word) < 2:
            return False

        if word.isdigit():
            return True
        vowels = set('aeiou')
        vowel_count = sum(1 for c in word.lower() if c in vowels)
        if vowel_count == 0:
            return False
        return self._max_consonant_run(word) <= 4

    @staticmethod
    def _max_consonant_run(word: str) -> int:
        """Return the longest consecutive consonant run in a label."""
        consonants = set('bcdfghjklmnpqrstvwxyz')
        longest = current = 0
        for character in word.lower():
            current = current + 1 if character in consonants else 0
            longest = max(longest, current)
        return longest

    def _is_valid_domain_part(self, part: str) -> bool:
        if len(part) < 2:
            return False
        return bool(re.match(r'^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$', part, re.IGNORECASE))

    def _is_valid_subdomain(self, subdomain: str) -> bool:
        subdomain_lower = subdomain.lower()
        if subdomain_lower in self.TRUNCATED_SUBDOMAINS:
            return False
        if len(subdomain) < self.MIN_SUBDOMAIN_LENGTH:
            return False
        if subdomain.isdigit():
            return True
        if not self._looks_like_real_word(subdomain):
            return False

        vowels = set('aeiou')
        vowel_count = sum(1 for c in subdomain_lower if c in vowels)
        return not (len(subdomain) >= 5 and vowel_count < 2)

    def extract_from_email_address(self, email: str) -> Set[str]:
        """Convert a valid email domain into an HTTPS URL candidate."""
        urls = set()
        if not email:
            return urls

        _, addr = parseaddr(email)
        if '@' in addr:
            test_url = f"https://{addr.split('@')[1]}"
            if self._is_valid_url(test_url):
                urls.add(test_url)

        return urls

    def extract_from_pdf(self, file_content: bytes) -> Set[str]:
        """Extract HTTP links from PDF annotations."""
        urls = set()
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                urls.update(self._extract_urls_from_pdf_annotations(page))
        except (OSError, ValueError, PdfReadError) as exc:
            logger.debug("PDF URL extraction failed: %s", exc)
        return urls

    def _extract_urls_from_pdf_annotations(self, page) -> Set[str]:
        """Extract URI actions from one PDF page annotation collection."""
        urls = set()
        try:
            if '/Annots' not in page:
                return urls
            annotations = page['/Annots']
            if not annotations:
                return urls
            for annot_ref in annotations:
                try:
                    annot = annot_ref.get_object()
                    uri = self._annotation_uri(annot)
                    if uri:
                        urls.add(uri)
                except Exception:
                    continue
        except (AttributeError, KeyError, TypeError) as exc:
            logger.debug("PDF annotation extraction failed: %s", exc)
        return urls

    @staticmethod
    def _annotation_uri(annotation):
        """Return an HTTP URI from a PDF link annotation, if present."""
        if annotation.get('/Subtype') != '/Link':
            return None
        action = annotation.get('/A')
        uri = action.get('/URI') if action and '/URI' in action else None
        return uri if isinstance(uri, str) and uri.startswith('http') else None

    def extract_from_docx(self, file_content: bytes) -> Set[str]:
        """Extract hyperlink relationships from a DOCX attachment."""
        urls = set()
        try:
            doc_file = BytesIO(file_content)
            doc = docx.Document(doc_file)
            for rel in doc.part.rels.values():
                if "hyperlink" in rel.reltype:
                    target = rel._target
                    if target and isinstance(target, str) and target.startswith('http'):
                        urls.add(target)
        except (OSError, ValueError, KeyError) as exc:
            logger.debug("DOCX URL extraction failed: %s", exc)
        return urls

    def extract_from_attachment(self, file_content: bytes, filename: str) -> Set[str]:
        """Dispatch attachment extraction by filename extension."""
        urls = set()
        filename_lower = filename.lower()

        if filename_lower.endswith('.pdf'):
            urls.update(self.extract_from_pdf(file_content))
        elif filename_lower.endswith(('.docx', '.doc')):
            urls.update(self.extract_from_docx(file_content))
        elif filename_lower.endswith('.txt'):
            try:
                urls.update(self.extract_from_text(file_content.decode('utf-8')))
            except UnicodeDecodeError as exc:
                logger.debug("Text attachment decoding failed: %s", exc)

        return urls

    def extract_all_urls(
        self,
        sender: str = None,
        receiver: str = None,
        content: str = None,
        attachments: List[dict] = None
    ) -> List[str]:
        """Collect unique URLs from sender, receiver, content and attachments."""
        all_urls = set()

        all_urls.update(self.extract_from_email_address(sender) if sender else set())
        all_urls.update(self.extract_from_email_address(receiver) if receiver else set())
        all_urls.update(self.extract_from_text(content) if content else set())
        all_urls.update(self._attachment_urls(attachments))

        return list(all_urls)

    def _attachment_urls(self, attachments):
        """Extract URLs from all attachments with content bytes."""
        urls = set()
        for attachment in attachments or []:
            if attachment.get('content'):
                urls.update(self.extract_from_attachment(attachment['content'], attachment.get('filename', '')))
        return urls
