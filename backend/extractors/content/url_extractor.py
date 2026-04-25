import re
from typing import List, Set
from email.utils import parseaddr
import PyPDF2
import docx
from io import BytesIO


class URLExtractor:
    VALID_TLDS = {
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'pl', 'de', 'uk', 'fr',
        'ru', 'cn', 'jp', 'br', 'au', 'in', 'it', 'es', 'nl', 'se', 'no', 'fi', 'dk',
        'ch', 'at', 'be', 'pt', 'cz', 'hu', 'ro', 'bg', 'gr', 'tr', 'ua', 'info', 'biz',
        'online', 'site', 'xyz', 'app', 'dev', 'tech', 'cloud', 'shop', 'store', 'blog',
        'news', 'email', 'link', 'click', 'eu', 'ca', 'mx', 'ar', 'cl', 'nz', 'sg', 'hk',
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

    def __init__(self):
        self.url_pattern = re.compile(
            r'https?://'
            r'(?:www\.)?'
            r'[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?'
            r'(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)*'
            r'\.[a-zA-Z]{2,}'
            r'(?::\d{1,5})?'
            r'(?:/[^\s\[\]<>\"\']*)?',
            re.IGNORECASE
        )

        self.ip_url_pattern = re.compile(
            r'https?://'
            r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)'
            r'(?::\d{1,5})?'
            r'(?:/[^\s\[\]<>\"\']*)?',
            re.IGNORECASE
        )

        self.bare_domain_pattern = re.compile(
            r'\b(?<![/@])'
            r'(?:www\.)?'
            r'([a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)'
            r'((?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)*)'
            r'\.([a-zA-Z]{2,})\b'
        )

    def extract_from_text(self, text: str) -> Set[str]:
        if not text:
            return set()

        text = self._clean_text(text)
        urls = set()

        for match in self.url_pattern.finditer(text):
            url = match.group(0)
            url = self._clean_url(url)
            if self._is_valid_url(url):
                urls.add(url)

        for match in self.ip_url_pattern.finditer(text):
            url = match.group(0)
            url = self._clean_url(url)
            urls.add(url)

        for match in self.bare_domain_pattern.finditer(text):
            full_match = match.group(0)
            tld = match.group(3).lower()

            if tld not in self.VALID_TLDS:
                continue

            tld_original = match.group(3)
            if tld_original != tld_original.lower() and tld_original != tld_original.upper():
                continue

            start_pos = match.start()
            if start_pos > 0 and text[start_pos - 1] == '@':
                continue

            full_url = f"https://{full_match.lower()}"
            if self._is_valid_url(full_url):
                urls.add(full_url)

        return urls

    def _clean_text(self, text: str) -> str:
        return re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)

    def _clean_url(self, url: str) -> str:
        while url and url[-1] in '.,;:!?)>\'"':
            url = url[:-1]

        while url.count('(') < url.count(')') and url.endswith(')'):
            url = url[:-1]

        while url.count('[') < url.count(']') and url.endswith(']'):
            url = url[:-1]

        url = re.sub(r'\[\d+\]$', '', url)
        url = re.sub(r'(\.(html?|php|aspx?|jsp|pdf|txt))([A-Z][a-zA-Z]+)$', r'\1', url, flags=re.IGNORECASE)
        url = re.sub(r'(/[a-z_\-]+)([A-Z][a-z]+)$', r'\1', url)
        url = re.sub(r'(#[\w\-]+)([A-Z][a-z]+.*)$', r'\1', url)
        url = url.rstrip('-')

        return url

    def _is_valid_url(self, url: str) -> bool:
        try:
            url_lower = url.lower()

            if url_lower.startswith('https://'):
                domain_part = url[8:]
            elif url_lower.startswith('http://'):
                domain_part = url[7:]
            else:
                return False

            if domain_part.lower().startswith('www.'):
                domain_part = domain_part[4:]

            domain = domain_part.split('/')[0].split(':')[0].split('?')[0].split('#')[0]
            domain_lower = domain.lower()

            if domain_lower in self.BLACKLISTED_DOMAINS:
                return False

            if '.' not in domain:
                return False

            if len(domain) < 4:
                return False

            if self.IP_PATTERN.match(domain_lower):
                return True

            parts = domain_lower.split('.')

            tld = parts[-1]
            if tld not in self.VALID_TLDS:
                return False

            sld = parts[-2] if len(parts) >= 2 else ''

            if sld.lower() in {'asp', 'vb', 'ado', 'wpf', 'linq', 'wcf', 'mvc', 'iplatforma'}:
                if tld == 'net':
                    return False

            full_domain = '.'.join(parts[-2:])
            if len(sld) < self.MIN_SLD_LENGTH:
                if full_domain not in self.KNOWN_SHORT_DOMAINS:
                    return False

            if not self._looks_like_real_word(sld):
                return False

            for i, part in enumerate(parts[:-1]):
                if not self._is_valid_domain_part(part):
                    return False

                if i == 0 and len(parts) > 2:
                    if not self._is_valid_subdomain(part):
                        return False

            return True

        except Exception:
            return False

    def _looks_like_real_word(self, word: str) -> bool:
        if len(word) < 2:
            return False

        if word.isdigit():
            return True

        vowels = set('aeiou')
        vowel_count = sum(1 for c in word.lower() if c in vowels)

        if vowel_count == 0:
            return False

        consonants = set('bcdfghjklmnpqrstvwxyz')
        max_consonant_run = 0
        current_run = 0
        for c in word.lower():
            if c in consonants:
                current_run += 1
                max_consonant_run = max(max_consonant_run, current_run)
            else:
                current_run = 0

        if max_consonant_run > 4:
            return False

        return True

    def _is_valid_domain_part(self, part: str) -> bool:
        if len(part) < 2:
            return False

        if not re.match(r'^[a-z0-9]([a-z0-9\-]*[a-z0-9])?$', part, re.IGNORECASE):
            return False

        return True

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

        if len(subdomain) >= 5 and vowel_count < 2:
            return False

        return True

    def extract_from_email_address(self, email: str) -> Set[str]:
        urls = set()
        if not email:
            return urls

        _, addr = parseaddr(email)
        if '@' in addr:
            domain = addr.split('@')[1]
            test_url = f"https://{domain}"
            if self._is_valid_url(test_url):
                urls.add(test_url)

        return urls

    def extract_from_pdf(self, file_content: bytes) -> Set[str]:
        urls = set()
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            for page in pdf_reader.pages:
                urls.update(self._extract_urls_from_pdf_annotations(page))

        except Exception:
            pass

        return urls

    def _extract_urls_from_pdf_annotations(self, page) -> Set[str]:
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
                    if annot.get('/Subtype') != '/Link':
                        continue

                    action = annot.get('/A')
                    if action and '/URI' in action:
                        uri = action['/URI']
                        if uri and isinstance(uri, str) and uri.startswith('http'):
                            urls.add(uri)
                except Exception:
                    continue

        except Exception:
            pass

        return urls

    def extract_from_docx(self, file_content: bytes) -> Set[str]:
        urls = set()
        try:
            doc_file = BytesIO(file_content)
            doc = docx.Document(doc_file)

            for rel in doc.part.rels.values():
                if "hyperlink" in rel.reltype:
                    target = rel._target
                    if target and isinstance(target, str) and target.startswith('http'):
                        urls.add(target)

        except Exception:
            pass

        return urls

    def extract_from_attachment(self, file_content: bytes, filename: str) -> Set[str]:
        urls = set()
        filename_lower = filename.lower()

        if filename_lower.endswith('.pdf'):
            urls.update(self.extract_from_pdf(file_content))
        elif filename_lower.endswith(('.docx', '.doc')):
            urls.update(self.extract_from_docx(file_content))
        elif filename_lower.endswith('.txt'):
            try:
                text = file_content.decode('utf-8')
                urls.update(self.extract_from_text(text))
            except Exception:
                pass

        return urls

    def extract_all_urls(
        self,
        sender: str = None,
        receiver: str = None,
        content: str = None,
        attachments: List[dict] = None
    ) -> List[str]:
        all_urls = set()

        if sender:
            all_urls.update(self.extract_from_email_address(sender))

        if receiver:
            all_urls.update(self.extract_from_email_address(receiver))

        if content:
            all_urls.update(self.extract_from_text(content))

        if attachments:
            for attachment in attachments:
                file_content = attachment.get('content')
                filename = attachment.get('filename', '')

                if file_content:
                    all_urls.update(
                        self.extract_from_attachment(file_content, filename)
                    )

        return list(all_urls)
