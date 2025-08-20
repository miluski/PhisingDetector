import re
from email.utils import parseaddr

class URLExtractor:
    @staticmethod
    def extract_url_from_message(sender, content, receiver=None, message_type='email', print_extracted=True):
        if not content:
            content = ""
        if not sender:
            sender = ""
        
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        
        extracted_url = None
        
        if urls:
            extracted_url = urls[0]
            if print_extracted:
                print(f"URL extracted from message content: {extracted_url}")
        elif message_type == 'email' and '@' in sender:
            addr = parseaddr(sender)[1] if parseaddr(sender)[1] else sender
            if '@' in addr:
                domain = addr.split('@', 1)[1]
                extracted_url = f"http://{domain}"
                if print_extracted:
                    print(f"URL derived from sender email domain: {extracted_url}")
                    print(f"Sender email: {sender}")
        
        if print_extracted and not extracted_url:
            if message_type == 'sms':
                print(f"No URL found in SMS content")
            else:
                print(f"No URL found in content or sender email: {sender}")
            
        return extracted_url

    @staticmethod
    def extract_both_urls_for_email(sender, content, receiver=None, print_extracted=True):
        url_data = {
            'content_url': None,
            'sender_domain_url': None
        }
        
        content_url = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        if content_url:
            url_data['content_url'] = content_url[0]
            if print_extracted:
                print(f"URL found in content: {url_data['content_url']}")
        
        if '@' in sender:
            addr = parseaddr(sender)[1] if parseaddr(sender)[1] else sender
            if '@' in addr:
                domain = addr.split('@', 1)[1]
                url_data['sender_domain_url'] = f"http://{domain}"
                if print_extracted:
                    print(f"Sender email: {sender}")
                    print(f"Sender domain URL: {url_data['sender_domain_url']}")
        
        if url_data['content_url'] and url_data['sender_domain_url']:
            if print_extracted:
                print("Both content URL and sender domain available for analysis")
        
        return url_data
