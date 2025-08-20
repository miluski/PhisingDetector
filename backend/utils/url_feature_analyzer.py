import re
from urllib.parse import urlparse
import requests

class URLFeatureAnalyzer:
    LEGITIMATE_TLDS = [
        'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'co.uk', 'de', 'fr', 'jp', 'cn'
    ]
    
    SUSPICIOUS_TLDS = [
        'tk', 'ml', 'ga', 'cf', 'pw', 'cc', 'ws', 'biz', 'info', 
        'click', 'download', 'loan', 'racing', 'review', 'science',
        'stream', 'trade', 'webcam', 'win', 'xyz', 'top'
    ]

    @staticmethod
    def get_feature_names():
        return [
            'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
            'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
            'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
            'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
            'qty_percent_url', 'qty_tld_url', 'length_url', 'qty_dot_domain',
            'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain',
            'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain',
            'qty_and_domain', 'qty_exclamation_domain', 'qty_space_domain',
            'qty_tilde_domain', 'qty_comma_domain', 'qty_plus_domain',
            'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
            'qty_percent_domain', 'qty_vowels_domain', 'domain_length',
            'domain_in_ip', 'server_client_domain', 'qty_dot_directory',
            'qty_hyphen_directory', 'qty_underline_directory', 'qty_slash_directory',
            'qty_questionmark_directory', 'qty_equal_directory', 'qty_at_directory',
            'qty_and_directory', 'qty_exclamation_directory', 'qty_space_directory',
            'qty_tilde_directory', 'qty_comma_directory', 'qty_plus_directory',
            'qty_asterisk_directory', 'qty_hashtag_directory', 'qty_dollar_directory',
            'qty_percent_directory', 'directory_length', 'qty_dot_file',
            'qty_hyphen_file', 'qty_underline_file', 'qty_slash_file',
            'qty_questionmark_file', 'qty_equal_file', 'qty_at_file',
            'qty_and_file', 'qty_exclamation_file', 'qty_space_file',
            'qty_tilde_file', 'qty_comma_file', 'qty_plus_file',
            'qty_asterisk_file', 'qty_hashtag_file', 'qty_dollar_file',
            'qty_percent_file', 'file_length', 'qty_dot_params',
            'qty_hyphen_params', 'qty_underline_params', 'qty_slash_params',
            'qty_questionmark_params', 'qty_equal_params', 'qty_at_params',
            'qty_and_params', 'qty_exclamation_params', 'qty_space_params',
            'qty_tilde_params', 'qty_comma_params', 'qty_plus_params',
            'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params',
            'qty_percent_params', 'params_length', 'tld_present_params',
            'qty_params', 'email_in_url', 'tls_ssl_certificate', 'qty_redirects',
            'url_shortened'
        ]

    @staticmethod
    def analyze_url_features(url):
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            path = parsed_url.path
            query = parsed_url.query
            fragment = parsed_url.fragment
            
            file_part = path.split('/')[-1] if path else ""
            directory_part = '/'.join(path.split('/')[:-1]) if path else ""
            
            tls_ssl_certificate = 1 if url.startswith('https') else 0
            
            qty_redirects = 0
            try:
                response = requests.get(url, timeout=2, allow_redirects=True, 
                                      headers={'User-Agent': 'Mozilla/5.0'})
                qty_redirects = len(response.history)
            except:
                qty_redirects = 0
            
            def count_chars_in_string(s):
                return {
                    'dot': s.count('.'),
                    'hyphen': s.count('-'),
                    'underline': s.count('_'),
                    'slash': s.count('/'),
                    'questionmark': s.count('?'),
                    'equal': s.count('='),
                    'at': s.count('@'),
                    'and': s.count('&'),
                    'exclamation': s.count('!'),
                    'space': s.count(' '),
                    'tilde': s.count('~'),
                    'comma': s.count(','),
                    'plus': s.count('+'),
                    'asterisk': s.count('*'),
                    'hashtag': s.count('#'),
                    'dollar': s.count('$'),
                    'percent': s.count('%')
                }
            
            url_counts = count_chars_in_string(url)
            domain_counts = count_chars_in_string(domain)
            directory_counts = count_chars_in_string(directory_part)
            file_counts = count_chars_in_string(file_part)
            params_counts = count_chars_in_string(query)
            
            vowels = 'aeiou'
            qty_vowels_domain = sum(1 for char in domain if char in vowels)
            
            tld = domain.split('.')[-1] if '.' in domain else ""
            qty_tld_url = len(re.findall(r'\.[a-zA-Z]{2,}', url))
            tld_present_params = 1 if any(tld_name in query for tld_name in ['.com', '.org', '.net']) else 0
            
            domain_in_ip = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain) else 0
            
            email_in_url = 1 if '@' in url and re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', url) else 0
            
            shortening_services = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'short.link']
            url_shortened = 1 if any(service in domain for service in shortening_services) else 0
            
            server_client_domain = 1 if any(keyword in domain for keyword in ['server', 'client', 'admin', 'secure']) else 0
            
            params_list = query.split('&') if query else []
            qty_params = len(params_list)
            
            features = {
                'qty_dot_url': url_counts['dot'],
                'qty_hyphen_url': url_counts['hyphen'],
                'qty_underline_url': url_counts['underline'],
                'qty_slash_url': url_counts['slash'],
                'qty_questionmark_url': url_counts['questionmark'],
                'qty_equal_url': url_counts['equal'],
                'qty_at_url': url_counts['at'],
                'qty_and_url': url_counts['and'],
                'qty_exclamation_url': url_counts['exclamation'],
                'qty_space_url': url_counts['space'],
                'qty_tilde_url': url_counts['tilde'],
                'qty_comma_url': url_counts['comma'],
                'qty_plus_url': url_counts['plus'],
                'qty_asterisk_url': url_counts['asterisk'],
                'qty_hashtag_url': url_counts['hashtag'],
                'qty_dollar_url': url_counts['dollar'],
                'qty_percent_url': url_counts['percent'],
                'qty_tld_url': qty_tld_url,
                'length_url': len(url),
                
                'qty_dot_domain': domain_counts['dot'],
                'qty_hyphen_domain': domain_counts['hyphen'],
                'qty_underline_domain': domain_counts['underline'],
                'qty_slash_domain': domain_counts['slash'],
                'qty_questionmark_domain': domain_counts['questionmark'],
                'qty_equal_domain': domain_counts['equal'],
                'qty_at_domain': domain_counts['at'],
                'qty_and_domain': domain_counts['and'],
                'qty_exclamation_domain': domain_counts['exclamation'],
                'qty_space_domain': domain_counts['space'],
                'qty_tilde_domain': domain_counts['tilde'],
                'qty_comma_domain': domain_counts['comma'],
                'qty_plus_domain': domain_counts['plus'],
                'qty_asterisk_domain': domain_counts['asterisk'],
                'qty_hashtag_domain': domain_counts['hashtag'],
                'qty_dollar_domain': domain_counts['dollar'],
                'qty_percent_domain': domain_counts['percent'],
                'qty_vowels_domain': qty_vowels_domain,
                'domain_length': len(domain),
                'domain_in_ip': domain_in_ip,
                'server_client_domain': server_client_domain,
                
                'qty_dot_directory': directory_counts['dot'],
                'qty_hyphen_directory': directory_counts['hyphen'],
                'qty_underline_directory': directory_counts['underline'],
                'qty_slash_directory': directory_counts['slash'],
                'qty_questionmark_directory': directory_counts['questionmark'],
                'qty_equal_directory': directory_counts['equal'],
                'qty_at_directory': directory_counts['at'],
                'qty_and_directory': directory_counts['and'],
                'qty_exclamation_directory': directory_counts['exclamation'],
                'qty_space_directory': directory_counts['space'],
                'qty_tilde_directory': directory_counts['tilde'],
                'qty_comma_directory': directory_counts['comma'],
                'qty_plus_directory': directory_counts['plus'],
                'qty_asterisk_directory': directory_counts['asterisk'],
                'qty_hashtag_directory': directory_counts['hashtag'],
                'qty_dollar_directory': directory_counts['dollar'],
                'qty_percent_directory': directory_counts['percent'],
                'directory_length': len(directory_part),
                
                'qty_dot_file': file_counts['dot'],
                'qty_hyphen_file': file_counts['hyphen'],
                'qty_underline_file': file_counts['underline'],
                'qty_slash_file': file_counts['slash'],
                'qty_questionmark_file': file_counts['questionmark'],
                'qty_equal_file': file_counts['equal'],
                'qty_at_file': file_counts['at'],
                'qty_and_file': file_counts['and'],
                'qty_exclamation_file': file_counts['exclamation'],
                'qty_space_file': file_counts['space'],
                'qty_tilde_file': file_counts['tilde'],
                'qty_comma_file': file_counts['comma'],
                'qty_plus_file': file_counts['plus'],
                'qty_asterisk_file': file_counts['asterisk'],
                'qty_hashtag_file': file_counts['hashtag'],
                'qty_dollar_file': file_counts['dollar'],
                'qty_percent_file': file_counts['percent'],
                'file_length': len(file_part),
                
                'qty_dot_params': params_counts['dot'],
                'qty_hyphen_params': params_counts['hyphen'],
                'qty_underline_params': params_counts['underline'],
                'qty_slash_params': params_counts['slash'],
                'qty_questionmark_params': params_counts['questionmark'],
                'qty_equal_params': params_counts['equal'],
                'qty_at_params': params_counts['at'],
                'qty_and_params': params_counts['and'],
                'qty_exclamation_params': params_counts['exclamation'],
                'qty_space_params': params_counts['space'],
                'qty_tilde_params': params_counts['tilde'],
                'qty_comma_params': params_counts['comma'],
                'qty_plus_params': params_counts['plus'],
                'qty_asterisk_params': params_counts['asterisk'],
                'qty_hashtag_params': params_counts['hashtag'],
                'qty_dollar_params': params_counts['dollar'],
                'qty_percent_params': params_counts['percent'],
                'params_length': len(query),
                'tld_present_params': tld_present_params,
                'qty_params': qty_params,
                
                'email_in_url': email_in_url,
                'tls_ssl_certificate': tls_ssl_certificate,
                'qty_redirects': qty_redirects,
                'url_shortened': url_shortened
            }
            
            all_features = URLFeatureAnalyzer.get_feature_names()
            for feature_name in all_features:
                if feature_name not in features:
                    features[feature_name] = URLFeatureAnalyzer._get_safe_default_value(feature_name)
            
            return features
            
        except Exception as e:
            print(f"Warning: Error analyzing URL {url}: {str(e)}")
            print("Using safe default features to allow prediction to continue...")
            return URLFeatureAnalyzer._get_safe_default_features()

    @staticmethod
    def _get_safe_default_features():
        feature_names = URLFeatureAnalyzer.get_feature_names()
        features = {}
        
        for feature in feature_names:
            features[feature] = URLFeatureAnalyzer._get_safe_default_value(feature)
        
        return features

    @staticmethod
    def _get_safe_default_value(feature_name):
        if 'length' in feature_name:
            return 0
        
        elif 'qty' in feature_name:
            return 0
        
        elif feature_name in ['domain_in_ip', 'url_shortened', 'email_in_url', 'tld_present_params', 'server_client_domain']:
            return 0
        
        elif feature_name == 'tls_ssl_certificate':
            return 0
        
        else:
            return 0

    @staticmethod
    def calculate_similarity_index(url):
        legitimate_domains = ['google.com', 'amazon.com', 'paypal.com', 'microsoft.com', 'apple.com']
        max_similarity = 0
        
        for domain in legitimate_domains:
            if domain in url.lower():
                max_similarity = max(max_similarity, 0.8)
        
        return max_similarity

    @staticmethod
    def calculate_char_continuation_rate(url):
        if len(url) <= 1:
            return 0
        
        continuations = 0
        for i in range(len(url) - 1):
            if url[i] == url[i + 1]:
                continuations += 1
        
        return continuations / len(url)

    @staticmethod
    def calculate_url_char_probability(url):
        if len(url) == 0:
            return 0
        
        suspicious_chars = set('!@#$%^&*()+=[]{}|\\:";\'<>?,`~')
        encoded_chars = re.findall(r'%[0-9a-fA-F]{2}', url)
        
        suspicious_count = sum(1 for char in url if char in suspicious_chars)
        suspicious_count += len(encoded_chars)
        
        consecutive_special = re.findall(r'[^a-zA-Z0-9]{2,}', url)
        suspicious_count += len(consecutive_special) * 2
        
        base_prob = 0.06
        penalty = suspicious_count / len(url) if len(url) > 0 else 0
        
        result = max(0.001, min(0.09, base_prob - penalty * 0.05))
        
        return result
