import re
from urllib.parse import urlparse

class RuleBasedDetector:
    @staticmethod
    def rule_based_phishing_check(url, sender=None):
        if not url:
            return False, 0.0, []
        
        risk_score = 0
        risk_factors = []
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'pw', 'click', 'download', 'racing', 'review', 'xyz']
            for tld in suspicious_tlds:
                if domain.endswith(f'.{tld}'):
                    risk_score += 4
                    risk_factors.append(f"Suspicious TLD: .{tld}")
                    break
            
            if re.match(r'^\d+\.\d+\.\d+\.\d+', domain):
                risk_score += 5
                risk_factors.append("Domain is IP address")
            
            if '%' in url and re.search(r'%[0-9a-fA-F]{2}', url):
                risk_score += 2
                risk_factors.append("URL contains encoding")
            
            legitimate_brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'twitter']
            for brand in legitimate_brands:
                if brand in domain and not domain.endswith(f'{brand}.com'):
                    if f'{brand}.' not in domain or domain.count('.') > 2:
                        risk_score += 3
                        risk_factors.append(f"Suspicious {brand} impersonation")
            
            if len(url) > 100:
                risk_score += 1
                risk_factors.append("Very long URL")
            
            suspicious_paths = ['verify', 'secure', 'update', 'confirm', 'login', 'signin', 'account']
            path = parsed.path.lower()
            for sus_path in suspicious_paths:
                if sus_path in path:
                    risk_score += 1
                    risk_factors.append(f"Suspicious path: {sus_path}")
                    break
            
            subdomain_count = domain.count('.') - 1
            if subdomain_count > 2:
                risk_score += 2
                risk_factors.append("Multiple subdomains")
            
            if sender and '@' in sender:
                sender_domain = sender.split('@')[1].lower()
                if sender_domain != domain and sender_domain not in domain:
                    risk_score += 2
                    risk_factors.append("Sender domain mismatch")
            
            max_score = 20
            probability = min(risk_score / max_score, 1.0)
            is_phishing = risk_score >= 5
            
            return is_phishing, probability, risk_factors
            
        except Exception as e:
            print(f"Error in rule-based check: {e}")
            return False, 0.0, []
