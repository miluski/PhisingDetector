export interface URLFeatures {
  url_length: number;
  num_dots: number;
  num_hyphens: number;
  num_underscores: number;
  num_slash: number;
  num_questionmarks: number;
  num_equal: number;
  num_at: number;
  num_and: number;
  num_exclamation: number;
  num_space: number;
  num_tilde: number;
  num_comma: number;
  num_plus: number;
  num_asterisk: number;
  num_hashtag: number;
  num_dollar: number;
  num_percent: number;
  has_ip: number;
  has_https: number;
  domain_length: number;
  subdomain_count: number;
  path_length: number;
  query_length: number;
  fragment_length: number;
}