export interface MessageFormData {
  message_type: 'email' | 'sms';
  content: string;
  sender: string;
  receiver: string;
}
