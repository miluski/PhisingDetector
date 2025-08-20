import { Component, WritableSignal, signal, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MessageFormData } from '../../types/message-form-data';

@Component({
  selector: 'app-message-form',
  imports: [CommonModule, FormsModule],
  templateUrl: './message-form.html',
  styleUrl: './message-form.scss'
})
export class MessageForm {
  protected formData = input.required<MessageFormData>();
  protected isLoading = input.required<boolean>();
  
  protected onFormSubmit = output<void>();
  protected onModelTest = output<string>();
  protected onEnhancedTest = output<void>();
  protected onFormDataChanged = output<MessageFormData>();

  protected messageType: WritableSignal<'email' | 'sms'> = signal('email');

  protected onMessageTypeChange(type: 'email' | 'sms') {
    this.messageType.set(type);
    const newData = { ...this.formData(), message_type: type };
    
    if (type === 'sms') {
      newData.sender = '';
      newData.receiver = '';
    }
    
    this.onFormDataChanged.emit(newData);
  }

  protected onContentChange(content: string) {
    this.onFormDataChanged.emit({ ...this.formData(), content });
  }

  protected onSenderChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.onFormDataChanged.emit({ ...this.formData(), sender: target.value });
  }

  protected onReceiverChange(event: Event) {
    const target = event.target as HTMLInputElement;
    this.onFormDataChanged.emit({ ...this.formData(), receiver: target.value });
  }

  protected onContentInputChange(event: Event) {
    const target = event.target as HTMLTextAreaElement;
    this.onFormDataChanged.emit({ ...this.formData(), content: target.value });
  }

  protected onSubmit() {
    this.onFormSubmit.emit();
  }

  protected testModel(model: string) {
    this.onModelTest.emit(model);
  }

  protected testEnhanced() {
    this.onEnhancedTest.emit();
  }
}
