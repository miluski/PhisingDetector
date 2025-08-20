import { HttpHeaders } from '@angular/common/http';

export interface Environment {
  production: boolean;
  serverUrl: string;
  httpOptions: {
    headers: HttpHeaders;
  };
}
