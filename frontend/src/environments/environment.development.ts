import { HttpHeaders } from "@angular/common/http";
import { Environment } from "../app/types/environment";

export const environment: Environment = {
    production: false,
    serverUrl: "http://localhost:5001",
    httpOptions: {
        headers: new HttpHeaders({
            'Content-Type': 'application/json'
        })
    }
};
