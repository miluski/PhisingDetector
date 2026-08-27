FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build -- --configuration=production

FROM node:20-alpine AS production
WORKDIR /app
RUN npm install -g http-server
COPY --from=build /app/dist/frontend/browser ./dist
RUN mkdir -p /app/certs
EXPOSE 4000
CMD ["sh", "-c", "http-server ./dist -p 4000 -S -C /app/certs/cert.pem -K /app/certs/key.pem --cors"]
