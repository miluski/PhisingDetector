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
RUN mkdir -p /app/certs/frontend
EXPOSE 4000
CMD ["sh", "-c", "http-server ./dist -p 4000 -S -C /app/certs/frontend/cert.pem -K /app/certs/frontend/key.pem --cors"]

