FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# The deployed application only serves static files; analysis runs in the browser.
FROM nginx:stable-alpine
ENV PORT=8080
ENV NGINX_ENVSUBST_FILTER=^PORT$
COPY deploy/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist/ /usr/share/nginx/html/
EXPOSE 8080
