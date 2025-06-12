FROM docker.io/library/node:21-alpine3.18 AS BUILDER

WORKDIR /app

COPY . .
RUN npm install
RUN npm run build

CMD ["nginx", "-g", "daemon off;"]

FROM nginx:stable-alpine as runner

COPY --from=builder /app/dist /usr/share/nginx/html

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]