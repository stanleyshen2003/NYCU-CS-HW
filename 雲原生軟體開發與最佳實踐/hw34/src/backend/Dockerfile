FROM docker.io/library/node:21-alpine3.18 AS BUILDER

WORKDIR /app

COPY . .

RUN npm install
RUN npm run build
EXPOSE 8888
CMD ["node", "dist/index.js"]