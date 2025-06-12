FROM docker.io/library/node:21-alpine3.18 AS BUILDER

WORKDIR /app

COPY . .

RUN npm install
EXPOSE 5000
CMD ["npm", "run", "dev"]