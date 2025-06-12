# Testing Lab

## Backend

Fastify Server

### Set your environment variable

```
cd backend
cp .env.sample .env
```

### Development

Run a mongo container
```
docker run -d -p 27017:27017 mongo
```

Install dependencies
```
npm install
```

Start development mode
```
npm run dev
```

### Run the test

```
npm run test
```

## Frontend

React (by vite)

### Development

Install dependencies
```
npm install
```

Start development mode
```
cd frontend
npm run dev
```

Visit
http://localhost:5173

### Run cypress test

```
npm run cy:test
```