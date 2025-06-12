# Backend

## File structure
```bash
src
├── index.js
├── models
│   ├── lab_staff.js
│   ├── order.js
│   └── qa_engineer.js
├── routes
│   ├── lab_staff.js
│   ├── order.js
│   └── qa_engineer.js
└── services
    ├── lab_staff.js
    ├── order.js
    └── qa_engineer.js
```

## How to use
1. build
```bash
npm install
```
2. open a mongo container
```bash
docker run --name my-mongodb -p 27017:27017 -d mongo
```
3. run backend server
```bash
npm run dev
```

## API routes
### routes
- /api/orders
- /api/staffs
- /api/qa_engineers

### action under routes
1. GET("/") : return the whole table
2. GET("/:id") : return the element
3. POST("/") : add entry
4. DELETE("/id") : delete the entry

## Note
Please follow the structure in src/models when testing DB