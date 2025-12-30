# Use the latest Node version automatically
FROM node:22-alpine

WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the code
COPY . .

# Run the app
# --host is CRITICAL for Vite to work inside Docker
CMD ["npm", "run", "dev", "--", "--host"]