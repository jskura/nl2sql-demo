#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory
mkdir -p logs

# Log files
BACKEND_LOG="logs/backend.log"
FRONTEND_LOG="logs/frontend.log"

# Function to handle cleanup on exit
cleanup() {
  echo -e "${YELLOW}Shutting down servers...${NC}"
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  echo -e "${GREEN}Servers stopped. Log files are available at:${NC}"
  echo -e "  - Backend: ${BLUE}$BACKEND_LOG${NC}"
  echo -e "  - Frontend: ${BLUE}$FRONTEND_LOG${NC}"
  exit 0
}

# Set up trap to catch Ctrl+C
trap cleanup INT TERM

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
  echo -e "${RED}Python 3 is required but not installed. Please install Python 3 and try again.${NC}"
  exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
  echo -e "${RED}Node.js is required but not installed. Please install Node.js and try again.${NC}"
  exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
  echo -e "${RED}npm is required but not installed. Please install npm and try again.${NC}"
  exit 1
fi

echo -e "${GREEN}Starting NL2SQL Agent Demo...${NC}"

# Check if root venv exists, create if it doesn't
if [ ! -d "venv" ]; then
  echo -e "${YELLOW}Creating virtual environment in project root...${NC}"
  python3 -m venv venv
fi

# Activate the root venv
source venv/bin/activate

# Start backend server with verbose output
echo -e "${BLUE}Starting backend server...${NC}"
cd backend
pip install -r requirements.txt
echo -e "${YELLOW}Starting backend with verbose logging...${NC}"
# Set environment variable for more verbose logging
export LOG_LEVEL=DEBUG
export DEBUG=true
uvicorn app.main:app --reload --log-level debug > ../$BACKEND_LOG 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to start...${NC}"
sleep 5

# Check if backend started successfully
if ! ps -p $BACKEND_PID > /dev/null; then
  echo -e "${RED}Backend failed to start. Check $BACKEND_LOG for details.${NC}"
  echo -e "${YELLOW}Last 10 lines of backend log:${NC}"
  tail -n 10 $BACKEND_LOG
  exit 1
fi

# Start frontend server with verbose output
echo -e "${BLUE}Starting frontend server...${NC}"
cd frontend
npm install
echo -e "${YELLOW}Starting frontend with verbose logging...${NC}"
# Set environment variable for more verbose logging
export REACT_APP_DEBUG=true
npm start > ../$FRONTEND_LOG 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait a moment to check if frontend started
sleep 5

# Check if frontend started successfully
if ! ps -p $FRONTEND_PID > /dev/null; then
  echo -e "${RED}Frontend failed to start. Check $FRONTEND_LOG for details.${NC}"
  echo -e "${YELLOW}Last 10 lines of frontend log:${NC}"
  tail -n 10 $FRONTEND_LOG
  kill $BACKEND_PID
  exit 1
fi

echo -e "${GREEN}NL2SQL Agent Demo is running!${NC}"
echo -e "${BLUE}Backend:${NC} http://localhost:8000"
echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "${YELLOW}Logs:${NC}"
echo -e "  - Backend: ${BLUE}$BACKEND_LOG${NC}"
echo -e "  - Frontend: ${BLUE}$FRONTEND_LOG${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the servers${NC}"

# Tail logs in real-time if requested
if [ "$1" == "--logs" ]; then
  echo -e "${YELLOW}Showing live logs (Ctrl+C to stop)...${NC}"
  tail -f $BACKEND_LOG $FRONTEND_LOG
else
  # Show log file locations and how to view them
  echo -e "${YELLOW}To view logs in real-time, run:${NC}"
  echo -e "  tail -f $BACKEND_LOG $FRONTEND_LOG"
fi

# Wait for both processes to finish
wait $BACKEND_PID $FRONTEND_PID 