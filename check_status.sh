#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log files
BACKEND_LOG="logs/backend.log"
FRONTEND_LOG="logs/frontend.log"

# Check if logs directory exists
if [ ! -d "logs" ]; then
  echo -e "${RED}Logs directory not found. Has the application been started?${NC}"
  exit 1
fi

# Function to check if a process is running
check_process() {
  local name=$1
  local count=$(ps aux | grep -v grep | grep -c "$name")
  if [ $count -gt 0 ]; then
    echo -e "${GREEN}$name is running${NC}"
    return 0
  else
    echo -e "${RED}$name is not running${NC}"
    return 1
  fi
}

# Function to display recent logs
show_recent_logs() {
  local log_file=$1
  local lines=${2:-20}
  local name=$3
  
  if [ -f "$log_file" ]; then
    echo -e "${YELLOW}Recent $name logs (last $lines lines):${NC}"
    tail -n $lines "$log_file"
  else
    echo -e "${RED}$name log file not found: $log_file${NC}"
  fi
}

# Function to check for errors in logs
check_for_errors() {
  local log_file=$1
  local name=$2
  
  if [ -f "$log_file" ]; then
    local error_count=$(grep -i "error\|exception\|fail" "$log_file" | wc -l)
    if [ $error_count -gt 0 ]; then
      echo -e "${RED}Found $error_count potential errors in $name logs${NC}"
      echo -e "${YELLOW}Recent errors:${NC}"
      grep -i "error\|exception\|fail" "$log_file" | tail -n 5
    else
      echo -e "${GREEN}No obvious errors found in $name logs${NC}"
    fi
  else
    echo -e "${RED}$name log file not found: $log_file${NC}"
  fi
}

echo -e "${BLUE}=== NL2SQL Agent Demo Status ===${NC}"

# Check backend status
echo -e "\n${BLUE}Checking backend status...${NC}"
check_process "uvicorn app.main:app"
backend_running=$?

# Check frontend status
echo -e "\n${BLUE}Checking frontend status...${NC}"
check_process "node.*start"
frontend_running=$?

# Check for errors in logs
echo -e "\n${BLUE}Checking logs for errors...${NC}"
check_for_errors "$BACKEND_LOG" "Backend"
check_for_errors "$FRONTEND_LOG" "Frontend"

# Show recent logs based on command line arguments
if [ "$1" == "--backend" ] || [ "$1" == "-b" ]; then
  echo -e "\n${BLUE}Showing backend logs...${NC}"
  show_recent_logs "$BACKEND_LOG" 50 "Backend"
elif [ "$1" == "--frontend" ] || [ "$1" == "-f" ]; then
  echo -e "\n${BLUE}Showing frontend logs...${NC}"
  show_recent_logs "$FRONTEND_LOG" 50 "Frontend"
elif [ "$1" == "--all" ] || [ "$1" == "-a" ]; then
  echo -e "\n${BLUE}Showing all logs...${NC}"
  show_recent_logs "$BACKEND_LOG" 25 "Backend"
  echo -e "\n${YELLOW}------------------------------${NC}\n"
  show_recent_logs "$FRONTEND_LOG" 25 "Frontend"
elif [ "$1" == "--live" ] || [ "$1" == "-l" ]; then
  echo -e "\n${BLUE}Showing live logs (Ctrl+C to exit)...${NC}"
  tail -f "$BACKEND_LOG" "$FRONTEND_LOG"
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo -e "\n${BLUE}Usage:${NC}"
  echo -e "  $0                  Show status only"
  echo -e "  $0 --backend, -b    Show backend logs"
  echo -e "  $0 --frontend, -f   Show frontend logs"
  echo -e "  $0 --all, -a        Show both backend and frontend logs"
  echo -e "  $0 --live, -l       Show live logs (tail -f)"
  echo -e "  $0 --help, -h       Show this help message"
fi

# Show application URLs if running
if [ $backend_running -eq 0 ] && [ $frontend_running -eq 0 ]; then
  echo -e "\n${GREEN}Application is running!${NC}"
  echo -e "${BLUE}Backend:${NC} http://localhost:8000"
  echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
else
  echo -e "\n${RED}Application is not fully running.${NC}"
  echo -e "${YELLOW}To start the application, run:${NC} ./run_demo.sh"
fi

echo -e "\n${YELLOW}To view logs in real-time, run:${NC} $0 --live" 