#!/bin/bash

# Super-Resolution Deployment Script for Linux/Mac
# This script sets up and deploys the Super-Resolution application

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="super-resolution"
PYTHON_VERSION="3.11"
VENV_DIR="venv"
PORT=8000

echo -e "${GREEN}Super-Resolution Deployment Script${NC}"
echo "===================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for uv (fast package manager)
USE_UV=false
if command_exists uv; then
    echo -e "${GREEN}✓ Found uv (Astral's fast Python package manager)${NC}"
    USE_UV=true
else
    echo -e "${YELLOW}uv not found, using pip. Install uv for faster dependency management:${NC}"
    echo -e "${YELLOW}  curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
fi

# Check Python installation
echo -e "${YELLOW}Checking Python installation...${NC}"
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python is not installed!${NC}"
    echo "Please install Python $PYTHON_VERSION or higher"
    exit 1
fi

PYTHON_VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Found Python $PYTHON_VER${NC}"

# Setup environment based on available tools
if [ "$USE_UV" = true ]; then
    # Using uv - much faster!
    echo -e "${YELLOW}Setting up environment with uv...${NC}"
    
    # Create venv with uv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        uv venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created with uv${NC}"
    else
        echo -e "${GREEN}✓ Virtual environment exists${NC}"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Install dependencies with uv (much faster than pip)
    echo -e "${YELLOW}Installing dependencies with uv...${NC}"
    uv pip install -e .
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    # Using traditional pip
    echo -e "${YELLOW}Checking pip installation...${NC}"
    if ! $PYTHON_CMD -m pip --version > /dev/null 2>&1; then
        echo -e "${RED}Error: pip is not installed!${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ pip is installed${NC}"

    # Create and activate virtual environment
    echo -e "${YELLOW}Setting up virtual environment...${NC}"
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✓ Virtual environment exists${NC}"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    echo -e "${YELLOW}Upgrading pip...${NC}"
    pip install --upgrade pip setuptools wheel

    # Install dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p uploads output Data
echo -e "${GREEN}✓ Directories created${NC}"

# Ask for deployment mode
echo ""
echo "Select deployment mode:"
echo "1) Development (with auto-reload)"
echo "2) Production (optimized)"
read -p "Enter choice [1-2]: " mode_choice

case $mode_choice in
    1)
        echo -e "${GREEN}Starting in Development mode...${NC}"
        echo ""
        echo -e "${YELLOW}Application will be available at: http://localhost:$PORT${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        uvicorn app:app --host 0.0.0.0 --port $PORT --reload
        ;;
    2)
        echo -e "${GREEN}Starting in Production mode...${NC}"
        
        # Ask about process manager
        echo ""
        echo "Select process manager:"
        echo "1) Run in foreground"
        echo "2) Use systemd service"
        echo "3) Use screen session"
        read -p "Enter choice [1-3]: " pm_choice
        
        case $pm_choice in
            1)
                echo ""
                echo -e "${YELLOW}Application will be available at: http://localhost:$PORT${NC}"
                echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
                echo ""
                uvicorn app:app --host 0.0.0.0 --port $PORT --workers 4
                ;;
            2)
                # Create systemd service
                SERVICE_FILE="/etc/systemd/system/${APP_NAME}.service"
                echo ""
                echo -e "${YELLOW}Creating systemd service...${NC}"
                
                sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Super-Resolution FastAPI Application
After=network.target

[Service]
Type=notify
User=$USER
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/$VENV_DIR/bin"
ExecStart=$SCRIPT_DIR/$VENV_DIR/bin/uvicorn app:app --host 0.0.0.0 --port $PORT --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
                
                sudo systemctl daemon-reload
                sudo systemctl enable "$APP_NAME"
                sudo systemctl start "$APP_NAME"
                
                echo -e "${GREEN}✓ Systemd service created and started${NC}"
                echo ""
                echo "Service commands:"
                echo "  Status:  sudo systemctl status $APP_NAME"
                echo "  Stop:    sudo systemctl stop $APP_NAME"
                echo "  Restart: sudo systemctl restart $APP_NAME"
                echo "  Logs:    sudo journalctl -u $APP_NAME -f"
                ;;
            3)
                echo ""
                echo -e "${YELLOW}Starting in screen session...${NC}"
                
                if ! command_exists screen; then
                    echo -e "${RED}Error: screen is not installed!${NC}"
                    echo "Install it with: sudo apt-get install screen (Ubuntu/Debian)"
                    exit 1
                fi
                
                screen -dmS "$APP_NAME" bash -c "source $VENV_DIR/bin/activate && uvicorn app:app --host 0.0.0.0 --port $PORT --workers 4"
                echo -e "${GREEN}✓ Application started in screen session${NC}"
                echo ""
                echo "Screen commands:"
                echo "  Attach:  screen -r $APP_NAME"
                echo "  Detach:  Ctrl+A, then D"
                echo "  List:    screen -ls"
                ;;
            *)
                echo -e "${RED}Invalid choice${NC}"
                exit 1
                ;;
        esac
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Deployment complete!${NC}"
