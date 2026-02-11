#!/bin/bash

#---------------------------------------------------------------
# Application Management for NVIDIA RAG & AI-Q
#---------------------------------------------------------------

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

# PID files (stored in /tmp to keep working directory clean)
RAG_FRONTEND_PID="/tmp/.port-forward-rag-frontend.pid"
RAG_INGESTOR_PID="/tmp/.port-forward-rag-ingestor.pid"
AIRA_FRONTEND_PID="/tmp/.port-forward-aira-frontend.pid"
RAG_ZIPKIN_PID="/tmp/.port-forward-rag-zipkin.pid"
RAG_GRAFANA_PID="/tmp/.port-forward-rag-grafana.pid"
AIRA_PHOENIX_PID="/tmp/.port-forward-aira-phoenix.pid"

show_help() {
    echo "Usage: ./app.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  port start <service>    Start port-forwarding (rag, aira, observability, all)"
    echo "  port stop <service>     Stop port-forwarding (rag, aira, observability, all)"
    echo "  port restart <service>  Restart port-forwarding (rag, aira, observability, all)"
    echo "  port status             Show port-forwarding status"
    echo "  ingest                  Ingest data from S3 to RAG"
    echo "  cleanup                 Clean up RAG and AI-Q applications"
    echo ""
    echo "Examples:"
    echo "  ./app.sh port start all            # Start all port-forwards"
    echo "  ./app.sh port start observability  # Start observability tools"
    echo "  ./app.sh port status               # Check port-forward status"
    echo "  ./app.sh ingest                    # Ingest documents from S3"
    echo "  ./app.sh cleanup                   # Remove all applications"
    echo ""
    echo "Port Mappings:"
    echo "  RAG Frontend:  localhost:3001 -> rag-frontend:3000"
    echo "  RAG Ingestor:  localhost:8082 -> ingestor-server:8082"
    echo "  AIRA Frontend: localhost:3000 -> aira-aira-frontend:3000"
    echo ""
    echo "Observability:"
    echo "  Zipkin (RAG):  localhost:9411 -> rag-zipkin:9411"
    echo "  Grafana (RAG): localhost:8080 -> rag-grafana:80"
    echo "  Phoenix (AIQ): localhost:6006 -> aira-phoenix:6006"
}

# ===== Port Forwarding Functions =====

check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

start_rag_ports() {
    # RAG Frontend
    if [ -f "$RAG_FRONTEND_PID" ] && kill -0 $(cat "$RAG_FRONTEND_PID") 2>/dev/null; then
        print_info "RAG frontend already running"
    else
        if check_port 3001; then
            print_error "Port 3001 in use"
        else
            kubectl port-forward -n rag svc/rag-frontend 3001:3000 >/dev/null 2>&1 &
            echo $! > "$RAG_FRONTEND_PID"
            sleep 1
            print_success "RAG frontend: http://localhost:3001"
        fi
    fi

    # Ingestor
    if [ -f "$RAG_INGESTOR_PID" ] && kill -0 $(cat "$RAG_INGESTOR_PID") 2>/dev/null; then
        print_info "Ingestor already running"
    else
        if check_port 8082; then
            print_error "Port 8082 in use"
        else
            kubectl port-forward -n rag svc/ingestor-server 8082:8082 >/dev/null 2>&1 &
            echo $! > "$RAG_INGESTOR_PID"
            sleep 1
            print_success "Ingestor: http://localhost:8082"
        fi
    fi
}

start_aira_ports() {
    if [ -f "$AIRA_FRONTEND_PID" ] && kill -0 $(cat "$AIRA_FRONTEND_PID") 2>/dev/null; then
        print_info "AIRA already running"
    else
        if check_port 3000; then
            print_error "Port 3000 in use"
        else
            kubectl port-forward -n nv-aira svc/aira-aira-frontend 3000:3000 >/dev/null 2>&1 &
            echo $! > "$AIRA_FRONTEND_PID"
            sleep 1
            print_success "AIRA frontend: http://localhost:3000"
        fi
    fi
}

stop_rag_ports() {
    for pid_file in "$RAG_FRONTEND_PID" "$RAG_INGESTOR_PID"; do
        if [ -f "$pid_file" ]; then
            kill $(cat "$pid_file") 2>/dev/null || true
            rm "$pid_file"
        fi
    done
    print_info "RAG port-forwards stopped"
}

stop_aira_ports() {
    if [ -f "$AIRA_FRONTEND_PID" ]; then
        kill $(cat "$AIRA_FRONTEND_PID") 2>/dev/null || true
        rm "$AIRA_FRONTEND_PID"
    fi
    print_info "AIRA port-forward stopped"
}

start_observability() {
    # RAG Zipkin
    if [ -f "$RAG_ZIPKIN_PID" ] && kill -0 $(cat "$RAG_ZIPKIN_PID") 2>/dev/null; then
        print_info "Zipkin already running"
    else
        if check_port 9411; then
            print_error "Port 9411 in use"
        else
            kubectl port-forward -n rag svc/rag-zipkin 9411:9411 >/dev/null 2>&1 &
            echo $! > "$RAG_ZIPKIN_PID"
            sleep 1
            print_success "Zipkin: http://localhost:9411"
        fi
    fi

    # RAG Grafana
    if [ -f "$RAG_GRAFANA_PID" ] && kill -0 $(cat "$RAG_GRAFANA_PID") 2>/dev/null; then
        print_info "Grafana already running"
    else
        if check_port 8080; then
            print_error "Port 8080 in use"
        else
            kubectl port-forward -n rag svc/rag-grafana 8080:80 >/dev/null 2>&1 &
            echo $! > "$RAG_GRAFANA_PID"
            sleep 1
            print_success "Grafana: http://localhost:8080"
        fi
    fi

    # AI-Q Phoenix (only if deployed)
    if kubectl get namespace nv-aira &>/dev/null; then
        if [ -f "$AIRA_PHOENIX_PID" ] && kill -0 $(cat "$AIRA_PHOENIX_PID") 2>/dev/null; then
            print_info "Phoenix already running"
        else
            if check_port 6006; then
                print_error "Port 6006 in use"
            else
                kubectl port-forward -n nv-aira svc/aira-phoenix 6006:6006 >/dev/null 2>&1 &
                echo $! > "$AIRA_PHOENIX_PID"
                sleep 1
                print_success "Phoenix: http://localhost:6006"
            fi
        fi
    else
        print_info "AI-Q not deployed, skipping Phoenix"
    fi
}

stop_observability() {
    for pid_file in "$RAG_ZIPKIN_PID" "$RAG_GRAFANA_PID" "$AIRA_PHOENIX_PID"; do
        if [ -f "$pid_file" ]; then
            kill $(cat "$pid_file") 2>/dev/null || true
            rm "$pid_file"
        fi
    done
    print_info "Observability port-forwards stopped"
}

show_port_status() {
    echo "RAG Services:"
    [ -f "$RAG_FRONTEND_PID" ] && kill -0 $(cat "$RAG_FRONTEND_PID") 2>/dev/null && \
        print_success "Frontend: http://localhost:3001" || print_info "Frontend not running"
    [ -f "$RAG_INGESTOR_PID" ] && kill -0 $(cat "$RAG_INGESTOR_PID") 2>/dev/null && \
        print_success "Ingestor: http://localhost:8082" || print_info "Ingestor not running"

    echo ""
    echo "AI-Q Services:"
    [ -f "$AIRA_FRONTEND_PID" ] && kill -0 $(cat "$AIRA_FRONTEND_PID") 2>/dev/null && \
        print_success "Frontend: http://localhost:3000" || print_info "Frontend not running"

    echo ""
    echo "Observability:"
    [ -f "$RAG_ZIPKIN_PID" ] && kill -0 $(cat "$RAG_ZIPKIN_PID") 2>/dev/null && \
        print_success "Zipkin: http://localhost:9411" || print_info "Zipkin not running"
    [ -f "$RAG_GRAFANA_PID" ] && kill -0 $(cat "$RAG_GRAFANA_PID") 2>/dev/null && \
        print_success "Grafana: http://localhost:8080" || print_info "Grafana not running"
    [ -f "$AIRA_PHOENIX_PID" ] && kill -0 $(cat "$AIRA_PHOENIX_PID") 2>/dev/null && \
        print_success "Phoenix: http://localhost:6006" || print_info "Phoenix not running"
}

handle_port_command() {
    local action=$1
    local service=$2

    case "$action" in
        start)
            case "$service" in
                rag) start_rag_ports ;;
                aira) start_aira_ports ;;
                observability) start_observability ;;
                all) start_rag_ports; start_aira_ports; start_observability ;;
                *) print_error "Invalid service: $service"; exit 1 ;;
            esac
            ;;
        stop)
            case "$service" in
                rag) stop_rag_ports ;;
                aira) stop_aira_ports ;;
                observability) stop_observability ;;
                all) stop_rag_ports; stop_aira_ports; stop_observability ;;
                *) print_error "Invalid service: $service"; exit 1 ;;
            esac
            ;;
        restart)
            case "$service" in
                rag) stop_rag_ports; sleep 1; start_rag_ports ;;
                aira) stop_aira_ports; sleep 1; start_aira_ports ;;
                observability) stop_observability; sleep 1; start_observability ;;
                all) stop_rag_ports; stop_aira_ports; stop_observability; sleep 1; start_rag_ports; start_aira_ports; start_observability ;;
                *) print_error "Invalid service: $service"; exit 1 ;;
            esac
            ;;
        status)
            show_port_status
            ;;
        *)
            print_error "Invalid port action: $action"
            echo "Valid actions: start, stop, restart, status"
            exit 1
            ;;
    esac
}

# ===== Data Ingestion Function =====

ingest_data() {
    # Prompt for S3 configuration if not set
    if [ -z "$S3_BUCKET_NAME" ]; then
        read -p "S3 bucket name: " S3_BUCKET_NAME
    fi

    if [ -z "$S3_PREFIX" ]; then
        read -p "S3 prefix (optional, press enter to skip): " S3_PREFIX
    fi

    # Set defaults
    INGESTOR_URL="${INGESTOR_URL:-localhost:8082}"
    RAG_COLLECTION_NAME="${RAG_COLLECTION_NAME:-multimodal_data}"
    LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/tmp/s3_ingestion}"
    UPLOAD_BATCH_SIZE="${UPLOAD_BATCH_SIZE:-100}"

    # Parse ingestor URL
    INGESTOR_HOST="${INGESTOR_URL%%:*}"
    INGESTOR_PORT="${INGESTOR_URL##*:}"

    print_info "Downloading from s3://$S3_BUCKET_NAME/$S3_PREFIX"

    # Download from S3
    mkdir -p "$LOCAL_DATA_DIR"
    aws s3 sync "s3://$S3_BUCKET_NAME/$S3_PREFIX" "$LOCAL_DATA_DIR" \
      --exclude "*" --include "*.pdf" --include "*.docx" --include "*.txt"

    # Download batch ingestion script from NVIDIA RAG repository
    TEMP_DIR="/tmp/rag_ingestion_$$"
    mkdir -p "$TEMP_DIR"

    print_info "Downloading batch ingestion script from NVIDIA RAG repository..."
    RAG_VERSION="v2.3.0"
    RAG_BASE_URL="https://raw.githubusercontent.com/NVIDIA-AI-Blueprints/rag/${RAG_VERSION}"

    # Download batch_ingestion.py and requirements.txt
    curl -sL "${RAG_BASE_URL}/scripts/batch_ingestion.py" -o "$TEMP_DIR/batch_ingestion.py"
    curl -sL "${RAG_BASE_URL}/scripts/requirements.txt" -o "$TEMP_DIR/requirements.txt"

    if [ ! -f "$TEMP_DIR/batch_ingestion.py" ]; then
        print_error "Failed to download batch_ingestion.py"
        if [ -d "$TEMP_DIR" ]; then
            rm -r "$TEMP_DIR"
        fi
        exit 1
    fi

    # Install dependencies
    print_info "Installing dependencies..."
    python3 -m pip install -q -r "$TEMP_DIR/requirements.txt" 2>/dev/null || true

    # Run batch ingestion
    print_info "Ingesting to collection: $RAG_COLLECTION_NAME"
    python3 "$TEMP_DIR/batch_ingestion.py" \
      --folder "$LOCAL_DATA_DIR" \
      --collection-name "$RAG_COLLECTION_NAME" \
      --create_collection \
      --ingestor-host "$INGESTOR_HOST" \
      --ingestor-port "$INGESTOR_PORT" \
      --upload-batch-size "$UPLOAD_BATCH_SIZE" \
      -v

    # Cleanup
    if [ -d "$LOCAL_DATA_DIR" ]; then
        rm -r "$LOCAL_DATA_DIR"
    fi
    if [ -d "$TEMP_DIR" ]; then
        rm -r "$TEMP_DIR"
    fi
    print_success "Ingestion complete"
}

# ===== Cleanup Function =====

cleanup_apps() {
    print_warning "This will remove all RAG and AI-Q applications and local files"
    read -p "Continue? (y/n) [n]: " CONFIRM

    # Accept y, Y, yes, Yes, YES
    if [[ ! "$CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
        print_info "Cleanup cancelled"
        exit 0
    fi

    # Stop port-forwards
    print_info "Stopping port-forwards..."
    stop_rag_ports 2>/dev/null || true
    stop_aira_ports 2>/dev/null || true
    stop_observability 2>/dev/null || true

    # Uninstall applications
    if kubectl get namespace nv-aira &>/dev/null; then
        print_info "Uninstalling AI-Q..."
        helm uninstall aira -n nv-aira 2>/dev/null || true
    fi

    if kubectl get namespace rag &>/dev/null; then
        print_info "Uninstalling RAG..."
        helm uninstall rag -n rag 2>/dev/null || true
    fi

    # Clean up temp files
    for pid_file in /tmp/.port-forward-*.pid; do
        if [ -f "$pid_file" ]; then
            rm "$pid_file"
        fi
    done

    print_success "Cleanup complete"
    print_info "GPU nodes will be terminated by Karpenter in 5-10 minutes"
    print_info "To clean up build artifacts (rag/, opensearch/, .env), run cleanup from infra/nvidia-deep-research/"
}

# ===== Main Script Logic =====

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [ -z "$1" ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

case "$COMMAND" in
    port)
        if [ -z "$1" ]; then
            print_error "Port command requires an action (start, stop, restart, status)"
            exit 1
        fi
        handle_port_command "$@"
        ;;
    ingest)
        ingest_data
        ;;
    cleanup)
        cleanup_apps
        ;;
    *)
        print_error "Invalid command: $COMMAND"
        echo ""
        show_help
        exit 1
        ;;
esac
