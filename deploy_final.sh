#!/bin/bash
 
# Simple CI/CD Deployment Script
# Logs into Azure, builds and pushes Docker image directly to ACR
 
set -e  # Exit on any error
 
# Configuration Variables
ACR_NAME="financenlpacr"
ACR_URL="financenlpacr.azurecr.io"
IMAGE_NAME="classification-app"
IMAGE_TAG="latest"
 
# Function to print section headers
print_section() {
    echo "=================================================="
    echo "$1"
    echo "=================================================="
}
 
# Function to execute commands with error handling
run_command() {
    echo "Executing: $1"
    if ! eval "$1"; then
        echo "Error: Command failed: $1"
        exit 1
    fi
}
 
# Step 1: Azure Login
azure_login() {
    print_section "STEP 1: Azure Authentication"
    
    # Check if already logged in
    if az account show >/dev/null 2>&1; then
        echo "✓ Already logged into Azure"
        az account show --query "name" -o tsv
    else
        echo "Logging into Azure..."
        run_command "az login"
    fi
}
 
# Step 2: ACR Login
acr_login() {
    print_section "STEP 2: Azure Container Registry Login"
    run_command "az acr login --name $ACR_NAME"
    echo "✓ ACR login successful"
}
 
# Step 3: Build and Push
build_and_push() {
    print_section "STEP 3: Build and Push Docker Image"
    
    FULL_IMAGE_NAME="$ACR_URL/$IMAGE_NAME:$IMAGE_TAG"
    echo "Building and pushing: $FULL_IMAGE_NAME"
    echo "Target platform: linux/amd64"
    
    BUILD_COMMAND="docker buildx build --platform linux/amd64 -t $FULL_IMAGE_NAME --push ."
    run_command "$BUILD_COMMAND"
    
    echo "✓ Build and push completed successfully"
}
 

# Step 4: Verify
verify_deployment() {
    print_section "STEP 4: Verification"
    echo "Verifying image in ACR..."
    if az acr repository show --name "$ACR_NAME" --image "$IMAGE_NAME:$IMAGE_TAG" >/dev/null 2>&1; then
        echo "✓ Image successfully pushed to ACR"
        echo "Image: $ACR_URL/$IMAGE_NAME:$IMAGE_TAG"
    else
        echo "Warning: Could not verify image in ACR"
    fi
}

# Step 5: Update Azure Container App
update_container_app() {
    print_section "STEP 5: Update Azure Container App"
    echo "Updating Azure Container App: finance-classifier-fast to use new image..."
    run_command "az containerapp update --name finance-classifier-update --resource-group finance-adb-rg --image $ACR_URL/$IMAGE_NAME:$IMAGE_TAG"
    # run_command "az containerapp revision restart --name finance-classifier-update --resource-group finance-adb-rg"
    # az container restart -g="XXX" -n="XXX"
    echo "✓ Azure Container App updated successfully"
}
 
# Main deployment pipeline
main() {
    echo "Starting Simple Deployment Pipeline"
    echo "ACR: $ACR_NAME"
    echo "Image: $IMAGE_NAME:$IMAGE_TAG"
    echo "Platform: linux/amd64"
    echo
    
    # Step 1: Azure login
    azure_login
    
    # Step 2: ACR login
    acr_login
    
    # Step 3: Build and push
    build_and_push
    
    # Step 4: Verify
    verify_deployment

    # Step 5: Update Azure Container App
    update_container_app

    print_section "DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "Your image is ready at: $ACR_URL/$IMAGE_NAME:$IMAGE_TAG"
}
 
# Handle script interruption
trap 'echo -e "\nDeployment cancelled by user."; exit 1' INT
 
# Run main function
main