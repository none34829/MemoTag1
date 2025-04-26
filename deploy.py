"""
Deployment script for the MemoTag cognitive decline detection API.
Supports deployment to cloud platforms like AWS, GCP, or Azure.
"""

import os
import sys
import logging
import argparse
import subprocess
import shutil
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Deployer:
    """Class to handle deployment of the API to cloud providers."""
    
    def __init__(self, platform, region=None, project_name="memotag-cognitive-api"):
        """
        Initialize the deployer.
        
        Args:
            platform: Cloud platform to deploy to ('aws', 'gcp', 'azure')
            region: Cloud region to deploy to
            project_name: Name of the project/application
        """
        self.platform = platform.lower()
        self.region = region
        self.project_name = project_name
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Validate platform
        valid_platforms = ['aws', 'gcp', 'azure', 'local']
        if self.platform not in valid_platforms:
            raise ValueError(f"Invalid platform: {platform}. Must be one of {valid_platforms}")
    
    def _create_dockerfile(self):
        """Create a Dockerfile for the application."""
        dockerfile_path = os.path.join(self.project_dir, "Dockerfile")
        
        dockerfile_content = """
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    libasound2-dev \\
    portaudio19-dev \\
    libsndfile1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create directories if they don't exist
RUN mkdir -p data/raw data/processed models

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000", "--no-reload"]
"""
        
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        logger.info(f"Created Dockerfile at {dockerfile_path}")
        return dockerfile_path
    
    def _create_aws_files(self):
        """Create AWS-specific deployment files."""
        # Create Elastic Beanstalk configuration
        eb_dir = os.path.join(self.project_dir, ".elasticbeanstalk")
        os.makedirs(eb_dir, exist_ok=True)
        
        config_path = os.path.join(eb_dir, "config.yml")
        config_content = f"""
branch-defaults:
  main:
    environment: {self.project_name}-env
    group_suffix: null
global:
  application_name: {self.project_name}
  branch: null
  default_ec2_keyname: null
  default_platform: Docker
  default_region: {self.region or "us-west-2"}
  include_git_submodules: true
  instance_profile: null
  platform_name: null
  platform_version: null
  profile: null
  repository: null
  sc: null
  workspace_type: Application
"""
        
        with open(config_path, "w") as f:
            f.write(config_content)
        
        # Create Docker compose file for EB
        compose_path = os.path.join(self.project_dir, "docker-compose.yml")
        compose_content = """
version: '3'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
"""
        
        with open(compose_path, "w") as f:
            f.write(compose_content)
        
        logger.info("Created AWS Elastic Beanstalk configuration files")
    
    def _create_gcp_files(self):
        """Create GCP-specific deployment files."""
        # Create app.yaml for Google App Engine
        app_yaml_path = os.path.join(self.project_dir, "app.yaml")
        app_yaml_content = """
runtime: custom
env: flex

# Persistent disk size in GB
resources:
  disk_size_gb: 10

# Add the minimum memory resources needed for your app
manual_scaling:
  instances: 1

# Configure network settings
network:
  forwarded_ports:
    - 8000:8000
"""
        
        with open(app_yaml_path, "w") as f:
            f.write(app_yaml_content)
        
        # Create cloudbuild.yaml
        cloudbuild_path = os.path.join(self.project_dir, "cloudbuild.yaml")
        cloudbuild_content = f"""
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/{self.project_name}', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/{self.project_name}']
- name: 'gcr.io/cloud-builders/gcloud'
  args: ['run', 'deploy', '{self.project_name}', '--image', 'gcr.io/$PROJECT_ID/{self.project_name}', '--platform', 'managed', '--region', '{self.region or "us-central1"}', '--allow-unauthenticated']

images:
- gcr.io/$PROJECT_ID/{self.project_name}
"""
        
        with open(cloudbuild_path, "w") as f:
            f.write(cloudbuild_content)
        
        logger.info("Created GCP deployment files")
    
    def _create_azure_files(self):
        """Create Azure-specific deployment files."""
        # Create azure-pipelines.yml
        azure_pipeline_path = os.path.join(self.project_dir, "azure-pipelines.yml")
        azure_pipeline_content = f"""
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  imageName: '{self.project_name}'
  containerRegistry: 'yourAzureContainerRegistry'

steps:
- task: Docker@2
  displayName: Build and push an image
  inputs:
    containerRegistry: $(containerRegistry)
    repository: $(imageName)
    command: buildAndPush
    Dockerfile: Dockerfile
    tags: |
      latest
      $(Build.BuildId)

- task: AzureWebAppContainer@1
  inputs:
    azureSubscription: 'YourAzureSubscription'
    appName: '{self.project_name}'
    containers: $(containerRegistry)/$(imageName):$(Build.BuildId)
"""
        
        with open(azure_pipeline_path, "w") as f:
            f.write(azure_pipeline_content)
        
        logger.info("Created Azure deployment files")
    
    def _create_platform_files(self):
        """Create platform-specific deployment files."""
        if self.platform == 'aws':
            self._create_aws_files()
        elif self.platform == 'gcp':
            self._create_gcp_files()
        elif self.platform == 'azure':
            self._create_azure_files()
        # No files needed for local deployment
    
    def prepare_deployment(self):
        """Prepare the application for deployment."""
        logger.info(f"Preparing deployment for {self.platform}...")
        
        # Create Dockerfile
        self._create_dockerfile()
        
        # Create platform-specific files
        self._create_platform_files()
        
        # Ensure requirements.txt is up to date
        # This is already created in our project
        
        logger.info("Deployment preparation complete")
    
    def deploy(self):
        """Deploy the application to the specified platform."""
        logger.info(f"Deploying to {self.platform}...")
        
        if self.platform == 'local':
            self._deploy_local()
        elif self.platform == 'aws':
            self._deploy_aws()
        elif self.platform == 'gcp':
            self._deploy_gcp()
        elif self.platform == 'azure':
            self._deploy_azure()
    
    def _deploy_local(self):
        """Deploy locally using Docker."""
        try:
            # Build the Docker image
            logger.info("Building Docker image...")
            subprocess.run(["docker", "build", "-t", self.project_name, "."], 
                          cwd=self.project_dir, check=True)
            
            # Run the container
            logger.info("Starting Docker container...")
            subprocess.run(["docker", "run", "-p", "8000:8000", "--name", self.project_name, 
                           "-d", self.project_name], 
                          cwd=self.project_dir, check=True)
            
            logger.info(f"API deployed locally at http://localhost:8000")
            logger.info(f"To stop the container, run: docker stop {self.project_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error deploying locally: {e}")
            raise
        except FileNotFoundError:
            logger.error("Docker not found. Please install Docker to deploy locally.")
            raise
    
    def _deploy_aws(self):
        """Deploy to AWS Elastic Beanstalk."""
        try:
            # Initialize EB if needed
            if not os.path.exists(os.path.join(self.project_dir, ".elasticbeanstalk")):
                logger.info("Initializing Elastic Beanstalk...")
                subprocess.run(["eb", "init", "-p", "docker", self.project_name, 
                               "--region", self.region or "us-west-2"], 
                              cwd=self.project_dir, check=True)
            
            # Create environment if it doesn't exist
            logger.info("Creating/updating Elastic Beanstalk environment...")
            subprocess.run(["eb", "create", f"{self.project_name}-env", "--single", 
                           "--instance-type", "t2.micro"], 
                          cwd=self.project_dir, check=False)
            
            # Deploy
            logger.info("Deploying to Elastic Beanstalk...")
            subprocess.run(["eb", "deploy"], cwd=self.project_dir, check=True)
            
            # Get the URL
            result = subprocess.run(["eb", "status", "--verbose"], 
                                  cwd=self.project_dir, capture_output=True, text=True, check=True)
            
            # Extract URL from output
            for line in result.stdout.splitlines():
                if "CNAME" in line:
                    url = line.split(":")[1].strip()
                    logger.info(f"API deployed to AWS at http://{url}")
                    break
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error deploying to AWS: {e}")
            raise
        except FileNotFoundError:
            logger.error("EB CLI not found. Install with: pip install awsebcli")
            raise
    
    def _deploy_gcp(self):
        """Deploy to Google Cloud Run."""
        try:
            # Verify gcloud is installed
            subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
            
            # Set project if provided
            if self.region:
                logger.info(f"Setting GCP region to {self.region}...")
                subprocess.run(["gcloud", "config", "set", "compute/region", self.region], 
                              check=True)
            
            # Deploy to Cloud Run
            logger.info("Deploying to Google Cloud Run...")
            subprocess.run(["gcloud", "builds", "submit", "--config", "cloudbuild.yaml"], 
                          cwd=self.project_dir, check=True)
            
            logger.info(f"API deployed to GCP Cloud Run. Check the GCP Console for the URL.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error deploying to GCP: {e}")
            raise
        except FileNotFoundError:
            logger.error("gcloud CLI not found. Install the Google Cloud SDK.")
            raise
    
    def _deploy_azure(self):
        """Deploy to Azure App Service."""
        try:
            # Verify Azure CLI is installed
            subprocess.run(["az", "--version"], capture_output=True, check=True)
            
            # Create resource group if it doesn't exist
            logger.info("Creating/checking Azure resource group...")
            resource_group = f"{self.project_name}-rg"
            region = self.region or "eastus"
            
            subprocess.run(["az", "group", "create", "--name", resource_group, 
                           "--location", region], check=True)
            
            # Create App Service plan
            logger.info("Creating/checking App Service plan...")
            plan_name = f"{self.project_name}-plan"
            subprocess.run(["az", "appservice", "plan", "create", "--name", plan_name, 
                           "--resource-group", resource_group, "--sku", "B1", 
                           "--is-linux"], check=True)
            
            # Create web app
            logger.info("Creating/checking web app...")
            subprocess.run(["az", "webapp", "create", "--name", self.project_name, 
                           "--resource-group", resource_group, "--plan", plan_name, 
                           "--deployment-container-image-name", f"{self.project_name}:latest"], 
                          check=True)
            
            # Configure the app to use the local Docker file
            logger.info("Configuring web app for Docker...")
            subprocess.run(["az", "webapp", "config", "container", "set", 
                           "--name", self.project_name, "--resource-group", resource_group, 
                           "--docker-custom-image-name", f"{self.project_name}:latest", 
                           "--docker-registry-server-url", "https://index.docker.io"], 
                          check=True)
            
            # Get the deployed URL
            result = subprocess.run(["az", "webapp", "show", "--name", self.project_name, 
                                   "--resource-group", resource_group, "--query", "defaultHostName", 
                                   "--output", "tsv"], 
                                  capture_output=True, text=True, check=True)
            
            url = result.stdout.strip()
            logger.info(f"API deployed to Azure at https://{url}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error deploying to Azure: {e}")
            raise
        except FileNotFoundError:
            logger.error("Azure CLI not found. Install with: pip install azure-cli")
            raise

def main():
    """Main function to deploy the application."""
    parser = argparse.ArgumentParser(description="Deploy the MemoTag Cognitive Decline Detection API")
    parser.add_argument("--platform", choices=["aws", "gcp", "azure", "local"], default="local",
                        help="Platform to deploy to")
    parser.add_argument("--region", help="Cloud region to deploy to")
    parser.add_argument("--project-name", default="memotag-cognitive-api",
                       help="Project/application name")
    
    args = parser.parse_args()
    
    try:
        deployer = Deployer(args.platform, args.region, args.project_name)
        deployer.prepare_deployment()
        deployer.deploy()
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)
    
    logger.info("Deployment completed successfully")

if __name__ == "__main__":
    main()
