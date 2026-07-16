pipeline {
    agent any

    environment {
        DOCKER_HUB_REPO = 'mustafakocaman/studymate'
        DOCKER_HUB_CREDENTIALS_ID = 'dockerhub-token'
        GITHUB_CREDENTIALS_ID = 'github-token'
        IMAGE_TAG = "v${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout GitHub') {
            steps {
                echo 'Checking out code from GitHub...'
                checkout scmGit(
                    branches: [[name: '*/main']],
                    extensions: [],
                    userRemoteConfigs: [[
                        credentialsId: "${GITHUB_CREDENTIALS_ID}",
                        url: 'https://github.com/MustafaKocamann/StudyMate-AI.git'
                    ]]
                )
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building ${DOCKER_HUB_REPO}:${IMAGE_TAG}..."
                    dockerImage = docker.build("${DOCKER_HUB_REPO}:${IMAGE_TAG}")
                }
            }
        }

        stage('Push Image to DockerHub') {
            steps {
                script {
                    echo 'Pushing Docker image to DockerHub...'
                    docker.withRegistry('https://registry.hub.docker.com', "${DOCKER_HUB_CREDENTIALS_ID}") {
                        dockerImage.push("${IMAGE_TAG}")
                    }
                }
            }
        }

        stage('Update Deployment YAML with New Tag') {
            steps {
                sh """
                    sed -i -E 's|(^[[:space:]]*image: ).*|\\1${DOCKER_HUB_REPO}:${IMAGE_TAG}|' manifests/deployment.yaml
                """
            }
        }

        stage('Commit Updated YAML') {
            steps {
                script {
                    withCredentials([usernamePassword(
                        credentialsId: "${GITHUB_CREDENTIALS_ID}",
                        usernameVariable: 'GIT_USER',
                        passwordVariable: 'GIT_PASS'
                    )]) {
                        sh '''
                            git config user.name "MustafaKocamann"
                            git config user.email "mustafakocaman789@gmail.com"
                            git add manifests/deployment.yaml
                            git diff --cached --quiet || git commit -m "Update image tag to ${IMAGE_TAG}"
                            git push https://${GIT_USER}:${GIT_PASS}@github.com/MustafaKocamann/StudyMate-AI.git HEAD:main
                        '''
                    }
                }
            }
        }

        stage('Install Kubectl & ArgoCD CLI Setup') {
            steps {
                sh '''
                    echo 'Installing Kubectl & ArgoCD CLI...'
                    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
                    chmod +x kubectl
                    mv kubectl /usr/local/bin/kubectl
                    curl -sSL -o /usr/local/bin/argocd https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
                    chmod +x /usr/local/bin/argocd
                '''
            }
        }

        stage('Apply Kubernetes & Sync App with ArgoCD') {
            steps {
                script {
                    kubeconfig(credentialsId: 'kubeconfig', serverUrl: 'https://192.168.49.2:8443') {
                        sh '''
                            argocd login 34.61.237.21:31704 --username admin --password $(kubectl get secret -n argocd argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d) --insecure
                            argocd app sync study
                        '''
                    }
                }
            }
        }
    }
}
