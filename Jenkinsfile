pipeline {
    agent any

    environment {
        DOCKER_HUB_REPO = 'mustafakocaman/studymate'
        DOCKER_HUB_CREDENTIALS_ID = 'dockerhub-token'
        GITHUB_CREDENTIALS_ID = 'github-token'
        IMAGE_TAG = "${BUILD_NUMBER}"
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
                            git config user.name "StudyMate Jenkins"
                            git config user.email "jenkins@studymate.local"
                            git add manifests/deployment.yaml
                            git diff --cached --quiet || git commit -m "Update image tag to ${IMAGE_TAG}"
                            git push https://${GIT_USER}:${GIT_PASS}@github.com/MustafaKocamann/StudyMate-AI.git HEAD:main
                        '''
                    }
                }
            }
        }
    }
}
