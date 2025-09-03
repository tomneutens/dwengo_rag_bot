#!/bin/bash


export DWENGO_REPO_URL=https://github.com/dwengovzw/learning_content
export DATA_DIR=./learning_content
export INDEX_PATH=./dwengo_faiss.index
export META_PATH=./dwengo_faiss.meta.json
export EMB_MODEL=intfloat/multilingual-e5-large-instruct
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export SYNC_REPO=true
export API_KEY=changeme
export CORS_ORIGINS=*

# Run this file with source setup_server_environment.sh