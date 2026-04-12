#!/bin/bash
fusermount3 -uz ../fuse_mnt || true
sleep 1
go build -o fuse_demon_check ./cmd/fuse_test/main.go
./fuse_demon_check &
SERVER_PID=$!
sleep 8

echo "╔═══════════════════════════════════════════════════════╗"
echo "║   🤖 RUNNING Llama.cpp INFERENCE VIA FUSE-MOUNT      ║"
echo "╚═══════════════════════════════════════════════════════╝"

LLAMA_BIN="/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"

echo "== Contexto Estático (Inicial) =="
$LLAMA_BIN -m ../fuse_mnt/virtual_brain.gguf -p "A linguagem de programação Go é " -n 12 --no-mmap 2>/dev/null

echo ""
echo "== Mudando para Persona Code =="
curl -s http://localhost:9999/context?persona=code
$LLAMA_BIN -m ../fuse_mnt/virtual_brain.gguf -p "Python is a language that " -n 12 --no-mmap 2>/dev/null

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
sleep 1
fusermount3 -uz ../fuse_mnt || true
