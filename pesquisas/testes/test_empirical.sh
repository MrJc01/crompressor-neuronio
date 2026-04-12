#!/bin/bash
fusermount3 -uz ../fuse_mnt || true
sleep 1
go build -o fuse_demon_check ./cmd/fuse_test/main.go
./fuse_demon_check &
SERVER_PID=$!
sleep 5

echo "== Contexto Estático (Inicial) =="
dd if=../fuse_mnt/virtual_brain.gguf iflag=direct bs=1M count=1 skip=1 2>/dev/null | md5sum

echo "== Mudando para Persona Code =="
curl -s http://localhost:9999/context?persona=code
dd if=../fuse_mnt/virtual_brain.gguf iflag=direct bs=1M count=1 skip=1 2>/dev/null | md5sum

echo "== Mudando para Persona Math =="
curl -s http://localhost:9999/context?persona=math
dd if=../fuse_mnt/virtual_brain.gguf iflag=direct bs=1M count=1 skip=1 2>/dev/null | md5sum

echo "== Mudando para Persona Creative =="
curl -s http://localhost:9999/context?persona=creative
dd if=../fuse_mnt/virtual_brain.gguf iflag=direct bs=1M count=1 skip=1 2>/dev/null | md5sum

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
sleep 1
fusermount3 -uz ../fuse_mnt || true
