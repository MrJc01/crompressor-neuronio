package pesquisa0

import (
	"fmt"
	"testing"
)

func TestDeltaBranchStore(t *testing.T) {
	base := make([]byte, 1024)
	for i := range base {
		base[i] = byte(i % 256)
	}

	store := NewDeltaBranchStore(base)

	// Create branch with 10 changes
	modified := make([]byte, len(base))
	copy(modified, base)
	modified[0] = 0xFF
	modified[100] = 0xAA
	modified[500] = 0xBB

	id := store.CreateBranch(modified)

	// Read back
	result, err := store.ReadBranch(id)
	if err != nil {
		t.Fatalf("ReadBranch failed: %v", err)
	}

	if result[0] != 0xFF || result[100] != 0xAA || result[500] != 0xBB {
		t.Error("Branch data mismatch")
	}

	// Verify unmodified bytes
	if result[1] != base[1] || result[50] != base[50] {
		t.Error("Unmodified bytes changed")
	}

	// Memory usage
	delta, full := store.MemoryUsage()
	if delta >= full {
		t.Errorf("Delta (%d) should be less than full (%d)", delta, full)
	}
	fmt.Printf("  Delta: %d bytes, Full: %d bytes (%.1f%% reduction)\n", delta, full, (1-float64(delta)/float64(full))*100)
}

func TestCollapse(t *testing.T) {
	base := make([]byte, 256)
	store := NewDeltaBranchStore(base)

	// Create 10 branches
	for i := 0; i < 10; i++ {
		mod := make([]byte, len(base))
		copy(mod, base)
		mod[i] = byte(i + 1)
		store.CreateBranch(mod)
	}

	if len(store.Branches) != 10 {
		t.Fatalf("Expected 10 branches, got %d", len(store.Branches))
	}

	// Collapse with observed reality (base state = all zeros)
	observed := make([]byte, len(base))
	collapsed := store.Collapse(observed, 0)

	fmt.Printf("  Collapsed %d branches, %d remaining\n", len(collapsed), len(store.Branches))
}

func TestSynapseProtocol(t *testing.T) {
	base := make([]byte, 256)

	nodeA := NewSynapseNode("A", base)
	nodeB := NewSynapseNode("B", base)
	nodeA.Connect(nodeB)

	done := make(chan struct{})
	go nodeB.Listen(done)

	// Send 10 updates
	for i := 0; i < 10; i++ {
		nodeA.Send(SynapseMessage{
			Type:  MSG_DELTA_UPDATE,
			Delta: map[int]byte{i: byte(i + 1)},
		})
	}

	// Wait for processing
	close(done)

	if nodeA.Stats.MessagesSent != 10 {
		t.Errorf("Expected 10 sent, got %d", nodeA.Stats.MessagesSent)
	}
	fmt.Printf("  Sent: %d, Received: %d, Bytes: %d\n",
		nodeA.Stats.MessagesSent, nodeB.Stats.MessagesReceived, nodeA.Stats.BytesSent)
}

func TestEd25519Signatures(t *testing.T) {
	import_test_only := false
	_ = import_test_only

	// Use RunBenchmarks for Ed25519 test
	results, err := RunBenchmarks("")
	if err != nil {
		t.Fatalf("Benchmark failed: %v", err)
	}

	if !results.Security.Valid {
		t.Error("Ed25519 signature verification failed")
	}

	fmt.Printf("  Sign: %.0fμs, Verify: %.0fμs, Valid: %v\n",
		results.Security.SignTimeUs, results.Security.VerifyTimeUs, results.Security.Valid)
	fmt.Printf("  Delta Store: 500 branches, %.0fμs/create, %.1f%% reduction\n",
		results.DeltaStore.CreateTimeUs, results.DeltaStore.ReductionPct)
	fmt.Printf("  Synapse: %d msgs, %.1fms total\n",
		results.Synapse.MessagesSent, results.Synapse.TotalTimeMs)
}
