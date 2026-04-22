// Package pesquisa0 implements the Delta Branch Store and Synapse Protocol
// in native Go, as required by the pesquisa0 experimental roadmap.
//
// Items covered:
//   3.2.4 - XOR Delta Branch Store (Go native)
//   6.2.3 - Synapse Protocol (goroutines + channels)
//   6.1.5 - Ed25519 signature integration
package pesquisa0

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// ============================================================================
// 3.2.4 — DELTA BRANCH STORE (XOR Delta nativo Go)
// ============================================================================

// DeltaBranchStore manages branches using XOR deltas against a base state.
type DeltaBranchStore struct {
	Base     []byte
	Branches map[int]*Branch
	mu       sync.RWMutex
	nextID   int
}

// Branch stores only the XOR delta against base.
type Branch struct {
	ID        int
	Delta     map[int]byte // sparse: only changed positions
	CreatedAt time.Time
}

// NewDeltaBranchStore creates a store with the given base state.
func NewDeltaBranchStore(base []byte) *DeltaBranchStore {
	baseCopy := make([]byte, len(base))
	copy(baseCopy, base)
	return &DeltaBranchStore{
		Base:     baseCopy,
		Branches: make(map[int]*Branch),
	}
}

// CreateBranch creates a new branch from a modified state.
// Only stores the XOR delta (positions that differ from base).
func (s *DeltaBranchStore) CreateBranch(modified []byte) int {
	s.mu.Lock()
	defer s.mu.Unlock()

	delta := make(map[int]byte)
	for i := 0; i < len(s.Base) && i < len(modified); i++ {
		xor := s.Base[i] ^ modified[i]
		if xor != 0 {
			delta[i] = xor
		}
	}

	id := s.nextID
	s.nextID++
	s.Branches[id] = &Branch{
		ID:        id,
		Delta:     delta,
		CreatedAt: time.Now(),
	}
	return id
}

// ReadBranch reconstructs the full state from base + delta.
func (s *DeltaBranchStore) ReadBranch(id int) ([]byte, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	branch, ok := s.Branches[id]
	if !ok {
		return nil, fmt.Errorf("branch %d not found", id)
	}

	result := make([]byte, len(s.Base))
	copy(result, s.Base)
	for pos, xor := range branch.Delta {
		result[pos] ^= xor
	}
	return result, nil
}

// Collapse removes all branches that don't match observed reality.
func (s *DeltaBranchStore) Collapse(observed []byte, tolerance int) []int {
	s.mu.Lock()
	defer s.mu.Unlock()

	var collapsed []int
	for id, branch := range s.Branches {
		mismatches := 0
		for pos, xor := range branch.Delta {
			if pos < len(observed) {
				actual := s.Base[pos] ^ xor
				if actual != observed[pos] {
					mismatches++
				}
			}
		}
		if mismatches > tolerance {
			delete(s.Branches, id)
			collapsed = append(collapsed, id)
		}
	}
	return collapsed
}

// MemoryUsage returns delta memory vs full copy memory.
func (s *DeltaBranchStore) MemoryUsage() (deltaBytes, fullBytes int) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, b := range s.Branches {
		deltaBytes += len(b.Delta) * 5 // 4 bytes index + 1 byte value
		fullBytes += len(s.Base)
	}
	return
}

// ============================================================================
// 6.2.3 — SYNAPSE PROTOCOL (goroutines + channels)
// ============================================================================

// MessageType defines synapse protocol message types.
type MessageType int

const (
	MSG_DELTA_UPDATE  MessageType = iota
	MSG_COLLAPSE      
	MSG_QUERY_STATE   
	MSG_MERGE_REQUEST 
)

// SynapseMessage is a message in the synapse protocol.
type SynapseMessage struct {
	Type      MessageType
	BranchID  int
	Delta     map[int]byte
	Timestamp time.Time
	Signature []byte
}

// SynapseNode is a participant in the synapse network.
type SynapseNode struct {
	ID       string
	Inbox    chan SynapseMessage
	Store    *DeltaBranchStore
	Peers    []*SynapseNode
	Stats    SynapseStats
	mu       sync.Mutex
}

// SynapseStats tracks protocol metrics.
type SynapseStats struct {
	MessagesSent     int
	MessagesReceived int
	BytesSent        int
	BytesReceived    int
	CollapsesSent    int
}

// NewSynapseNode creates a node with a branch store.
func NewSynapseNode(id string, base []byte) *SynapseNode {
	return &SynapseNode{
		ID:    id,
		Inbox: make(chan SynapseMessage, 100),
		Store: NewDeltaBranchStore(base),
	}
}

// Connect adds a peer to this node.
func (n *SynapseNode) Connect(peer *SynapseNode) {
	n.Peers = append(n.Peers, peer)
}

// Send broadcasts a message to all peers.
func (n *SynapseNode) Send(msg SynapseMessage) {
	n.mu.Lock()
	n.Stats.MessagesSent++
	n.Stats.BytesSent += len(msg.Delta) * 5
	n.mu.Unlock()

	for _, peer := range n.Peers {
		select {
		case peer.Inbox <- msg:
		default:
			// Drop if buffer full
		}
	}
}

// Listen processes incoming messages (run as goroutine).
func (n *SynapseNode) Listen(done <-chan struct{}) {
	for {
		select {
		case msg := <-n.Inbox:
			n.mu.Lock()
			n.Stats.MessagesReceived++
			n.Stats.BytesReceived += len(msg.Delta) * 5
			n.mu.Unlock()

			switch msg.Type {
			case MSG_DELTA_UPDATE:
				modified := make([]byte, len(n.Store.Base))
				copy(modified, n.Store.Base)
				for pos, xor := range msg.Delta {
					modified[pos] ^= xor
				}
				n.Store.CreateBranch(modified)

			case MSG_COLLAPSE:
				n.Store.Collapse(n.Store.Base, 0)
			}

		case <-done:
			return
		}
	}
}

// ============================================================================
// 6.1.5 — ED25519 SIGNATURE INTEGRATION
// ============================================================================

// SignedDelta wraps a delta with Ed25519 signature.
type SignedDelta struct {
	Delta     map[int]byte `json:"delta"`
	BranchID  int          `json:"branch_id"`
	Timestamp int64        `json:"timestamp"`
	PublicKey []byte       `json:"public_key"`
	Signature []byte       `json:"signature"`
}

// SignDelta signs a branch delta with Ed25519.
func SignDelta(priv ed25519.PrivateKey, branchID int, delta map[int]byte) (*SignedDelta, error) {
	sd := &SignedDelta{
		Delta:     delta,
		BranchID:  branchID,
		Timestamp: time.Now().UnixNano(),
		PublicKey: priv.Public().(ed25519.PublicKey),
	}

	payload, err := json.Marshal(struct {
		BranchID  int          `json:"b"`
		Delta     map[int]byte `json:"d"`
		Timestamp int64        `json:"t"`
	}{sd.BranchID, sd.Delta, sd.Timestamp})
	if err != nil {
		return nil, err
	}

	hash := sha256.Sum256(payload)
	sd.Signature = ed25519.Sign(priv, hash[:])
	return sd, nil
}

// VerifyDelta verifies the Ed25519 signature of a delta.
func VerifyDelta(sd *SignedDelta) bool {
	payload, err := json.Marshal(struct {
		BranchID  int          `json:"b"`
		Delta     map[int]byte `json:"d"`
		Timestamp int64        `json:"t"`
	}{sd.BranchID, sd.Delta, sd.Timestamp})
	if err != nil {
		return false
	}

	hash := sha256.Sum256(payload)
	return ed25519.Verify(sd.PublicKey, hash[:], sd.Signature)
}

// ============================================================================
// BENCHMARK — Run all tests and output results
// ============================================================================

// BenchmarkResults holds all benchmark data.
type BenchmarkResults struct {
	DeltaStore DeltaStoreResults `json:"delta_store"`
	Synapse    SynapseResults    `json:"synapse"`
	Security   SecurityResults   `json:"security"`
}

type DeltaStoreResults struct {
	BranchesCreated int     `json:"branches_created"`
	CreateTimeUs    float64 `json:"create_time_us"`
	CollapseTimeUs  float64 `json:"collapse_time_us"`
	DeltaBytes      int     `json:"delta_bytes"`
	FullBytes       int     `json:"full_bytes"`
	ReductionPct    float64 `json:"reduction_pct"`
}

type SynapseResults struct {
	NodeCount    int     `json:"node_count"`
	MessagesSent int     `json:"messages_sent"`
	TotalTimeMs  float64 `json:"total_time_ms"`
	PerMsgUs     float64 `json:"per_msg_us"`
}

type SecurityResults struct {
	SignTimeUs   float64 `json:"sign_time_us"`
	VerifyTimeUs float64 `json:"verify_time_us"`
	Valid        bool    `json:"valid"`
}

// RunBenchmarks runs all pesquisa0 Go benchmarks and saves results.
func RunBenchmarks(outputPath string) (*BenchmarkResults, error) {
	results := &BenchmarkResults{}

	// ── Delta Branch Store ────────────────────────────
	base := make([]byte, 1024*1024) // 1MB base state
	rand.Read(base)
	store := NewDeltaBranchStore(base)

	// Create 500 branches with 1% divergence
	t0 := time.Now()
	for i := 0; i < 500; i++ {
		modified := make([]byte, len(base))
		copy(modified, base)
		nChanges := len(base) / 100 // 1%
		for j := 0; j < nChanges; j++ {
			pos := j * 100 % len(base)
			modified[pos] ^= byte(i + j)
		}
		store.CreateBranch(modified)
	}
	createTime := time.Since(t0)

	deltaBytes, fullBytes := store.MemoryUsage()
	reduction := (1.0 - float64(deltaBytes)/float64(fullBytes)) * 100

	// Collapse
	observed := make([]byte, len(base))
	copy(observed, base)
	t0 = time.Now()
	store.Collapse(observed, 100)
	collapseTime := time.Since(t0)

	results.DeltaStore = DeltaStoreResults{
		BranchesCreated: 500,
		CreateTimeUs:    float64(createTime.Microseconds()) / 500,
		CollapseTimeUs:  float64(collapseTime.Microseconds()),
		DeltaBytes:      deltaBytes,
		FullBytes:       fullBytes,
		ReductionPct:    reduction,
	}

	// ── Synapse Protocol ─────────────────────────────
	base2 := make([]byte, 4096)
	rand.Read(base2)

	nodeA := NewSynapseNode("A", base2)
	nodeB := NewSynapseNode("B", base2)
	nodeC := NewSynapseNode("C", base2)
	nodeA.Connect(nodeB)
	nodeA.Connect(nodeC)

	done := make(chan struct{})
	go nodeB.Listen(done)
	go nodeC.Listen(done)

	t0 = time.Now()
	nMsgs := 100
	for i := 0; i < nMsgs; i++ {
		delta := map[int]byte{i % 4096: byte(i)}
		nodeA.Send(SynapseMessage{
			Type:      MSG_DELTA_UPDATE,
			Delta:     delta,
			Timestamp: time.Now(),
		})
	}
	time.Sleep(50 * time.Millisecond) // Let goroutines process
	close(done)
	synapseTime := time.Since(t0)

	results.Synapse = SynapseResults{
		NodeCount:    3,
		MessagesSent: nodeA.Stats.MessagesSent,
		TotalTimeMs:  float64(synapseTime.Milliseconds()),
		PerMsgUs:     float64(synapseTime.Microseconds()) / float64(nMsgs),
	}

	// ── Ed25519 Signatures ───────────────────────────
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	_ = pub

	testDelta := map[int]byte{0: 1, 1: 2, 2: 3}
	t0 = time.Now()
	sd, _ := SignDelta(priv, 42, testDelta)
	signTime := time.Since(t0)

	t0 = time.Now()
	valid := VerifyDelta(sd)
	verifyTime := time.Since(t0)

	results.Security = SecurityResults{
		SignTimeUs:   float64(signTime.Microseconds()),
		VerifyTimeUs: float64(verifyTime.Microseconds()),
		Valid:        valid,
	}

	// ── Save results ─────────────────────────────────
	if outputPath != "" {
		data, _ := json.MarshalIndent(results, "", "  ")
		os.WriteFile(outputPath, data, 0644)
	}

	return results, nil
}
