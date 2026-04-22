// Package lab19 implements the .crom v2 binary format.
//
// Item 1.3.1: Definir header binário
// Item 1.3.2: Serialização/deserialização de codebook
//
// The .crom v2 format is designed to store trained VQ codebooks and
// metadata compactly, acting as the "frozen brain" for CROM Agents.
package lab19

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	// CromMagic is the magic byte sequence identifying a .crom file.
	CromMagic = "CROM"
	// CromVersion is the current format version.
	CromVersion uint16 = 2
)

// Flags for the header bitfield.
const (
	FlagFrozen     uint16 = 1 << 0
	FlagCompressed uint16 = 1 << 1
	FlagSigned     uint16 = 1 << 2
)

// CromV2Header represents the 16-byte binary header of a .crom v2 file.
//
// Layout (Little Endian):
// 0-3: Magic ("CROM")
// 4-5: Version (uint16)
// 6-7: K (codebook size, uint16)
// 8-9: D (dimension, uint16)
// 10-11: Flags (bitfield, uint16)
// 12-15: MetaLen (metadata JSON length, uint32)
type CromV2Header struct {
	Magic   [4]byte
	Version uint16
	K       uint16
	D       uint16
	Flags   uint16
	MetaLen uint32
}

// CromFile represents a fully parsed .crom v2 file.
type CromFile struct {
	Header    CromV2Header
	Metadata  map[string]interface{}
	Centroids [][]float32 // Stored as float32 to save space vs float64
}

var (
	ErrInvalidMagic   = errors.New("invalid magic bytes, not a .crom file")
	ErrUnsupportedVer = errors.New("unsupported .crom version")
	ErrCorruptData    = errors.New("file corrupted or truncated")
)

// WriteCromV2 serializes a codebook and metadata to a .crom v2 file.
func WriteCromV2(path string, k, d int, flags uint16, centroids [][]float64, meta map[string]interface{}) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if len(centroids) != k {
		return fmt.Errorf("centroid count %d does not match K=%d", len(centroids), k)
	}

	// 1. Serialize Metadata
	var metaBytes []byte
	if meta != nil {
		metaBytes, err = json.Marshal(meta)
		if err != nil {
			return fmt.Errorf("failed to marshal metadata: %w", err)
		}
	}

	// 2. Build Header
	header := CromV2Header{
		Version: CromVersion,
		K:       uint16(k),
		D:       uint16(d),
		Flags:   flags,
		MetaLen: uint32(len(metaBytes)),
	}
	copy(header.Magic[:], CromMagic)

	// 3. Write Header
	if err := binary.Write(f, binary.LittleEndian, &header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// 4. Write Metadata
	if len(metaBytes) > 0 {
		if _, err := f.Write(metaBytes); err != nil {
			return fmt.Errorf("failed to write metadata: %w", err)
		}
	}

	// 5. Write Centroids (convert float64 -> float32)
	// Write in a single batch for speed
	floatBuf := make([]float32, k*d)
	for i := 0; i < k; i++ {
		if len(centroids[i]) < d {
			return fmt.Errorf("centroid %d has dimension %d, expected %d", i, len(centroids[i]), d)
		}
		for j := 0; j < d; j++ {
			floatBuf[i*d+j] = float32(centroids[i][j])
		}
	}

	if err := binary.Write(f, binary.LittleEndian, floatBuf); err != nil {
		return fmt.Errorf("failed to write centroids: %w", err)
	}

	return nil
}

// ReadCromV2 reads and parses a .crom v2 file.
func ReadCromV2(path string) (*CromFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	buf := bytes.NewReader(data)

	// 1. Read Header
	var header CromV2Header
	if err := binary.Read(buf, binary.LittleEndian, &header); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	if string(header.Magic[:]) != CromMagic {
		return nil, ErrInvalidMagic
	}
	if header.Version != CromVersion {
		return nil, fmt.Errorf("%w: got %d", ErrUnsupportedVer, header.Version)
	}

	// 2. Read Metadata
	meta := make(map[string]interface{})
	if header.MetaLen > 0 {
		metaBytes := make([]byte, header.MetaLen)
		if _, err := io.ReadFull(buf, metaBytes); err != nil {
			return nil, fmt.Errorf("failed to read metadata: %w", err)
		}
		if err := json.Unmarshal(metaBytes, &meta); err != nil {
			return nil, fmt.Errorf("failed to parse metadata JSON: %w", err)
		}
	}

	// 3. Read Centroids
	k, d := int(header.K), int(header.D)
	floatBuf := make([]float32, k*d)
	if err := binary.Read(buf, binary.LittleEndian, &floatBuf); err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return nil, ErrCorruptData
		}
		return nil, fmt.Errorf("failed to read centroids: %w", err)
	}

	centroids := make([][]float32, k)
	for i := 0; i < k; i++ {
		centroids[i] = make([]float32, d)
		copy(centroids[i], floatBuf[i*d:(i+1)*d])
	}

	return &CromFile{
		Header:    header,
		Metadata:  meta,
		Centroids: centroids,
	}, nil
}
