package hnsw

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// Binary layout of hnsw.bin (little-endian, all section offsets aligned to
// 16 bytes so that int32 sections can be zero-copy cast via unsafe.Slice on
// 64-bit amd64):
//
//	Header (256 bytes, padded)
//	[align] Vectors    [N × 14]uint8
//	[align] Conn0Cnt   [N]uint8
//	[align] Conn0      [N × M0]int32
//	[align] Labels     [(N+7)/8]uint8
//	[align] UpperOff   [maxLayer+1]int32
//	[align] UpperNodes [totalUpperNodes]int32
//	[align] UpperCnt   [totalUpperNodes]uint8
//	[align] UpperConn  [totalUpperNodes × M]int32
//
// "totalUpperNodes" is encoded indirectly via UpperOff[maxLayer]; UpperCnt's
// length is the same value.

const (
	magicStr  = "RINHA26"
	binFormat = uint32(1)

	headerSize  = 256
	sectionAlgn = 16
)

// Header is the persisted binary header. Field order is part of the on-disk
// format — do not reorder.
type header struct {
	Magic      [8]byte
	Version    uint32
	N          uint32
	M          uint32
	M0         uint32
	EntryPoint uint32
	MaxLayer   int32

	VectorsOff    uint64
	VectorsLen    uint64
	Conn0CntOff   uint64
	Conn0CntLen   uint64
	Conn0Off      uint64
	Conn0Len      uint64
	LabelsOff     uint64
	LabelsLen     uint64
	UpperOffOff   uint64
	UpperOffLen   uint64
	UpperNodesOff uint64
	UpperNodesLen uint64
	UpperCntOff   uint64
	UpperCntLen   uint64
	UpperConnOff  uint64
	UpperConnLen  uint64
}

// Save writes the graph to path in the binary layout above. Returns the
// number of bytes written, or an error.
func (g *Graph) Save(path string) (int64, error) {
	f, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	// Reserve header space — we'll seek back and rewrite it once offsets are
	// known.
	zero := make([]byte, headerSize)
	if _, err := f.Write(zero); err != nil {
		return 0, err
	}

	pos := int64(headerSize)

	writeSection := func(buf []byte) (offset, length uint64, err error) {
		// Align.
		if pad := pos % sectionAlgn; pad != 0 {
			padBuf := make([]byte, sectionAlgn-pad)
			if _, err := f.Write(padBuf); err != nil {
				return 0, 0, err
			}
			pos += int64(len(padBuf))
		}
		offset = uint64(pos)
		length = uint64(len(buf))
		n, err := f.Write(buf)
		if err != nil {
			return 0, 0, err
		}
		pos += int64(n)
		return offset, length, nil
	}

	// Each int32 section is reinterpreted as bytes via unsafe — no copy,
	// no allocation.
	int32Bytes := func(s []int32) []byte {
		if len(s) == 0 {
			return nil
		}
		return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
	}

	h := header{
		Magic:      [8]byte{magicStr[0], magicStr[1], magicStr[2], magicStr[3], magicStr[4], magicStr[5], magicStr[6], 0},
		Version:    binFormat,
		N:          g.N,
		M:          uint32(g.M),
		M0:         uint32(g.M0),
		EntryPoint: g.EntryPoint,
		MaxLayer:   g.MaxLayer,
	}

	if h.VectorsOff, h.VectorsLen, err = writeSection(g.Vectors); err != nil {
		return 0, err
	}
	if h.Conn0CntOff, h.Conn0CntLen, err = writeSection(g.Conn0Cnt); err != nil {
		return 0, err
	}
	if h.Conn0Off, h.Conn0Len, err = writeSection(int32Bytes(g.Conn0)); err != nil {
		return 0, err
	}
	if h.LabelsOff, h.LabelsLen, err = writeSection(g.Labels); err != nil {
		return 0, err
	}
	if h.UpperOffOff, h.UpperOffLen, err = writeSection(int32Bytes(g.UpperOff)); err != nil {
		return 0, err
	}
	if h.UpperNodesOff, h.UpperNodesLen, err = writeSection(int32Bytes(g.UpperNodes)); err != nil {
		return 0, err
	}
	if h.UpperCntOff, h.UpperCntLen, err = writeSection(g.UpperCnt); err != nil {
		return 0, err
	}
	if h.UpperConnOff, h.UpperConnLen, err = writeSection(int32Bytes(g.UpperConn)); err != nil {
		return 0, err
	}

	// Seek back, write the header.
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return 0, err
	}
	if err := binary.Write(f, binary.LittleEndian, &h); err != nil {
		return 0, err
	}

	return pos, nil
}

// Load reads an hnsw.bin file from path and returns a Graph whose slices
// alias the underlying byte buffer (zero-copy where possible). The Graph
// retains the buffer in g.Raw so the slices remain valid.
//
// This implementation uses os.ReadFile and is portable; the runtime API can
// substitute an mmap'd buffer using the same loadFromBytes core.
func Load(path string) (*Graph, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return loadFromBytes(data)
}

// loadFromBytes parses the header and creates slice views over the buffer.
func loadFromBytes(data []byte) (*Graph, error) {
	if len(data) < headerSize {
		return nil, errors.New("hnsw: file too small for header")
	}
	var h header
	if err := binary.Read(byteReader(data[:headerSize]), binary.LittleEndian, &h); err != nil {
		return nil, fmt.Errorf("hnsw: header read: %w", err)
	}
	if string(h.Magic[:len(magicStr)]) != magicStr {
		return nil, errors.New("hnsw: bad magic")
	}
	if h.Version != binFormat {
		return nil, fmt.Errorf("hnsw: unsupported version %d", h.Version)
	}

	g := &Graph{
		N:          h.N,
		M:          int(h.M),
		M0:         int(h.M0),
		EntryPoint: h.EntryPoint,
		MaxLayer:   h.MaxLayer,
	}

	// Slice views — bounds-check each section, then bind.
	if err := bindBytes(&g.Vectors, data, h.VectorsOff, h.VectorsLen); err != nil {
		return nil, fmt.Errorf("Vectors: %w", err)
	}
	if err := bindBytes(&g.Conn0Cnt, data, h.Conn0CntOff, h.Conn0CntLen); err != nil {
		return nil, fmt.Errorf("Conn0Cnt: %w", err)
	}
	if err := bindInt32(&g.Conn0, data, h.Conn0Off, h.Conn0Len); err != nil {
		return nil, fmt.Errorf("Conn0: %w", err)
	}
	if err := bindBytes(&g.Labels, data, h.LabelsOff, h.LabelsLen); err != nil {
		return nil, fmt.Errorf("Labels: %w", err)
	}
	if err := bindInt32(&g.UpperOff, data, h.UpperOffOff, h.UpperOffLen); err != nil {
		return nil, fmt.Errorf("UpperOff: %w", err)
	}
	if err := bindInt32(&g.UpperNodes, data, h.UpperNodesOff, h.UpperNodesLen); err != nil {
		return nil, fmt.Errorf("UpperNodes: %w", err)
	}
	if err := bindBytes(&g.UpperCnt, data, h.UpperCntOff, h.UpperCntLen); err != nil {
		return nil, fmt.Errorf("UpperCnt: %w", err)
	}
	if err := bindInt32(&g.UpperConn, data, h.UpperConnOff, h.UpperConnLen); err != nil {
		return nil, fmt.Errorf("UpperConn: %w", err)
	}

	// Sanity-check expected sizes.
	expectedVec := int(g.N) * config.VectorDim
	if len(g.Vectors) != expectedVec {
		return nil, fmt.Errorf("Vectors size mismatch: got %d, want %d", len(g.Vectors), expectedVec)
	}
	expectedConn0 := int(g.N) * g.M0
	if len(g.Conn0) != expectedConn0 {
		return nil, fmt.Errorf("Conn0 size mismatch: got %d, want %d", len(g.Conn0), expectedConn0)
	}
	if len(g.Conn0Cnt) != int(g.N) {
		return nil, fmt.Errorf("Conn0Cnt size mismatch: got %d, want %d", len(g.Conn0Cnt), g.N)
	}
	expectedLabels := int((g.N + 7) / 8)
	if len(g.Labels) != expectedLabels {
		return nil, fmt.Errorf("Labels size mismatch: got %d, want %d", len(g.Labels), expectedLabels)
	}

	// Retain the byte buffer so GC doesn't reclaim the storage behind the
	// unsafe-derived int32 slices.
	g.raw = data
	return g, nil
}

func bindBytes(dst *[]uint8, data []byte, off, length uint64) error {
	end := off + length
	if end > uint64(len(data)) {
		return errors.New("section out of bounds")
	}
	*dst = data[off:end]
	return nil
}

func bindInt32(dst *[]int32, data []byte, off, length uint64) error {
	end := off + length
	if end > uint64(len(data)) {
		return errors.New("section out of bounds")
	}
	if length%4 != 0 {
		return fmt.Errorf("int32 section length %d not multiple of 4", length)
	}
	if length == 0 {
		*dst = nil
		return nil
	}
	if off%4 != 0 {
		return fmt.Errorf("int32 section offset %d not 4-aligned", off)
	}
	*dst = unsafe.Slice((*int32)(unsafe.Pointer(&data[off])), length/4)
	return nil
}

// byteReader is a tiny io.Reader over a byte slice, avoiding the bytes pkg.
type byteSliceReader struct {
	b []byte
	o int
}

func byteReader(b []byte) *byteSliceReader { return &byteSliceReader{b: b} }
func (r *byteSliceReader) Read(p []byte) (int, error) {
	if r.o >= len(r.b) {
		return 0, io.EOF
	}
	n := copy(p, r.b[r.o:])
	r.o += n
	return n, nil
}
