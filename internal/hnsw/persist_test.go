package hnsw

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// TestPersist_RoundTrip builds a small graph, saves it, loads it back, and
// asserts every persisted field matches.
func TestPersist_RoundTrip(t *testing.T) {
	const N = uint32(150)
	const M = 4
	const M0 = 8

	b := NewBuilder(N, M, M0, 30, 17)
	rng := rand.New(rand.NewSource(31))
	for id := uint32(0); id < N; id++ {
		var v [config.VectorDim]uint8
		for d := range config.VectorDim {
			v[d] = uint8(rng.Intn(256))
		}
		label := LabelLegit
		if rng.Intn(4) == 0 {
			label = LabelFraud
		}
		b.Insert(id, &v, label)
	}
	b.Finalize()
	orig := b.G

	tmp := filepath.Join(t.TempDir(), "test.bin")
	if _, err := orig.Save(tmp); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load(tmp)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Header fields.
	if loaded.N != orig.N {
		t.Errorf("N: %d vs %d", loaded.N, orig.N)
	}
	if loaded.M != orig.M {
		t.Errorf("M: %d vs %d", loaded.M, orig.M)
	}
	if loaded.M0 != orig.M0 {
		t.Errorf("M0: %d vs %d", loaded.M0, orig.M0)
	}
	if loaded.EntryPoint != orig.EntryPoint {
		t.Errorf("EntryPoint: %d vs %d", loaded.EntryPoint, orig.EntryPoint)
	}
	if loaded.MaxLayer != orig.MaxLayer {
		t.Errorf("MaxLayer: %d vs %d", loaded.MaxLayer, orig.MaxLayer)
	}

	// Slice equality.
	if string(loaded.Vectors) != string(orig.Vectors) {
		t.Errorf("Vectors differ")
	}
	if string(loaded.Conn0Cnt) != string(orig.Conn0Cnt) {
		t.Errorf("Conn0Cnt differs")
	}
	if string(loaded.Labels) != string(orig.Labels) {
		t.Errorf("Labels differ")
	}
	if string(loaded.UpperCnt) != string(orig.UpperCnt) {
		t.Errorf("UpperCnt differs")
	}
	for i := range orig.Conn0 {
		if loaded.Conn0[i] != orig.Conn0[i] {
			t.Errorf("Conn0[%d]: %d vs %d", i, loaded.Conn0[i], orig.Conn0[i])
			break
		}
	}
	for i := range orig.UpperOff {
		if loaded.UpperOff[i] != orig.UpperOff[i] {
			t.Errorf("UpperOff[%d]: %d vs %d", i, loaded.UpperOff[i], orig.UpperOff[i])
		}
	}
	for i := range orig.UpperNodes {
		if loaded.UpperNodes[i] != orig.UpperNodes[i] {
			t.Errorf("UpperNodes[%d] differs", i)
			break
		}
	}
	for i := range orig.UpperConn {
		if loaded.UpperConn[i] != orig.UpperConn[i] {
			t.Errorf("UpperConn[%d]: %d vs %d", i, loaded.UpperConn[i], orig.UpperConn[i])
			break
		}
	}
}

// TestPersist_BadMagic ensures we error cleanly on a corrupt file.
func TestPersist_BadMagic(t *testing.T) {
	tmp := filepath.Join(t.TempDir(), "junk.bin")
	if err := os.WriteFile(tmp, make([]byte, headerSize+100), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Load(tmp)
	if err == nil {
		t.Fatal("expected error on bad magic, got nil")
	}
}
