//go:build unix

package hnsw

import (
	"fmt"
	"os"
	"syscall"
)

// LoadMmap memory-maps an hnsw.bin file read-only and returns a Graph whose
// slices alias the mapped bytes. On Linux containers this keeps the large
// index file out of the Go heap and lets both API instances share file-backed
// pages through the kernel page cache.
func LoadMmap(path string) (*Graph, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() <= 0 {
		return nil, fmt.Errorf("hnsw: empty index file %s", path)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}

	g, err := loadFromBytes(data)
	if err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}
	return g, nil
}
