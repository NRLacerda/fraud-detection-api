//go:build !unix

package hnsw

// LoadMmap falls back to Load on platforms where the standard library does
// not expose mmap. The production Docker image is Linux and uses the mmap
// implementation.
func LoadMmap(path string) (*Graph, error) {
	return Load(path)
}
