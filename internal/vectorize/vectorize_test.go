package vectorize

import (
	"math"
	"testing"
	"time"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// approxEq compares to ~4 decimal places — same precision the docs print.
func approxEq(t *testing.T, idx int, got, want float32) {
	t.Helper()
	if math.Abs(float64(got)-float64(want)) > 1e-4 {
		t.Errorf("dim[%d]: got %v, want %v", idx, got, want)
	}
}

func mustTime(t *testing.T, s string) time.Time {
	t.Helper()
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return tm
}

// TestVectorize_LegitExample is the worked example from DETECTION_RULES.md.
// Expected vector:
//
//	[0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1,
//	 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
func TestVectorize_LegitExample(t *testing.T) {
	r := &Request{
		Transaction: Transaction{
			Amount:       41.12,
			Installments: 2,
			RequestedAt:  mustTime(t, "2026-03-11T18:45:53Z"),
		},
		Customer: Customer{
			AvgAmount:      82.24,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: Merchant{
			ID:        "MERC-016",
			MCC:       "5411",
			AvgAmount: 60.25,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  29.23,
		},
		HasLastTransaction: false,
	}
	var v [config.VectorDim]float32
	Vectorize(r, &v)

	want := [config.VectorDim]float32{
		0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1,
		0.0292, 0.15, 0, 1, 0, 0.15, 0.006,
	}
	for i := range config.VectorDim {
		approxEq(t, i, v[i], want[i])
	}
}

// TestVectorize_FraudExample is the fraud worked example from
// DETECTION_RULES.md.
//
// Expected vector:
//
//	[0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1,
//	 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
func TestVectorize_FraudExample(t *testing.T) {
	r := &Request{
		Transaction: Transaction{
			Amount:       9505.97,
			Installments: 10,
			RequestedAt:  mustTime(t, "2026-03-14T05:15:12Z"),
		},
		Customer: Customer{
			AvgAmount:      81.28,
			TxCount24h:     20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: Merchant{
			ID:        "MERC-068",
			MCC:       "7802",
			AvgAmount: 54.86,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  952.27,
		},
		HasLastTransaction: false,
	}
	var v [config.VectorDim]float32
	Vectorize(r, &v)

	want := [config.VectorDim]float32{
		0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1,
		0.9523, 1.0, 0, 1, 1, 0.75, 0.0055,
	}
	for i := range config.VectorDim {
		approxEq(t, i, v[i], want[i])
	}
}

// TestVectorize_WithLastTransaction confirms minutes_since_last_tx and
// km_from_last_tx are computed (not -1) when HasLastTransaction is true.
//
// Built from the API.md example payload:
//
//	requested_at = 2026-03-11T20:23:35Z
//	last_transaction.timestamp = 2026-03-11T14:58:35Z
//	→ diff = 5h25m = 325 min → dim5 = 325/1440 = 0.2257
//	km_from_current = 18.8626479774 → dim6 = 0.01886
func TestVectorize_WithLastTransaction(t *testing.T) {
	r := &Request{
		Transaction: Transaction{
			Amount:       384.88,
			Installments: 3,
			RequestedAt:  mustTime(t, "2026-03-11T20:23:35Z"),
		},
		Customer: Customer{
			AvgAmount:      769.76,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-009", "MERC-001", "MERC-001"},
		},
		Merchant: Merchant{
			ID:        "MERC-001",
			MCC:       "5912",
			AvgAmount: 298.95,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  13.7090520965,
		},
		LastTransaction: LastTransaction{
			Timestamp:     mustTime(t, "2026-03-11T14:58:35Z"),
			KmFromCurrent: 18.8626479774,
		},
		HasLastTransaction: true,
	}
	var v [config.VectorDim]float32
	Vectorize(r, &v)

	approxEq(t, 5, v[5], 0.2257) // 325 / 1440
	approxEq(t, 6, v[6], 0.01886)
	approxEq(t, 11, v[11], 0) // MERC-001 IS in known_merchants → known → 0
	approxEq(t, 12, v[12], 0.20) // 5912 → 0.20

	if v[5] == -1 || v[6] == -1 {
		t.Errorf("dims 5/6 should not be sentinel when HasLastTransaction is true")
	}
}

// TestVectorize_ClampUpper exercises the upper clamp on multiple dims.
func TestVectorize_ClampUpper(t *testing.T) {
	r := &Request{
		Transaction: Transaction{
			Amount:       50000, // >> max_amount (10000)
			Installments: 24,    // >> max_installments (12)
			RequestedAt:  mustTime(t, "2026-03-11T00:00:00Z"),
		},
		Customer: Customer{
			AvgAmount:  100,
			TxCount24h: 100, // >> max (20)
		},
		Merchant: Merchant{
			ID:        "MERC-X",
			MCC:       "9999", // not in table → default 0.5
			AvgAmount: 99999,  // >> max
		},
		Terminal: Terminal{
			KmFromHome: 99999, // >> max_km
		},
	}
	var v [config.VectorDim]float32
	Vectorize(r, &v)

	approxEq(t, 0, v[0], 1.0)
	approxEq(t, 1, v[1], 1.0)
	approxEq(t, 2, v[2], 1.0)
	approxEq(t, 7, v[7], 1.0)
	approxEq(t, 8, v[8], 1.0)
	approxEq(t, 12, v[12], config.MCCRiskDefault)
	approxEq(t, 13, v[13], 1.0)
}

// TestVectorize_ZeroAvgAmount: customer.avg_amount=0 should give dim2=1.0
// rather than NaN or +Inf.
func TestVectorize_ZeroAvgAmount(t *testing.T) {
	r := &Request{
		Transaction: Transaction{
			Amount:       100,
			RequestedAt:  mustTime(t, "2026-03-11T00:00:00Z"),
		},
		Customer: Customer{AvgAmount: 0},
	}
	var v [config.VectorDim]float32
	Vectorize(r, &v)
	approxEq(t, 2, v[2], 1.0)
}

// TestVectorize_ZeroAlloc verifies Vectorize itself does not allocate.
// Pre-populate the slice once; the test loop never touches the heap.
func TestVectorize_ZeroAlloc(t *testing.T) {
	r := &Request{
		Customer: Customer{KnownMerchants: []string{"MERC-001", "MERC-002"}},
		Merchant: Merchant{ID: "MERC-001", MCC: "5411"},
		Transaction: Transaction{
			Amount:       100,
			Installments: 3,
			RequestedAt:  mustTime(t, "2026-03-11T12:00:00Z"),
		},
	}
	var v [config.VectorDim]float32

	allocs := testing.AllocsPerRun(1000, func() {
		Vectorize(r, &v)
	})
	if allocs != 0 {
		t.Errorf("Vectorize allocated %v times per call, want 0", allocs)
	}
}

func BenchmarkVectorize(b *testing.B) {
	r := &Request{
		Customer: Customer{
			AvgAmount:      82.24,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: Merchant{ID: "MERC-016", MCC: "5411", AvgAmount: 60.25},
		Terminal: Terminal{CardPresent: true, KmFromHome: 29.23},
		Transaction: Transaction{
			Amount:       41.12,
			Installments: 2,
			RequestedAt:  time.Date(2026, 3, 11, 18, 45, 53, 0, time.UTC),
		},
	}
	var v [config.VectorDim]float32
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Vectorize(r, &v)
	}
}
