// Package vectorize turns a parsed fraud-score request into a 14-d float32
// vector, following the formulas in docs/DETECTION_RULES.md.
//
// Vectorize is intentionally caller-owned: the output array lives on the
// stack (or in a pooled request struct) of the caller, so this function
// performs zero heap allocations.
package vectorize

import (
	"time"

	"github.com/nrlacerda/fraud-detection-api/internal/config"
)

// Transaction is the inner "transaction" object of the payload.
type Transaction struct {
	Amount       float32   `json:"amount"`
	Installments int32     `json:"installments"`
	RequestedAt  time.Time `json:"requested_at"`
}

// Customer is the inner "customer" object of the payload.
//
// KnownMerchants is a slice because JSON arrays are decoded into slices;
// when a request is reused via a pool, the slice is reset to len=0 and the
// underlying array is retained to avoid reallocation.
type Customer struct {
	AvgAmount      float32  `json:"avg_amount"`
	TxCount24h     int32    `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

// Merchant is the inner "merchant" object of the payload.
type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

// Terminal is the inner "terminal" object of the payload.
type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float32 `json:"km_from_home"`
}

// LastTransaction is the inner "last_transaction" object — may be null in the
// JSON. Nullness is signaled by Request.HasLastTransaction, set by the
// handler during unmarshal (we cannot use *LastTransaction because that
// would allocate on every present-last-transaction request).
type LastTransaction struct {
	Timestamp     time.Time `json:"timestamp"`
	KmFromCurrent float32   `json:"km_from_current"`
}

// Request is the full /fraud-score payload, decoded.
//
// HasLastTransaction is not a JSON field; it is set to true by the unmarshal
// path when last_transaction is a non-null object, false when null.
type Request struct {
	ID                 string          `json:"id"`
	Transaction        Transaction     `json:"transaction"`
	Customer           Customer        `json:"customer"`
	Merchant           Merchant        `json:"merchant"`
	Terminal           Terminal        `json:"terminal"`
	LastTransaction    LastTransaction `json:"last_transaction"`
	HasLastTransaction bool            `json:"-"`
}

// Reset prepares a pooled Request for reuse — keeps the KnownMerchants slice
// capacity, zeros everything else.
func (r *Request) Reset() {
	km := r.Customer.KnownMerchants[:0]
	*r = Request{}
	r.Customer.KnownMerchants = km
}

// Vectorize turns r into the 14-d vector defined in DETECTION_RULES.md and
// writes it into *out. Zero allocations on the hot path.
//
// Indices 5 and 6 receive the sentinel value -1 when r.HasLastTransaction is
// false (i.e., the payload had `last_transaction: null`).
func Vectorize(r *Request, out *[config.VectorDim]float32) {
	out[0] = clamp(r.Transaction.Amount / config.MaxAmount)
	out[1] = clamp(float32(r.Transaction.Installments) / config.MaxInstallments)

	// amount_vs_avg: customer.avg_amount can be zero in degenerate payloads;
	// divide-by-zero produces +Inf and clamp pins it to 1.0, which is the
	// correct semantic (massively-above-average spend).
	if r.Customer.AvgAmount > 0 {
		out[2] = clamp((r.Transaction.Amount / r.Customer.AvgAmount) / config.AmountVsAvgRatio)
	} else {
		out[2] = 1.0
	}

	// requested_at is parsed as UTC because the contract is "...Z" RFC3339.
	out[3] = float32(r.Transaction.RequestedAt.Hour()) / 23.0

	// Go's Weekday: Sunday=0..Saturday=6. The challenge wants Mon=0..Sun=6.
	w := (int(r.Transaction.RequestedAt.Weekday()) + 6) % 7
	out[4] = float32(w) / 6.0

	if r.HasLastTransaction {
		minutes := r.Transaction.RequestedAt.Sub(r.LastTransaction.Timestamp).Minutes()
		out[5] = clamp(float32(minutes) / config.MaxMinutes)
		out[6] = clamp(r.LastTransaction.KmFromCurrent / config.MaxKm)
	} else {
		out[5] = -1
		out[6] = -1
	}

	out[7] = clamp(r.Terminal.KmFromHome / config.MaxKm)
	out[8] = clamp(float32(r.Customer.TxCount24h) / config.MaxTxCount24h)
	out[9] = boolToFloat(r.Terminal.IsOnline)
	out[10] = boolToFloat(r.Terminal.CardPresent)
	out[11] = boolToFloat(!contains(r.Customer.KnownMerchants, r.Merchant.ID))

	if risk, ok := config.MCCRisk[r.Merchant.MCC]; ok {
		out[12] = risk
	} else {
		out[12] = config.MCCRiskDefault
	}

	out[13] = clamp(r.Merchant.AvgAmount / config.MaxMerchantAvgAmount)
}

func clamp(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func boolToFloat(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

// contains is a linear scan over a small slice (typical len < 20). A map
// lookup would allocate during unmarshal, which is the more expensive end of
// the tradeoff.
func contains(xs []string, target string) bool {
	for i := range xs {
		if xs[i] == target {
			return true
		}
	}
	return false
}
