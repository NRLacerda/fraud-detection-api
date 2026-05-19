// Package config holds the challenge's normalization constants and MCC risk
// table, baked in as Go literals so no JSON file I/O happens at runtime.
//
// Source: docs/DATASET.md (normalization.json + mcc_risk.json).
package config

const (
	MaxAmount            = 10000.0
	MaxInstallments      = 12.0
	AmountVsAvgRatio     = 10.0
	MaxMinutes           = 1440.0
	MaxKm                = 1000.0
	MaxTxCount24h        = 20.0
	MaxMerchantAvgAmount = 10000.0

	// MCCRiskDefault is used when merchant.mcc is not in MCCRisk.
	MCCRiskDefault = 0.5

	// VectorDim is the fixed dimensionality of the fraud vector.
	VectorDim = 14

	// KNN is the fixed neighbor count used for the fraud_score computation.
	KNN = 5

	// FraudThreshold: approved = fraud_score < FraudThreshold.
	FraudThreshold = 0.6
)

// MCCRisk maps a 4-digit MCC string to its risk weight. MCCs outside this
// table use MCCRiskDefault.
//
// Kept as a small map (10 entries) because lookups during vectorization are
// off the hot scoring path for builds (build-index doesn't use it — references
// are pre-vectorized) and once per request on the runtime path. A linear scan
// would also work; a map keeps the code readable without measurable cost at
// this size.
var MCCRisk = map[string]float32{
	"5411": 0.15,
	"5812": 0.30,
	"5912": 0.20,
	"5944": 0.45,
	"7801": 0.80,
	"7802": 0.75,
	"7995": 0.85,
	"4511": 0.35,
	"5311": 0.25,
	"5999": 0.50,
}
