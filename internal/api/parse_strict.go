package api

import (
	"errors"
	"strconv"
	"time"
	"unsafe"

	"github.com/nrlacerda/fraud-detection-api/internal/vectorize"
)

var errBadPayload = errors.New("bad fraud-score payload")

func expectKeyAt(body []byte, p int, key string) (int, error) {
	p = skipSpaces(body, p)
	if !hasPrefix(body[p:], key) {
		return 0, errBadPayload
	}
	p = skipSpaces(body, p+len(key))
	if p >= len(body) || body[p] != ':' {
		return 0, errBadPayload
	}
	return skipSpaces(body, p+1), nil
}

func expectByte(body []byte, p int, want byte) (int, error) {
	p = skipSpaces(body, p)
	if p >= len(body) || body[p] != want {
		return 0, errBadPayload
	}
	return skipSpaces(body, p+1), nil
}

func expectComma(body []byte, p int) (int, error) {
	return expectByte(body, p, ',')
}

func readStringAt(body []byte, p int) (string, int, error) {
	p = skipSpaces(body, p)
	if p >= len(body) || body[p] != '"' {
		return "", 0, errBadPayload
	}
	start := p + 1
	for i := start; i < len(body); i++ {
		if body[i] == '\\' {
			return "", 0, errBadPayload
		}
		if body[i] == '"' {
			return unsafe.String(unsafe.SliceData(body[start:i]), i-start), skipSpaces(body, i+1), nil
		}
	}
	return "", 0, errBadPayload
}

func readStringArrayAt(body []byte, p int, dst []string) ([]string, int, error) {
	p = skipSpaces(body, p)
	if p >= len(body) || body[p] != '[' {
		return nil, 0, errBadPayload
	}
	p++
	for {
		p = skipSpaces(body, p)
		if p >= len(body) {
			return nil, 0, errBadPayload
		}
		if body[p] == ']' {
			return dst, skipSpaces(body, p+1), nil
		}
		var s string
		var err error
		s, p, err = readStringAt(body, p)
		if err != nil {
			return nil, 0, err
		}
		dst = append(dst, s)
		p = skipSpaces(body, p)
		if p >= len(body) {
			return nil, 0, errBadPayload
		}
		if body[p] == ',' {
			p++
			continue
		}
		if body[p] == ']' {
			return dst, skipSpaces(body, p+1), nil
		}
		return nil, 0, errBadPayload
	}
}

func readFloat32At(body []byte, p int) (float32, int, error) {
	p = skipSpaces(body, p)
	sign := float64(1)
	if p < len(body) && body[p] == '-' {
		sign = -1
		p++
	}
	if p >= len(body) || body[p] < '0' || body[p] > '9' {
		return 0, 0, errBadPayload
	}
	var n float64
	for p < len(body) && body[p] >= '0' && body[p] <= '9' {
		n = n*10 + float64(body[p]-'0')
		p++
	}
	if p < len(body) && body[p] == '.' {
		p++
		scale := float64(0.1)
		for p < len(body) && body[p] >= '0' && body[p] <= '9' {
			n += float64(body[p]-'0') * scale
			scale *= 0.1
			p++
		}
	}
	return float32(sign * n), skipSpaces(body, p), nil
}

func readInt32At(body []byte, p int) (int32, int, error) {
	p = skipSpaces(body, p)
	if p >= len(body) || body[p] < '0' || body[p] > '9' {
		return 0, 0, errBadPayload
	}
	var n int32
	for p < len(body) && body[p] >= '0' && body[p] <= '9' {
		n = n*10 + int32(body[p]-'0')
		p++
	}
	return n, skipSpaces(body, p), nil
}

func readBoolAt(body []byte, p int) (bool, int, error) {
	p = skipSpaces(body, p)
	if hasPrefix(body[p:], "true") {
		return true, skipSpaces(body, p+len("true")), nil
	}
	if hasPrefix(body[p:], "false") {
		return false, skipSpaces(body, p+len("false")), nil
	}
	return false, 0, errBadPayload
}

func readTimeAt(body []byte, p int) (time.Time, int, error) {
	s, next, err := readStringAt(body, p)
	if err != nil {
		return time.Time{}, 0, err
	}
	t, err := parseFixedUTC(s)
	if err != nil {
		return time.Time{}, 0, err
	}
	return t, next, nil
}

func parseStrictRequest(body []byte, r *vectorize.Request) error {
	p := skipSpaces(body, 0)
	var err error
	if p, err = expectByte(body, p, '{'); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"id"`); err != nil {
		return err
	}
	if r.ID, p, err = readStringAt(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"transaction"`); err != nil {
		return err
	}
	if p, err = expectByte(body, p, '{'); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"amount"`); err != nil {
		return err
	}
	if r.Transaction.Amount, p, err = readFloat32At(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"installments"`); err != nil {
		return err
	}
	if r.Transaction.Installments, p, err = readInt32At(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"requested_at"`); err != nil {
		return err
	}
	if r.Transaction.RequestedAt, p, err = readTimeAt(body, p); err != nil {
		return err
	}
	if p, err = expectByte(body, p, '}'); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"customer"`); err != nil {
		return err
	}
	if p, err = expectByte(body, p, '{'); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"avg_amount"`); err != nil {
		return err
	}
	if r.Customer.AvgAmount, p, err = readFloat32At(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"tx_count_24h"`); err != nil {
		return err
	}
	if r.Customer.TxCount24h, p, err = readInt32At(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"known_merchants"`); err != nil {
		return err
	}
	if r.Customer.KnownMerchants, p, err = readStringArrayAt(body, p, r.Customer.KnownMerchants[:0]); err != nil {
		return err
	}
	if p, err = expectByte(body, p, '}'); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"merchant"`); err != nil {
		return err
	}
	if p, err = expectByte(body, p, '{'); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"id"`); err != nil {
		return err
	}
	if r.Merchant.ID, p, err = readStringAt(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"mcc"`); err != nil {
		return err
	}
	if r.Merchant.MCC, p, err = readStringAt(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"avg_amount"`); err != nil {
		return err
	}
	if r.Merchant.AvgAmount, p, err = readFloat32At(body, p); err != nil {
		return err
	}
	if p, err = expectByte(body, p, '}'); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"terminal"`); err != nil {
		return err
	}
	if p, err = expectByte(body, p, '{'); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"is_online"`); err != nil {
		return err
	}
	if r.Terminal.IsOnline, p, err = readBoolAt(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"card_present"`); err != nil {
		return err
	}
	if r.Terminal.CardPresent, p, err = readBoolAt(body, p); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"km_from_home"`); err != nil {
		return err
	}
	if r.Terminal.KmFromHome, p, err = readFloat32At(body, p); err != nil {
		return err
	}
	if p, err = expectByte(body, p, '}'); err != nil {
		return err
	}
	if p, err = expectComma(body, p); err != nil {
		return err
	}
	if p, err = expectKeyAt(body, p, `"last_transaction"`); err != nil {
		return err
	}
	p = skipSpaces(body, p)
	if hasPrefix(body[p:], "null") {
		r.HasLastTransaction = false
		p += len("null")
	} else {
		r.HasLastTransaction = true
		if p, err = expectByte(body, p, '{'); err != nil {
			return err
		}
		if p, err = expectKeyAt(body, p, `"timestamp"`); err != nil {
			return err
		}
		if r.LastTransaction.Timestamp, p, err = readTimeAt(body, p); err != nil {
			return err
		}
		if p, err = expectComma(body, p); err != nil {
			return err
		}
		if p, err = expectKeyAt(body, p, `"km_from_current"`); err != nil {
			return err
		}
		if r.LastTransaction.KmFromCurrent, p, err = readFloat32At(body, p); err != nil {
			return err
		}
		if p, err = expectByte(body, p, '}'); err != nil {
			return err
		}
	}
	_, err = expectByte(body, p, '}')
	return err
}

func findObject(body []byte, key string) ([]byte, error) {
	pos := indexKey(body, key)
	if pos < 0 {
		return nil, errBadPayload
	}
	p := skipSpaces(body, pos+len(key))
	if p >= len(body) || body[p] != ':' {
		return nil, errBadPayload
	}
	p = skipSpaces(body, p+1)
	return objectAt(body, p)
}

func objectAt(body []byte, p int) ([]byte, error) {
	if p >= len(body) || body[p] != '{' {
		return nil, errBadPayload
	}
	depth := 0
	inString := false
	for i := p; i < len(body); i++ {
		c := body[i]
		if inString {
			if c == '\\' {
				i++
				continue
			}
			if c == '"' {
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return body[p : i+1], nil
			}
		}
	}
	return nil, errBadPayload
}

func findString(body []byte, key string) (string, error) {
	p, err := valueStart(body, key)
	if err != nil {
		return "", err
	}
	if p >= len(body) || body[p] != '"' {
		return "", errBadPayload
	}
	start := p + 1
	for i := start; i < len(body); i++ {
		if body[i] == '\\' {
			return "", errBadPayload
		}
		if body[i] == '"' {
			return unsafe.String(unsafe.SliceData(body[start:i]), i-start), nil
		}
	}
	return "", errBadPayload
}

func findStringArray(body []byte, key string, dst []string) ([]string, error) {
	p, err := valueStart(body, key)
	if err != nil {
		return nil, err
	}
	if p >= len(body) || body[p] != '[' {
		return nil, errBadPayload
	}
	p++
	for {
		p = skipSpaces(body, p)
		if p >= len(body) {
			return nil, errBadPayload
		}
		if body[p] == ']' {
			return dst, nil
		}
		if body[p] != '"' {
			return nil, errBadPayload
		}
		start := p + 1
		i := start
		for ; i < len(body); i++ {
			if body[i] == '\\' {
				return nil, errBadPayload
			}
			if body[i] == '"' {
				break
			}
		}
		if i >= len(body) {
			return nil, errBadPayload
		}
		dst = append(dst, unsafe.String(unsafe.SliceData(body[start:i]), i-start))
		p = skipSpaces(body, i+1)
		if p >= len(body) {
			return nil, errBadPayload
		}
		if body[p] == ',' {
			p++
			continue
		}
		if body[p] == ']' {
			return dst, nil
		}
		return nil, errBadPayload
	}
}

func findFloat32(body []byte, key string) (float32, error) {
	p, err := valueStart(body, key)
	if err != nil {
		return 0, err
	}
	end := numberEnd(body, p)
	if end == p {
		return 0, errBadPayload
	}
	v, err := strconv.ParseFloat(unsafe.String(unsafe.SliceData(body[p:end]), end-p), 32)
	if err != nil {
		return 0, errBadPayload
	}
	return float32(v), nil
}

func findInt32(body []byte, key string) (int32, error) {
	p, err := valueStart(body, key)
	if err != nil {
		return 0, err
	}
	end := numberEnd(body, p)
	if end == p {
		return 0, errBadPayload
	}
	v, err := strconv.ParseInt(unsafe.String(unsafe.SliceData(body[p:end]), end-p), 10, 32)
	if err != nil {
		return 0, errBadPayload
	}
	return int32(v), nil
}

func findBool(body []byte, key string) (bool, error) {
	p, err := valueStart(body, key)
	if err != nil {
		return false, err
	}
	if hasPrefix(body[p:], "true") {
		return true, nil
	}
	if hasPrefix(body[p:], "false") {
		return false, nil
	}
	return false, errBadPayload
}

func findTime(body []byte, key string) (time.Time, error) {
	s, err := findString(body, key)
	if err != nil {
		return time.Time{}, err
	}
	return parseFixedUTC(s)
}

func valueStart(body []byte, key string) (int, error) {
	pos := indexKey(body, key)
	if pos < 0 {
		return 0, errBadPayload
	}
	p := skipSpaces(body, pos+len(key))
	if p >= len(body) || body[p] != ':' {
		return 0, errBadPayload
	}
	return skipSpaces(body, p+1), nil
}

func indexKey(body []byte, key string) int {
	k := []byte(key)
outer:
	for i := 0; i+len(k) <= len(body); i++ {
		for j := range k {
			if body[i+j] != k[j] {
				continue outer
			}
		}
		return i
	}
	return -1
}

func numberEnd(body []byte, p int) int {
	for p < len(body) {
		c := body[p]
		if (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' {
			p++
			continue
		}
		break
	}
	return p
}

func skipSpaces(body []byte, p int) int {
	for p < len(body) {
		switch body[p] {
		case ' ', '\n', '\r', '\t':
			p++
		default:
			return p
		}
	}
	return p
}

func hasPrefix(body []byte, s string) bool {
	if len(body) < len(s) {
		return false
	}
	for i := range s {
		if body[i] != s[i] {
			return false
		}
	}
	return true
}

func parseFixedUTC(s string) (time.Time, error) {
	if len(s) != len("2006-01-02T15:04:05Z") ||
		s[4] != '-' || s[7] != '-' || s[10] != 'T' ||
		s[13] != ':' || s[16] != ':' || s[19] != 'Z' {
		return time.Time{}, errBadPayload
	}
	year, ok := atoi4(s[0:4])
	if !ok {
		return time.Time{}, errBadPayload
	}
	month, ok := atoi2(s[5:7])
	if !ok {
		return time.Time{}, errBadPayload
	}
	day, ok := atoi2(s[8:10])
	if !ok {
		return time.Time{}, errBadPayload
	}
	hour, ok := atoi2(s[11:13])
	if !ok {
		return time.Time{}, errBadPayload
	}
	minute, ok := atoi2(s[14:16])
	if !ok {
		return time.Time{}, errBadPayload
	}
	second, ok := atoi2(s[17:19])
	if !ok {
		return time.Time{}, errBadPayload
	}
	if month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 59 {
		return time.Time{}, errBadPayload
	}
	return time.Date(year, time.Month(month), day, hour, minute, second, 0, time.UTC), nil
}

func atoi2(s string) (int, bool) {
	if len(s) != 2 || s[0] < '0' || s[0] > '9' || s[1] < '0' || s[1] > '9' {
		return 0, false
	}
	return int(s[0]-'0')*10 + int(s[1]-'0'), true
}

func atoi4(s string) (int, bool) {
	if len(s) != 4 {
		return 0, false
	}
	n := 0
	for i := 0; i < 4; i++ {
		if s[i] < '0' || s[i] > '9' {
			return 0, false
		}
		n = n*10 + int(s[i]-'0')
	}
	return n, true
}
